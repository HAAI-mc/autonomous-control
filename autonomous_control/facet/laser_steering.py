"""Laser steering solenoid alignment control script.

This module defines a BAX optimization routine for minimizing the kick from
the solenoid using Xopt.
"""

import os
import time
import logging
from xopt import Xopt, Evaluator, VOCS

import torch
from botorch.acquisition.multi_objective.analytic import (
    MultiObjectiveAnalyticAcquisitionFunction,
)
from botorch.models.model import Model, ModelList
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

import pickle
from copy import deepcopy
from typing import Dict, List

from botorch.models import ModelListGP, SingleTaskGP
from pydantic import (
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from xopt.errors import VOCSError
from xopt.generators.bayesian.bax.algorithms import Algorithm
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.turbo import EntropyTurboController, SafetyTurboController

from bax_algorithms.solenoid_alignment import PathwiseSolenoidAlignment
from bax_algorithms.pathwise.optimize import DifferentialEvolution
from bax_algorithms.utils import get_bax_mean_prediction, tuning_input_tensor_to_dict
from bax_algorithms.visualize import visualize_virtual_measurement_result

from xopt.generators.bayesian.bax_generator import BaxGenerator

from autonomous_control.facet.optimization_utils import safe_evaluate_best_point

from xopt.numerical_optimizer import LBFGSOptimizer

import epics

logger = logging.getLogger("solenoid_alignment_opt")



def optimize_laser_steering(
    env,
    dump_location=None,
    initial_random_evaluations=2,
    n_steps=30,
    mirror_range_fraction=0.01,
    solenoid_range_fraction=0.03,
):
    """Run BAX optimization for solenoid alignment.

    Parameters
    ----------
    env : Any
        Injector control environment that provides variable and observable
        interfaces used by this routine.
    dump_location : str or pathlib.Path, optional
        Xopt dump file path.
    initial_random_evaluations : int, optional
        Number of random warm-start evaluations.
    n_steps : int, optional
        Number of BAX optimization iterations.
    mirror_range_fraction : float, optional
        Fractional +/- range for mirror adjustments.
    solenoid_range_fraction : float, optional
        Fractional +/- range for solenoid adjustments.

    Returns
    -------
    Xopt
        Configured and executed Xopt instance containing optimization data.
    """
    run_start_time = time.time()

    output_directory = os.path.dirname(dump_location) if dump_location else "."

    # TODO: check data folder exists

    logger.info("Starting BAX solenoid alignment optimization.")
    logger.info(
        "Solenoid alignment config: initial_random_evaluations=%d n_steps=%d dump_location=%s",
        initial_random_evaluations,
        n_steps,
        dump_location,
    )
    env.save_directory = os.path.join(output_directory, "data/")
    logger.debug(
        "Configured solenoid alignment optimization with save_directory=%s dump_location=%s output_directory=%s",
        env.save_directory,
        dump_location,
        output_directory,
    )

    def evaluate(inputs):
        """Evaluate solenoid alignment observables at a candidate setting.

        Parameters
        ----------
        inputs : dict[str, float]
            Mapping of control variable names to values.

        Returns
        -------
        dict[str, float]
            Observable dictionary returned by the control environment.
        """
        logger.debug("Evaluating alignment settings: %s", inputs)

        # env.set_variables(inputs)
        for key, val in inputs.items():
            epics.caput(key, val)

        time.sleep(2.0) # wait for steering to settle

        # Get the output from the environment
        # note that output will contain many results, not just emittance_x
        # see FACET-II injector badger environment for details
        output = env.get_observables(
            ["BPMS:IN10:221:X", "BPMS:IN10:221:Y", "BPMS:IN10:221:TMIT"]
        )
        output["BPMS:IN10:221:TMIT"] *= 1e-10
        logger.debug("Evaluation output keys: %s", list(output.keys()))

        return output

    # set up VOCS
    meas_param = "SOLN:IN10:121:BCTRL"
    variable_names = [
        "MIRR:LT10:770:M2_MOTR_H",
        "MIRR:LT10:770:M2_MOTR_V",
        "SOLN:IN10:121:BCTRL",
    ]
    init_settings = {var_name: epics.caget(var_name) for var_name in variable_names}
    variables = {
        var_name: sorted(
            [
                init_settings[var_name] * (1 - mirror_range_fraction),
                init_settings[var_name] * (1 + mirror_range_fraction),
            ]
        )
        for var_name in variable_names[:2]
    }
    variables["SOLN:IN10:121:BCTRL"] = [
        init_settings["SOLN:IN10:121:BCTRL"] * (1 - solenoid_range_fraction),
        init_settings["SOLN:IN10:121:BCTRL"] * (1 + solenoid_range_fraction),
    ]
    # construct vocs
    vocs = VOCS(
        variables=variables,
        observables=["BPMS:IN10:221:X", "BPMS:IN10:221:Y", "BPMS:IN10:221:TMIT"],
    )

    meas_dim = sorted(vocs.variable_names).index(meas_param)

    # Prepare Algorithm
    algo_kwargs = {
        "x_key": "BPMS:IN10:221:X",
        "y_key": "BPMS:IN10:221:Y",
        "n_samples": 2,
        "meas_dim": meas_dim,
        "n_steps_measurement_param": 5,
        "observable_names_ordered": ["BPMS:IN10:221:X", "BPMS:IN10:221:Y"],
        "optimizer": DifferentialEvolution(minimize=True, maxiter=10, verbose=False),
        "n_batch": 5,
    }
    algo = PathwiseSolenoidAlignment(**algo_kwargs)

    numerical_optimizer = LBFGSOptimizer(n_restarts=10, max_time=1)

    # construct BAX generator
    generator = BaxGenerator(
        vocs=vocs,
        numerical_optimizer=numerical_optimizer,
        algorithm=algo,
    )

    generator.gp_constructor.use_low_noise_prior = False
    # construct evaluator
    evaluator = Evaluator(function=evaluate)

    ts = int(time.time())

    # construct Xopt optimizer
    X = Xopt(
        vocs=vocs,
        evaluator=evaluator,
        generator=generator,
        dump_file=dump_location,
    )
    logger.debug("Created Xopt object.")

    # evaluate the current point and two random points
    logger.info(
        "Running initial evaluations (current + %d random points).",
        initial_random_evaluations,
    )
    X.evaluate_data(env.get_variables(X.vocs.variable_names))
    X.random_evaluate(initial_random_evaluations)

    for i in range(n_steps):
        logger.debug("Running optimization step %d/%d", i + 1, n_steps)
        X.step()

    mean_optimizer = DifferentialEvolution(
        minimize=True, popsize=100, maxiter=100, verbose=True
    )
    x_tuning = get_bax_mean_prediction(X.generator, mean_optimizer)
    x_tuning_dict = tuning_input_tensor_to_dict(X.generator, x_tuning)
    best = x_tuning_dict | {"SOLN:IN10:121:BCTRL": init_settings["SOLN:IN10:121:BCTRL"]}

    safe_evaluate_best_point(
        X,
        logger,
        best_inputs=best,
        context="solenoid alignment optimization",
    )
    logger.info("Completed solenoid alignment optimization.")

    fig, ax = visualize_virtual_measurement_result(
        X.generator,
        variable_names=["MIRR:LT10:770:M2_MOTR_H", "MIRR:LT10:770:M2_MOTR_V"],
        reference_point=best,
        n_grid=10,
        n_samples=1000,
        result_keys=["objective", "misalignment_x", "misalignment_y"],
    )

    fig.savefig(os.path.join(output_directory, f"solenoid_alignment_opt_{ts}.png"))
    logger.info(
        "Solenoid alignment summary: evaluations=%d dump_file=%s png=%s duration=%.2f s",
        len(X.data),
        dump_location,
        os.path.join(output_directory, f"solenoid_alignment_opt_{ts}.png"),
        time.time() - run_start_time,
    )
    return X
