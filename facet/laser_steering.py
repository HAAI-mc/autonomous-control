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

try:
    from facet.optimization_utils import merge_config, safe_evaluate_best_point
except ImportError:
    from optimization_utils import merge_config, safe_evaluate_best_point

from xopt.numerical_optimizer import LBFGSOptimizer

import epics

logger = logging.getLogger("solenoid_alignment_opt")


class ModelListExpectedInformationGain(MultiObjectiveAnalyticAcquisitionFunction):
    r"""Single-outcome expected information gain for independent
        multi-output (ModelListGP) models.

    Example:
        >>> model1 = SingleTaskGP(train_X, train_Y1)
        >>> model2 = SingleTaskGP(train_X, train_Y2)
        >>> model = ModelList(model1, model2)
        >>> EIG = ExpectedInformationGain(model, algo)
        >>> eig = EIG(test_X)

        Parameters
        ----------
            model: A fitted independent multi-output (ModelList) model.
    """

    def __init__(self, model: Model, algorithm: Algorithm, bounds: Tensor) -> None:
        super().__init__(model=model)
        self.algorithm = algorithm

        # get sample-wise algorithm execution (BAX) results
        (
            self.xs_exe,
            self.ys_exe,
            self.algorithm_results,
        ) = self.algorithm.get_execution_paths(self.model, bounds)

        # Need to call the model on some data before we can condition_on_observations
        self.model.posterior(self.xs_exe[:1, 0:1, 0:])

        # construct a batch of size n_samples fantasy models,
        # where each fantasy model is produced by taking the model
        # at the current iteration and conditioning it
        # on one of the sampled execution path subsequences:
        xs_exe_list = [self.xs_exe for i in range(len(model.models))]
        ys_exe_list = [
            torch.index_select(self.ys_exe, dim=-1, index=torch.tensor([i]))
            for i in range(len(model.models))
        ]
        fantasy_models = [
            m.condition_on_observations(x, y)
            for m, x, y in zip(model.models, xs_exe_list, ys_exe_list)
        ]
        self.fantasy_models = ModelList(*fantasy_models)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Information Gain on the candidate set X.

        Parameters
        ----------
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Information Gain is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns
        -------
            A `(b1 x ... bk)`-dim tensor of Expected Information Gain values at the
            given design points `X`.
        """

        # Use the current & fantasy models to compute a
        # Monte-Carlo estimate of the Expected Information Gain:
        # see https://arxiv.org/pdf/2104.09460.pdf:
        # eq (4) and the last sentence of page 7)

        # calculcate the variance of the posterior for each input x
        post = self.model.posterior(X)
        var_post = post.variance

        # calculcate the variance of the fantasy posteriors
        fantasy_posts = self.fantasy_models.posterior(
            (
                X.reshape(*X.shape[:-2], 1, *X.shape[-2:]).expand(
                    *X.shape[:-2], self.xs_exe.shape[0], *X.shape[-2:]
                )
            )
        )
        var_fantasy_posts = fantasy_posts.variance

        # calculate Shannon entropy for posterior given the current data
        h_current = 0.5 * torch.log(2 * torch.pi * var_post) + 0.5
        # sum the entropies from each independent posterior in the ModelList
        h_current_scalar = torch.sum(h_current, dim=-1)

        # calculate the Shannon entropy for the fantasy posteriors
        h_fantasies = 0.5 * torch.log(2 * torch.pi * var_fantasy_posts) + 0.5
        # sum the entropies from each independent posterior in the fantasy ModelList
        h_fantasies_scalar = torch.sum(h_fantasies, dim=-1)

        # compute the Monte-Carlo estimate of the Expected value of the entropy
        avg_h_fantasy = torch.mean(h_fantasies_scalar, dim=-2)

        # use the above entropies to compute the Expected Information Gain,
        # where the terms in the equation below correspond to the terms in
        # eq (4) of https://arxiv.org/pdf/2104.09460.pdf
        # (avg_h_fantasy is a Monte-Carlo estimate of the second term on the right)
        eig = h_current_scalar - avg_h_fantasy

        return eig.reshape(X.shape[:-2])


class BaxGenerator(BayesianGenerator):
    """
    BAX Generator for Bayesian optimization.

    Attributes
    ----------
    name : str
        The name of the generator.
    algorithm : Algorithm
        Algorithm evaluated in the BAX process.
    algorithm_results : Dict
        Dictionary results from the algorithm.
    algorithm_results_file : str
        File name to save algorithm results at every step.
    _n_calls : int
        Internal counter for the number of calls to the generate method.

    Methods
    -------
    validate_turbo_controller(cls, value, info: ValidationInfo) -> Any
        Validate the turbo controller.
    validate_vocs(cls, v, info: ValidationInfo) -> VOCS
        Validate the VOCS object.
    generate(self, n_candidates: int) -> List[Dict]
        Generate a specified number of candidate samples.
    _get_acquisition(self, model) -> ModelListExpectedInformationGain
        Get the acquisition function.
    """

    name = "BAX"
    supports_constraints: bool = True
    algorithm: Algorithm = Field(description="algorithm evaluated in the BAX process")
    algorithm_results: Dict = Field(
        None, description="dictionary results from algorithm", exclude=True
    )
    algorithm_results_file: str = Field(
        None, description="file name to save algorithm results at every step"
    )
    _n_calls: int = 0
    _compatible_turbo_controllers = [EntropyTurboController, SafetyTurboController]

    @field_validator("vocs", mode="after")
    def validate_vocs(cls, v, info: ValidationInfo):
        if v.n_constraints > 0 and not info.data["supports_constraints"]:
            raise VOCSError("this generator does not support constraints")

        # assert that the generator had no objectives
        if not v.n_objectives == 0:
            raise VOCSError("BAX generator only supports problems with no objectives")

        return v

    @model_validator(mode="after")
    def validate_model_after(self):
        # validate turbo controller center if it exists
        # validate_turbo_controller_center(self)

        return self

    def generate(self, n_candidates: int) -> List[Dict]:
        """
        Generate a specified number of candidate samples.

        Parameters
        ----------
        n_candidates : int
            The number of candidate samples to generate.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing the generated samples.
        """
        self._n_calls += 1
        return super().generate(n_candidates)

    def _get_acquisition(self, model) -> ModelListExpectedInformationGain:
        """
        Get the acquisition function.

        Parameters
        ----------
        model : Model
            The model to use for the acquisition function.

        Returns
        -------
        ModelListExpectedInformationGain
            The acquisition function.
        """
        bax_model_ids = [
            self.vocs.output_names.index(name)
            for name in self.algorithm.observable_names_ordered
        ]
        bax_model = model.subset_output(bax_model_ids)

        if isinstance(bax_model, SingleTaskGP):
            bax_model = ModelListGP(bax_model)

        eig = ModelListExpectedInformationGain(
            bax_model, self.algorithm, self._get_optimization_bounds()
        )
        self.algorithm_results = eig.algorithm_results
        if self.algorithm_results_file is not None:
            results = deepcopy(self.algorithm_results)

            with open(
                f"{self.algorithm_results_file}_{self._n_calls}.pkl", "wb"
            ) as outfile:
                pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        return eig


def optimize_solenoid_alignment(env, dump_location=None, **kwargs):
    """Run BAX optimization for solenoid alignment.

    Parameters
    ----------
    env : Any
        Injector control environment that provides variable and observable
        interfaces used by this routine.
    dump_location : str or pathlib.Path
        Requested output location for optimization artifacts.
    **kwargs
        Configuration overrides. Supported keys include
        ``initial_random_evaluations`` and ``n_steps``.

    Returns
    -------
    Xopt
        Configured and executed Xopt instance containing optimization data.
    """

    if dump_location is None:
        dump_location = "."

    settings = merge_config(
        {
            "initial_random_evaluations": 2,
            "n_steps": 30,
        },
        kwargs,
    )

    # TODO: check data folder exists

    logger.info("Starting BAX solenoid alignment optimization.")
    env.save_directory = os.path.join(dump_location, "data/")
    logger.debug(
        "Configured solenoid alignment optimization with save_directory=%s dump_location=%s",
        env.save_directory,
        dump_location,
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

        time.sleep(2.0)

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
            [0.99 * init_settings[var_name], 1.01 * init_settings[var_name]]
        )
        for var_name in variable_names[:2]
    }
    variables["SOLN:IN10:121:BCTRL"] = [0.97 * init_settings["SOLN:IN10:121:BCTRL"], 1.03 * init_settings["SOLN:IN10:121:BCTRL"]]
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
        dump_file=os.path.join(dump_location, f"solenoid_alignment_opt_{ts}.yaml"),
    )
    logger.debug("Created Xopt object with dump file: %s", X.dump_file)

    # evaluate the current point and two random points
    logger.info("Running initial evaluations (current + 2 random points).")
    X.evaluate_data(env.get_variables(X.vocs.variable_names))
    X.random_evaluate(settings["initial_random_evaluations"])

    for i in range(settings["n_steps"]):
        logger.debug("Running optimization step %d/5", i + 1)
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

    fig.savefig(os.path.join(dump_location, f"solenoid_alignment_opt_{ts}.png"))
    return X
