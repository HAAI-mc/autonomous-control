"""Injector emittance optimization control script.

This module defines a Bayesian optimization routine for minimizing measured
injector emittance using Xopt.
"""

import os
import time
import logging
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator


from autonomous_control.facet.optimization_utils import (
    restore_on_error,
    safe_evaluate_best_point,
)

logger = logging.getLogger("injector_emittance_opt")


@restore_on_error(context="emittance_opt")
def minimize_injector_emittance(
    env,
    variables,
    dump_location=None,
    n_steps=3,
    min_joint_bmag_constraint=1.5,
    n_initial=2,
):
    """Run Bayesian optimization for injector emittance.

    Parameters
    ----------
    env : Any
        Injector control environment that provides variable and observable
        interfaces used by this routine.
    dump_location : str or pathlib.Path, optional
        Xopt dump file path.
    variables : dict
        Mapping of variable names to bounds for optimization.
    n_steps : int, optional
        Number of optimization steps to perform, by default 3.
    min_joint_bmag_constraint : float, optional
        Maximum allowed value for `min_joint_bmag` inside Xopt constraints, by default 1.5.
    n_initial : int, optional
        Number of initial random evaluations to perform on top of current point, by default 2.

    Returns
    -------
    Xopt
        Configured and executed Xopt instance containing optimization data.
    """

    # TODO: check data folder exists

    logger.info("Starting injector emittance optimization.")
    output_directory = os.path.dirname(dump_location) if dump_location else "."
    env.save_directory = os.path.join(output_directory)

    logger.debug(
        "Configured emittance optimization with config=%s save_directory=%s dump_location=%s",
        env.emittance_config_fname,
        env.save_directory,
        dump_location,
    )

    def evaluate(inputs):
        """Evaluate injector observables at a candidate setting.

        Parameters
        ----------
        inputs : dict[str, float]
            Mapping of control variable names to values.

        Returns
        -------
        dict[str, float]
            Observable dictionary returned by the control environment.
        """
        logger.debug("Evaluating injector settings: %s", inputs)
        env.set_variables(inputs)

        time.sleep(0.1)

        # Get the output from the environment
        # note that output will contain many results, not just emittance_x
        # see FACET-II injector badger environment for details
        output = env.get_observables(["emittance_x"])
        logger.debug("Evaluation output keys: %s", list(output.keys()))

        return output

    vocs = VOCS(
        variables={
            **variables,
        },
        objectives={"emittance_mean": "MINIMIZE"},
        constraints={"min_joint_bmag": ["LESS_THAN", min_joint_bmag_constraint]},
    )

    evaluator = Evaluator(function=evaluate)
    generator = ExpectedImprovementGenerator(vocs=vocs)

    X = Xopt(
        vocs=vocs,
        evaluator=evaluator,
        generator=generator,
        dump_file=dump_location,
    )
    logger.debug("Created Xopt object.")

    # evaluate the current point and two random points
    logger.info("Running initial evaluations (current + %d random points).", n_initial)
    X.evaluate_data(env.get_variables(X.vocs.variable_names))
    X.random_evaluate(n_initial)

    for i in range(n_steps):
        logger.debug("Running optimization step %d/%d", i + 1, n_steps)
        X.step()

    safe_evaluate_best_point(
        X,
        logger,
        use_select_best=True,
        context="injector emittance optimization",
    )

    logger.info("Completed injector emittance optimization.")

    return X
