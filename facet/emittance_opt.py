"""Injector emittance optimization control script.

This module defines a Bayesian optimization routine for minimizing measured
injector emittance using Xopt.
"""

import os
import time
import logging
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator


from .optimization_utils import restore_on_error, safe_evaluate_best_point

logger = logging.getLogger("injector_emittance_opt")


@restore_on_error(context="emittance_opt")
def optimize_injector_emittance(env, dump_location=None, *, n_steps=3):
    """Run Bayesian optimization for injector emittance.

    Parameters
    ----------
    env : Any
        Injector control environment that provides variable and observable
        interfaces used by this routine.
    dump_location : str or pathlib.Path
        Requested output location for optimization artifacts.

    n_steps : int, optional
        Number of Bayesian optimization steps.

    Returns
    -------
    Xopt
        Configured and executed Xopt instance containing optimization data.
    """
    run_start_time = time.time()

    if dump_location is None:
        dump_location = "."

    # TODO: check data folder exists

    logger.info("Starting injector emittance optimization.")
    logger.info(
        "Injector emittance config: n_steps=%d dump_location=%s",
        n_steps,
        dump_location,
    )
    env.emittance_config_fname = "/home/fphysics/rroussel/e331/Badger-Resources/facet/plugins/environments/inj_emit/emittance_measurement_configs/PROF10571.yaml"
    env.save_directory = os.path.join(dump_location, "data/")
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
            "SOLN:IN10:121:BCTRL": [0.39, 0.41],
            "QUAD:IN10:121:BCTRL": [-0.008, 0.0085],
            "QUAD:IN10:122:BCTRL": [-0.008, 0.0085],
            # "QUAD:IN10:361:BCTRL": [-3, -2.5],
            # "QUAD:IN10:371:BCTRL": [2.5, 3],
        },
        objectives={"emittance_mean": "MINIMIZE"},
        constraints={"min_joint_bmag": ["LESS_THAN", 1.5]},
    )

    evaluator = Evaluator(function=evaluate)
    generator = ExpectedImprovementGenerator(vocs=vocs)

    X = Xopt(
        vocs=vocs,
        evaluator=evaluator,
        generator=generator,
        dump_file=os.path.join(
            dump_location, f"5d_emittance_opt_{int(time.time())}.yaml"
        ),
    )
    logger.debug("Created Xopt object with dump file: %s", X.dump_file)

    # evaluate the current point and two random points
    logger.info("Running initial evaluations (current + 2 random points).")
    X.evaluate_data(env.get_variables(X.vocs.variable_names))
    X.random_evaluate(2)

    for i in range(n_steps):
        logger.debug("Running optimization step %d/5", i + 1)
        X.step()

    safe_evaluate_best_point(
        X,
        logger,
        use_select_best=True,
        context="injector emittance optimization",
    )

    logger.info("Completed injector emittance optimization.")
    logger.info(
        "Injector emittance summary: evaluations=%d duration=%.2f s",
        len(X.data),
        time.time() - run_start_time,
    )

    return X
