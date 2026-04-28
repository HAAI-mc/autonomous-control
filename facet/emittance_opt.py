"""Injector emittance optimization control script.

This module defines a Bayesian optimization routine for minimizing measured
injector emittance using Xopt.
"""

import os
import time
import logging
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator


logger = logging.getLogger("injector_emittance_opt")

def optimize_injector_emittance(env, dump_location):
    """Run Bayesian optimization for injector emittance.

    Parameters
    ----------
    env : Any
        Injector control environment that provides variable and observable
        interfaces used by this routine.
    dump_location : str or pathlib.Path
        Requested output location for optimization artifacts.

    Returns
    -------
    Xopt
        Configured and executed Xopt instance containing optimization data.
    """
    logger.info("Starting injector emittance optimization.")
    env.emittance_config_fname = (
        "/home/fphysics/rroussel/e331/Badger-Resources/facet/plugins/environments/inj_emit/emittance_measurement_configs/PROF10571.yaml"
    )
    env.save_directory = "data/"
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
            "SOLN:IN10:121:BCTRL": [0.390,0.405],
            "QUAD:IN10:121:BCTRL": [-0.008,0.0085],
            "QUAD:IN10:122:BCTRL": [-0.008,0.0085],
            "QUAD:IN10:361:BCTRL": [-3, -2.5],
            "QUAD:IN10:371:BCTRL": [2.5,3],
        },
        objectives={"emittance_mean": "MINIMIZE"},
        constraints={"min_joint_bmag":["LESS_THAN",1.5]},
    )
    
    evaluator = Evaluator(function=evaluate)
    generator = ExpectedImprovementGenerator(vocs=vocs)
    
    X = Xopt(
        vocs=vocs,
        evaluator=evaluator,
        generator=generator,
        dump_file=os.fspath(dump_location / f"5d_emittance_opt_{int(time.time())}.yaml"),
    )
    logger.debug("Created Xopt object with dump file: %s", X.dump_file)

    # evaluate the current point and two random points
    logger.info("Running initial evaluations (current + 2 random points).")
    X.evaluate_data(env.get_variables(X.vocs.variable_names))
    X.random_evaluate(2)

    for i in range(5):
        logger.debug("Running optimization step %d/5", i + 1)
        X.step()

    # evaluate the best point
    best = X.vocs.select_best(X.data)[2]
    logger.info("Evaluating best point from optimization: %s", best)
    X.evaluate_data(best)

    logger.info("Completed injector emittance optimization.")

    return X

    

    