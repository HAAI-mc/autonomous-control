"""Energy spread optimization control script.

This module performs a phase scan optimization to minimize transverse beam
size as a proxy for energy spread.
"""

import os
import numpy as np
import time
import logging

from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator


logger = logging.getLogger("energy_spread_opt")


def optimize_energy_spread(env, dump_location):
    """Optimize beam energy spread using klystron phase control.

    Parameters
    ----------
    env : Any
        Control environment with variable, observable, and beam profile
        measurement interfaces.
    dump_location : str or pathlib.Path
        Requested output location for optimization artifacts.

    Returns
    -------
    Xopt
        Executed optimizer instance containing all evaluations.

    Raises
    ------
    RuntimeError
        If the dipole current state is not suitable for energy measurements.
    """
    logger.info("Starting energy spread optimization.")
    dipole_correct_state = 0.125
    dipole_current_state = env.get_variables(["BEND:IN10:661:BACT"])[
        "BEND:IN10:661:BACT"
    ]

    klys_phase_set_pv = "KLYS:LI10:41:SFB_PDES"
    klys_phase_readback_pv = "ACCL:LI10:41:PHASE_W0CH0"

    if not np.isclose(dipole_correct_state, dipole_current_state, rtol=1e-2):
        logger.error(
            "Dipole state check failed: expected=%s actual=%s",
            dipole_correct_state,
            dipole_current_state,
        )
        raise RuntimeError("dipole not in correct state for energy measurements")
    logger.debug("Dipole state check passed: %s", dipole_current_state)

    measurement = env.create_beamprofile_measurement("PROF10711")
    logger.debug("Created beam profile measurement for PROF10711.")

    def evaluate(inputs):
        """Evaluate beam size metrics for a trial phase setting.

        Parameters
        ----------
        inputs : dict[str, float]
            Mapping that includes the phase setpoint PV value.

        Returns
        -------
        dict[str, float]
            Dictionary with measured RMS beam sizes in x and y.
        """
        logger.debug("Evaluating inputs: %s", inputs)
        env.set_variables(inputs)
        logger.debug("Waiting for klystron phase to settle.")
        time.sleep(0.1)  # Simulate some processing time
        settle_polls = 0
        while not np.isclose(
            env.get_observables([klys_phase_readback_pv])[klys_phase_readback_pv],
            inputs[klys_phase_set_pv],
            atol=0.05,
        ):
            time.sleep(0.1)
            settle_polls += 1
        logger.debug("Phase settled after %d polls.", settle_polls)

        # Get the output from the environment
        results = measurement.measure()
        output = results.rms_sizes_all
        output_dict = dict(zip(["rms_x", "rms_y"], list(output.flatten())))
        logger.debug("Measurement result: %s", output_dict)

        return output_dict

    initial_phase = env.get_variables([klys_phase_set_pv])[klys_phase_set_pv]
    logger.debug(
        "Initial phase readback for optimization variable %s: %s",
        klys_phase_set_pv,
        initial_phase,
    )

    # Define the VOCS for the optimization problem
    vocs = VOCS(
        variables={
            klys_phase_set_pv: [initial_phase - 5, initial_phase + 5],
        },
        objectives={"rms_x": "MINIMIZE"},
    )

    evaluator = Evaluator(function=evaluate)
    generator = ExpectedImprovementGenerator(vocs=vocs, n_interpolate_points=5)

    X = Xopt(
        vocs=vocs,
        evaluator=evaluator,
        generator=generator,
        dump_file=os.fspath(
            dump_location / f"energy_spread_minimization_{int(time.time())}.yaml"
        ),
    )
    logger.debug(
        "Created Xopt object with dump file: %s (dump_location=%s)",
        X.dump_file,
        dump_location,
    )

    logger.info("Running initial random evaluations.")
    X.random_evaluate(3)

    for i in range(2):
        logger.debug("Running optimization step %d/2", i + 1)
        X.step()

    logger.info("Optimization loop complete; evaluating best point.")
    best = X.vocs.select_best(X.data)[2]
    logger.debug("Best point selected: %s", best)
    X.evaluate_data(best)
    logger.info("Energy spread optimization finished.")

    return X
