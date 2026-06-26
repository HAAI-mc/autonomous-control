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


from .optimization_utils import restore_on_error, safe_evaluate_best_point

logger = logging.getLogger("energy_spread_opt")


@restore_on_error(context="e_spread_opt")
def optimize_energy_spread(
    env,
    dump_location=None,
    *,
    dipole_correct_state=0.125,
    dipole_current_pv="BEND:IN10:661:BACT",
    phase_set_pv="KLYS:LI10:41:SFB_PDES",
    phase_readback_pv="ACCL:LI10:41:PHASE_W0CH0",
    phase_span=5.0,
    variables=None,
    objectives=None,
    measurement_screen="PROF10711",
    initial_random_evaluations=3,
    n_steps=5,
    settle_wait=0.1,
    poll_interval=0.5,
    phase_tolerance=0.05,
    max_settle_polls=40,
):
    """Optimize beam energy spread using klystron phase control.

    Parameters
    ----------
    env : Any
        Control environment with variable, observable, and beam profile
        measurement interfaces.
    dump_location : str or pathlib.Path
        Requested output location for optimization artifacts.
    dipole_correct_state : float, optional
        Required dipole current state for energy measurements.
    dipole_current_pv : str, optional
        Dipole current readback PV.
    phase_set_pv : str, optional
        Klystron phase setpoint PV.
    phase_readback_pv : str, optional
        Klystron phase readback PV.
    phase_span : float, optional
        Symmetric optimization span around the current phase.
    variables : dict, optional
        Explicit variable bounds override for VOCS.
    objectives : dict, optional
        VOCS objective mapping.
    measurement_screen : str, optional
        Screen used for beam profile measurements.
    initial_random_evaluations : int, optional
        Number of initial random evaluations.
    n_steps : int, optional
        Number of Bayesian optimization steps.
    settle_wait : float, optional
        Initial wait before settling loop.
    poll_interval : float, optional
        Poll interval for phase settling.
    phase_tolerance : float, optional
        Absolute tolerance for phase settling.
    max_settle_polls : int, optional
        Maximum number of settling polls before timeout.

    Returns
    -------
    Xopt
        Executed optimizer instance containing all evaluations.

    Raises
    ------
    RuntimeError
        If the dipole current state is not suitable for energy measurements.
    """
    if dump_location is None:
        dump_location = "."

    if objectives is None:
        objectives = {"rms_x": "MINIMIZE"}

    settings = {
        "dipole_correct_state": dipole_correct_state,
        "dipole_current_pv": dipole_current_pv,
        "phase_set_pv": phase_set_pv,
        "phase_readback_pv": phase_readback_pv,
        "phase_span": phase_span,
        "variables": variables,
        "objectives": objectives,
        "measurement_screen": measurement_screen,
        "initial_random_evaluations": initial_random_evaluations,
        "n_steps": n_steps,
        "settle_wait": settle_wait,
        "poll_interval": poll_interval,
        "phase_tolerance": phase_tolerance,
        "max_settle_polls": max_settle_polls,
    }
    logger.info("Starting energy spread optimization.")
    dipole_correct_state = settings["dipole_correct_state"]
    dipole_current_state = env.get_variables([settings["dipole_current_pv"]])[
        settings["dipole_current_pv"]
    ]

    klys_phase_set_pv = settings["phase_set_pv"]
    klys_phase_readback_pv = settings["phase_readback_pv"]
    max_settle_polls = settings["max_settle_polls"]

    if not np.isclose(dipole_correct_state, dipole_current_state, rtol=1e-2):
        logger.error(
            "Dipole state check failed: expected=%s actual=%s",
            dipole_correct_state,
            dipole_current_state,
        )
        raise RuntimeError("dipole not in correct state for energy measurements")
    logger.debug("Dipole state check passed: %s", dipole_current_state)

    measurement = env.create_beamprofile_measurement(settings["measurement_screen"])
    logger.debug(
        "Created beam profile measurement for %s.",
        settings["measurement_screen"],
    )

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
        time.sleep(settings["settle_wait"])
        settle_polls = 0
        readback_value = env.get_observables([klys_phase_readback_pv])[
            klys_phase_readback_pv
        ]
        while not np.isclose(
            readback_value,
            inputs[klys_phase_set_pv],
            atol=settings["phase_tolerance"],
        ):
            time.sleep(settings["poll_interval"])
            readback_value = env.get_observables([klys_phase_readback_pv])[
                klys_phase_readback_pv
            ]
            logger.debug(f"measured readback: {readback_value}")
            settle_polls += 1
            if settle_polls >= max_settle_polls:
                msg = (
                    "Klystron phase failed to settle within timeout: "
                    f"target={inputs[klys_phase_set_pv]} readback={readback_value} "
                    f"polls={settle_polls}"
                )
                logger.error(msg)
                raise TimeoutError(msg)
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

    variable_bounds = settings["variables"]
    if variable_bounds is None:
        variable_bounds = {
            klys_phase_set_pv: [
                initial_phase - settings["phase_span"],
                initial_phase + settings["phase_span"],
            ]
        }

    # Define the VOCS for the optimization problem
    vocs = VOCS(
        variables=variable_bounds,
        objectives=settings["objectives"],
    )

    evaluator = Evaluator(function=evaluate)
    generator = ExpectedImprovementGenerator(vocs=vocs)

    X = Xopt(
        vocs=vocs,
        evaluator=evaluator,
        generator=generator,
        dump_file=os.path.join(
            dump_location, f"energy_spread_minimization_{int(time.time())}.yaml"
        ),
    )
    logger.debug(
        "Created Xopt object with dump file: %s (dump_location=%s)",
        X.dump_file,
        dump_location,
    )

    logger.info("Running initial random evaluations.")
    X.random_evaluate(settings["initial_random_evaluations"])

    for i in range(settings["n_steps"]):
        logger.debug("Running optimization step %d/%d", i + 1, settings["n_steps"])
        X.step()

    logger.info("Optimization loop complete.")
    safe_evaluate_best_point(
        X,
        logger,
        use_select_best=True,
        context="energy spread optimization",
    )
    logger.info("Energy spread optimization finished.")

    return X
