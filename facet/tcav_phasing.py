"""TCAV phasing optimization control script.

This module provides a model-based controller that tunes TCAV phase using
Bayesian optimization with transmission constraints.
"""

import os
import time
import logging
from typing import Any, Optional, Callable
import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import (
    ExpectedImprovementGenerator,
)
from xopt.vocs import select_best

from ml_tto.automatic_emittance.transmission import TransmissionMeasurement
from lcls_tools.common.devices.bpm import BPM


# Setup logging
logger = logging.getLogger("auto_tcav_phasing")


class MLTCAVPhasing(BaseModel):
    """Bayesian optimization routine for tuning TCAV phase.

    Parameters
    ----------
    tcav : Any
        TCAV device interface with phase, amplitude, and mode configuration.
    bpm : BPM
        Downstream BPM used to measure centroid response.
    transmission_measurement : TransmissionMeasurement
        Measurement helper for beam transmission during scans.
    n_measurement_shots : int, optional
        Number of shots per measurement, by default 1.
    wait_time : float, optional
        Settling time in seconds after actuator changes, by default 2.0.
    n_initial_points : int, optional
        Number of points in the initial coarse scan, by default 10.
    n_iterations : int, optional
        Maximum Bayesian optimization iterations, by default 10.
    max_scan_range : list[float], optional
        Bounds for TCAV phase optimization, by default [-10, 10].
    evaluate_callback : callable, optional
        Optional callback for adding extra observables to each evaluation.
    min_transmission : float, optional
        Minimum allowable transmission constraint, by default 0.8.
    """

    tcav: Any
    bpm: BPM
    transmission_measurement: TransmissionMeasurement

    n_measurement_shots: PositiveInt = 1
    wait_time: PositiveFloat = 2.0
    phase_tolerance: PositiveFloat = 0.05
    phase_settle_poll_interval: PositiveFloat = 0.5
    max_phase_settle_polls: PositiveInt = 40

    n_initial_points: PositiveInt = 10
    n_iterations: PositiveInt = 10

    X: Optional[Xopt] = None

    name: str = "automatic_phase_scan"
    nominal_centroid: Optional[float] = None
    # max_scan_range: list[float] = [-10, 10]
    max_scan_range: list[float] = [-5, 5]
    evaluate_callback: Optional[Callable] = None
    # min_transmission: float = 0.8
    min_transmission: float = 0.4
    dump_location: Optional[str] = None

    verbose: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def optimized_phase(self):
        """Return the best phase found so far.

        Returns
        -------
        float
            Best phase according to the VOCS objective.
        """
        return float(select_best(self.X.vocs, self.X.data)[2]["phase"])

    def run(self):
        """Execute the TCAV phase optimization routine.

        Returns
        -------
        Xopt or None
            Optimizer object containing collected data.

        Raises
        ------
        RuntimeError
            If TCAV is not in ACCEL_STDBY mode when optimization starts.
        Exception
            Propagates runtime failures after restoring machine settings.
        """
        logger.info("Starting TCAV phase optimization....")
        # make sure that the tcav is in accel mode

        if self.tcav.mode_config != "ACCEL_STDBY":
            logger.error("TCAV is not in ACCEL_STDBY model")
            raise RuntimeError("tcav must be in ACCEL_STDBY mode config")

        # acquire the beam posisition without the TCAV on
        self.nominal_centroid = self.acquire_nominal_centroid()
        logger.debug(f"Acquired nominal centroid: {self.nominal_centroid}")

        # create xopt object
        self.X = self.create_xopt_object()

        # get origonal values
        start_amp = self.tcav.amplitude
        start_phase = self.tcav.phase
        logger.info(f"Initial TCAV amplitude: {start_amp}, phase: {start_phase}")
        logger.debug(
            "Optimization settings: n_initial_points=%s n_iterations=%s scan_range=%s min_transmission=%s",
            self.n_initial_points,
            self.n_iterations,
            self.max_scan_range,
            self.min_transmission,
        )

        # run optimization - if an error is raised, reset the scan values
        try:
            # initial coarse scan
            initial_scan_values = np.linspace(
                start_phase - 5.0, start_phase + 5.0, self.n_initial_points
            )

            # evaluate current point
            self.X.evaluate_data({"phase": start_phase})
            logger.debug(f"Initial scan values: {initial_scan_values}")

            # do scan for initialization + TCAV calibration
            self.X.evaluate_data({"phase": initial_scan_values})

            # run optimization
            for i in range(self.n_iterations):
                best_offset = self.X.data["offset"].min()
                logger.debug(
                    "Iteration %d/%d current best offset=%s",
                    i + 1,
                    self.n_iterations,
                    best_offset,
                )
                if best_offset < 1e-2:
                    logger.info("Converged")
                    break

                logger.debug(f"Optimization step:{i}")
                self.X.step()

            final_phase = float(select_best(self.X.vocs, self.X.data)[2]["phase"])
            logger.info(f"setting final phase to {final_phase}")
            logger.debug("Optimization data points collected: %s", len(self.X.data))

            self._set_phase_and_wait(final_phase)

        except Exception:
            logger.exception(
                "Error during TCAV optimization, resetting to original phase"
            )
            self.tcav.phase = start_phase
            raise

        finally:
            self.tcav.amplitude = start_amp
            logger.info("Restored original TCAV amplitude.")
            logger.info("TCAV phase optimization complete.")

        return self.X

    def create_xopt_object(self):
        """Instantiate an Xopt optimizer configured for TCAV phase.

        Returns
        -------
        Xopt
            Configured optimizer instance.
        """
        logger.debug("Creating Xopt optimizer object.")
        vocs = VOCS(
            variables={"phase": self.max_scan_range},
            objectives={"offset": "MINIMIZE"},
            constraints={"transmission": ["GREATER_THAN", self.min_transmission]},
        )

        evaluator = Evaluator(function=self._evaluate)

        generator = ExpectedImprovementGenerator(vocs=vocs)
        logger.debug("Xopt object created.")
        return Xopt(
            vocs=vocs,
            evaluator=evaluator,
            generator=generator,
            dump_file=os.path.join(
                self.dump_location, f"tcav_phasing_{int(time.time())}.yaml"
            )
            if self.dump_location
            else None,
        )

    def acquire_nominal_centroid(self) -> float:
        """Get centroid without TCAV streaking influence."""
        logger.info("Acquiring nominal centroid.")
        self.tcav.mode_config = "STDBY"
        time.sleep(self.wait_time)

        result = self.bpm.y

        self.tcav.mode_config = "ACCEL_STDBY"
        time.sleep(self.wait_time)

        logger.debug(f"Nominal centroid value: {result}")
        return result

    def _evaluate(self, inputs: dict[str, Any]) -> dict[str, float]:
        """Evaluate the objective function for Bayesian optimization."""
        logger.debug(f"Evaluating input: {inputs}")

        self._set_phase_and_wait(inputs["phase"])

        logger.debug(f"TCAV Phase set to {inputs['phase']} degrees")

        transmission = self.transmission_measurement.measure()["transmission"]
        if transmission > self.min_transmission:
            offset = (self.nominal_centroid - self.bpm.y) ** 2
            centroid = self.bpm.y
        else:
            offset = np.nan
            centroid = np.nan
            logger.warning(
                f"Low transmission ({transmission:.2f}), skipping centroid measurement."
            )

        result = {"offset": offset, "centroid": centroid, "transmission": transmission}
        if self.evaluate_callback is not None:
            result.update(self.evaluate_callback(inputs))

        logger.debug(f"Evaluation result: {result}")
        return result

    def _set_phase_and_wait(self, target_phase: float) -> None:
        """Set TCAV phase and verify readback settles within retry budget."""
        self.tcav.phase = target_phase
        settle_polls = 0
        readback_phase = self.tcav.phase

        while not np.isclose(readback_phase, target_phase, atol=self.phase_tolerance):
            time.sleep(self.phase_settle_poll_interval)
            readback_phase = self.tcav.phase
            settle_polls += 1

            if settle_polls >= self.max_phase_settle_polls:
                msg = (
                    "TCAV phase failed to settle within timeout: "
                    f"target={target_phase} readback={readback_phase} "
                    f"polls={settle_polls}"
                )
                logger.error(msg)
                raise TimeoutError(msg)

        if self.wait_time > 0:
            time.sleep(self.wait_time)


def run_automatic_tcav_phasing(env, dump_location=None):
    """Create and run the automatic TCAV phasing controller.

    Parameters
    ----------
    env : Any
        Environment that provides TCAV, BPM, transmission measurement, and
        callback interfaces.
    dump_location : str or Path, optional
        Directory to save optimization data dumps, by default None (no dumps).

    Returns
    -------
    Xopt or None
        Optimization object from the phasing run.
    """
    tcav = env.tcav
    logger.info(f"Starting automatic TCAV phasing. Current TCAV phase: {tcav.phase}")

    def eval_callback(inputs):
        return env._evaluate_callback(inputs, None)

    phaser = MLTCAVPhasing(
        bpm=env.downstream_bpm,
        tcav=tcav,
        transmission_measurement=env.transmission_measurement,
        wait_time=0.5,
        evaluate_callback=eval_callback,
        verbose=False,
        max_scan_range=[35.0, 55.0],
        dump_location=dump_location,
    )
    logger.debug(
        "Configured MLTCAVPhasing with wait_time=%s max_scan_range=%s",
        phaser.wait_time,
        phaser.max_scan_range,
    )

    X = phaser.run()
    logger.info(
        "Automatic TCAV phasing finished. Optimized phase: %s", phaser.optimized_phase
    )

    return X
