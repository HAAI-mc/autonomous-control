"""TCAV phasing optimization control script.

This module provides a model-based controller that tunes TCAV phase using
Bayesian optimization with transmission constraints.
"""

import time
import logging
from typing import Any, Optional, Callable
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    retry_if_result,
    stop_after_attempt,
    wait_fixed,
)


from autonomous_control.facet.optimization_utils import restore_on_error
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

# Module-level retry/settle configuration for TCAV property set/read behavior.
TCAV_SETTLE_ATTEMPTS = 40
TCAV_SETTLE_WAIT = 0.5
TCAV_READ_RETRY_ATTEMPTS = 5
TCAV_READ_RETRY_WAIT = 0.25
TCAV_STATE_CHANGE_WAIT = (
    10.0  # seconds to wait after changing TCAV mode before reading values
)


def _is_timeout_exception(exc: BaseException) -> bool:
    """Return True when an exception appears to be timeout related."""
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc).lower()
    return "timeout" in message or "timed out" in message


def read_tcav_attr_with_retry(
    tcav: Any,
    attr_name: str,
    *,
    log: logging.Logger,
):
    """Read a TCAV attribute with retries on None values and timeout exceptions."""
    retrying = Retrying(
        stop=stop_after_attempt(TCAV_READ_RETRY_ATTEMPTS),
        wait=wait_fixed(TCAV_READ_RETRY_WAIT),
        retry=(
            retry_if_result(lambda value: value is None)
            | retry_if_exception(_is_timeout_exception)
        ),
        retry_error_callback=lambda _retry_state: None,
        before_sleep=before_sleep_log(log, logging.DEBUG),
    )
    value = retrying(lambda: getattr(tcav, attr_name))

    if value is None:
        raise RuntimeError(
            f"TCAV {attr_name} readback unavailable after {TCAV_READ_RETRY_ATTEMPTS} attempts"
        )

    return value


def _set_tcav_property_and_wait(
    tcav: Any,
    *,
    set_attr: str,
    target_value: Any,
    readback_attr: str,
    log: logging.Logger,
    atol: Optional[float] = None,
    cast: Optional[Callable[[Any], Any]] = None,
    label: Optional[str] = None,
) -> None:
    """Set a TCAV property and wait until the specified readback matches."""
    logger.debug(
        "Setting TCAV %s to %s and waiting for %s readback to match",
        set_attr,
        target_value,
        readback_attr,
    )
    setattr(tcav, set_attr, target_value)

    readback_value = None

    def _is_settled() -> bool:
        nonlocal readback_value
        readback_value = read_tcav_attr_with_retry(
            tcav,
            readback_attr,
            log=log,
        )

        if cast is not None:
            try:
                readback_value = cast(readback_value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"TCAV {readback_attr} readback is not valid: {readback_value!r}"
                ) from exc

        if atol is None:
            return readback_value == target_value

        return bool(np.isclose(readback_value, target_value, atol=atol))

    retrying = Retrying(
        stop=stop_after_attempt(TCAV_SETTLE_ATTEMPTS),
        wait=wait_fixed(TCAV_SETTLE_WAIT),
        retry=retry_if_result(lambda settled: not settled),
        retry_error_callback=lambda _retry_state: False,
        before_sleep=before_sleep_log(log, logging.DEBUG),
    )
    settled = retrying(_is_settled)

    if not settled:
        value_label = label or set_attr
        raise TimeoutError(
            f"TCAV {value_label} failed to settle within timeout: "
            f"target={target_value} readback={readback_value} polls={TCAV_SETTLE_ATTEMPTS}"
        )

    logger.debug(
        "TCAV %s successfully set to %s (readback: %s)",
        set_attr,
        target_value,
        readback_value,
    )


def set_tcav_phase_and_wait(
    tcav: Any,
    target_phase: float,
    *,
    phase_tolerance: float = 1e-3,
    log: logging.Logger = logger,
) -> None:
    """Set TCAV phase and wait for phase_avgnt readback to match."""
    target_phase = float(target_phase)
    _set_tcav_property_and_wait(
        tcav,
        set_attr="phase",
        target_value=target_phase,
        readback_attr="phase_avgnt",
        log=log,
        atol=phase_tolerance,
        cast=float,
        label="phase",
    )


def set_tcav_amplitude_and_wait(
    tcav: Any,
    target_amplitude: float,
    *,
    amplitude_tolerance: float = 1e-3,
    log: logging.Logger = logger,
) -> None:
    """Set TCAV amplitude and wait for amplitude_wocho readback to match."""
    target_amplitude = float(target_amplitude)
    _set_tcav_property_and_wait(
        tcav,
        set_attr="amplitude",
        target_value=target_amplitude,
        readback_attr="amplitude_wocho",
        log=log,
        atol=amplitude_tolerance,
        cast=float,
        label="amplitude",
    )


def set_tcav_mode_config_and_wait(
    tcav: Any,
    target_mode: str,
    *,
    log: logging.Logger = logger,
) -> None:
    """Set TCAV mode_config and wait for mode_config readback to match."""
    current_mode = read_tcav_attr_with_retry(tcav, "mode_config", log=log)
    _set_tcav_property_and_wait(
        tcav,
        set_attr="mode_config",
        target_value=target_mode,
        readback_attr="mode_config",
        log=log,
        cast=str,
        label="mode",
    )
    # NOTE: there are no readbacks for the mode change, so we just wait a few seconds to let the TCAV update
    if current_mode != target_mode:
        logger.debug(
            "waiting %s seconds for TCAV to update after mode change",
            TCAV_STATE_CHANGE_WAIT,
        )
        time.sleep(TCAV_STATE_CHANGE_WAIT)


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

    tcav_on_amplitude: PositiveFloat = 0.3
    n_measurement_shots: PositiveInt = 1
    amplitude_tolerance: PositiveFloat = 1e-3
    phase_tolerance: PositiveFloat = 0.05

    n_initial_points: PositiveInt = 10
    n_iterations: PositiveInt = 10

    X: Optional[Xopt] = None

    name: str = "automatic_phase_scan"
    nominal_centroid: Optional[float] = None
    max_scan_range: list[float] = [-10, 10]
    evaluate_callback: Optional[Callable] = None
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

        # acquire the beam posisition without the TCAV on
        self.nominal_centroid = self.acquire_nominal_centroid()
        logger.debug(f"Acquired nominal centroid: {self.nominal_centroid}")

        # set the TCAV to ACCEL_STDBY mode and verify
        mode_config = read_tcav_attr_with_retry(self.tcav, "mode_config", log=logger)
        if mode_config != "ACCEL_STDBY":
            logger.error("TCAV is not in ACCEL_STDBY mode")
            raise RuntimeError("tcav must be in ACCEL_STDBY mode config")

        # create xopt object
        self.X = self.create_xopt_object()

        # get origonal values
        start_amp = float(read_tcav_attr_with_retry(self.tcav, "amplitude", log=logger))
        start_phase = float(read_tcav_attr_with_retry(self.tcav, "phase", log=logger))
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
            # set the TCAV amplitude to the desired value for optimization
            set_tcav_amplitude_and_wait(
                self.tcav,
                self.tcav_on_amplitude,
                amplitude_tolerance=self.amplitude_tolerance,
            )

            # initial coarse scan
            initial_scan_values = np.linspace(
                np.clip(
                    start_phase - 5.0, self.max_scan_range[0], self.max_scan_range[1]
                ),
                np.clip(
                    start_phase + 5.0, self.max_scan_range[0], self.max_scan_range[1]
                ),
                self.n_initial_points,
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

            set_tcav_phase_and_wait(
                self.tcav, final_phase, phase_tolerance=self.phase_tolerance
            )

        except Exception:
            logger.exception(
                "Error during TCAV optimization, resetting to original phase"
            )
            set_tcav_phase_and_wait(
                self.tcav, start_phase, phase_tolerance=self.phase_tolerance
            )
            raise

        finally:
            set_tcav_amplitude_and_wait(
                self.tcav, start_amp, amplitude_tolerance=self.amplitude_tolerance
            )
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
            evaluator=evaluator,
            generator=generator,
            dump_file=self.dump_location,
        )

    def acquire_nominal_centroid(self) -> float:
        """Get centroid without TCAV streaking influence."""
        logger.info("Acquiring nominal centroid.")
        set_tcav_mode_config_and_wait(self.tcav, "STDBY")

        result = self.bpm.y
        logger.debug(f"Acquired nominal centroid: {result}")

        set_tcav_mode_config_and_wait(self.tcav, "ACCEL_STDBY")

        logger.debug(f"Nominal centroid value: {result}")
        return result

    def _evaluate(self, inputs: dict[str, Any]) -> dict[str, float]:
        """Evaluate the objective function for Bayesian optimization."""
        logger.debug(f"Evaluating input: {inputs}")

        set_tcav_phase_and_wait(
            self.tcav,
            target_phase=inputs["phase"],
            phase_tolerance=self.phase_tolerance,
        )

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


@restore_on_error(context="tcav_phasing")
def run_automatic_tcav_phasing(
    env,
    dump_location=None,
    *,
    tcav_on_amplitude=0.3,
    n_measurement_shots=1,
    amplitude_tolerance=1e-3,
    phase_tolerance=0.05,
    n_initial_points=10,
    n_iterations=10,
    name="automatic_phase_scan",
    max_scan_range=None,
    min_transmission=0.4,
    verbose=False,
):
    """Create and run the automatic TCAV phasing controller.

    Parameters
    ----------
    env : Any
        Environment that provides TCAV, BPM, transmission measurement, and
        callback interfaces.
    dump_location : str or Path, optional
        Directory to save optimization dumps, by default None (no dumps).
    tcav_on_amplitude : float, optional
        Target TCAV amplitude used during phasing scans.
    n_measurement_shots : int, optional
        Number of shots averaged per measurement.
    amplitude_tolerance : float, optional
        TCAV amplitude restoration tolerance.
    phase_tolerance : float, optional
        TCAV phase-settle tolerance.
    n_initial_points : int, optional
        Number of initial scan points.
    n_iterations : int, optional
        Maximum optimization iterations.
    name : str, optional
        Name label used by the phasing controller.
    max_scan_range : list[float], optional
        Phase bounds for optimization.
    min_transmission : float, optional
        Minimum transmission constraint.
    verbose : bool, optional
        Verbosity flag for the phasing controller.

    Returns
    -------
    Xopt or None
        Optimization object from the phasing run.
    """
    run_start_time = time.time()
    if max_scan_range is None:
        max_scan_range = [-10, 10]

    tcav = env.tcav
    logger.info(f"Starting automatic TCAV phasing. Current TCAV phase: {tcav.phase}")
    logger.info(
        "TCAV phasing config: tcav_on_amplitude=%s n_initial_points=%d n_iterations=%d max_scan_range=%s min_transmission=%s dump_location=%s",
        tcav_on_amplitude,
        n_initial_points,
        n_iterations,
        max_scan_range,
        min_transmission,
        dump_location,
    )

    def eval_callback(inputs):
        return env._evaluate_callback(inputs, None)

    phaser = MLTCAVPhasing(
        bpm=env.downstream_bpm,
        tcav=tcav,
        tcav_on_amplitude=tcav_on_amplitude,
        transmission_measurement=env.transmission_measurement,
        n_measurement_shots=n_measurement_shots,
        amplitude_tolerance=amplitude_tolerance,
        phase_tolerance=phase_tolerance,
        n_initial_points=n_initial_points,
        n_iterations=n_iterations,
        name=name,
        max_scan_range=max_scan_range,
        evaluate_callback=eval_callback,
        min_transmission=min_transmission,
        dump_location=dump_location,
        verbose=verbose,
    )
    logger.debug(
        "Configured MLTCAVPhasing with max_scan_range=%s",
        phaser.max_scan_range,
    )

    X = phaser.run()
    logger.info(
        "Automatic TCAV phasing finished. Optimized phase: %s duration=%.2f s",
        phaser.optimized_phase,
        time.time() - run_start_time,
    )

    return X
