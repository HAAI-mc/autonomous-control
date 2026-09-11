import logging
import time
from pathlib import Path

import yaml

from autonomous_control.facet.optimization_utils import restore_on_error

logger = logging.getLogger("auto_emittance")


@restore_on_error(context="auto_emittance")
def run_automatic_emittance(
    env,
    config_file,
    dump_location=None,
    screen_settle_time=2.0,
):
    """
    Run an automatic emittance measurement using the specified config file.

    Reads the measurement screen and screen insertion targets from the YAML
    configuration, configures ``env``, executes the measurement, and returns
    the results.

    Parameters
    ----------
    env : Any
        Control environment with screen insertion, emittance configuration,
        and measurement interfaces.
    config_file : str or pathlib.Path
        Required emittance configuration file. The file must contain
        ``screen.name`` and ``screen_targets``.
    dump_location : str or pathlib.Path, optional
        Directory where environment-managed outputs should be saved.
    screen_settle_time : float, optional
        Wait time in seconds after changing screen targets, by default 2.0.

    Returns
    -------
    emittance_result : ScreenBeamProfileMeasurementResult
        Result object from the beam profile measurement.
    fname : str
        Path to the file where results were saved.
    X : Xopt
        Optimizer instance from the emittance measurement.
    """

    config_path, screen_name, screen_targets = resolve_emittance_config(config_file)

    if dump_location is not None:
        env.save_directory = str(dump_location)
    elif getattr(env, "save_directory", None) in (None, ""):
        env.save_directory = "."

    logger.info(f"Starting automatic emittance measurement on screen: {screen_name}")

    for name, target in screen_targets.items():
        env.screens[name].target = target

    # wait for screen to settle after changing targets
    logger.info(f"Waiting for {screen_settle_time} seconds for screen to settle...")
    time.sleep(screen_settle_time)
    env.emittance_config_fname = str(config_path)
    logger.info("Configured environment for %s", screen_name)

    env._create_emittance_object()
    emittance_result, fname = env.run_emittance_measurement()
    logger.info(f"Emittance measurement complete. Results saved to: {fname}")
    return emittance_result, fname, env._emittance_measurement_object.X


def measure_emittance(
    env,
    config_file,
    dump_location=None,
    *,
    screen_settle_time=2.0,
):
    """Run automatic emittance and return only the Xopt object.

    This is a thin compatibility wrapper for workflow runners that expect each
    top-level step callable to return a single Xopt instance.
    """
    _, _, xopt = run_automatic_emittance(
        env,
        config_file,
        dump_location=dump_location,
        screen_settle_time=screen_settle_time,
    )
    return xopt


def resolve_emittance_config(config_file):
    """Resolve an emittance config file and extract screen setup metadata."""
    if config_file is None or str(config_file) == "":
        raise ValueError("config_file must be a non-empty path")

    config_path = Path(config_file).resolve()
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    screen_name = config.get("screen", {}).get("name") if config else None
    if not screen_name:
        raise ValueError(f"Config file {config_path} must define screen.name")

    screen_targets = config.get("screen_targets") if config else None
    if not isinstance(screen_targets, dict) or not screen_targets:
        raise ValueError(
            f"Config file {config_path} must define non-empty screen_targets"
        )
    if screen_name not in screen_targets:
        raise ValueError(
            f"Config file {config_path} defines screen.name={screen_name!r}, "
            "but screen_targets does not include that screen."
        )

    return config_path, screen_name, screen_targets
