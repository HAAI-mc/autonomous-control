import logging
import os
import time

from autonomous_control.facet.optimization_utils import (
    merge_config,
    restore_on_error,
)

logger = logging.getLogger("auto_emittance")


@restore_on_error(context="auto_emittance")
def run_automatic_emittance(
    env,
    dump_location,
    screen_name,
    config_directory=None,
    screen_settle_time=2.0,
    screens=None,
):
    """
    Run an automatic emittance measurement for the specified screen using
    the quadrupole scan method defined in the environment's emittance configuration.

    Inserts the requested screen, configures the emittance measurement object
    on ``env``, executes the measurement, and returns the results.

    Parameters
    ----------
    env : Any
        Control environment with screen insertion, emittance configuration,
        and measurement interfaces.
    dump_location : str or pathlib.Path
        Directory where environment-managed outputs should be saved.
    screen_name : str
        Name of the screen device to use. Supported values are
        ``"PR10571"`` and ``"PR10711"``.
    config_directory : str or pathlib.Path, optional
        Directory containing per-screen emittance configuration YAML files.
        Defaults to the FACET badger resources emittance config directory.
    screen_settle_time : float, optional
        Wait time in seconds after changing screen targets, by default 2.0.
    screens : dict, optional
        Per-screen mapping of insertion targets and config file names.
        Keys are screen names and values must include ``targets`` and
        ``config_file`` entries.

    Returns
    -------
    emittance_result : ScreenBeamProfileMeasurementResult
        Result object from the beam profile measurement.
    fname : str
        Path to the file where results were saved.
    X : Xopt
        Optimizer instance from the emittance measurement.
    """

    if config_directory is None:
        config_directory = f"{os.environ['BADGER_RESOURCES']}/facet/plugins/environments/inj_emit/emittance_measurement_configs/"

    default_screens = {
        "PR10571": {
            "targets": {"PR10571": 1},
            "config_file": "PR10571.yaml",
        },
        "PR10711": {
            "targets": {"PR10571": 0, "PR10711": 1},
            "config_file": "PR10711.yaml",
        },
    }
    screen_settings = merge_config(default_screens, screens)

    env.save_directory = str(dump_location)

    logger.info(f"Starting automatic emittance measurement on screen: {screen_name}")

    screen_config = screen_settings.get(screen_name)
    if screen_config is None:
        raise ValueError(f"Unsupported screen_name: {screen_name}")

    for name, target in screen_config["targets"].items():
        env.screens[name].target = target

    # wait for screen to settle after changing targets
    logger.info(f"Waiting for {screen_settle_time} seconds for screen to settle...")
    time.sleep(screen_settle_time)
    env.emittance_config_fname = os.path.join(
        config_directory, screen_config["config_file"]
    )
    logger.info("Configured environment for %s", screen_name)

    env._create_emittance_object()
    emittance_result, fname = env.run_emittance_measurement()
    logger.info(f"Emittance measurement complete. Results saved to: {fname}")
    return emittance_result, fname, env._emittance_measurement_object.X


def run_automatic_emittance_xopt(
    env,
    dump_location,
    screen_name,
    config_directory=None,
    screen_settle_time=2.0,
    screens=None,
):
    """Run automatic emittance and return only the Xopt object.

    This is a thin compatibility wrapper for workflow runners that expect each
    top-level step callable to return a single Xopt instance.
    """
    _, _, xopt = run_automatic_emittance(
        env,
        dump_location,
        screen_name,
        config_directory=config_directory,
        screen_settle_time=screen_settle_time,
        screens=screens,
    )
    return xopt
