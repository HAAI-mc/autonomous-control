import logging
import os
import time

try:
    from facet.optimization_utils import merge_config, restore_on_error
except ImportError:
    from optimization_utils import merge_config, restore_on_error

logger = logging.getLogger("auto_emittance")


@restore_on_error(context="auto_emittance")
def run_automatic_emittance(env, screen_name, config=None):
    """Run an automatic emittance measurement for the specified screen.

    Inserts the requested screen, configures the emittance measurement object
    on ``env``, executes the measurement, and returns the results.

    Parameters
    ----------
    env : Any
        Control environment with screen insertion, emittance configuration,
        and measurement interfaces.
    screen_name : str
        Name of the screen device to use.  Supported values are
        ``"PROF10571"`` and ``"PROF10711"``.
    config : dict, optional
        Configuration overrides, typically loaded from a config file. Supported
        keys include ``config_directory``, ``screen_settle_time``, and a nested
        ``screens`` mapping with per-screen insertion targets and config files.

    Returns
    -------
    emittance_result : ScreenBeamProfileMeasurementResult
        Result object from the beam profile measurement.
    fname : str
        Path to the file where results were saved.
    X : Xopt
        Optimizer instance from the emittance measurement.
    """

    logger.info(f"Starting automatic emittance measurement on screen: {screen_name}")

    settings = merge_config(
        {
            "config_directory": f"{os.environ['BADGER_RESOURCES']}/facet/plugins/environments/inj_emit/emittance_measurement_configs/",
            "screen_settle_time": 2.0,
            "screens": {
                "PROF10571": {
                    "targets": {"PROF10571": 1},
                    "config_file": "PROF10571.yaml",
                },
                "PROF10711": {
                    "targets": {"PROF10571": 0, "PROF10711": 1},
                    "config_file": "PROF10711.yaml",
                },
            },
        },
        config,
    )

    screen_config = settings["screens"].get(screen_name)
    if screen_config is None:
        raise ValueError(f"Unsupported screen_name: {screen_name}")

    for name, target in screen_config["targets"].items():
        env.screens[name].target = target
    time.sleep(settings["screen_settle_time"])
    env.emittance_config_fname = os.path.join(
        settings["config_directory"], screen_config["config_file"]
    )
    logger.info("Configured environment for %s", screen_name)

    env._create_emittance_object()
    emittance_result, fname = env.run_emittance_measurement()
    logger.info(f"Emittance measurement complete. Results saved to: {fname}")
    return emittance_result, fname, env._emittance_measurement_object.X
