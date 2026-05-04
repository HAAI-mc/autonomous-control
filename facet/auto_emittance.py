import logging
import os
from time import time

logger = logging.getLogger("auto_emittance")


def run_automatic_emittance(env, screen_name):
    """
    Run automatic emittance measurement using the specified environment and screen name.

    Parameters:
        env (Environment): The environment in which the measurement is performed.
        screen_name (str): The name of the screen device to be used for measurements.

    Returns:
        ScreenBeamProfileMeasurementResult: The result of the beam profile measurement.
        fname (str): The filename where the results are saved.
        X: Xopt object from the emittance measurement.
    """

    logger.info(f"Starting automatic emittance measurement on screen: {screen_name}")

    config_directory = "/home/fphysics/rroussel/e331/Badger-Resources/facet/plugins/environments/inj_emit/emittance_measurement_configs/"

    if screen_name == "PROF10571":
        env.screens["PROF10571"].target = 1
        time.sleep(2.0)
        # emittance_file = os.path.join(config_directory, "PROF10571.yaml")
        # emittance_config = yaml.safe_load(open(emittance_file))
        env.emittance_config_fname = os.path.join(config_directory, "PROF10571.yaml")

        # env.transmission_measurement_constraint = 0.7
        # env.min_beamsize_cutoff = 1000
        logger.info("Configured environment for PROF10571")

    elif screen_name == "PROF10711":
        env.screens["PROF10571"].target = 0
        env.screens["PROF10711"].target = 1
        time.sleep(2.0)
        # emittance_file = os.path.join(config_directory, "PROF10711.yaml")
        # emittance_config = yaml.safe_load(open(emittance_file))
        env.emittance_config_fname = os.path.join(config_directory, "PROF10711.yaml")

        # env.transmission_measurement_constraint = 0.7
        # env.min_beamsize_cutoff = 1000
        logger.info("Configured environment for PROF10711")
    env._create_emittance_object()
    emittance_result, fname = env.run_emittance_measurement()
    logger.info(f"Emittance measurement complete. Results saved to: {fname}")
    return emittance_result, fname, env._emittance_measurement_object.X
