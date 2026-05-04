import logging
import os
import yaml

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

    env.set_screen(screen_name)

    config_directory = "/home/physics/badger/resources/facet/environments/inj_emit/emittance_measurement_configs/"

    if screen_name == "PROF10571":
        emittance_file = os.path.join(config_directory, "PROF10571.yaml")
        emittance_config = yaml.safe_load(open(emittance_file))
        env.emittance_config = emittance_config

        # env.transmission_measurement_constraint = 0.7
        # env.min_beamsize_cutoff = 1000
        logger.info("Configured environment for PROF10571")

    elif screen_name == "PROF10711":
        emittance_file = os.path.join(config_directory, "PROF10711.yaml")
        emittance_config = yaml.safe_load(open(emittance_file))
        env.emittance_config = emittance_config

        # env.transmission_measurement_constraint = 0.7
        # env.min_beamsize_cutoff = 1000
        logger.info("Configured environment for PROF10711")

    emittance_result, fname = env.run_emittance_measurement()
    logger.info(f"Emittance measurement complete. Results saved to: {fname}")
    return emittance_result, fname, env._emittance_measurement_object.X
    
