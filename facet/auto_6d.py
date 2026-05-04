import logging

from auto_emittance import run_automatic_emittance
from lcls_tools.common.data.saver import H5Saver
import time
import pandas as pd

logger = logging.getLogger("auto_6d")


def run_automatic_6d_measurement(env, save_filename):
    """
    Does the following:
    1. Insert PROF10571
    2. Run automatic emittance measurement with TCAV off.
    3. Run automatic emittance measurement with TCAV on.
    4. Remove PROF10571
    5. Run automatic emittance measurement with TCAV off.
    6. Run automatic emittance measurement with TCAV on.

    """
    saver = H5Saver()

    # turn off TCAV
    env.tcav.mode_config = "STDBY"
    time.sleep(2.0)

    data = {}

    # run automatic emittance measurement with TCAV off
    logger.info("running PROF10571 quad scan tcav off")
    emittance_result_PROF10571_off, _, X = run_automatic_emittance(env, "PROF10571")
    data["PROF10571_off"] = emittance_result_PROF10571_off.model_dump() | {
        "environment_variables": env.get_variables(env.variables.keys())
    }
    # save the results
    tracking_data = X.data
    saver.dump(data, save_filename)

    # turn on TCAV
    env.tcav.mode_config = "ACCEL_STDBY"
    time.sleep(2.0)

    # run automatic emittance measurement with TCAV on
    logger.info("running PROF10571 quad scan tcav on")
    emittance_result_PROF10571_on, _, X = run_automatic_emittance(env, "PROF10571")
    data["PROF10571_on"] = emittance_result_PROF10571_on.model_dump() | {
        "environment_variables": env.get_variables(env.variables.keys())
    }
    # save the results
    tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)
    saver.dump(data, save_filename)

    # remove PROF10571 and insert PROF10711
    env.screens["PROF10571"].target = 0
    env.screens["PROF10711"].target = 1

    # turn off TCAV
    env.tcav.mode_config = "STDBY"
    time.sleep(2.0)

    # run automatic emittance measurement with TCAV off
    logger.info("running PROF10711 quad scan tcav off")
    emittance_result_PROF10711_off, _, X = run_automatic_emittance(env, "PROF10711")
    data["PROF10711_off"] = emittance_result_PROF10711_off.model_dump() | {
        "environment_variables": env.get_variables(env.variables.keys())
    }
    # save the results
    tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)
    saver.dump(data, save_filename)

    # turn on TCAV
    env.tcav.mode_config = "ACCEL_STDBY"
    time.sleep(2.0)

    # run automatic emittance measurement with TCAV on
    logger.info("running PROF10711 quad scan tcav on")
    emittance_result_PROF10711_on, _, X = run_automatic_emittance(env, "PROF10711")
    data["PROF10711_on"] = emittance_result_PROF10711_on.model_dump() | {
        "environment_variables": env.get_variables(env.variables.keys())
    }

    # set the tcav amp back to 0.0
    env.tcav.mode_config = "STDBY"
    time.sleep(2.0)

    # save the results
    tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)
    saver.dump(data, save_filename)

    return data, tracking_data
