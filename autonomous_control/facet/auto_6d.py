import logging

from autonomous_control.facet.auto_emittance import run_automatic_emittance

try:
    from autonomous_control.facet.optimization_utils import restore_on_error
except ImportError:
    from autonomous_control.facet.optimization_utils import restore_on_error
from lcls_tools.common.data.saver import H5Saver
import time
import pandas as pd

logger = logging.getLogger("auto_6d")


@restore_on_error(context="auto_6d")
def run_automatic_6d_measurement(env, save_filename):
    """Run a full 6D emittance measurement sequence.

    Performs quad scans on PROF10571 and PR10711 with the TCAV both off and
    on, saving incremental results after each step.

    Sequence
    --------
    1. Insert PROF10571, TCAV off  — quad scan.
    2. Insert PROF10571, TCAV on   — quad scan.
    3. Swap to PR10711, TCAV off — quad scan.
    4. Swap to PR10711, TCAV on  — quad scan.

    Parameters
    ----------
    env : Any
        Control environment providing TCAV control, screen insertion,
        variable access, and emittance measurement interfaces.
    save_filename : str or pathlib.Path
        Output path for the HDF5 results file.  Intermediate results are
        written after every measurement step.

    Returns
    -------
    data : dict
        Dictionary with keys ``"PROF10571_off"``, ``"PROF10571_on"``,
        ``"PR10711_off"``, ``"PR10711_on"``; each value is a dict
        containing the serialized emittance result and captured environment
        variables.
    tracking_data : pandas.DataFrame
        Concatenated Xopt data frames from all four quad scans.
    """
    saver = H5Saver()

    # turn off TCAV
    env.tcav.mode_config = "STDBY"
    time.sleep(2.0)

    data = {}

    # run automatic emittance measurement with TCAV off
    logger.info("running PROF10571 quad scan tcav off")
    emittance_result_PROF10571_off, _, X = run_automatic_emittance(
        env,
        dump_location=env.save_directory,
        screen_name="PROF10571",
    )
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
    emittance_result_PROF10571_on, _, X = run_automatic_emittance(
        env,
        dump_location=env.save_directory,
        screen_name="PROF10571",
    )
    data["PROF10571_on"] = emittance_result_PROF10571_on.model_dump() | {
        "environment_variables": env.get_variables(env.variables.keys())
    }
    # save the results
    tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)
    saver.dump(data, save_filename)

    # remove PROF10571 and insert PR10711
    env.screens["PROF10571"].target = 0
    env.screens["PR10711"].target = 1

    # turn off TCAV
    env.tcav.mode_config = "STDBY"
    time.sleep(2.0)

    # run automatic emittance measurement with TCAV off
    logger.info("running PR10711 quad scan tcav off")
    emittance_result_PR10711_off, _, X = run_automatic_emittance(
        env,
        dump_location=env.save_directory,
        screen_name="PR10711",
    )
    data["PR10711_off"] = emittance_result_PR10711_off.model_dump() | {
        "environment_variables": env.get_variables(env.variables.keys())
    }
    # save the results
    tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)
    saver.dump(data, save_filename)

    # turn on TCAV
    env.tcav.mode_config = "ACCEL_STDBY"
    time.sleep(2.0)

    # run automatic emittance measurement with TCAV on
    logger.info("running PR10711 quad scan tcav on")
    emittance_result_PR10711_on, _, X = run_automatic_emittance(
        env,
        dump_location=env.save_directory,
        screen_name="PR10711",
    )
    data["PR10711_on"] = emittance_result_PR10711_on.model_dump() | {
        "environment_variables": env.get_variables(env.variables.keys())
    }

    # set the tcav amp back to 0.0
    env.tcav.mode_config = "STDBY"
    time.sleep(2.0)

    # save the results
    tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)
    saver.dump(data, save_filename)

    return data, tracking_data
