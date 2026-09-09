import logging

from autonomous_control.facet.auto_emittance import run_automatic_emittance
from autonomous_control.facet.env_utils import validate_environment

try:
    from autonomous_control.facet.optimization_utils import restore_on_error
except ImportError:
    from autonomous_control.facet.optimization_utils import restore_on_error
from lcls_tools.common.data.saver import H5Saver
import time
import pandas as pd

logger = logging.getLogger("auto_6d")


@restore_on_error(context="auto_6d")
def run_automatic_6d_measurement(
    env,
    save_filename,
    screen_names,
    tcav_modes=("STDBY", "ACCEL_STDBY"),
    reset_tcav_mode="STDBY",
    tcav_settle_time=2.0,
    dump_location=None,
):
    """Run a full 6D emittance measurement sequence.

    Performs a quad scan on each screen in ``screen_names``, with the TCAV
    set to each mode in ``tcav_modes`` in turn, saving incremental results
    after every step. Screen insertion/retraction targets for each scan are
    resolved by ``run_automatic_emittance``, so this function does not
    hardcode any device names itself.

    Parameters
    ----------
    env : Any
        Control environment providing TCAV control, screen insertion,
        variable access, and emittance measurement interfaces. See
        ``EnvironmentInterface`` in ``env_utils.py``. Assumes ``env.screens``
        is a dict mapping screen name (str) to a Screen object.
    save_filename : str or pathlib.Path
        Output path for the HDF5 results file.  Intermediate results are
        written after every measurement step.
    screen_names : tuple of str, optional
        Names of the two screens to scan, in order; each must be a key of
        ``env.screens``.
    tcav_modes : tuple of str, optional
        ``env.tcav.mode_config`` values to scan through, in order; each value
        is also used verbatim as the result-key suffix. Defaults to
        ``("STDBY", "ACCEL_STDBY")``.
    reset_tcav_mode : str, optional
        ``mode_config`` value applied to the TCAV after all scans complete,
        by default ``"STDBY"``.
    tcav_settle_time : float, optional
        Wait time in seconds after changing the TCAV mode, by default 2.0.
    dump_location : str or pathlib.Path, optional
        Forwarded to ``run_automatic_emittance`` as the measurement dump
        directory.

    Returns
    -------
    data : dict
        Keyed by ``f"{screen_name}_{mode}"`` for every ``screen_names`` x
        ``tcav_modes`` combination; each value is a dict containing the
        serialized emittance result and captured environment variables.
    tracking_data : pandas.DataFrame
        Concatenated Xopt data frames from all quad scans.
    """
    validate_environment(env)

    saver = H5Saver()
    data = {}
    tracking_data = None

    for name in screen_names:
        screen_name = env.screens[name].name
        for mode in tcav_modes:
            env.tcav.mode_config = mode
            time.sleep(tcav_settle_time)

            logger.info(f"running {screen_name} quad scan tcav {mode}")
            emittance_result, _, X = run_automatic_emittance(
                env,
                screen_name=screen_name,
                dump_location=dump_location,
            )
            data[f"{screen_name}_{mode}"] = emittance_result.model_dump() | {
                "environment_variables": env.get_variables(env.variables.keys())
            }
            tracking_data = (
                X.data
                if tracking_data is None
                else pd.concat([tracking_data, X.data], ignore_index=True)
            )
            saver.dump(data, save_filename)

    # return the TCAV to a safe state after the sequence completes
    env.tcav.mode_config = reset_tcav_mode
    time.sleep(tcav_settle_time)

    return data, tracking_data

