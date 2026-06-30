"""Utilities to create the FACET-II badger environment for use in autonomous workflows."""

import epics
import sys
import os


def create_env():
    """
    Create and configure the FACET-II badger environment for use in autonomous workflows.

    """

    # add the path that contains the facet environment
    sys.path.insert(0, os.path.join(os.environ["BADGER_RESOURCES"], "facet"))

    from plugins.environments.inj_emit import Environment
    from plugins.interfaces.epics import Interface

    env = Environment(interface=Interface())

    import torch

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "5"

    return env


def capture_env_state(env) -> dict:
    """
    Capture the current state of the FACET-II badger environment.
    """
    state = env.get_variables(env.variables.keys())
    return state


def restore_env_state(env, state: dict):
    """
    Restore the FACET-II badger environment to a previously captured state.
    """
    env.set_variables(state)


def reset_env(env):
    """
    Reset the FACET-II badger environment to a safe state for autonomous workflows.
    This includes setting the TCAV to standby mode,
    retracting screens, and removing the Faraday cup from the beam path.
    """
    env.tcav.mode_config = "STDBY"
    env.screens["PR10571"].target = 0
    env.screens["PR10711"].target = 0
    epics.caput("FARC:IN10:241:PNEUMATIC", 0)
