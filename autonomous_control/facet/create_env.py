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


def reset_env(env):
    """Reset the FACET-II badger environment to a safe state for autonomous workflows."""
    env.tcav.mode_config = "STDBY"
    env.screens["PR10571"].target = 0
    env.screens["PR10711"].target = 0
    epics.caput("FARC:IN10:241:PNEUMATIC", 0)
