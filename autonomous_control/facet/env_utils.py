"""Utilities to create the FACET-II badger environment for use in autonomous workflows."""

import epics
import sys
import os
from typing import Any, Iterable, Protocol


class ScreenInterface(Protocol):
    """Minimal interface for a screen device used by auto_6d/auto_emittance."""

    name: str
    target: int


class TCAVInterface(Protocol):
    """Minimal interface for a TCAV device used by auto_6d/auto_emittance."""

    mode_config: str


class EnvironmentInterface(Protocol):
    """Informal environment contract required by auto_6d.py and auto_emittance.py.

    Documents (via structural typing) the attributes/methods those modules rely
    on, so other control environments can be swapped in without code changes.

    Assumes ``screens`` is a dict mapping screen name (str) to a Screen object
    (``ScreenInterface``); callers look up individual screens by name via
    ``env.screens[name]``.
    """

    screens: dict[str, ScreenInterface]
    tcav: TCAVInterface
    variables: dict[str, Any]
    save_directory: str
    emittance_config_fname: str

    def get_variables(self, keys: Iterable[str]) -> dict[str, Any]: ...

    def set_variables(self, state: dict[str, Any]) -> None: ...

    def _create_emittance_object(self) -> None: ...

    def run_emittance_measurement(self) -> tuple[Any, str]: ...


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
