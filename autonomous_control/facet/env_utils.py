"""Utilities to create the FACET-II badger environment for use in autonomous workflows."""

import epics
import sys
import os
from typing import Any, Iterable, Protocol

from lcls_tools.common.devices.screen import Screen
from lcls_tools.common.devices.tcav import TCAV


class EnvironmentInterface(Protocol):
    """Informal environment contract required by auto_6d.py and auto_emittance.py.

    Documents (via structural typing) the attributes/methods those modules rely
    on, so other control environments can be swapped in without code changes.

    Assumes ``screens`` is a dict mapping screen name (str) to a
    ``lcls_tools`` ``Screen`` object; callers look up individual screens by
    name via ``env.screens[name]``.
    """

    screens: dict[str, Screen]
    tcav: TCAV
    variables: dict[str, Any]
    save_directory: str
    emittance_config_fname: str

    def get_variables(self, keys: Iterable[str]) -> dict[str, Any]: ...

    def set_variables(self, state: dict[str, Any]) -> None: ...

    def _create_emittance_object(self) -> None: ...

    def run_emittance_measurement(self) -> tuple[Any, str]: ...


def validate_environment(env) -> None:
    """Validate that ``env`` satisfies the ``EnvironmentInterface`` contract.

    Parameters
    ----------
    env : Any
        Candidate control environment to validate.

    Raises
    ------
    TypeError
        If ``env`` is missing required attributes/methods, or ``env.screens``
        values / ``env.tcav`` are not ``lcls_tools`` ``Screen``/``TCAV``
        instances.
    """
    errors = []

    screens = getattr(env, "screens", None)
    if not isinstance(screens, dict):
        errors.append(
            f"env.screens must be a dict[str, Screen], got {type(screens).__name__}"
        )
    else:
        for name, screen in screens.items():
            if not isinstance(screen, Screen):
                errors.append(
                    f"env.screens[{name!r}] must be a Screen instance, "
                    f"got {type(screen).__name__}"
                )

    tcav = getattr(env, "tcav", None)
    if not isinstance(tcav, TCAV):
        errors.append(f"env.tcav must be a TCAV instance, got {type(tcav).__name__}")

    for attr in ("variables", "save_directory", "emittance_config_fname"):
        if not hasattr(env, attr):
            errors.append(f"env.{attr} is required but missing")

    for method in (
        "get_variables",
        "set_variables",
        "_create_emittance_object",
        "run_emittance_measurement",
    ):
        if not callable(getattr(env, method, None)):
            errors.append(f"env.{method}() is required but missing or not callable")

    if errors:
        raise TypeError(
            "Invalid environment; does not satisfy EnvironmentInterface:\n  - "
            + "\n  - ".join(errors)
        )


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
