"""Hardware-free tests for run_automatic_6d_measurement's generalized sequence."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from lcls_tools.common.devices.screen import Screen
from lcls_tools.common.devices.tcav import TCAV

import autonomous_control.auto_6d as auto_6d_module
from autonomous_control.auto_6d import run_automatic_6d_measurement


class _FakeResult:
    def __init__(self, screen_name, mode):
        self._screen_name = screen_name
        self._mode = mode

    def model_dump(self):
        return {"screen_name": self._screen_name, "mode": self._mode}


class _FakeXopt:
    def __init__(self, row):
        self.data = pd.DataFrame([row])


def _make_fake_screen(name):
    screen = MagicMock(spec=Screen)
    screen.name = name
    screen.target = 0
    return screen


def _make_fake_tcav():
    tcav = MagicMock(spec=TCAV)
    tcav.mode_config = None
    return tcav


class _FakeEnv:
    def __init__(self, screen_names=("PR10571", "PR10711")):
        self.tcav = _make_fake_tcav()
        self.screens = {name: _make_fake_screen(name) for name in screen_names}
        self.save_directory = "."
        self.emittance_config_fname = ""
        self.variables = {"PV1": 1.0}

    def get_variables(self, keys):
        return {k: self.variables[k] for k in keys}

    def set_variables(self, state):
        self.variables.update(state)

    def _create_emittance_object(self):
        pass

    def run_emittance_measurement(self):
        pass


@pytest.fixture(autouse=True)
def _no_disk_io(monkeypatch):
    # avoid writing real HDF5 files during these unit tests
    monkeypatch.setattr(
        "lcls_tools.common.data.saver.H5Saver.dump", lambda self, data, filepath: None
    )


def _fake_run_automatic_emittance(
    env,
    config_file,
    dump_location=None,
):
    mode = env.tcav.mode_config
    screen_name = Path(config_file).stem
    return _FakeResult(screen_name, mode), "fake.h5", _FakeXopt(
        {"screen_name": screen_name, "mode": mode}
    )


class TestAutomaticSixD:
    def test_default_args_preserve_existing_key_order(self, monkeypatch):
        monkeypatch.setattr(
            auto_6d_module, "run_automatic_emittance", _fake_run_automatic_emittance
        )
        monkeypatch.setattr(
            auto_6d_module,
            "resolve_emittance_config",
            lambda config_file: (
                config_file,
                Path(config_file).stem,
                {Path(config_file).stem: 1},
            ),
        )
        env = _FakeEnv()

        data, tracking_data = run_automatic_6d_measurement(
            env,
            "unused.h5",
            config_files=("PR10571.yaml", "PR10711.yaml"),
        )

        assert list(data.keys()) == [
            "PR10571_STDBY",
            "PR10571_ACCEL_STDBY",
            "PR10711_STDBY",
            "PR10711_ACCEL_STDBY",
        ]
        assert len(tracking_data) == 4
        assert env.tcav.mode_config == "STDBY"

    def test_custom_screen_names_and_tcav_modes(self, monkeypatch):
        calls = []

        def recording_run_automatic_emittance(
            env,
            config_file,
            dump_location=None,
        ):
            screen_name = Path(config_file).stem
            calls.append(
                (
                    screen_name,
                    env.tcav.mode_config,
                    dump_location,
                    config_file,
                )
            )
            return _fake_run_automatic_emittance(
                env,
                config_file,
                dump_location=dump_location,
            )

        monkeypatch.setattr(
            auto_6d_module,
            "run_automatic_emittance",
            recording_run_automatic_emittance,
        )
        monkeypatch.setattr(
            auto_6d_module,
            "resolve_emittance_config",
            lambda config_file: (
                config_file,
                Path(config_file).stem,
                {Path(config_file).stem: 1},
            ),
        )
        env = _FakeEnv(screen_names=("SCREEN_A", "SCREEN_B", "SCREEN_C"))

        data, _ = run_automatic_6d_measurement(
            env,
            "unused.h5",
            config_files=("SCREEN_A.yaml", "SCREEN_B.yaml", "SCREEN_C.yaml"),
            tcav_modes=("MODE_LO", "MODE_HI"),
            reset_tcav_mode="MODE_RESET",
        )

        assert list(data.keys()) == [
            "SCREEN_A_MODE_LO",
            "SCREEN_A_MODE_HI",
            "SCREEN_B_MODE_LO",
            "SCREEN_B_MODE_HI",
            "SCREEN_C_MODE_LO",
            "SCREEN_C_MODE_HI",
        ]
        assert calls == [
            (
                "SCREEN_A",
                "MODE_LO",
                None,
                "SCREEN_A.yaml",
            ),
            (
                "SCREEN_A",
                "MODE_HI",
                None,
                "SCREEN_A.yaml",
            ),
            (
                "SCREEN_B",
                "MODE_LO",
                None,
                "SCREEN_B.yaml",
            ),
            (
                "SCREEN_B",
                "MODE_HI",
                None,
                "SCREEN_B.yaml",
            ),
            (
                "SCREEN_C",
                "MODE_LO",
                None,
                "SCREEN_C.yaml",
            ),
            (
                "SCREEN_C",
                "MODE_HI",
                None,
                "SCREEN_C.yaml",
            ),
        ]
        assert env.tcav.mode_config == "MODE_RESET"

    def test_invalid_environment_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            auto_6d_module, "run_automatic_emittance", _fake_run_automatic_emittance
        )
        monkeypatch.setattr(
            auto_6d_module,
            "resolve_emittance_config",
            lambda config_file: (
                config_file,
                Path(config_file).stem,
                {Path(config_file).stem: 1},
            ),
        )
        env = _FakeEnv()
        env.screens["PR10571"] = object()  # not a Screen instance

        with pytest.raises(TypeError, match="Screen instance"):
            run_automatic_6d_measurement(
                env,
                "unused.h5",
                config_files=("PR10571.yaml", "PR10711.yaml"),
            )
