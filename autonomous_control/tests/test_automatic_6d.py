"""Hardware-free tests for run_automatic_6d_measurement's generalized sequence."""

import pandas as pd
import pytest

import autonomous_control.facet.auto_6d as auto_6d_module
from autonomous_control.facet.auto_6d import run_automatic_6d_measurement


class _FakeResult:
    def __init__(self, screen_name, mode):
        self._screen_name = screen_name
        self._mode = mode

    def model_dump(self):
        return {"screen_name": self._screen_name, "mode": self._mode}


class _FakeXopt:
    def __init__(self, row):
        self.data = pd.DataFrame([row])


class _FakeScreen:
    def __init__(self, name):
        self.name = name
        self.target = 0


class _FakeTCAV:
    def __init__(self):
        self.mode_config = None


class _FakeEnv:
    def __init__(self, screen_names=("PR10571", "PR10711")):
        self.tcav = _FakeTCAV()
        self.screens = {name: _FakeScreen(name) for name in screen_names}
        self.save_directory = "."
        self.variables = {"PV1": 1.0}

    def get_variables(self, keys):
        return {k: self.variables[k] for k in keys}


@pytest.fixture(autouse=True)
def _no_disk_io(monkeypatch):
    # avoid writing real HDF5 files during these unit tests
    monkeypatch.setattr(
        "lcls_tools.common.data.saver.H5Saver.dump", lambda self, data, filepath: None
    )


def _fake_run_automatic_emittance(env, screen_name, dump_location=None):
    mode = env.tcav.mode_config
    return _FakeResult(screen_name, mode), "fake.h5", _FakeXopt(
        {"screen_name": screen_name, "mode": mode}
    )


class TestAutomaticSixD:
    def test_default_args_preserve_existing_key_order(self, monkeypatch):
        monkeypatch.setattr(
            auto_6d_module, "run_automatic_emittance", _fake_run_automatic_emittance
        )
        env = _FakeEnv()

        data, tracking_data = run_automatic_6d_measurement(env, "unused.h5")

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

        def recording_run_automatic_emittance(env, screen_name, dump_location=None):
            calls.append((screen_name, env.tcav.mode_config, dump_location))
            return _fake_run_automatic_emittance(
                env, screen_name, dump_location=dump_location
            )

        monkeypatch.setattr(
            auto_6d_module,
            "run_automatic_emittance",
            recording_run_automatic_emittance,
        )
        env = _FakeEnv(screen_names=("SCREEN_A", "SCREEN_B", "SCREEN_C"))

        data, _ = run_automatic_6d_measurement(
            env,
            "unused.h5",
            screen_names=("SCREEN_A", "SCREEN_B", "SCREEN_C"),
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
            ("SCREEN_A", "MODE_LO", None),
            ("SCREEN_A", "MODE_HI", None),
            ("SCREEN_B", "MODE_LO", None),
            ("SCREEN_B", "MODE_HI", None),
            ("SCREEN_C", "MODE_LO", None),
            ("SCREEN_C", "MODE_HI", None),
        ]
        assert env.tcav.mode_config == "MODE_RESET"
