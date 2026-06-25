import os
import sys
import time
import pytest
import numpy as np

# add the path that contains the facet environment
sys.path.insert(0, os.path.join(os.environ["BADGER_RESOURCES"], "facet"))

# add the autonomous control directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from plugins.environments.inj_emit import Environment
from plugins.interfaces.epics import Interface

from facet.auto_emittance import run_automatic_emittance
from facet.tcav_phasing import (
    MLTCAVPhasing,
    run_automatic_tcav_phasing,
    set_tcav_amplitude_and_wait,
    set_tcav_mode_config_and_wait,
    set_tcav_phase_and_wait,
)

import logging
logging.basicConfig(level=logging.DEBUG)


def _get_tcav_or_fail(env):
    """Return a stable TCAV instance, failing hard when unavailable."""
    try:
        tcav = env.tcav
    except Exception as exc:
        pytest.fail(f"Unable to create TCAV object for utility test: {exc}")

    # Ensure repeated property access in a test reuses the same TCAV object.
    if hasattr(env, "_tcav"):
        env._tcav = tcav

    return tcav


class TestAutomaticTcavPhasing:
    @pytest.fixture
    def env(self):
        environment = Environment(interface=Interface())
        environment.measure_background = False
        environment.save_directory = "."
        environment.median_filter_size = None
        environment.min_beamsize_cutoff = 2000
        environment.upstream_bpm_name = "BPM10371"
        environment.downstream_bpm_name = "BPM10651"

        # remove PVs that are not supported by the VA
        for name in list(environment.variables.keys()):
            if (
                "IN10:12" in name
                or "BEND" in name
                or "XCOR" in name
                or "YCOR" in name
                or "KLYS" in name
            ):
                del environment.variables[name]

        return environment

    def test_set_tcav_mode_config_and_wait(self, env):
        tcav = _get_tcav_or_fail(env)

        for mode in ["STDBY", "ACCEL_STDBY"]:
            set_tcav_mode_config_and_wait(tcav, mode)
            assert tcav.mode_config == mode

    def test_set_tcav_amplitude_and_wait(self, env):
        tcav = _get_tcav_or_fail(env)

        for target_amplitude in [0.0, 0.3, 0.0]:
            set_tcav_mode_config_and_wait(tcav, "ACCEL_STDBY")
            set_tcav_amplitude_and_wait(
                tcav,
                target_amplitude,
            )
            assert np.isclose(float(tcav.amplitude_wocho), target_amplitude, atol=1e-3)

    def test_set_tcav_phase_and_wait(self, env):
        tcav = _get_tcav_or_fail(env)

        for target_phase in [0.0, 8.0, 0.0]:
            set_tcav_mode_config_and_wait(tcav, "ACCEL_STDBY")
            set_tcav_phase_and_wait(
                tcav,
                target_phase,
            )
            assert np.isclose(float(tcav.phase_avgnt), target_phase, atol=0.5)

    def test_acquire_nominal_centroid(self, env):
        tcav = _get_tcav_or_fail(env)
        transmission_measurement = env.transmission_measurement

        env.upstream_bpm_name = "BPM10371"
        env.downstream_bpm_name = "BPM10651"

        assert env.downstream_bpm is not None

        # Start from streaking-like conditions before calling acquire_nominal_centroid
        set_tcav_mode_config_and_wait(tcav, "ACCEL_STDBY")
        set_tcav_amplitude_and_wait(tcav, 0.3)
        set_tcav_phase_and_wait(tcav, 8.0)

        phaser = MLTCAVPhasing(
            bpm=env.downstream_bpm,
            tcav=tcav,
            transmission_measurement=transmission_measurement,
            wait_time=1e-6,
            max_scan_range=[-10, 10],
            verbose=False,
        )
        nominal_centroid = phaser.acquire_nominal_centroid()

        assert np.isclose(nominal_centroid, 0.0, atol=1e-3), (
            f"Nominal centroid in STDBY should be near 0.0, got {nominal_centroid}"
        )



    def test_run_tcav_phasing(self, env):
        env.upstream_bpm_name = "BPM10371"
        env.downstream_bpm_name = "BPM10651"

        assert env.upstream_bpm is not None
        assert env.downstream_bpm is not None

        assert env.upstream_bpm.x is not None
        assert env.upstream_bpm.y is not None
        assert env.upstream_bpm.tmit is not None

        # assert the TCAV attributes are not none
        assert env.tcav.amplitude is not None
        assert env.tcav.phase is not None

        # set the tcav amplitude and phase
        tcav = _get_tcav_or_fail(env)
        set_tcav_mode_config_and_wait(tcav, "ACCEL_STDBY")
        set_tcav_amplitude_and_wait(tcav, 0.3)
        set_tcav_phase_and_wait(tcav, 8.0)        

        X = run_automatic_tcav_phasing(env, max_scan_range=[-10, 10])

        # final phase value should be zero and tcav amplitude should be set to 0.3
        assert np.isclose(env.tcav.amplitude, 0.3, atol=1e-3)
        assert np.isclose(env.tcav.phase, 0.0, atol=0.5), (
            f"Final TCAV phase should be close to 0, but got {env.tcav.phase}"
        )

