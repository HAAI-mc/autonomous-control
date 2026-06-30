import time
import pytest
import numpy as np

from autonomous_control.facet.env_utils import create_env
from autonomous_control.facet.tcav_phasing import (
    MLTCAVPhasing,
    tcav_phasing,
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
        environment = create_env()

        # testing config for VA
        environment.measure_background = False
        environment.save_directory = "."
        environment.median_filter_size = None
        environment.min_beamsize_cutoff = 2000

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
            max_scan_range=[-10, 10],
            verbose=False,
        )
        nominal_centroid = phaser.acquire_nominal_centroid()

        assert np.isclose(nominal_centroid, 0.0, atol=1e-3), (
            f"Nominal centroid in STDBY should be near 0.0, got {nominal_centroid}"
        )

    def test_run_tcav_phasing(self, env):
        # set the tcav amplitude and phase
        tcav = _get_tcav_or_fail(env)
        set_tcav_mode_config_and_wait(tcav, "ACCEL_STDBY")
        set_tcav_amplitude_and_wait(tcav, 0.3)
        set_tcav_phase_and_wait(tcav, 8.0)

        time.sleep(5.0)  # wait for the VA to settle after changing the TCAV settings

        X = tcav_phasing(
            env,
            max_scan_range=[-10, 10],
            n_iterations=3,
            n_initial_points=3,
        )

        # final phase value should be zero and tcav amplitude should be set to 0.3
        assert np.isclose(env.tcav.amplitude, 0.3, atol=1e-3)
        assert np.isclose(env.tcav.phase, 0.0, atol=0.5), (
            f"Final TCAV phase should be close to 0, but got {env.tcav.phase}"
        )
