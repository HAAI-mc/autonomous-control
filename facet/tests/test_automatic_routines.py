import os
import sys
import pytest
import numpy as np

# add the path that contains the facet environment
sys.path.insert(0, os.path.join(os.environ["BADGER_RESOURCES"], "facet"))

# add the autonomous control directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from plugins.environments.inj_emit import Environment
from plugins.interfaces.epics import Interface

from facet.auto_emittance import run_automatic_emittance
from facet.tcav_phasing import run_automatic_tcav_phasing

import logging
logging.basicConfig(level=logging.DEBUG)


class TestAutomaticRoutines:
    @pytest.fixture
    def env(self):
        environment = Environment(interface=Interface())
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

    def test_screen_measurement(self, env):
        assert env.screens["PROF10571"].image.sum() > 0, (
            "Screen PROF10571 should have a non-empty image"
        )
        assert env.screens["PROF10711"].image.sum() > 0, (
            "Screen PROF10711 should have a non-empty image"
        )

        env.create_beamprofile_measurement("PROF10571").measure()
        env.create_beamprofile_measurement("PROF10711").measure()

    def test_run_automatic_emittance_on_va(self, env):
        result, fname, X = run_automatic_emittance(env, "PROF10571")
        print(X)

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

        # set the tcav amplitude
        env.tcav.amplitude = 0.3

        X = run_automatic_tcav_phasing(env, max_scan_range=[-10, 10])
        
        # final phase value should be zero and tcav amplitude should be set to 0.3
        assert np.isclose(env.tcav.amplitude, 0.3, atol=1e-3)
        assert np.isclose(env.tcav.phase, 0.0, atol=1.0), (
            f"Final TCAV phase should be close to 0, but got {env.tcav.phase}"
        )

