import pytest

import logging

from autonomous_control.facet.auto_emittance import run_automatic_emittance
from autonomous_control.facet.env_utils import create_env

logging.basicConfig(level=logging.DEBUG)


class TestAutomaticEmittance:
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

    def test_screen_measurement(self, env):
        assert env.screens["PR10571"].image.sum() > 0, (
            "Screen PR10571 should have a non-empty image"
        )
        assert env.screens["PR10711"].image.sum() > 0, (
            "Screen PR10711 should have a non-empty image"
        )

        env.create_beamprofile_measurement("PR10571").measure()
        env.create_beamprofile_measurement("PR10711").measure()

    def test_run_automatic_emittance_on_va(self, env):
        result, fname, X = run_automatic_emittance(env, screen_name="PR10571")
        print(X)
