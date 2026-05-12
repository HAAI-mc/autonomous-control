import os
import sys
import pytest

# add the path that contains the facet environment
sys.path.insert(0, os.path.join(os.environ["BADGER_RESOURCES"], "facet"))

# add the autonomous control directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from plugins.environments.inj_emit import Environment
from plugins.interfaces.epics import Interface

from facet.auto_emittance import run_automatic_emittance


class TestAutomaticEmittance:
    @pytest.fixture
    def env(self):
        environment = Environment(interface=Interface())
        environment.measure_background = False
        environment.save_directory = "."

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
