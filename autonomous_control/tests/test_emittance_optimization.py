import os
import sys
import pytest

# add the path that contains the facet environment
sys.path.insert(0, os.path.join(os.environ["BADGER_RESOURCES"], "facet"))

from plugins.environments.inj_emit import Environment
from plugins.interfaces.epics import Interface

from autonomous_control.facet.emittance_opt import optimize_injector_emittance

import logging

logging.basicConfig(level=logging.DEBUG)


class TestAutomaticEmittance:
    @pytest.fixture
    def env(self):
        environment = Environment(interface=Interface())
        environment.measure_background = False
        environment.save_directory = "."
        environment.median_filter_size = None
        environment.min_beamsize_cutoff = 2000
        environment.min_bmag_threshold = 2000
        environment.n_iterations = 1
        environment.n_interpolate_points = 1


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

    def test_run_emittance_opt_on_va(self, env):
        X = optimize_injector_emittance(
            env,
            variables={"QUAD:IN10:511:BCTRL": [5.0, 6.0]},
            n_steps=1,
            min_joint_bmag_constraint=2000,
        )
        assert len(X.data) == 5
