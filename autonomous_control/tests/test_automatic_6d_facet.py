import logging

import pytest

from autonomous_control.facet.auto_6d import run_automatic_6d_measurement
from autonomous_control.facet.env_utils import create_facet_env

logging.basicConfig(level=logging.DEBUG)


class TestAutomaticSixDFacet:
    @pytest.fixture
    def env(self):
        environment = create_facet_env()

        # testing config for VA
        environment.measure_background = False
        environment.save_directory = "."
        environment.median_filter_size = None
        environment.min_beamsize_cutoff = 2000
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

    def test_run_automatic_6d_measurement_on_va(self, env):
        data, tracking_data = run_automatic_6d_measurement(
            env, "unused.h5", screen_names=("PR10571", "PR10711")
        )
        print(data)
        print(tracking_data)
