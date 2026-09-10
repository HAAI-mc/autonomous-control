import logging
from pathlib import Path
import pytest

from autonomous_control.facet.auto_6d import run_automatic_6d_measurement
from autonomous_control.facet.env_utils import create_cuhxr_env

logging.basicConfig(level=logging.DEBUG)


class TestAutomaticSixDCUHXR:
    @pytest.fixture
    def env(self):
        environment = create_cuhxr_env()

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
                "IN20:12" in name
                or "BEND" in name
                or "XCOR" in name
                or "YCOR" in name
                or "KLYS" in name
            ):
                del environment.variables[name]

        return environment

    @pytest.xfail("Expected to fail due to incomplete VA for lcls inj")
    def test_run_automatic_6d_measurement_on_va(self, env):
        config_dir = Path(env.emittance_config_fname).parent

        data, tracking_data = run_automatic_6d_measurement(
            env,
            "unused.h5",
            config_files=(config_dir / "OTR2.yaml", config_dir / "OTR2.yaml"), # until we fix the simulation for 711
            tcav_amplitude=0.0,
        )
        print(data)
        print(tracking_data)
