import pytest

import logging

from autonomous_control.facet.env_utils import create_env
from autonomous_control.facet.alignment_opt_es import optimize_alignment

logging.basicConfig(level=logging.INFO)


class TestAutomaticAlignment:
    @pytest.fixture
    def env(self):
        environment = create_env()

        # remove PVs that are not supported by the VA
        for name in list(environment.variables.keys()):
            if (
                "XCOR:IN10:121:BCTRL" in name
                or "XCOR:IN10:221:BCTRL" in name
                or "YCOR:IN10:122:BCTRL" in name
                or "YCOR:IN10:222:BCTRL" in name
                or "BEND" in name
                or "KLYS" in name
            ):
                del environment.variables[name]

        return environment

    def test_automatic_alignment_on_va(self, env):
        # peturbation variables for alignment optimization
        env.set_variables({"XCOR:IN10:311:BCTRL": 0.001})

        custom_corrector_pvs = [
            f"XCOR:IN10:{ele}:BCTRL" for ele in [311, 381, 411, 491, 521, 641]
        ] + [f"YCOR:IN10:{ele}:BCTRL" for ele in [312, 382, 412, 492, 522, 642]]

        # the VA has default +/- 100 value range for correctors
        n_steps = 10
        X = optimize_alignment(
            env,
            region_fraction=1e-4,
            n_steps=n_steps,
            custom_corrector_pvs=custom_corrector_pvs,
        )
        assert len(X.data) == n_steps + 2  # +2 for the initial measurement

        # assert that BPM readings are unique after the second row
        # this ensures that the optimization is waiting for readbacks
        # to stabilize before taking the next measurement
        bpm_readings = X.data.filter(like="BPMS:IN10", axis=1)
        assert bpm_readings.iloc[1:].nunique().sum() == bpm_readings.iloc[1:].size
