import pytest

import logging

from autonomous_control.env_utils import create_facet_env, create_cuhxr_env
from autonomous_control.alignment_opt_es import optimize_alignment

logging.basicConfig(level=logging.DEBUG)


class TestAutomaticAlignment:
    def test_automatic_alignment_on_facet(self):
        environment = create_facet_env()

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


        # peturbation variables for alignment optimization
        environment.set_variables({"XCOR:IN10:311:BCTRL": 0.001})

        corrector_pvs = [
            f"XCOR:IN10:{ele}:BCTRL" for ele in [311, 381, 411, 491, 521, 641]
        ] + [f"YCOR:IN10:{ele}:BCTRL" for ele in [312, 382, 412, 492, 522, 642]]

        bpm_pvs = [f"BPMS:IN10:{ele}:X" for ele in [371, 425, 511, 525, 581, 631, 651]]

        # the VA has default +/- 100 value range for correctors
        n_steps = 10
        X = optimize_alignment(
            environment,
            corrector_pvs=corrector_pvs,
            bpm_observable_pvs=bpm_pvs,
            region_fraction=1e-4,
            n_steps=n_steps,
            steering_settle_time=5.0,  # VA is slow, so we have to wait a bit for the readbacks to settle
        )
        assert len(X.data) == n_steps + 2  # +2 for the initial measurement

        # assert that BPM readings are unique after the second row and before the last row
        # this ensures that the optimization is waiting for readbacks
        # to stabilize before taking the next measurement
        bpm_readings = X.data.filter(like="BPMS:IN10", axis=1)
        assert bpm_readings.iloc[1:-1].nunique().sum() == bpm_readings.iloc[1:-1].size

    def test_automatic_alignment_on_cuhxr(self):
        environment = create_cuhxr_env()

        # remove PVs that are not supported by the VA
        for name in list(environment.variables.keys()):
            if (
                "XCOR:IN20:121:BCTRL" in name
                or "XCOR:IN20:221:BCTRL" in name
                or "YCOR:IN20:122:BCTRL" in name
                or "YCOR:IN20:222:BCTRL" in name
                or "BEND" in name
                or "KLYS" in name
                or "IN20:12" in name
            ):
                del environment.variables[name]


        # peturbation variables for alignment optimization
        environment.set_variables({"XCOR:IN20:491:BCTRL": 0.005})

        corrector_pvs = [
            f"XCOR:IN20:{ele}:BCTRL" for ele in [491, 521, 641]
        ] + [f"YCOR:IN20:{ele}:BCTRL" for ele in [492, 522, 642]]
        bpm_pvs = [f"BPMS:IN20:{ele}:X" for ele in [581, 631, 651]]

        # the VA has default +/- 100 value range for correctors
        n_steps = 10
        X = optimize_alignment(
            environment,
            corrector_pvs=corrector_pvs,
            bpm_observable_pvs=bpm_pvs,
            upstream_charge_pv="BPMS:IN20:371:TMIT",
            downstream_charge_pv="BPMS:IN20:651:TMIT",
            region_fraction=1e-4,
            n_steps=n_steps,
            steering_settle_time=5.0,  # VA is slow, so we have to wait a bit for the readbacks to settle
            target_value=1.0e-8, # never going to converge so it runs 10 iterations
        )
        assert len(X.data) == n_steps + 2  # +2 for the initial measurement

        # assert that BPM readings are unique after the second row and before the last row
        # this ensures that the optimization is waiting for readbacks
        # to stabilize before taking the next measurement
        bpm_readings = X.data.filter(like="BPMS:IN20", axis=1)
        assert bpm_readings.iloc[1:-1].nunique().sum() == bpm_readings.iloc[1:-1].size