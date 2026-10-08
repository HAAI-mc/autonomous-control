from dataclasses import dataclass, field

import pytest

import logging

from autonomous_control.env_utils import create_facet_env, create_cuhxr_env
from autonomous_control.alignment_opt_es import optimize_alignment

logging.basicConfig(level=logging.DEBUG)


@dataclass
class _AlignmentCase:
    area: str
    create_env: callable
    perturbation_element: int
    perturbation_value: float
    corrector_x_elements: list
    corrector_y_elements: list
    bpm_elements: list
    extra_remove_patterns: tuple = field(default_factory=tuple)


# shared VA-unsupported-PV patterns; per-case extras are appended in _build_environment
_COMMON_REMOVE_PATTERNS = ("BEND", "KLYS")

_FACET_CASE = _AlignmentCase(
    area="IN10",
    create_env=create_facet_env,
    perturbation_element=311,
    perturbation_value=0.001,
    corrector_x_elements=[311, 381, 411, 491, 521, 641],
    corrector_y_elements=[312, 382, 412, 492, 522, 642],
    bpm_elements=[371, 425, 511, 525, 581, 631, 651],
)

_CUHXR_CASE = _AlignmentCase(
    area="IN20",
    create_env=create_cuhxr_env,
    perturbation_element=491,
    perturbation_value=0.005,
    corrector_x_elements=[491, 521, 641],
    corrector_y_elements=[492, 522, 642],
    bpm_elements=[581, 631, 651],
    extra_remove_patterns=("IN20:12",),
)


def _build_environment(case: _AlignmentCase):
    environment = case.create_env()

    # remove PVs that are not supported by the VA (same XCOR/YCOR element numbers removed per area)
    remove_patterns = (
        f"XCOR:{case.area}:121:BCTRL",
        f"XCOR:{case.area}:221:BCTRL",
        f"YCOR:{case.area}:122:BCTRL",
        f"YCOR:{case.area}:222:BCTRL",
    ) + _COMMON_REMOVE_PATTERNS + case.extra_remove_patterns
    for name in list(environment.variables.keys()):
        if any(pattern in name for pattern in remove_patterns):
            del environment.variables[name]

    return environment


class TestAutomaticAlignment:
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(_FACET_CASE, id="facet", marks=pytest.mark.facet_va),
            pytest.param(_CUHXR_CASE, id="cuhxr", marks=pytest.mark.lcls_va),
        ],
    )
    def test_automatic_alignment(self, case):
        environment = _build_environment(case)

        # peturbation variable for alignment optimization
        environment.set_variables(
            {f"XCOR:{case.area}:{case.perturbation_element}:BCTRL": case.perturbation_value}
        )

        corrector_pvs = [
            f"XCOR:{case.area}:{ele}:BCTRL" for ele in case.corrector_x_elements
        ] + [f"YCOR:{case.area}:{ele}:BCTRL" for ele in case.corrector_y_elements]

        bpm_pvs = [f"BPMS:{case.area}:{ele}:X" for ele in case.bpm_elements]

        # the VA has default +/- 100 value range for correctors
        n_steps = 10
        X = optimize_alignment(
            environment,
            corrector_pvs=corrector_pvs,
            bpm_observable_pvs=bpm_pvs,
            upstream_charge_pv=f"BPMS:{case.area}:371:TMIT",
            downstream_charge_pv=f"BPMS:{case.area}:651:TMIT",
            region_fraction=1e-4,
            n_steps=n_steps,
            steering_settle_time=5.0,  # VA is slow, so we have to wait a bit for the readbacks to settle
            target_value=1.0e-8,  # never going to converge so it runs 10 iterations
        )
        assert len(X.data) == n_steps + 2  # +2 for the initial measurement

        # assert that BPM readings are unique after the second row and before the last row
        # this ensures that the optimization is waiting for readbacks
        # to stabilize before taking the next measurement
        bpm_readings = X.data.filter(like=f"BPMS:{case.area}", axis=1)
        assert bpm_readings.iloc[1:-1].nunique().sum() == bpm_readings.iloc[1:-1].size