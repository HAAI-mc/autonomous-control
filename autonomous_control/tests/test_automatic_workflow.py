import os
import sys
import pytest

badger_resources = os.getenv("BADGER_RESOURCES")
if badger_resources is None:
    pytest.skip("BADGER_RESOURCES is not configured", allow_module_level=True)

# add the path that contains the facet environment
sys.path.insert(0, os.path.join(badger_resources, "facet"))

# add the autonomous control directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from autonomous_control.facet.runner import run_automatic_workflow

from plugins.environments.inj_emit import Environment
from plugins.interfaces.epics import Interface


class TestAutomaticWorkflow:
    @pytest.fixture
    def env(self):
        environment = Environment(interface=Interface())
        environment.measure_background = False
        environment.save_directory = "."
        environment.median_filter_size = None
        environment.min_beamsize_cutoff = 2000
        environment.upstream_bpm_name = "BPM10371"
        environment.downstream_bpm_name = "BPM10651"

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

    def test_run_automatic_workflow_on_va(self, env):
        # define a simple workflow with two steps
        workflow = [
            {
                "type": "measure_emittance",
                "screen_name": "PROF10571",
            },
            {
                "type": "tcav_phasing",
                "max_scan_range": [-10, 10],
                "n_iterations": 3,
                "n_initial_points": 3,
                "tcav_on_amplitude": 0.3,
            },
        ]
        log_file = run_automatic_workflow(workflow, env)

        # check to make sure that info has been logged for each step
        with open(log_file, "r") as f:
            log_contents = f.read()
            assert "Starting automatic workflow..." in log_contents
            assert "Running workflow step: measure_emittance" in log_contents
            assert "Running workflow step: tcav_phasing" in log_contents
            assert "Automatic workflow completed." in log_contents
