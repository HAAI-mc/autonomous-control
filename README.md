Code used to run autonomous operations at FACET-II and AWA.

## Installation
You can install this package by git cloning it and then executing
```bash
pip install -e .
```
in the origin directory.

## Setup
This package requires the environment variable `BADGER_RESOURCES` which points to the location of badger environments on the local machine.
For example, on facet ACR servers this should point to `/home/fphysics/badger/resources/`

## GitHub Actions Integration Testing
VA-backed tests run against two virtual accelerators (VAs), each via its own thin
workflow that calls a shared reusable template:
- `.github/workflows/_va-test-template.yml`: reusable `workflow_call` template with all
  the VA bring-up/test logic (clone resources, start the VA, wait for EPICS, run pytest,
  upload failure artifacts).
- `.github/workflows/facet-va-tests.yml`: runs the `facet_staged` VA (clones
  `slaclab/facet2-lattice`) and the tests marked `facet_va`.
- `.github/workflows/lcls-va-tests.yml`: runs the `cu_hxr_staged` VA (clones
  `slaclab/lcls-lattice`) and the tests marked `lcls_va`.

Both workflows always clone `slaclab/Badger-Resources` (private) and
`slaclab/virtual-accelerator`, set `BADGER_RESOURCES`/`<FACILITY>_LATTICE`/`PYTHONPATH`,
start their VA, wait for EPICS connectivity, and run
`pytest autonomous_control/tests -m <marker>`.

Required repository secret:
- `BADGER_RESOURCES_SSH_KEY`: read-only private SSH deploy key that can clone `git@github.com:slaclab/Badger-Resources.git`

### Tagging tests to a VA
Tests are assigned to a VA with a pytest marker, not file naming, since a single file can
mix tests for multiple VAs (see `test_automatic_alignment.py`). Markers are registered in
`pyproject.toml` under `[tool.pytest.ini_options]` with `--strict-markers` enabled, so an
unregistered marker fails collection instead of silently doing nothing.

### Adding a new VA
1. Register a new `<facility>_va` marker in `pyproject.toml`.
2. Tag the relevant tests with `pytestmark = pytest.mark.<facility>_va` (module/class
   level) or `@pytest.mark.<facility>_va` (per test, for files mixing multiple VAs).
3. Add a new thin caller workflow (copy `lcls-va-tests.yml`) that calls
   `_va-test-template.yml` with that VA's `va_model`, `lattice_repo`, `lattice_env_var`,
   `pythonpath_subdir`, `end_element`, `readiness_pv`, `pytest_marker`, and
   `artifact_name_suffix` inputs. No changes to the template itself are needed unless the
   new VA requires a different simulator backend (see `extra_conda_packages` input).

The workflow triggers on pull requests, pushes, manual dispatch, and a nightly schedule.

## Usage
See AGENT.md for a usage guide via CLI.

For using the python interface, see the code snippet below for an example
```python
from autonomous_control.facet.runner import run_automatic_workflow

workflow = [
    {
        "type": "measure_emittance",
        "config_file": "PR10571.yaml",
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
```
