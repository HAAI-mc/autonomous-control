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
The repository includes a workflow at `.github/workflows/facet-va-tests.yml` that runs the VA-backed test suite.

The workflow:
- clones `slaclab/Badger-Resources` (private), `slaclab/facet2-lattice`, and `slaclab/virtual-accelerator`
- sets `BADGER_RESOURCES`, `FACET2_LATTICE`, and `PYTHONPATH`
- starts the FACET staged virtual accelerator
- waits for EPICS connectivity
- runs `pytest autonomous_control/tests`

Required repository secret:
- `BADGER_RESOURCES_SSH_KEY`: read-only private SSH deploy key that can clone `git@github.com:slaclab/Badger-Resources.git`

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
