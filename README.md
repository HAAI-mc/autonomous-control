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

## Usage
See AGENT.md for a usage guide via CLI.

For using the python interface, see the code snippet below for an example
```python
from autonomous_control.facet.runner import run_automatic_workflow

workflow = [
    {
        "type": "measure_emittance",
        "screen_name": "PR10571",
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
