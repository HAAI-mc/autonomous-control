# Autonomous Control Workflow Guide

This guide explains the high-level FACET workflows supported by the workflow runner and how to execute them from a YAML file.

## Badger resources
Make sure that the environment variable `BADGER_RESOURCES` points to the local Badger-Resources directory.

## YAML Interface

The workflow runner (`autonomous_control.runner`) expects a YAML list of steps.

Each step is a mapping with:
- `type`: workflow step name
- additional keys: parameters passed to that step

General shape:

```yaml
- type: some_workflow_type
  param_a: value
  param_b: value
- type: another_workflow_type
  param_c: value
```

Supported `type` values. These are the only types registered in `STEP_HANDLERS` in `autonomous_control/runner.py`, each mapping 1:1 to a top-level callable:
- `measure_emittance` -> `autonomous_control.auto_emittance.measure_emittance`
- `optimize_alignment` -> `autonomous_control.alignment_opt_es.optimize_alignment`
- `tcav_phasing` -> `autonomous_control.auto_tcav_phasing.tcav_phasing`
- `optimize_laser_steering` -> `autonomous_control.auto_laser_steering.optimize_laser_steering`

Additional automation routines exist under `autonomous_control.facet.*` (and `autonomous_control.auto_6d`) but are **not** wired into `STEP_HANDLERS`, so they cannot be used as a YAML step `type`. Call them directly from Python instead — see [Facet-Only Automation Functions](#facet-only-automation-functions-not-available-as-yaml-steps).

A per-step `dump_location` key is honored by every handler above; each writes its own Xopt dump file there if set. This is independent of the runner's top-level `--dump_location`, which only controls where the runner writes its own aggregate `automatic_workflow_xopt_<timestamp>.yaml` file (see below).

## How To Run A YAML Workflow

The CLI entry point does not currently accept an `--env` argument, and `run_automatic_workflow_from_file` has no default for its required `env` parameter, so invoking `python -m autonomous_control.runner ...` directly will raise `TypeError: missing 1 required positional argument: 'env'`. Until the CLI is updated to construct or accept an environment, run workflows from Python instead:

```python
from autonomous_control.env_utils import create_facet_env
from autonomous_control.runner import run_automatic_workflow_from_file

env = create_facet_env()
log_file = run_automatic_workflow_from_file(
    "autonomous_control/facet/workflows/start_to_end.yml",
    env=env,
    dump_location="results",
    reset_env_after=True,
    logging_level="INFO",
)
```

Notes:
- `dump_location` is optional (default `"."`). When provided, the runner creates the directory and writes its aggregate Xopt dump file there; it is not forwarded to individual step handlers (set a per-step `dump_location` key for that).
- `reset_env_after` defaults to `True` in `run_automatic_workflow`/`run_automatic_workflow_from_file`, but defaults to `False` when run through the CLI argparse flag (`--reset_env_after` is `store_true`).
- A log file named `automatic_workflow_<timestamp>.log` is written in the current working directory.
- The example workflow files under `autonomous_control/facet/workflows/` (e.g. `start_to_end.yml`) currently reference step types (`optimize_schottky`, `minimize_energy_spread`, `minimize_injector_emittance`) that are not registered in `STEP_HANDLERS`; running them as-is with `autonomous_control.runner` will raise `ValueError: Unknown workflow step type`.

## High-Level Workflow Steps

### 1) `measure_emittance`
Purpose:
- Perform quadrupole-scan emittance measurement at a selected screen.

Main parameters:
- `config_file` (required): path to the emittance YAML config; the file defines `screen.name` and `screen_targets`
- `screen_settle_time` (optional, default `2.0`): seconds to wait after screen target change
- `dump_location` (optional): Xopt dump file path

YAML step example:

```yaml
- type: measure_emittance
  config_file: /path/to/emittance_measurement_configs/PR10571.yaml
  screen_settle_time: 2.0
```

### 2) `optimize_alignment`
Purpose:
- Perform extremum-seeking beam alignment using correctors and BPM norms.

Main parameters (`corrector_pvs`, `bpm_observable_pvs`, `upstream_charge_pv`, and `downstream_charge_pv` have no default and must be supplied explicitly — there is no `to_screen_name` shortcut in the current implementation):
- `corrector_pvs` (required): list of corrector PVs to use for alignment
- `bpm_observable_pvs` (required): list of BPM observable PVs to use for alignment
- `upstream_charge_pv` (required): upstream charge PV for transmission measurements
- `downstream_charge_pv` (required): downstream charge PV for transmission measurements
- `n_steps` (optional, default `100`): maximum number of extremum-seeking steps
- `target_value` (optional, default `1.0`): BPM-norm convergence threshold
- `region_fraction` (optional, default `0.15`): half-width of the search region as a fraction of variable range
- `oscillation_size` (optional, default `0.01`)
- `steering_settle_time` (optional, default `0.2`): seconds to wait for steering adjustments to settle
- `dump_location` (optional): Xopt dump file path

`autonomous_control.alignment_opt_es.DEFAULT_ALIGNMENT_PVS` holds reference corrector/BPM PV lists for known screens (`PR10571`, `LI11312`) that can be copied into a step's `corrector_pvs`/`bpm_observable_pvs` values.

YAML step example (PR10571 screen):

```yaml
- type: optimize_alignment
  corrector_pvs:
    - XCOR:IN10:221:BCTRL
    - XCOR:IN10:311:BCTRL
    - YCOR:IN10:222:BCTRL
    - YCOR:IN10:312:BCTRL
  bpm_observable_pvs:
    - BPMS:IN10:371:X
    - BPMS:IN10:425:X
    - BPMS:IN10:371:Y
    - BPMS:IN10:425:Y
  upstream_charge_pv: BPMS:IN10:371:TMIT
  downstream_charge_pv: BPMS:IN10:651:TMIT
  n_steps: 100
  region_fraction: 0.15
```

### 3) `tcav_phasing`
Purpose:
- Automatically phase the TCAV by optimizing downstream centroid behavior with transmission constraints.

Main parameters:
- `tcav_on_amplitude` (optional, default `0.3`)
- `n_measurement_shots` (optional, default `1`): number of shots averaged per measurement
- `amplitude_tolerance` (optional, default `1e-3`): TCAV amplitude restoration tolerance
- `phase_tolerance` (optional, default `0.05`): TCAV phase-settle tolerance
- `n_initial_points` (optional, default `10`)
- `n_iterations` (optional, default `10`)
- `name` (optional, default `"automatic_phase_scan"`)
- `max_scan_range` (optional, default `[-10, 10]`): phase bounds for optimization
- `min_transmission` (optional, default `0.4`)
- `upstream_bpm_name` (optional, default `"BPM10371"`)
- `downstream_bpm_name` (optional, default `"BPM10651"`)
- `verbose` (optional, default `false`)
- `dump_location` (optional): directory to save optimization dumps

YAML step example:

```yaml
- type: tcav_phasing
  max_scan_range: [-10.0, 10.0]
  n_initial_points: 5
  n_iterations: 3
  tcav_on_amplitude: 0.3
```

### 4) `optimize_laser_steering`
Purpose:
- Run BAX-based solenoid alignment with laser mirror and solenoid control variables.

Main parameters (`bpm_x_pv`, `bpm_y_pv`, `bpm_tmit_pv`, `solenoid_pv`, and `mirror_pvs` have no default and must be supplied explicitly):
- `bpm_x_pv` (required): PV for the BPM X position
- `bpm_y_pv` (required): PV for the BPM Y position
- `bpm_tmit_pv` (required): PV for the BPM transmitted intensity
- `solenoid_pv` (required): PV for the solenoid control
- `mirror_pvs` (required): list of PVs for the mirror controls
- `initial_random_evaluations` (optional, default `2`): number of random warm-start evaluations
- `n_steps` (optional, default `30`): number of BAX optimization iterations
- `mirror_range_fraction` (optional, default `0.01`): fractional +/- range for mirror adjustments
- `solenoid_range_fraction` (optional, default `0.03`): fractional +/- range for solenoid adjustments
- `dump_location` (optional): Xopt dump file path

YAML step example:

```yaml
- type: optimize_laser_steering
  bpm_x_pv: BPMS:IN10:221:X
  bpm_y_pv: BPMS:IN10:221:Y
  bpm_tmit_pv: BPMS:IN10:221:TMIT
  solenoid_pv: SOLN:IN10:121:BCTRL
  mirror_pvs:
    - MIRR:IN10:101:MOTR_H
    - MIRR:IN10:101:MOTR_V
  initial_random_evaluations: 10
  n_steps: 30
  mirror_range_fraction: 0.01
  solenoid_range_fraction: 0.03
```

## Facet-Only Automation Functions (not available as YAML steps)

The following routines live under `autonomous_control.facet` (and `autonomous_control.auto_6d`) and implement additional FACET automation, but they are not registered in `autonomous_control.runner.STEP_HANDLERS`. Call them directly from Python rather than through a workflow YAML `type`:
- `autonomous_control.facet.auto_schottky.optimize_schottky(environment, dump_location=None, config=None)` — Amortized-BOED Schottky scan; `config` keys include `model_dir`, `design_range`, `observable_name`, `variable_name`, `max_measure`, `visualize`.
- `autonomous_control.facet.e_spread_opt.minimize_energy_spread(env, dump_location=None, config=None)` — minimizes energy spread proxy (beam size) by optimizing klystron phase; `config` keys include `phase_span`, `measurement_screen`, `initial_random_evaluations`, `n_steps`, `phase_tolerance`.
- `autonomous_control.facet.emittance_opt.minimize_injector_emittance(env, variables, dump_location=None, n_steps=3, min_joint_bmag_constraint=1.5, n_initial=2)` — Bayesian optimization over injector controls; `variables` maps PV name to `[min, max]` bounds.
- `autonomous_control.facet.l0_phasing.l0_phasing(env, *, k=4, p0=-15, pf=15, Nshots=100, makeplot=False)` — automatic L0 (10-4/10-8) RF phasing via a fast BSA phase scan and sinusoidal fit; `k` must be `4` or `8`.
- `autonomous_control.auto_6d.run_automatic_6d_measurement(env, save_filename, config_files, tcav_modes=("STDBY", "ACCEL_STDBY"), reset_tcav_mode="STDBY", tcav_settle_time=2.0, tcav_amplitude=0.4, dump_location=None)` — runs a quad scan for each config file across each TCAV mode, saving results to an HDF5 file.

## Full Example

```yaml
- type: optimize_laser_steering
  bpm_x_pv: BPMS:IN10:221:X
  bpm_y_pv: BPMS:IN10:221:Y
  bpm_tmit_pv: BPMS:IN10:221:TMIT
  solenoid_pv: SOLN:IN10:121:BCTRL
  mirror_pvs:
    - MIRR:IN10:101:MOTR_H
    - MIRR:IN10:101:MOTR_V
  initial_random_evaluations: 10
  n_steps: 30

- type: optimize_alignment
  corrector_pvs:
    - XCOR:IN10:221:BCTRL
    - YCOR:IN10:222:BCTRL
  bpm_observable_pvs:
    - BPMS:IN10:371:X
    - BPMS:IN10:371:Y
  upstream_charge_pv: BPMS:IN10:371:TMIT
  downstream_charge_pv: BPMS:IN10:651:TMIT
  n_steps: 100
  region_fraction: 0.15

- type: tcav_phasing
  max_scan_range: [-10.0, 10.0]
  n_iterations: 3
  n_initial_points: 5
  tcav_on_amplitude: 0.3

- type: measure_emittance
  config_file: /path/to/emittance_measurement_configs/PR10571.yaml
```

## Troubleshooting

- Unknown workflow type:
  - Ensure `type` matches one of the supported values exactly (see [Supported `type` values](#yaml-interface)).
- Unexpected keyword argument:
  - Ensure step keys match that step's supported parameters.
- Runtime machine errors:
  - The runner can reset the machine state after run if `reset_env_after=True` is passed to `run_automatic_workflow`/`run_automatic_workflow_from_file` (or `--reset_env_after` is passed on the CLI).

## Capturing and restoring machine state
Before running an autonomous workflow, it is a good idea to capture the current machine state using `autonomous_control.env_utils.capture_env_state` and store the state (a Python dict) in memory. If something goes wrong with the autonomous workflow then you can restore the machine state by passing the state to `autonomous_control.env_utils.restore_env_state`.
