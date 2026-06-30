# Autonomous Control Workflow Guide

This guide explains the high-level FACET workflows supported by the workflow runner and how to execute them from a YAML file.

## Badger resources
Make sure that the environment variable `BADGER_RESOURCES` points to the local Badger-Resources directory.

## YAML Interface

The workflow runner expects a YAML list of steps.

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

Supported `type` values (each maps 1:1 to a top-level callable with the same name):
- `measure_emittance` -> `autonomous_control.facet.auto_emittance.measure_emittance`
- `optimize_schottky` -> `autonomous_control.facet.auto_schottky.optimize_schottky`
- `optimize_alignment` -> `autonomous_control.facet.alignment_opt_es.optimize_alignment`
- `minimize_energy_spread` -> `autonomous_control.facet.e_spread_opt.minimize_energy_spread`
- `minimize_injector_emittance` -> `autonomous_control.facet.emittance_opt.minimize_injector_emittance`
- `tcav_phasing` -> `autonomous_control.facet.tcav_phasing.tcav_phasing`
- `l0_phasing` -> `autonomous_control.facet.l0_phasing.l0_phasing`
- `optimize_laser_steering` -> `autonomous_control.facet.laser_steering.optimize_laser_steering`


## How To Run A YAML Workflow

From repository root:

```bash
python -m autonomous_control.facet.runner autonomous_control/facet/workflows/start_to_end.yml \
  --dump_location results \
  --logging_level INFO \
  --reset_env_after
```

Notes:
- `--dump_location` is optional. When provided, the runner creates the directory.
- `--reset_env_after` is a flag. Include it to force a post-workflow reset.
- A log file named `automatic_workflow_<timestamp>.log` is written in the current working directory.

## High-Level Workflow Steps

### 1) `measure_emittance`
Purpose:
- Perform quadrupole-scan emittance measurement at a selected screen.

Main parameters:
- `screen_name` (required): typically `PR10571` or `PR10711`
- `config_directory` (optional): override config file directory
- `screen_settle_time` (optional): seconds to wait after screen target change
- `screens` (optional): override per-screen target/config mapping

YAML step example:

```yaml
- type: measure_emittance
  screen_name: PR10571
  screen_settle_time: 2.0
```

### 2) `optimize_schottky`
Purpose:
- Run an automatic Schottky scan using the Amortized BOED generator.

Main parameters:
- `config` (optional): dictionary of scan settings and model options

Common `config` keys:
- `model_dir`
- `design_range`
- `observable_name`
- `variable_name`
- `max_measure`
- `visualize`

YAML step example:

```yaml
- type: optimize_schottky
  config:
    max_measure: 80
    design_range: [-25.0, 45.0]
    visualize: false
```

### 3) `optimize_alignment`
Purpose:
- Perform extremum-seeking beam alignment using correctors and BPM norms.

Main parameters:
- `to_screen_name` (optional, default `PR10571`)
- `n_steps` (optional)
- `target_value` (optional): convergence threshold on BPM norm
- `region_fraction` (optional): local search region size

YAML step example:

```yaml
- type: optimize_alignment
  to_screen_name: PR10571
  n_steps: 100
  region_fraction: 0.15
```

### 4) `minimize_energy_spread`
Purpose:
- Minimize energy spread proxy (beam size) by optimizing klystron phase.

Main parameters:
- `config` (optional): dictionary of optimization and machine settings

Common `config` keys:
- `phase_span`
- `measurement_screen`
- `initial_random_evaluations`
- `n_steps`
- `phase_tolerance`

YAML step example:

```yaml
- type: minimize_energy_spread
  config:
    phase_span: 4.0
    n_steps: 7
    measurement_screen: PR10711
```

### 5) `minimize_injector_emittance`
Purpose:
- Run Bayesian optimization over injector controls to reduce emittance metrics.

Main parameters:
- `variables` (required): mapping from PV name to `[min, max]` bounds
- `n_steps` (optional)

YAML step example:

```yaml
- type: minimize_injector_emittance
  n_steps: 3
  variables:
    SOLN:IN10:121:BCTRL: [0.39, 0.41]
    QUAD:IN10:121:BCTRL: [-0.008, 0.0085]
    QUAD:IN10:122:BCTRL: [-0.008, 0.0085]
```

Important:
- Do not include a `screen` key for this step with the current implementation. It is fixed to the PR10571 screen.

### 6) `tcav_phasing`
Purpose:
- Automatically phase the TCAV by optimizing downstream centroid behavior with transmission constraints.

Main parameters:
- `tcav_on_amplitude` (optional)
- `n_initial_points` (optional)
- `n_iterations` (optional)
- `max_scan_range` (optional)
- `min_transmission` (optional)
- `name` (optional)
- `verbose` (optional)

YAML step example:

```yaml
- type: tcav_phasing
  max_scan_range: [-10.0, 10.0]
  n_initial_points: 5
  n_iterations: 3
  tcav_on_amplitude: 0.3
```

### 7) `l0_phasing`
Purpose:
- Run automatic L0 (10-4 / 10-8) RF phasing using a fast BSA phase scan and sinusoidal fit.

Main parameters:
- `k` (optional, default `4`): klystron identifier, must be `4` or `8`
- `p0` (optional, default `-15`): initial waveguide phase for the scan range
- `pf` (optional, default `15`): final waveguide phase for the scan range
- `Nshots` (optional, default `100`): number of beam-synchronous points to acquire
- `makeplot` (optional, default `false`): display scan and fit plot

YAML step example:

```yaml
- type: l0_phasing
  k: 4
  p0: -15
  pf: 15
  Nshots: 100
  makeplot: false
```

### 8) `optimize_laser_steering`
Purpose:
- Run BAX-based solenoid alignment with laser mirror and solenoid control variables.

Main parameters:
- `initial_random_evaluations` (optional)
- `n_steps` (optional)
- `mirror_range_fraction` (optional)
- `solenoid_range_fraction` (optional)

YAML step example:

```yaml
- type: optimize_laser_steering
  initial_random_evaluations: 10
  n_steps: 30
  mirror_range_fraction: 0.01
  solenoid_range_fraction: 0.03
```

## Full Example

```yaml
- type: optimize_laser_steering
  initial_random_evaluations: 10
  n_steps: 30

- type: optimize_alignment
  to_screen_name: PR10571
  n_steps: 100
  region_fraction: 0.15

- type: minimize_injector_emittance
  n_steps: 3
  variables:
    SOLN:IN10:121:BCTRL: [0.39, 0.41]
    QUAD:IN10:121:BCTRL: [-0.008, 0.0085]
    QUAD:IN10:122:BCTRL: [-0.008, 0.0085]

- type: tcav_phasing
  max_scan_range: [-10.0, 10.0]
  n_iterations: 3
  n_initial_points: 5
  tcav_on_amplitude: 0.3

- type: minimize_energy_spread
```

## Troubleshooting

- Unknown workflow type:
  - Ensure `type` matches one of the supported values exactly.
- Unexpected keyword argument:
  - Ensure step keys match that step's supported parameters.
- Runtime machine errors:
  - The runner can reset the machine state after run if `--reset_env_after` is provided.

## Capturing and restoring machine state
Before running an autonomous workflow, it is a good idea to capture the current machine state using `autonomous_control.facet.env_utils.capture_env_state` and store the state (a Python dict) in memory. If something goes wrong with the autonomous workflow then you can restore the machine state by passing the state to `autonomous_control.facet.env_utils.restore_env_state`.