import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import logging
import os

from xopt import Evaluator, VOCS, Xopt

from optimization_utils import merge_config, restore_on_error
from two_bunch_boed_utils import (
    AmortizedBOEDBunchGenerator,
    get_results,
)

logger = logging.getLogger("auto_schottky_scan")

@restore_on_error(context="auto_schottky_scan")
def run_automatic_schottky_scan(environment, dump_location=None, config=None):

    settings = merge_config(
        {
            "model_dir": 'aboed',  # directory with traced .pt models
            "design_range": [-25.0, 45.0],   # [t_min, t_max] in physical gunphase degrees
            "observable_name": 'charge',
            "variable_name": 'gunphase',
            "max_measure": 100,              # total measurements (grid + BOED)
            "visualize": True,                # whether to show diagnostic plots during BOED phase
            "n_posterior_samples": 1000,       # number of posterior samples to draw for T0 histogram and predictive curves
            "n_predictive_curves": 100,        # number of posterior predictive curves to
        },
        config,
    )


    def evaluate(inputs):
        environment.set_variables(inputs)

        time.sleep(0.5)  # wait for settings to take effect and measurements to stabilize
        output = environment.get_observables(settings["observable_name"])

        return {settings["observable_name"]: output[settings["observable_name"]]}


    vocs = VOCS(
        variables={settings["variable_name"]: settings["design_range"]},
        observables=[settings["observable_name"]],
    )
    generator = AmortizedBOEDBunchGenerator(
        model_dir=settings["model_dir"],
        design_range=settings["design_range"],
        observable_name=settings["observable_name"],
        vocs=vocs,
    )
    # Extract the number of grid steps from the generator for use in the experiment loop.
    grid_steps = generator.grid_steps
    evaluator = Evaluator(function=evaluate)
    X = Xopt(
        vocs=vocs, 
        generator=generator, 
        evaluator=evaluator, 
        dump_file=os.path.join(dump_location, f"schottky_scan_data_{int(time.time())}.yaml") if dump_location else None
    )

    logger.info("Starting automatic Schottky scan with Amortized BOED generator.")
    logger.info(f'Running {settings["max_measure"]} steps ({grid_steps} grid + {settings["max_measure"] - grid_steps} BOED)...')

    for step in range(settings["max_measure"]):
        X.step()

        t_vals = X.data[settings["variable_name"]].values
        y_vals = X.data[settings["observable_name"]].values
        n_obs  = len(t_vals)
        order  = np.arange(n_obs)

        # After grid scan completes, show grid overview once
        if n_obs == grid_steps:
            if settings["visualize"]:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(t_vals, y_vals, c='steelblue', s=40, zorder=3)
                ax.set_xlabel('gunphase (°)')
                ax.set_ylabel(settings["observable_name"])
                ax.set_title('Grid scan')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.show()
                    

        # Still in grid phase — no per-step plot
        if n_obs < grid_steps:
            continue

        # BOED per-step diagnostic
        boed_step = n_obs - grid_steps

        if boed_step % 5 == 0:
            logger.info(f'BOED step {boed_step} / {settings["max_measure"] - grid_steps}')

            if settings["visualize"]:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                ax = axes[0]
                ax.scatter(t_vals[:grid_steps], y_vals[:grid_steps], c='steelblue', s=30, zorder=4, label='grid')
                sc = ax.scatter(t_vals[grid_steps:], y_vals[grid_steps:],
                                c=order[grid_steps:], cmap='YlOrRd', s=30, zorder=5, label='BOED')
                fig.colorbar(sc, ax=ax, label='BOED step')
                ax.set_xlabel('gunphase (°)')
                ax.set_ylabel(settings["observable_name"])
                ax.set_title('Signal vs gunphase')
                ax.legend(fontsize=8)
                ax.grid(True)

                ax = axes[1]
                ax.plot(order, t_vals, 'o-', ms=4)
                ax.set_xlabel('step')
                ax.set_ylabel('gunphase (°)')
                ax.set_title('Sampling sequence')
                ax.grid(True)
                plt.tight_layout()
                plt.show()

    # get results
    results = get_results(X, grid_steps, settings)

    # set the phase to the median of the last round T0 posterior samples for best estimate
    t0_samples = results["t0_samples"]
    best_t0 = np.median(t0_samples)

    logger.info(f"Setting best T0 estimate from BOED scan: {best_t0:.2f}° gunphase")
    environment.set_variables({settings["variable_name"]: best_t0})

    logger.info("Automatic Schottky scan complete.")
    return X