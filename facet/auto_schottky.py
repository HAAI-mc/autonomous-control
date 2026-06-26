import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import logging
import os
import epics
import traceback
from tenacity import retry, stop_after_attempt, wait_fixed

from xopt import Evaluator, VOCS, Xopt

from .optimization_utils import restore_on_error
from .two_bunch_boed_utils import (
    AmortizedBOEDBunchGenerator,
    get_results,
)

logger = logging.getLogger("auto_schottky_scan")

@restore_on_error(context="auto_schottky_scan")
def run_automatic_schottky_scan(
    env,
    dump_location=None,
    *,
    model_dir='/home/fphysics/rroussel/e331/facet/aboed',
    design_range=(-25.0, 45.0),
    observable_name='BPMS:IN10:221:TMIT',
    variable_name='control_phase',
    max_measure=100,
    visualize=True,
    n_posterior_samples=1000,
    n_predictive_curves=100,
):
    """Run the automatic Schottky scan with BOED-guided phase selection.

    Parameters
    ----------
    env : Any
        Control environment that provides variable setting and observable reads.
    dump_location : str or pathlib.Path, optional
        Directory where optimization dump artifacts are written.
    model_dir : str, optional
        Directory containing traced BOED models.
    design_range : tuple[float, float], optional
        Gunphase search bounds in degrees.
    observable_name : str, optional
        Observable PV to optimize.
    variable_name : str, optional
        Optimization variable name for Xopt.
    max_measure : int, optional
        Total number of measurements (grid + BOED).
    visualize : bool, optional
        If True, render BOED diagnostic plots during execution.
    n_posterior_samples : int, optional
        Posterior sample count used in downstream result summaries.
    n_predictive_curves : int, optional
        Number of posterior predictive curves used for diagnostics.

    Returns
    -------
    Xopt
        Optimizer object containing collected scan data.
    """

    settings = {
        "model_dir": model_dir,
        "design_range": list(design_range),
        "observable_name": observable_name,
        "variable_name": variable_name,
        "max_measure": max_measure,
        "visualize": visualize,
        "n_posterior_samples": n_posterior_samples,
        "n_predictive_curves": n_predictive_curves,
    }
    old_feedback_state = epics.caget("KLYS:LI10:31:SFB_PDIS")
    old_charge_feedback_state = epics.caget("SIOC:SYS1:ML03:AO502")
    old_fcup_state = epics.caget("FARC:IN10:241:PNEUMATIC")

    logging.info("inserting Faraday cup, interrupting feedbacks")
    epics.caput("KLYS:LI10:31:SFB_PDIS", 0)
    epics.caput("SIOC:SYS1:ML03:AO502", 0)
    epics.caput("FARC:IN10:241:PNEUMATIC", 1)
    phase_w0ch6 = epics.caget("ACCL:LI10:31:PHASE_W0CH6")
    init_pdes_value = epics.caget("KLYS:LI10:31:PDES")
    time.sleep(10.0)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5.0))
    def evaluate(inputs):
        logging.debug(inputs)
        
        true_val = inputs['control_phase'] + (init_pdes_value - phase_w0ch6)
        
        env.set_variables({"KLYS:LI10:31:PDES": true_val})

        time.sleep(0.5)  # wait for settings to take effect and measurements to stabilize
        output = env.get_observables([settings["observable_name"]])

        # get readbacks of variables
        output.update({
            "ACCL:LI10:31:PHASE_W0CH6":epics.caget("ACCL:LI10:31:PHASE_W0CH6"),
            "KLYS:LI10:31:PDES":epics.caget("KLYS:LI10:31:PDES"),
        })
        
        output[settings["observable_name"]] *= 1.6e-19 * 1e9 # return charge in nC

        return output

    try:
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

        # set RF back
        env.set_variables({"KLYS:LI10:31:PDES": init_pdes_value})

        # set laser timing
        tlaser_initial = epics.caget("OSC:LT10:20:FS_TGT_TIME")
        tlaser_final = tlaser_initial + (1.0742 * best_t0/1e3)
        epics.caput("OSC:LT10:20:FS_TGT_TIME", tlaser_final) # cannot use env unless we specify a range
    
        logger.info(f"best T0 estimate from BOED scan: {best_t0:.2f}° gunphase")
        #environment.set_variables({settings["variable_name"]: best_t0})
    except Exception as e:
        logger.error("Exception:")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("removing the faraday cup and restarting feedbacks")
        epics.caput("KLYS:LI10:31:SFB_PDIS", 1)
        epics.caput("FARC:IN10:241:PNEUMATIC", 0)
        epics.caput("SIOC:SYS1:ML03:AO502", 1)


    logger.info("Automatic Schottky scan complete.")
    return X