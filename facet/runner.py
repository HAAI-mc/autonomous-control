import os
import sys
import time
import logging
from typing import Any

from .laser_steering import optimize_solenoid_alignment
from .auto_emittance import run_automatic_emittance
from .auto_schottky import run_automatic_schottky_scan
from .alignment_opt_es import run_automatic_alignment
from .e_spread_opt import optimize_energy_spread
from .emittance_opt import optimize_injector_emittance
from .tcav_phasing import run_automatic_tcav_phasing
from .create_env import create_env, reset_env

STEP_HANDLERS = {
    "measure_emittance": run_automatic_emittance,
    "optimize_schottky": run_automatic_schottky_scan,
    "optimize_alignment": run_automatic_alignment,
    "minimize_energy_spread": optimize_energy_spread,
    "minimize_injector_emittance": optimize_injector_emittance,
    "tcav_phasing": run_automatic_tcav_phasing,
    "optimize_laser_steering": optimize_solenoid_alignment,
}

def run_automatic_workflow(
        workflow: list[dict], 
        env: Any = None,
        dump_location: str = None, 
        reset_env_after: bool = True, 
        logging_level: int = logging.INFO
    ):
    """
    Run a sequence of automatic workflows in the FACET-II badger environment.
    
    Iterates through the provided list of workflow steps, executing each step in order. 
    Each step is a dictionary that specifies the type of workflow to run and any necessary parameters.

    Example
    -------
    ```python
    >>> from facet.runner import run_automatic_workflow
    >>> workflow = [
    >>>     {"type": "measure_emittance", "screen_name": "PROF10571"},
    >>>     {"type": "tcav_phasing", "max_scan_range": [-10, 10], "n_iterations": 3, "n_initial_points": 3},
    >>> ]
    >>> run_automatic_workflow(workflow, dump_location="results.h5", reset_env_after=True, logging_level=logging.INFO)
    ```
    
    Parameters
    ----------
    workflow : list of dict
        A list of dictionaries, where each dictionary represents a workflow step. 
        Each dictionary must contain a 'type' key that specifies the type of workflow to run, 
        and may contain additional keys for parameters required by that workflow.
    env : Any, optional
        An existing FACET-II badger environment. If not provided, a new environment will be created.
    dump_location : str, optional
        If provided, the path to a file where the results of each workflow step will be saved. If not provided, results will not be saved to a file.
    reset_env_after : bool, optional
        If True, the FACET-II badger environment will be reset to a safe state after all workflow steps have been executed. Default is True.
    logging_level : int, optional
        The logging level to use for the workflow execution. Default is logging.INFO.

    """

    ts = time.time()
    log_file = f"automatic_workflow_{int(ts)}.log"
    force_reconfigure_logging = "PYTEST_CURRENT_TEST" in os.environ

    logging.basicConfig(
        level=logging_level,
        handlers=[
            logging.FileHandler(log_file), # Writes to file
            logging.StreamHandler(sys.stdout)    # Writes to console
        ],
        encoding='utf-8',
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=force_reconfigure_logging,
    )
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    # create and configure the FACET-II badger environment
    if env is None:
        logging.info("Creating new FACET-II badger environment.")
        env = create_env()

    # reset the environment to a safe state before starting the workflow
    reset_env(env)

    for step in workflow:
        step_kwargs = dict(step)
        step_type = step_kwargs.pop("type", None)
        logging.info(f"Starting workflow step: {step_type}")

        step_handler = STEP_HANDLERS.get(step_type)
        if step_handler is None:
            logging.error(f"Unknown workflow type: {step_type}")
            raise ValueError(f"Unknown workflow type: {step_type}")

        step_handler(env, dump_location, **step_kwargs)

    if reset_env_after:
        logging.info("Resetting environment to safe state.")
        reset_env(env)

    # return logging file name for reference
    return log_file
