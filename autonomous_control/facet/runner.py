import os
import sys
import time
import logging
from typing import Any
import yaml
import argparse


from autonomous_control.facet.laser_steering import optimize_solenoid_alignment
from autonomous_control.facet.auto_emittance import run_automatic_emittance_xopt
from autonomous_control.facet.auto_schottky import run_automatic_schottky_scan
from autonomous_control.facet.alignment_opt_es import run_automatic_alignment
from autonomous_control.facet.e_spread_opt import optimize_energy_spread
from autonomous_control.facet.emittance_opt import optimize_injector_emittance
from autonomous_control.facet.tcav_phasing import run_automatic_tcav_phasing
from autonomous_control.facet.create_env import create_env, reset_env

STEP_HANDLERS = {
    "measure_emittance": run_automatic_emittance_xopt,
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
    dump_location: str = ".",
    reset_env_after: bool = True,
    logging_level: int = logging.INFO,
):
    """
    Run a sequence of automatic workflows in the FACET-II badger environment.

    If no environment is provided, a new FACET-II badger environment will be created.
    The workflow is defined as a list of dictionaries, where each dictionary specifies
    the type of workflow to run and any necessary parameters. We iterate through the workflow steps,
    executing each one in order. After all steps are completed, the
    environment can be reset to a safe state if requested.

    Example
    -------
    ```python
    >>> from autonomous_control.facet.runner import run_automatic_workflow
    >>> workflow = [
    >>>     {"type": "measure_emittance", "screen_name": "PR10571"},
    >>>     {"type": "tcav_phasing", "max_scan_range": [-10, 10], "n_iterations": 3, "n_initial_points": 3},
    >>> ]
    >>> run_automatic_workflow(workflow, dump_location="results", reset_env_after=True, logging_level=logging.INFO)
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
        If provided, the path to an output directory where workflow steps may write
        their own result artifacts. The directory will be created automatically if
        needed. If not provided, step handlers will manage their own default output
        locations.
    reset_env_after : bool, optional
        If True, the FACET-II badger environment will be reset to a safe state after all workflow steps have been executed. Default is True.
    logging_level : int, optional
        The logging level to use for the workflow execution. Default is logging.INFO.

    """

    ts = time.time()
    workflow_timestamp = int(ts)
    log_file = f"automatic_workflow_{workflow_timestamp}.log"
    workflow_start_time = time.time()
    force_reconfigure_logging = "PYTEST_CURRENT_TEST" in os.environ
    workflow_xopt_dump_file = f"automatic_workflow_xopt_{workflow_timestamp}.yaml"
    workflow_xopt_dump_data = {
        "workflow_timestamp": workflow_timestamp,
        "task_handlers": {},
    }

    logging.basicConfig(
        level=logging_level,
        handlers=[
            logging.FileHandler(log_file),  # Writes to file
            logging.StreamHandler(sys.stdout),  # Writes to console
        ],
        encoding="utf-8",
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=force_reconfigure_logging,
    )
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.info("Starting automatic workflow...")
    logging.info(
        "Workflow context: steps=%d dump_location=%s reset_env_after=%s logging_level=%s log_file=%s",
        len(workflow),
        dump_location,
        reset_env_after,
        logging.getLevelName(logging_level),
        log_file,
    )

    # create and configure the FACET-II badger environment
    if env is None:
        logging.info("Creating new FACET-II badger environment.")
        env = create_env()
    else:
        logging.info("Using provided FACET-II badger environment.")

    if dump_location is not None:
        os.makedirs(dump_location, exist_ok=True)
        logging.info("Ensured workflow output directory exists: %s", dump_location)

        workflow_xopt_dump_file = os.path.join(
            dump_location,
            f"automatic_workflow_xopt_{workflow_timestamp}.yaml",
        )
        with open(workflow_xopt_dump_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(workflow_xopt_dump_data, f, sort_keys=False)
        logging.info("Workflow Xopt dump file: %s", workflow_xopt_dump_file)

    # reset the environment to a safe state before starting the workflow
    pre_reset_start = time.time()
    logging.info("Resetting environment to safe state before workflow.")
    try:
        reset_env(env)
    except Exception:
        logging.exception("Pre-workflow environment reset failed.")
        raise
    logging.info(
        "Pre-workflow environment reset completed in %.2f s.",
        time.time() - pre_reset_start,
    )

    completed_steps = 0
    total_steps = len(workflow)

    for i, step in enumerate(workflow, start=1):
        step_kwargs = dict(step)
        step_type = step_kwargs.pop("type", None)
        logging.info(f"Running workflow step: {step_type}")
        logging.info(
            "Step %d/%d parameters: %s",
            i,
            total_steps,
            step_kwargs if step_kwargs else "{}",
        )

        step_handler = STEP_HANDLERS.get(step_type)
        if step_handler is None:
            logging.error(f"Unknown workflow type: {step_type}")
            raise ValueError(f"Unknown workflow type: {step_type}")

        step_start = time.time()
        try:
            step_result = step_handler(env, dump_location, **step_kwargs)
        except Exception:
            logging.exception(
                "Workflow step failed: %s (step %d/%d) after %.2f s",
                step_type,
                i,
                total_steps,
                time.time() - step_start,
            )
            raise

        if workflow_xopt_dump_file is not None:
            serialized_xopt = step_result.model_dump(mode="json")

            handler_name = getattr(step_handler, "__name__", step_type)
            handler_records = workflow_xopt_dump_data["task_handlers"].setdefault(
                handler_name,
                [],
            )
            handler_records.append(
                {
                    "step_index": i,
                    "step_type": step_type,
                    "xopt": serialized_xopt,
                }
            )
            with open(workflow_xopt_dump_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(workflow_xopt_dump_data, f, sort_keys=False)
            logging.info(
                "Appended Xopt serialization for %s to %s",
                handler_name,
                workflow_xopt_dump_file,
            )

        completed_steps += 1
        logging.info(
            "Completed workflow step: %s (step %d/%d) in %.2f s",
            step_type,
            i,
            total_steps,
            time.time() - step_start,
        )

    if reset_env_after:
        logging.info("Resetting environment to safe state.")
        post_reset_start = time.time()
        try:
            reset_env(env)
        except Exception:
            logging.exception("Post-workflow environment reset failed.")
            raise
        logging.info(
            "Post-workflow environment reset completed in %.2f s.",
            time.time() - post_reset_start,
        )
    else:
        logging.info(
            "Skipping post-workflow environment reset (reset_env_after=False)."
        )

    logging.info("Automatic workflow completed.")
    logging.info(
        "Workflow summary: completed_steps=%d total_steps=%d duration=%.2f s xopt_dump_file=%s",
        completed_steps,
        total_steps,
        time.time() - workflow_start_time,
        workflow_xopt_dump_file,
    )

    # return logging file name for reference
    return log_file


def run_automatic_workflow_from_file(
    workflow_file: str,
    env: Any = None,
    dump_location: str = None,
    reset_env_after: bool = True,
    logging_level: int = logging.INFO,
):
    """
    Run a sequence of automatic workflows in the FACET-II badger environment from a YAML file.

    Parameters
    ----------
    workflow_file : str
        Path to a YAML file that defines the workflow steps. The file should contain a list of dictionaries,
        where each dictionary represents a workflow step with a 'type' key and any necessary parameters.
    env : Any, optional
        An existing FACET-II badger environment. If not provided, a new environment will be created.
    dump_location : str, optional
        If provided, the path to an output directory where workflow steps may write
        their own result artifacts. The directory will be created automatically if
        needed. If not provided, step handlers will manage their own default output
        locations.
    reset_env_after : bool, optional
        If True, the FACET-II badger environment will be reset to a safe state after all workflow steps have been executed. Default is True.
    logging_level : int, optional
        The logging level to use for the workflow execution. Default is logging.INFO.
    """

    with open(workflow_file, "r") as f:
        workflow = yaml.safe_load(f)

    return run_automatic_workflow(
        workflow=workflow,
        env=env,
        dump_location=dump_location,
        reset_env_after=reset_env_after,
        logging_level=logging_level,
    )


if __name__ == "__main__":
    # CLI interface for running automatic workflows from a YAML file

    parser = argparse.ArgumentParser(
        description="Run an automatic workflow in the FACET-II badger environment from a YAML file."
    )
    parser.add_argument(
        "workflow_file",
        type=str,
        help="Path to the YAML file defining the workflow steps.",
    )
    parser.add_argument(
        "--dump_location",
        type=str,
        default=None,
        help="Optional output directory for workflow step artifacts.",
    )
    parser.add_argument(
        "--reset_env_after",
        action="store_true",
        help="Reset the environment to a safe state after workflow completion.",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is INFO.",
    )

    args = parser.parse_args()

    # Convert logging level string to logging module constant
    logging_level = getattr(logging, args.logging_level.upper(), logging.INFO)

    run_automatic_workflow_from_file(
        workflow_file=args.workflow_file,
        dump_location=args.dump_location,
        reset_env_after=args.reset_env_after,
        logging_level=logging_level,
    )
