"""Shared utilities for robust post-optimization reevaluation."""

import functools
import logging
import traceback
from typing import Any, Callable, Optional

from xopt.vocs import select_best

_logger = logging.getLogger(__name__)


def restore_on_error(context: Optional[str] = None):
    """Decorator factory that snapshots machine state and restores it on failure.

    Wraps an optimization entry-point function whose first argument is ``env``.
    Before the function runs, all variables tracked by ``env`` are captured.
    If the function raises, the snapshot is applied via ``env.set_variables``
    and the original exception is re-raised with a diagnostic log message.

    Parameters
    ----------
    context : str, optional
        Human-readable label for log messages.  Defaults to the decorated
        function's ``__name__``.

    Examples
    --------
    >>> @restore_on_error(context="alignment_opt")
    ... def run_automatic_alignment(env, to_screen_name="PROF571"):
    ...     ...
    """

    def decorator(fn: Callable) -> Callable:
        label = context or fn.__name__

        @functools.wraps(fn)
        def wrapper(env: Any, *args: Any, **kwargs: Any) -> Any:
            return run_script_with_restore(fn, env, *args, context=label, **kwargs)

        return wrapper

    return decorator


def run_script_with_restore(
    script_fn: Callable,
    env: Any,
    *args: Any,
    context: Optional[str] = None,
    logger: Optional[Any] = None,
    **kwargs: Any,
):
    """Run an optimization script and restore the machine state on failure.

    Before calling ``script_fn``, the current values of all variables tracked
    by ``env`` are captured.  If ``script_fn`` raises, the saved state is
    applied via ``env.set_variables`` and a diagnostic message is logged
    before the exception is re-raised.

    Parameters
    ----------
    script_fn : Callable
        The optimization script entry-point to run. Its first positional
        argument must be ``env``.
    env : Any
        Control environment with ``variables``, ``get_variables``, and
        ``set_variables`` interfaces.
    *args : Any
        Positional arguments forwarded to ``script_fn`` after ``env``.
    context : str, optional
        Human-readable label used in log messages, e.g. ``"alignment_opt"``.
        Defaults to ``script_fn.__name__``.
    logger : logging.Logger, optional
        Logger to use.  Defaults to the module-level logger.
    **kwargs : Any
        Keyword arguments forwarded to ``script_fn``.

    Returns
    -------
    Any
        Whatever ``script_fn`` returns on success.

    Raises
    ------
    Exception
        The original exception from ``script_fn``, re-raised after machine
        restore is attempted.
    """
    log = logger or _logger
    label = context or getattr(script_fn, "__name__", repr(script_fn))

    # --- snapshot initial machine state ---
    try:
        initial_state = env.get_variables(env.variables.keys())
    except Exception:
        log.warning(
            "Could not snapshot machine state before running %s; "
            "restore on failure will be skipped.\n%s",
            label,
            traceback.format_exc(),
        )
        initial_state = None

    log.info("Starting %s.", label)

    try:
        return script_fn(env, *args, **kwargs)

    except Exception:
        log.error(
            "%s failed with an unhandled exception:\n%s",
            label,
            traceback.format_exc(),
        )

        # --- attempt machine restore ---
        if initial_state is not None:
            log.warning(
                "Attempting to restore machine to state captured before %s: %s",
                label,
                initial_state,
            )
            try:
                env.set_variables(initial_state)
                log.info("Machine state restored successfully after %s failure.", label)
            except Exception:
                log.error(
                    "Machine restore FAILED after %s error — manual intervention may be required.\n%s",
                    label,
                    traceback.format_exc(),
                )
        else:
            log.error(
                "No snapshot available; machine state was NOT restored after %s failure.",
                label,
            )

        raise


def safe_evaluate_best_point(
    X: Any,
    logger: Any,
    *,
    metric_name: Optional[str] = None,
    best_inputs: Optional[dict[str, float]] = None,
    use_select_best: bool = False,
    context: str = "optimization",
):
    """Safely evaluate a best candidate point without crashing cleanup paths.

    Parameters
    ----------
    X : Any
        Xopt instance with ``data``, ``vocs``, and ``evaluate_data``.
    logger : Any
        Logger used for warnings and exception reporting.
    metric_name : str, optional
        Column name in ``X.data`` used to find ``idxmin`` when ``best_inputs``
        is not supplied.
    best_inputs : dict[str, float], optional
        Explicit input dictionary to evaluate.
    use_select_best : bool, optional
        If True and ``best_inputs`` is not provided, select candidate with
        ``xopt.vocs.select_best``.
    context : str, optional
        Human-readable context label for logging.

    Returns
    -------
    Any | None
        Result from ``X.evaluate_data`` or ``None`` when no valid candidate is
        available or reevaluation fails.
    """
    try:
        if best_inputs is None:
            if X.data is None or X.data.empty:
                logger.warning(
                    "Skipping best-point reevaluation during %s: no optimization data.",
                    context,
                )
                return None

            if use_select_best:
                best_inputs = select_best(X.vocs, X.data)[2]
            else:
                if metric_name is None:
                    logger.warning(
                        "Skipping best-point reevaluation during %s: no metric_name provided.",
                        context,
                    )
                    return None

                if metric_name not in X.data:
                    logger.warning(
                        "Skipping best-point reevaluation during %s: missing '%s' column.",
                        context,
                        metric_name,
                    )
                    return None

                valid_metric = X.data[metric_name].dropna()
                if valid_metric.empty:
                    logger.warning(
                        "Skipping best-point reevaluation during %s: all '%s' values are NaN.",
                        context,
                        metric_name,
                    )
                    return None

                best_idx = valid_metric.idxmin()
                best_inputs = X.data.loc[best_idx, X.vocs.variable_names].to_dict()

        logger.info("Evaluating best point during %s.", context)
        result = X.evaluate_data(best_inputs)

        if metric_name is not None and metric_name in result:
            logger.info(
                "Best point evaluated during %s: %s=%s",
                context,
                metric_name,
                result[metric_name][0],
            )
        else:
            logger.info("Best point evaluated during %s.", context)

        return result

    except Exception:
        logger.exception("Best-point reevaluation failed during %s.", context)
        return None
