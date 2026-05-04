"""Shared utilities for robust post-optimization reevaluation."""

from typing import Any, Optional

from xopt.vocs import select_best


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
