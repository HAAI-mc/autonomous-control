import logging
import numpy as np

from xopt import VOCS

logger = logging.getLogger(__name__)


def get_local_region(center_point: dict, vocs: VOCS, fraction: float = 0.1) -> dict:
    """Calculate bounds of a local region around a center point.

    Side lengths equal a fixed fraction of the full input-space range for each
    variable, clamped to the VOCS bounds.

    Parameters
    ----------
    center_point : dict
        Mapping of variable name to current value.  Keys must exactly match
        ``vocs.variable_names``.
    vocs : VOCS
        Xopt VOCS object defining variable names and bounds.
    fraction : float, optional
        Half-width of the local region as a fraction of the full variable range,
        by default 0.1.

    Returns
    -------
    dict
        Mapping of variable name to ``[lower, upper]`` bound lists.

    Raises
    ------
    KeyError
        If ``center_point`` keys do not match ``vocs.variable_names``.
    """
    logger.debug("Calculating local region bounds.")
    if not center_point.keys() == set(vocs.variable_names):
        logger.error("Center point keys must match VOCS variable names")
        raise KeyError("Center point keys must match vocs variable names")

    bounds = {}
    widths = {
        ele: vocs.variables[ele].domain[1] - vocs.variables[ele].domain[0]
        for ele in vocs.variable_names
    }

    for name in vocs.variable_names:
        bounds[name] = [
            np.max(
                (
                    center_point[name] - widths[name] * fraction,
                    vocs.variables[name].domain[0],
                )
            ),
            np.min(
                (
                    center_point[name] + widths[name] * fraction,
                    vocs.variables[name].domain[1],
                )
            ),
        ]

    logger.debug(f"Local region: {bounds}")
    return bounds
