import logging
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.objectives import CustomXoptObjective
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from botorch.exceptions.errors import OptimizationGradientError
import numpy as np
import torch
from gpytorch.kernels import ScaleKernel, PolynomialKernel
import traceback


from optimization_utils import merge_config, restore_on_error, safe_evaluate_best_point

from ml_tto.errors import TransmissionError

# Setup Logging
logger = logging.getLogger("auto_alignment")


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
        ele: vocs.variables[ele][1] - vocs.variables[ele][0]
        for ele in vocs.variable_names
    }

    for name in vocs.variable_names:
        bounds[name] = [
            np.max(
                (center_point[name] - widths[name] * fraction, vocs.variables[name][0])
            ),
            np.min(
                (center_point[name] + widths[name] * fraction, vocs.variables[name][1])
            ),
        ]

    logger.debug(f"Local region: {bounds}")
    return bounds


bpms = [221, 371, 425, 511, 525]
alignment_pvs = {
    "PROF571": {
        "corrector_pvs": [
            f"XCOR:DIAG0:{ele}:BCTRL" for ele in [221, 311, 381, 411, 491, 521]
        ]
        + [f"YCOR:DIAG0:{ele}:BCTRL" for ele in [222, 312, 382, 412, 492, 522]],
        "bpms": [f"BPMS:DIAG0:{ele}:XSCDTH" for ele in bpms]
        + [f"BPMS:DIAG0:{ele}:YSCDTH" for ele in bpms],
    },
}


@restore_on_error(context="alignment_opt")
def run_automatic_alignment(
    env,
    to_screen_name="PROF571",
    n_steps=20,
    old_data=None,
    target_value=1.0,
    config=None,
):
    """Run the Bayesian alignment optimization process on DIAG0.

    Parameters
    ----------
    env : Any
        Control environment providing ``set_screen``, ``get_bounds``,
        ``get_variables``, ``set_variables``, and ``get_observables``.
    to_screen_name : str, optional
        Screen name to align to, by default ``"PROF571"``.
    n_steps : int, optional
        Maximum number of Bayesian optimization steps, by default 20.
    old_data : pandas.DataFrame or None, optional
        Previously collected data to seed the optimizer.  When ``None``,
        random initial evaluations are performed instead.
    target_value : float, optional
        BPM-norm convergence threshold; optimization stops early when
        the best ``norm`` falls below this value, by default 1.0.
    config : dict, optional
        Configuration overrides, typically loaded from a config file. Supported
        keys include ``screens``, ``constraints``, BPM weighting overrides, and
        search-region or generator options.

    Returns
    -------
    Xopt
        Optimizer instance containing all collected evaluations.
    """
    env.set_screen(to_screen_name)

    settings = merge_config(
        {
            "screens": alignment_pvs,
            "constraints": {"transmission": ["GREATER_THAN", 0.9]},
            "bpm_weight_overrides": {"330": 2.0, "390": 2.0},
            "local_region_fraction": 0.15,
            "random_sample_fraction": 0.1,
            "initial_random_evaluations": 10,
            "generator": {"n_interpolate_points": 4, "max_time": 2.5},
        },
        config,
    )

    logger.info(f"Starting automatic alignment for screen: {to_screen_name}")
    # if just transporting beam to OTRDG02, use all BPMs except 470 and 520
    screen_config = settings["screens"][to_screen_name]
    pvs = screen_config["corrector_pvs"]
    bpm_observables = screen_config["bpms"]

    # set biasing for certain bpms
    bpm_weights = {name: 1.0 for name in bpm_observables}
    for name in bpm_weights:
        for token, weight in settings["bpm_weight_overrides"].items():
            if token in name:
                bpm_weights[name] = weight
    formatted_string = "\n".join([f"{name}:{val}" for name, val in bpm_weights.items()])
    logger.debug(f"weighting bpm signal as follows:\n{formatted_string}")

    temp_vocs = VOCS(variables=env.get_bounds(pvs), observables=[])
    local_region = get_local_region(
        env.get_variables(temp_vocs.variables.keys()),
        temp_vocs,
        settings["local_region_fraction"],
    )

    def eval(inputs):
        logger.debug("evaluating point")
        try:
            env.set_variables(inputs)
        except TransmissionError:
            logger.warning("Transmission error while setting variables.")
            # transmission below 0.8
            norm = np.nan
            bpm_signals = {name: np.nan for name in bpm_observables}
            transmission = 0.5
            return {"norm": norm, "transmission": transmission} | bpm_signals

        transmission = env.get_observables(["transmission"])["transmission"]
        try:
            bpm_signals = env.get_observables(bpm_observables)
            norm = np.linalg.norm([bpm_signals[name] for name in bpm_observables])
        except KeyError:
            logger.warning("Error while getting observables")
            norm = np.nan
            bpm_signals = {name: np.nan for name in bpm_observables}

        # pop input keys from bpm_signals
        for name in inputs.keys():
            if name in bpm_signals:
                bpm_signals.pop(name)

        return {"norm": norm, "transmission": transmission} | bpm_signals

    vocs = VOCS(
        variables=local_region,
        observables=bpm_observables,
        constraints=settings["constraints"],
    )

    # create custom objective
    class MyObjective(CustomXoptObjective):
        def forward(self, samples, X=None):
            return -torch.norm(
                torch.stack(
                    [
                        samples[..., self.vocs.observable_names.index(name)]
                        * bpm_weights[name]
                        for name in bpm_observables
                    ]
                ),
                dim=0,
            )

    class WeightedPolynomialKernel(PolynomialKernel):
        has_lengthscale = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def forward(self, x1, x2, **params):
            return super().forward(x1 * self.lengthscale**2, x2, **params)

    covar_modules = {
        name: ScaleKernel(
            WeightedPolynomialKernel(power=1, ard_num_dims=vocs.n_variables)
        )
        for name in bpm_observables
    }
    gp_constructor = StandardModelConstructor(
        covar_modules=covar_modules,
    )

    generator = ExpectedImprovementGenerator(
        vocs=vocs,
        custom_objective=MyObjective(vocs),
        gp_constructor=gp_constructor,
        n_interpolate_points=settings["generator"]["n_interpolate_points"],
    )
    generator.numerical_optimizer.max_time = settings["generator"]["max_time"]

    evaluator = Evaluator(function=eval)

    X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator, strict=False)

    logger.info("Starting evaluation")
    # evaluate
    X.evaluate_data(env.get_variables(vocs.variables.keys()))
    if X.data.min()["norm"] < target_value:
        logger.info("converged")
        return X

    random_sample_region = get_local_region(
        env.get_variables(vocs.variables.keys()),
        X.vocs,
        fraction=settings["random_sample_fraction"],
    )

    if old_data is not None:
        logger.info("Adding old data.")
        X.add_data(old_data)
    else:
        logger.info("Generating and evaluating random points.")
        X.random_evaluate(
            settings["initial_random_evaluations"],
            custom_bounds=random_sample_region,
        )

    try:
        for i in range(n_steps):
            # if any of the evaluations are close to the objective value - use max travel distances
            # to restrict exploration
            if (
                np.any(X.data["norm"] < target_value * 3.0)
                and X.generator.max_travel_distances is None
            ):
                logger.info(
                    "found a point close to the optimum, evaluating that point and restricting max travel distances"
                )
                X.evaluate_data(
                    X.data[X.vocs.variable_names]
                    .iloc[X.data.idxmin()["norm"]]
                    .to_dict()
                )
                X.generator.max_travel_distances = [0.25] * X.vocs.n_variables

            logger.info(f"At step {i}")
            if X.data.min()["norm"] < target_value:
                logger.info("Converged")
                break

            # try running a bo step until we succeed -- max 5 tries
            for _ in range(5):
                try:
                    X.step()
                    break
                except OptimizationGradientError:
                    logger.warning(
                        "gradient error, adding random evals and then trying again"
                    )
                    random_sample_region = get_local_region(
                        env.get_variables(vocs.variables.keys()), X.vocs, fraction=0.1
                    )
                    X.random_evaluate(1, custom_bounds=random_sample_region)

    except Exception:
        logger.error("Exception:")
        logger.error(traceback.format_exc())
        raise
    finally:
        safe_evaluate_best_point(
            X,
            logger,
            metric_name="norm",
            context="alignment finalization",
        )

    return X
