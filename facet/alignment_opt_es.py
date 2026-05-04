import logging
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.sequential import ExtremumSeekingGenerator
from botorch.exceptions.errors import OptimizationGradientError
import numpy as np
import traceback
import epics
import time
from ml_tto.errors import TransmissionError
import os

# Setup Logging
logger = logging.getLogger("auto_alignment")


def get_local_region(center_point: dict, vocs: VOCS, fraction: float = 0.1) -> dict:
    """
    calculates the bounds of a local region around a center point with side lengths
    equal to a fixed fraction of the input space for each variable

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


bpms = [371, 425, 511, 525, 581, 631, 651]
alignment_pvs = {
    "PROF571": {
        "corrector_pvs": [
            f"XCOR:IN10:{ele}:BCTRL" for ele in [221, 311, 381, 411, 491, 521, 641]
        ]
        + [f"YCOR:IN10:{ele}:BCTRL" for ele in [222, 312, 382, 412, 492, 522, 642]],
        "bpms": [f"BPMS:IN10:{ele}:X" for ele in bpms]
        + [f"BPMS:IN10:{ele}:Y" for ele in bpms],
    },
    "LI11312": {
        "corrector_pvs": [f"XCOR:IN10:{ele}:BCTRL" for ele in [721, 761]]
        + [f"XCOR:LI11:{ele}:BCTRL" for ele in [104, 140, 202, 272, 304]]
        + [f"YCOR:IN10:{ele}:BCTRL" for ele in [722, 762]]
        + [f"YCOR:LI11:{ele}:BCTRL" for ele in [105, 141, 203, 273, 305, 321]],
        "bpms": [
            "BPMS:IN10:771:X",
            "BPMS:IN10:781:X",
            "BPMS:IN10:771:Y",
            "BPMS:IN10:781:Y",
            "BPMS:LI11:132:X",
            "BPMS:LI11:201:X",
            "BPMS:LI11:265:X",
            "BPMS:LI11:301:X",
            "BPMS:LI11:312:X",
            "BPMS:LI11:132:Y",
            "BPMS:LI11:201:Y",
            "BPMS:LI11:265:Y",
            "BPMS:LI11:301:Y",
            "BPMS:LI11:312:Y",
            "BPMS:LI11:333:X",
            "BPMS:LI11:358:X",
            "BPMS:LI11:362:X",
            "BPMS:LI11:333:Y",
            "BPMS:LI11:358:Y",
            "BPMS:LI11:362:Y",
        ],
    },
}


def run_automatic_alignment(
    env,
    to_screen_name="PROF571",
    n_steps=100,
    old_data=None,
    target_value=1.0,
    region_fraction=0.15,
    dump_location=".",
):
    """
    Runs the automatic alignment optimization process on DIAG0 to
    `to_screen_name`.

    Parameters:
        env (Environment): The environment in which the optimization is performed.
        to_screen_name (str): The name of the screen to align to. Default is "PROF571".

    """
    # env.set_screen(to_screen_name)

    logger.info(f"Starting automatic alignment for screen: {to_screen_name}")
    # if just transporting beam to OTRDG02, use all BPMs except 470 and 520
    pvs = alignment_pvs[to_screen_name]["corrector_pvs"]
    bpm_observables = alignment_pvs[to_screen_name]["bpms"]

    temp_vocs = VOCS(variables=env.get_bounds(pvs), observables=[])
    local_region = get_local_region(
        env.get_variables(temp_vocs.variables.keys()), temp_vocs, region_fraction
    )

    def eval(inputs):
        logger.debug("evaluating point")
        try:
            for name, val in inputs.items():
                epics.caput(name, val)

            time.sleep(0.25)
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
        objectives={"norm": "MINIMIZE"},
    )

    generator = ExtremumSeekingGenerator(
        vocs=vocs,
    )
    evaluator = Evaluator(function=eval)

    X = Xopt(
        vocs=vocs,
        generator=generator,
        evaluator=evaluator,
        strict=True,
        dump_file=os.path.join(
            dump_location, f"beam_steering_{to_screen_name}_{int(time.time())}.yaml"
        ),
    )

    logger.info("Starting evaluation")
    # evaluate
    X.evaluate_data(env.get_variables(vocs.variables.keys()))

    if X.data.min()["norm"] < target_value:
        logger.info("converged")
        return X

    random_sample_region = get_local_region(
        env.get_variables(vocs.variables.keys()), X.vocs, fraction=0.1
    )

    if old_data is not None:
        logger.info("Adding old data.")
        X.add_data(old_data)
    else:
        pass
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
        X.generator.reset()
        result = X.evaluate_data(
            X.data[X.vocs.variable_names].iloc[X.data.idxmin()["norm"]].to_dict()
        )
        logger.info(f"evaluated the best point: norm={result['norm'][0]}")

    return X
