import logging
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.sequential import ExtremumSeekingGenerator
import numpy as np
import traceback
import epics
import time
from ml_tto.errors import TransmissionError
import os


from .optimization_utils import restore_on_error, safe_evaluate_best_point
from .utils import get_local_region

# Setup Logging
logger = logging.getLogger("auto_alignment")


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


@restore_on_error(context="alignment_opt_es")
def run_automatic_alignment(
    env,
    dump_location=None,
    *,
    to_screen_name="PROF571",
    n_steps=100,
    target_value=1.0,
    region_fraction=0.15,
):
    """Run the extremum-seeking alignment optimization process.

    Parameters
    ----------
    env : Any
        Control environment providing ``get_bounds``, ``get_variables``,
        and ``get_observables``.
    dump_location : str or pathlib.Path, optional
        Directory for optimization dump files.
    to_screen_name : str, optional
        Screen key from ``alignment_pvs`` used for the alignment optimization.
    n_steps : int, optional
        Number of optimization steps.
    target_value : float, optional
        Early-stop threshold for the ``norm`` objective.
    region_fraction : float, optional
        Fractional local-region width around the current machine state.

    Returns
    -------
    Xopt
        Optimizer instance containing all collected evaluations.
    """
    run_start_time = time.time()
    if dump_location is None:
        dump_location = "."

    # env.set_screen(to_screen_name)

    logger.info(f"Starting automatic alignment for screen: {to_screen_name}")
    logger.info(
        "Alignment ES config: n_steps=%d target_value=%s region_fraction=%s dump_location=%s",
        n_steps,
        target_value,
        region_fraction,
        dump_location,
    )
    # if just transporting beam to OTRDG02, use all BPMs except 470 and 520
    pvs = alignment_pvs[to_screen_name]["corrector_pvs"]
    bpm_observables = alignment_pvs[to_screen_name]["bpms"]
    logger.info(
        "Using %d correctors and %d BPM observables for %s.",
        len(pvs),
        len(bpm_observables),
        to_screen_name,
    )

    temp_vocs = VOCS(variables=env.get_bounds(pvs), observables=[])
    local_region = get_local_region(
        env.get_variables(temp_vocs.variables.keys()), temp_vocs, region_fraction
    )

    def eval(inputs):
        logger.debug("evaluating point")
        try:
            epics.caput_many(list(inputs.keys()), list(inputs.values()))
            time.sleep(0.2)

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
        oscillation_size=0.01,
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
        logger.info(
            "Converged immediately with norm=%s in %.2f s.",
            X.data.min()["norm"],
            time.time() - run_start_time,
        )
        return X

    try:
        for n in range(n_steps):
            logger.info(n)
            X.step()

    except Exception:
        logger.error("Exception:")
        logger.error(traceback.format_exc())
        raise
    finally:
        X.generator.reset()
        safe_evaluate_best_point(
            X,
            logger,
            metric_name="norm",
            context="extremum-seeking alignment finalization",
        )

    logger.info(
        "Automatic alignment (ES) complete: evaluations=%d best_norm=%s duration=%.2f s",
        len(X.data),
        X.data["norm"].min() if "norm" in X.data else "N/A",
        time.time() - run_start_time,
    )

    return X
