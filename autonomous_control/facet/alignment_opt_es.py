import logging
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.sequential import ExtremumSeekingGenerator
import numpy as np
import traceback
import time
from ml_tto.errors import TransmissionError
import epics

from xopt.vocs import (
    get_local_region,
)


from autonomous_control.facet.optimization_utils import (
    restore_on_error,
    safe_evaluate_best_point,
)

# Setup Logging
logger = logging.getLogger("auto_alignment")


bpms = [371, 425, 511, 525, 581, 631, 651]
DEFAULT_ALIGNMENT_PVS = {
    "PR10571": {
        "corrector_pvs": [
            f"XCOR:IN10:{ele}:BCTRL" for ele in [221, 311, 381, 411, 491, 521, 641]
        ]
        + [f"YCOR:IN10:{ele}:BCTRL" for ele in [222, 312, 382, 412, 492, 522, 642]],
        "bpms": [f"BPMS:IN10:{ele}:X" for ele in bpms]
        + [f"BPMS:IN10:{ele}:Y" for ele in bpms],
        "upstream_bpm": "BPM10371",
        "downstream_bpm": "BPM10651",
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
        "upstream_bpm": "BPM10371",  # TODO: check if this is correct for LI11312
        "downstream_bpm": "BPM10651",
    },
}


@restore_on_error(context="alignment_opt_es")
def optimize_alignment(
    env,
    dump_location=None,
    to_screen_name="PR10571",
    custom_corrector_pvs=None,
    custom_bpm_observable_pvs=None,
    custom_upstream_bpm_name=None,
    custom_downstream_bpm_name=None,
    n_steps=100,
    target_value=1.0,
    region_fraction=0.15,
    oscillation_size=0.01,
):
    """Run the extremum-seeking alignment optimization process.

    Users specify a default set of corrector PVs and BPM observable PVs for alignment by setting the
    ``to_screen_name`` argument. If custom corrector PVs or BPM observable PVs are desired,
    they can be specified with the ``custom_corrector_pvs``, ``custom_bpm_observable_pvs``,
    ``custom_upstream_bpm``, and ``custom_downstream_bpm`` arguments and will override the default
    PVs for the specified screen.

    Parameters
    ----------
    env : Any
        Control environment providing ``get_bounds``, ``get_variables``,
        and ``get_observables``.
    dump_location : str or pathlib.Path, optional
        Xopt dump file path, by default None.
    to_screen_name : str, optional
        Screen name to align to, by default ``"PR10571"``.
    custom_corrector_pvs : list of str, optional
        Custom corrector PVs to use for alignment, by default None.
    custom_bpm_observable_pvs : list of str, optional
        Custom BPM observable PVs to use for alignment, by default None.
    custom_upstream_bpm_name : str, optional
        Custom upstream BPM PV to use for alignment, by default None.
    custom_downstream_bpm_name : str, optional
        Custom downstream BPM PV to use for alignment, by default None.
    n_steps : int, optional
        Maximum number of extremum-seeking steps, by default 100.
    target_value : float, optional
        BPM-norm convergence threshold, by default 1.0.
    region_fraction : float, optional
        Half-width of the search region as a fraction of variable range,
        by default 0.15.
    oscillation_size : float, optional
        Size of the oscillation for extremum-seeking, by default 0.01.
    Returns
    -------
    Xopt
        Optimizer instance containing all collected evaluations.
    """
    # env.set_screen(to_screen_name)

    logger.info(f"Starting automatic alignment for screen: {to_screen_name}")

    # load in default corrector and BPM PVs for the specified screen
    # override with custom PVs if provided
    pvs = custom_corrector_pvs or DEFAULT_ALIGNMENT_PVS[to_screen_name]["corrector_pvs"]
    bpm_observables = (
        custom_bpm_observable_pvs or DEFAULT_ALIGNMENT_PVS[to_screen_name]["bpms"]
    )
    upstream_bpm_name = (
        custom_upstream_bpm_name
        or DEFAULT_ALIGNMENT_PVS[to_screen_name]["upstream_bpm"]
    )
    downstream_bpm_name = (
        custom_downstream_bpm_name
        or DEFAULT_ALIGNMENT_PVS[to_screen_name]["downstream_bpm"]
    )

    temp_vocs = VOCS(variables=env.get_bounds(pvs), observables=[])
    local_region = get_local_region(
        temp_vocs, env.get_variables(temp_vocs.variables.keys()), region_fraction
    )

    # set environment BPMs for transmission measurement
    env.upstream_bpm_name = upstream_bpm_name
    env.downstream_bpm_name = downstream_bpm_name

    def eval(inputs):
        logger.info(f"evaluating point: {inputs}")
        try:
            epics.caput_many(list(inputs.keys()),list(inputs.values()))
            #env.set_variables(inputs)
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
        oscillation_size=oscillation_size,
    )
    evaluator = Evaluator(function=eval)

    X = Xopt(
        vocs=vocs,
        generator=generator,
        evaluator=evaluator,
        dump_file=dump_location,
        strict=True,
    )
    logger.info("Created alignment Xopt object.")

    logger.info("Starting evaluation")
    # evaluate
    X.evaluate_data(env.get_variables(vocs.variables.keys()))

    if X.data.min()["norm"] < target_value:
        logger.info("converged")
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

    return X
