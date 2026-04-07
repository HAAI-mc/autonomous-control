import numpy as np
import time

from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator

def optimize_energy_spread(env, dump_location):
    dipole_correct_state = 0.125
    dipole_current_state = env.get_variables(
        ["BEND:IN10:661:BACT"]
    )["BEND:IN10:661:BACT"]

    klys_phase_set_pv = "KLYS:LI10:41:SFB_PDES"
    klys_phase_readback_pv = 'ACCL:LI10:41:PHASE_W0CH0'

    if not np.isclose(
        dipole_correct_state, 
        dipole_current_state,
        rtol=1e-2
    ):
        raise RuntimeError("dipole not in correct state for energy measurements")

    measurement = env.create_beamprofile_measurement("PROF10711")

    def evaluate(inputs):
        env.set_variables(inputs)
        print('changing phase')
        time.sleep(0.1)  # Simulate some processing time
        while not np.isclose(
            env.get_observables(
                [klys_phase_readback_pv]
            )[klys_phase_readback_pv]
        , inputs[klys_phase_set_pv], atol=0.05
        ):
            time.sleep(0.1)
        print('phase changed')
            
        # Get the output from the environment
        results = measurement.measure()
        output = results.rms_sizes_all
        output_dict = dict(zip(['rms_x', 'rms_y'], list(output.flatten())))
    
        return output_dict

    initial_phase = env.get_variables([klys_phase_set_pv])[klys_phase_set_pv]

    # Define the VOCS for the optimization problem
    # NOTE: only QE01-04 are currently available in the simulated server
    vocs = VOCS(
        variables={
            klys_phase_set_pv: [initial_phase-5, initial_phase+5],
        },
        objectives={"rms_x": "MINIMIZE"},
    )
    
    evaluator = Evaluator(function=evaluate)
    generator = ExpectedImprovementGenerator(
        vocs=vocs, 
        n_interpolate_points=5
    )
    
    X = Xopt(
        vocs=vocs,
        evaluator=evaluator,
        generator=generator,
        dump_file=f"energy_spread_minimization_{int(time.time())}.yaml"
    )
    X.random_evaluate(3)

    for i in range(2):
        X.step()
    print("_____________FINISHED______________")

    X.evaluate_data(X.vocs.select_best(X.data)[2])
    return X
