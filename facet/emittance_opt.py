import sys
import time

def optimize_injector_emittance(env, dump_location):
    env.emittance_config_fname = (
        "/home/fphysics/rroussel/e331/Badger-Resources/facet/plugins/environments/inj_emit/emittance_measurement_configs/PROF10571.yaml"
    )
    env.save_directory = "data/"

    def evaluate(inputs):
        env.set_variables(inputs)
    
        time.sleep(0.1)  # Simulate some processing time
    
        # Get the output from the environment
        # note that output will contain many results, not just emittance_x
        # see FACET-II injector badger environment for details
        output = env.get_observables(["emittance_x"])
    
        return output

    vocs = VOCS(
        variables={
            "SOLN:IN10:121:BCTRL": [0.390,0.405],
            "QUAD:IN10:121:BCTRL": [-0.008,0.0085],
            "QUAD:IN10:122:BCTRL": [-0.008,0.0085],
            "QUAD:IN10:361:BCTRL": [-3, -2.5],
            "QUAD:IN10:371:BCTRL": [2.5,3],
        },
        objectives={"emittance_mean": "MINIMIZE"},
        constraints={"min_joint_bmag":["LESS_THAN",1.5]},
    )
    
    evaluator = Evaluator(function=evaluate)
    generator = ExpectedImprovementGenerator(vocs=vocs)
    
    X = Xopt(
        vocs=vocs,
        evaluator=evaluator,
        generator=generator,
        dump_file=f"5d_emittance_opt_{int(time.time())}.yaml",
    )

    # evaluate the current point and two random points
    X.evaluate_data(env.get_variables(X.vocs.variable_names))
    X.random_evaluate(2)

    for i in range(1):
        print(i)
        X.step()

    # evaluate the best point
    X.evaluate_data(X.vocs.select_best(X.data)[2])

    return X

    

    