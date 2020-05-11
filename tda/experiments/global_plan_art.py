from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from copy import deepcopy
from tda.experiments.global_plan import experiment_plan as original_experiment_plan

new_experiments = list()
for experiment in original_experiment_plan.experiments:

    new_config = deepcopy(experiment.config)
    new_config["attack_type"] = experiment.config["attack_type"]+"_art"

    new_experiments.append(R3D3Experiment(
        binary=experiment.binary,
        config=new_config
    ))

experiment_plan = R3D3ExperimentPlan(
    max_nb_processes=original_experiment_plan.max_nb_processes,
    db_path=original_experiment_plan.db_path,
    experiments=new_experiments
)
