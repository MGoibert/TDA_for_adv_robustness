from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product

from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import (
    mnist_mlp,
    mnist_lenet,
    fashion_mnist_lenet,
    fashion_mnist_mlp,
    svhn_lenet,
    svhn_lenet_bandw,
    cifar_lenet,
)
from tda.rootpath import rootpath, db_path
from copy import deepcopy

#threshold_list = ["0:0.05_1:0.05_2:0.05_3:0.05_4:0.0", "0:0.1_1:0.1_2:0.1_3:0.1_4:0.0", "0:0.2_1:0.2_2:0.2_3:0.2_4:0.0",
#                  "0:0.3_1:0.3_2:0.3_3:0.3_4:0.0", "0:0.4_1:0.4_2:0.4_3:0.4_4:0.0", "0:0.5_1:0.5_2:0.5_3:0.5_4:0.0"]
#threshold_list = ["0:0.05_2:0.0_4:0.0_5:0.0_6:0.0", "0:0.1_2:0.0_4:0.0_5:0.0_6:0.0", "0:0.2_2:0.0_4:0.0_5:0.0_6:0.0", "0:0.3_2:0.0_4:0.0_5:0.0_6:0.0", "0:0.4_2:0.0_4:0.0_5:0.0_6:0.0", "0:0.5_2:0.0_4:0.0_5:0.0_6:0.0",
#                  "0:0.05_2:0.05_4:0.0_5:0.0_6:0.0", "0:0.1_2:0.1_4:0.0_5:0.0_6:0.0", "0:0.2_2:0.2_4:0.0_5:0.0_6:0.0", "0:0.3_2:0.3_4:0.0_5:0.0_6:0.0", "0:0.4_2:0.4_4:0.0_5:0.0_6:0.0", "0:0.5_2:0.5_4:0.0_5:0.0_6:0.0",
#                  "0:0.05_2:0.05_4:0.05_5:0.0_6:0.0", "0:0.1_2:0.1_4:0.1_5:0.0_6:0.0", "0:0.2_2:0.2_4:0.2_5:0.0_6:0.0", "0:0.3_2:0.3_4:0.3_5:0.0_6:0.0", "0:0.4_2:0.4_4:0.4_5:0.0_6:0.0", "0:0.5_2:0.5_4:0.5_5:0.0_6:0.0",
#                  "0:0.05_2:0.05_4:0.05_5:0.05_6:0.0", "0:0.1_2:0.1_4:0.1_5:0.1_6:0.0", "0:0.2_2:0.2_4:0.2_5:0.2_6:0.0", "0:0.3_2:0.3_4:0.3_5:0.3_6:0.0", "0:0.4_2:0.4_4:0.4_5:0.4_6:0.0", "0:0.5_2:0.5_4:0.5_5:0.5_6:0.0",
#                  "0:0.05_2:0.05_4:0.05_5:0.05_6:0.05", "0:0.1_2:0.1_4:0.1_5:0.1_6:0.1", "0:0.2_2:0.2_4:0.2_5:0.2_6:0.2", "0:0.3_2:0.3_4:0.3_5:0.3_6:0.3", "0:0.4_2:0.4_4:0.4_5:0.4_6:0.4", "0:0.5_2:0.5_4:0.5_5:0.5_6:0.5"]
threshold_list = ["0:0.1_2:0.0_4:0.0_5:0.0_6:0.0", "0:0.1_2:0.1_4:0.0_5:0.0_6:0.0", "0:0.1_2:0.1_4:0.1_5:0.0_6:0.0", "0:0.1_2:0.1_4:0.1_5:0.1_6:0.0", "0:0.1_2:0.1_4:0.1_5:0.1_6:0.1",
"0:0.05_2:0.0_4:0.0_5:0.0_6:0.0", "0:0.05_2:0.05_4:0.0_5:0.0_6:0.0", "0:0.05_2:0.05_4:0.05_5:0.0_6:0.0", "0:0.05_2:0.05_4:0.05_5:0.05_6:0.0", "0:0.05_2:0.05_4:0.05_5:0.05_6:0.05"]
#threshold_list = [
#        "0:0.025_1:0.0_2:0.0_3:0.0_4:0.0", "0:0.025_1:0.025_2:0.0_3:0.0_4:0.0", "0:0.025_1:0.025_2:0.025_3:0.0_4:0.0", "0:0.025_1:0.025_2:0.025_3:0.025_4:0.0", "0:0.025_1:0.025_2:0.025_3:0.025_4:0.025",
#        "0:0.05_1:0.0_2:0.0_3:0.0_4:0.0", "0:0.05_1:0.05_2:0.0_3:0.0_4:0.0", "0:0.05_1:0.05_2:0.05_3:0.0_4:0.0", "0:0.05_1:0.05_2:0.05_3:0.05_4:0.0", "0:0.05_1:0.05_2:0.05_3:0.05_4:0.05", 
#        "0:0.75_1:0.0_2:0.0_3:0.0_4:0.0", "0:0.75_1:0.75_2:0.0_3:0.0_4:0.0", "0:0.75_1:0.75_2:0.75_3:0.0_4:0.0", "0:0.75_1:0.75_2:0.75_3:0.75_4:0.0", "0:0.75_1:0.75_2:0.75_3:0.75_4:0.75", 
#        ]

base_configs = cartesian_product(
    {
        "embedding_type": [EmbeddingType.PersistentDiagram],
        "kernel_type": [KernelType.SlicedWasserstein],
        "dataset_size": [500],
        "attack_type": ["FGSM"],
        "noise": [0.0],
        "n_jobs": [6],
        "all_epsilons": ["0.01;0.1;0.4"],
        "sigmoidize": [True],
        "thresholds": threshold_list
    }
)

binary = f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py"

all_experiments = list()

for model, dataset, nb_epochs, threshold_strategy in [
    [   # AUC : 0.01: 0.975, 0.1: 0.975
        svhn_lenet_bandw.name,
        "SVHN_BandW",
        200,
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        False
    ],
]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs
        #config["thresholds"] = best_threshold
        config["threshold_strategy"] = threshold_strategy
        config["sigmoidize"] = sigmoidize

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=5, db_path=db_path
)
