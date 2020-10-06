from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product

from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import (
    mnist_mlp,
    mnist_lenet,
    fashion_mnist_lenet,
    fashion_mnist_mlp,
    svhn_lenet,
    cifar_lenet,
)
from tda.rootpath import rootpath, db_path
from copy import deepcopy

def generate_threshold(val_list, which_layer):
    all_layer = [0,2,4,5,6]
    list_threshold = list()
    for val in val_list:
        current_threshold = ""
        for layer in all_layer:
            layer = str(layer)
            if layer == "0":
                if layer == str(which_layer):
                    strg = str(which_layer) + ":" + str(val)
                else:
                    strg = layer + ":" + "0.0"
            else:
                if layer == str(which_layer):
                    strg = "_" + str(which_layer) + ":" + str(val)
                else:
                    strg = "_" + layer + ":" + "0.0"
            current_threshold += strg
        list_threshold.append(current_threshold)
    return list_threshold

#val_list = [0.01, 0.05, 0.1, 0.2, 0.3]  
#which_layer = 6
#list_thresholds = generate_threshold(val_list, which_layer)
#list_thresholds = ["0:0.0_2:0.0_4:0.001_5:0.01_6:0.1", "0:0.0_2:0.0_4:0.001_5:0.01_6:0.2", "0:0.0_2:0.0_4:0.001_5:0.01_6:0.3",
#"0:0.0_2:0.0_4:0.001_5:0.005_6:0.1", "0:0.0_2:0.0_4:0.001_5:0.02_6:0.1", "0:0.0_2:0.0_4:0.001_5:0.03_6:0.1",
#"0:0.0_2:0.0_4:0.0005_5:0.01_6:0.1", "0:0.0_2:0.0_4:0.002_5:0.01_6:0.1", "0:0.0_2:0.0_4:0.003_5:0.01_6:0.1",
#"0:0.0_2:0.00000001_4:0.001_5:0.01_6:0.1", "0:0.00000001_2:0.0_4:0.001_5:0.01_6:0.1", "0:0.000000001_2:0.00000001_4:0.001_5:0.01_6:0.1"]
#list_thresholds = ["0:0.0_2:0.0_4:0.001_5:0.01_6:0.1"]
#list_thresholds = ["0;1;0.0_-1;0;0.0_1;2;0_2;3;0.0_3;4;0.01_4;5;0.01_5;6;0.1"]
#list_thresholds = ["0;1;0.0_-1;0;0.0_1;2;10000_2;3;10000_3;4;10000_4;5;200000_5;6;15000"]
#list_thresholds = ["0;1;0.0_-1;0;0_1;2;0_2;3;0_3;4;0_4;5;0_5;6;30000"]
list_thresholds = ["0:0_2:0_4:0_5:0_6:0.1"]

base_configs = cartesian_product(
    {
        "embedding_type": [EmbeddingType.PersistentDiagram],
        "kernel_type": [KernelType.SlicedWasserstein],
        "dataset_size": [5],
        "attack_type": ["FGSM"],
        "noise": [0.0],
        "n_jobs": [2],
        "all_epsilons": ["0.05"],
        "dataset": ["CIFAR10"],
        "architecture": [cifar_lenet.name],
        "epochs": [300],
        "sigmoidize": [True],
        #"threshold_strategy": [ThresholdStrategy.ActivationValue] #[ThresholdStrategy.UnderoptimizedMagnitudeIncrease]
        "threshold_strategy": [ThresholdStrategy.UnderoptimizedMagnitudeIncrease]
    }
)

binary = f"{rootpath}/tda/experiments/visualization/visualization_binary_bis.py"
all_experiments = list()

for best_threshold in list_thresholds:
    for config in base_configs:
        config = deepcopy(config)
        config["thresholds"] = best_threshold
        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
