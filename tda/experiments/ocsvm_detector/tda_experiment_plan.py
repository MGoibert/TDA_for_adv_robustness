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

base_configs = cartesian_product(
    {
        "embedding_type": [EmbeddingType.PersistentDiagram],
        "kernel_type": [KernelType.SlicedWasserstein],
        "dataset_size": [500],
        "attack_type": ["FGSM"],
        "noise": [0.0],
        "n_jobs": [8],
        "all_epsilons": ["0.01;0.1;0.4"]
    }
)

binary = f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py"

all_experiments = list()

for model, dataset, nb_epochs, best_threshold, threshold_strategy in [
    [
        mnist_mlp.name,
        "MNIST",
        50,
        "0:0.1_1:0.1_2:0.0",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
    ],
    [
        mnist_lenet.name,
        "MNIST",
        50,
        "0:0.05_2:0.05_4:0.05_5:0.0",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
    ],
    [
        fashion_mnist_mlp.name,
        "FashionMNIST",
        50,
        "0:0.1_1:0.1_2:0.0",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
    ],
    [   # AUC : 0.01: 0.975, 0.1: 0.975
        fashion_mnist_lenet.name,
        "FashionMNIST",
        200,
        "0:0.05_2:0.05_4:0.0_5:0.0",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
    ],
    [
        svhn_lenet.name,
        "SVHN",
        250,
        "0;1;0.02_-1;0;0.3_1;2;0.3_2;3;0.02",
        ThresholdStrategy.ActivationValue,
    ],
    [
        svhn_lenet.name,
        "SVHN",
        250,
        "0:0.1_2:0.1_4:0.1_5:0.1_6:0.0",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
    ],
    [
        cifar_lenet.name,
        "CIFAR10",
        300,
        "0:0.1_2:0.1_4:0.1_5:0.1_6:0.0",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
    ],
]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs
        config["thresholds"] = best_threshold
        config["threshold_strategy"] = threshold_strategy

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=4, db_path=db_path
)
