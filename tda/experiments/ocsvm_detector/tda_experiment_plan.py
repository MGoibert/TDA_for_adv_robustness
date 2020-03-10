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
import numpy as np

base_configs = cartesian_product(
    {
        "embedding_type": [EmbeddingType.PersistentDiagram],
        "kernel_type": [KernelType.SlicedWasserstein],
        "threshold_strategy": [
            ThresholdStrategy.ActivationValue,
            ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        ],
        "dataset_size": [500],
        "attack_type": ["FGSM"],
        "thresholds": [str(round(t, 2)) for t in np.linspace(0.1, 0.9, 9)],
        "noise": [0.0],
        "n_jobs": [24],
    }
)

binary = f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py"

all_experiments = list()

for model, dataset, nb_epochs in [
    [mnist_mlp.name, "MNIST", 50],
    [mnist_lenet.name, "MNIST", 50],
    [fashion_mnist_mlp.name, "FashionMNIST", 50],
    [fashion_mnist_lenet.name, "FashionMNIST", 200],
    [svhn_lenet.name, "SVHN", 250],
    [cifar_lenet.name, "CIFAR10", 300],
]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
