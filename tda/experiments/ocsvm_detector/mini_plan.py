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
        "dataset_size": [10],
        "attack_type": ["FGSM"],
        "noise": [0.01],
        "n_jobs": [1],
        "all_epsilons": ["0.1"]
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
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
