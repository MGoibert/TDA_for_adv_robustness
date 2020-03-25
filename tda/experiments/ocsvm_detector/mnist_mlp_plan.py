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
        "n_jobs": [5],
        "all_epsilons": ["0.01;0.1"],
    }
)

binary = f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py"

all_experiments = list()

thresholds = cartesian_product(
    {"0": [0, 0.05, 0.1, 0.2, 0.5, 0.7], "1": [0, 0.05, 0.1, 0.2, 0.5, 0.7]}
)

thresholds = [f"0:{z['0']}_1:{z['1']}" for z in thresholds]


for threshold in thresholds:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = mnist_mlp.name
        config["dataset"] = "MNIST"
        config["epochs"] = 50
        config["thresholds"] = threshold
        config["threshold_strategy"] = ThresholdStrategy.UnderoptimizedMagnitudeIncrease

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
