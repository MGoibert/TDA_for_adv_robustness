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
        "embedding_type": [
            EmbeddingType.PersistentDiagram,
            EmbeddingType.RawGraph,
            EmbeddingType.WeisfeilerLehman,
        ],
        "dataset_size": [500],
        "attack_type": ["FGSM"],
        "noise": [0.0],
        "n_jobs": [24],
        "threshold_strategy": [
            ThresholdStrategy.QuantilePerGraphLayer,
            ThresholdStrategy.ActivationValue,
        ],
        "all_epsilons": ["0.01;0.1"],
        "architecture": [mnist_mlp.name],
        "epochs": [50],
        "dataset": ["MNIST"],
    }
)

binary = f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py"

all_experiments = list()

for threshold in [
    "0;1;0.1_-1;0;0.1_1;2;0.1",
    "0;1;0.5_-1;0;0.5_1;2;0.5",
    "0;1;0.9_-1;0;0.9_1;2;0.9",
]:
    for config in base_configs:
        config = deepcopy(config)
        config["thresholds"] = threshold
        if config["embedding_type"] == EmbeddingType.PersistentDiagram:
            config["kernel_type"] = KernelType.SlicedWasserstein
        else:
            config["kernel_type"] = KernelType.RBF

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
