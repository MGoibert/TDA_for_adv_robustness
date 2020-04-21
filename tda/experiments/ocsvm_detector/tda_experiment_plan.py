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

threshold_list = [
"0:0_2:0_4:0_5:0.1_6:0.2", "0:0_2:0_4:0_5:0.1_6:0.4", "0:0_2:0_4:0_5:0.2_6:0.1", "0:0_2:0_4:0_5:0.3_6:0.1", "0:0_2:0_4:0_5:0.3_6:0.3"
]
base_configs = cartesian_product(
    {
        "embedding_type": [EmbeddingType.PersistentDiagram],
        "dataset_size": [500],
        "attack_type": ["FGSM"],
        "noise": [0.0],
        "all_epsilons": ["0.1"],
        "n_jobs": [20],
        "thresholds": threshold_list
    }
)

binary = f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py"

all_experiments = list()

for model, dataset, nb_epochs, threshold_strategy, sigmoidize in [
    [
        cifar_lenet.name,
        "CIFAR10",
        300,
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        True
    ],
]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs
        config["threshold_strategy"] = threshold_strategy
        config["sigmoidize"] = sigmoidize

        if config["embedding_type"] == EmbeddingType.PersistentDiagram:
            config["kernel_type"] = KernelType.SlicedWasserstein
        elif config["embedding_type"] == EmbeddingType.RawGraph:
            config["kernel_type"] = KernelType.RBF

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
