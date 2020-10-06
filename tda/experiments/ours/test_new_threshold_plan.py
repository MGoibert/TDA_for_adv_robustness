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

from tda.dataset.adversarial_generation import AttackType, AttackBackend


base_configs = cartesian_product(
    {
        "embedding_type": [EmbeddingType.PersistentDiagram], #, EmbeddingType.RawGraph],
        "dataset_size": [500],
        "attack_type": ["FGSM"],
        "attack_backend": [AttackBackend.FOOLBOX],
        "noise": [0.0],
        "n_jobs": [2],
        "all_epsilons": ["0.01;0.1;0.4"],
        "raw_graph_pca": [-1]
    }
)

binary = f"{rootpath}/tda/experiments/ours/our_binary.py"

all_experiments = list()

for model, dataset, nb_epochs, best_threshold, threshold_strategy, sigmoidize_rawgraph in [
    [
        mnist_lenet.name,
        "MNIST",
        50,
        "0:0.025_2:0.025_4:0.025_6:0.025",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        True,
    ],
    [
        mnist_lenet.name,
        "MNIST",
        50,
        "0:0.025_2:0.025_4:0.025_6:0.025", # true threshold
        ThresholdStrategy.UnderoptimizedMagnitudeIncreaseV3,
        True,
    ],
    
]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs
        config["thresholds"] = best_threshold
        config["threshold_strategy"] = threshold_strategy
        config["sigmoidize"] = sigmoidize_rawgraph

        if config["embedding_type"] == EmbeddingType.PersistentDiagram:
            config["kernel_type"] = KernelType.SlicedWasserstein
            config["sigmoidize"] = True
        elif config["embedding_type"] == EmbeddingType.RawGraph:
            config["kernel_type"] = KernelType.RBF

        if config["attack_type"] not in ["FGSM", "PGD"]:
            config["all_epsilons"] = "1.0"

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=2, db_path=db_path
)
