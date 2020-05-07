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
        "embedding_type": [EmbeddingType.PersistentDiagram, EmbeddingType.RawGraph],
        "dataset_size": [300],
        "attack_type": ["FGSM"],
        "noise": [0.0],
        "n_jobs": [8],
        "all_epsilons": ["0.05;0.07;0.1;0.4"],
        "raw_graph_pca": [-1],
        "transfered_attacks": [True],
        "archi_trsf_attack": ["None"],
    }
)

binary = f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py"

all_experiments = list()

for model, dataset, nb_epochs, best_threshold, threshold_strategy, sigmoidize_rawgraph in[
    [
        mnist_lenet.name,
        "MNIST",
        50,
        "0:0.05_2:0.05_4:0.05_5:0.0",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        True,
    ],
    [  # AUC : 0.01: 0.975, 0.1: 0.975
        fashion_mnist_lenet.name,
        "FashionMNIST",
        200,
        "0:0.05_2:0.05_4:0.0_5:0.0",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        True,
    ],
    [
        svhn_lenet.name,
        "SVHN",
        250,
        "0:0_2:0.5_4:0.5_5:0_6:0",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        False,
    ],
    [
        cifar_lenet.name,
        "CIFAR10",
        300,
        "0:0_2:0_4:0_5:0.1_6:0.3",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        True
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
 
        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=5, db_path=db_path
)
