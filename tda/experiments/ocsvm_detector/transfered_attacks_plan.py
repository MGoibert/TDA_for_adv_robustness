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
        "dataset_size": [300],
        "attack_type": ["FGSM"],
        "noise": [0.0],
        "n_jobs": [6],
        "all_epsilons": ["0.05;0.07;0.1;0.4"],
        "transfered_attacks": [True],
        "archi_trsf_attack": ["None"],
        "perc_of_nn": [0.2],
        "batch_size": [150],
        "number_of_samples_for_mu_sigma": [500],
        "preproc_epsilon": [1e-2],
        "raw_graph_pca": [-1]
    }
)

binaries = list()
#for _ in range(len(threshold_list)):
binaries.append(f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py")
binaries.append(f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py")
binaries.append(f"{rootpath}/tda/experiments/lid/lid_binary.py")
binaries.append(f"{rootpath}/tda/experiments/mahalanobis/mahalanobis_binary.py")

embeddings = [EmbeddingType.PersistentDiagram, EmbeddingType.RawGraph, EmbeddingType.PersistentDiagram, EmbeddingType.PersistentDiagram]
kernels = [KernelType.SlicedWasserstein, KernelType.RBF, KernelType.SlicedWasserstein, KernelType.SlicedWasserstein]
sig1 = [True, True, False, False]
sig2 = [True, False, False, False]

all_experiments = list()

for i, (binary, embedding, kernel) in enumerate(zip(binaries, embeddings, kernels)):
    for model, dataset, nb_epochs, threshold, threshold_strategy, sig in [
        [
            mnist_lenet.name,
            "MNIST",
            50,
            "0:0.05_2:0.05_4:0.05_5:0.0",
            ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
            sig1[i]
        ],
        [
            fashion_mnist_lenet.name,
            "FashionMNIST",
            200,
            "0:0.05_2:0.05_4:0.0_5:0.0",
            ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
            sig1[i]
        ],
        [
            svhn_lenet.name,
            "SVHN",
            250,
            "0:0_2:0.5_4:0.5_5:0_6:0",
            ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
            sig2[i]
        ],
        [
            cifar_lenet.name,
            "CIFAR10",
            300,
            "0:0_2:0_4:0_5:0.1_6:0.3",
            ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
            sig2[i]
        ]
    ]:
        for config in base_configs:
            config = deepcopy(config)
            config["architecture"] = model
            config["dataset"] = dataset
            config["epochs"] = nb_epochs
            config["embedding_type"] = embedding
            config["kernel_type"] = kernel
            config["thresholds"] = threshold
            config["threshold_strategy"] = threshold_strategy
            config["sigmoidize"] = sig
 
            all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
