from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product

from tda.dataset.adversarial_generation import AttackType, AttackBackend
from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import (
    cifar_resnet_1,
    mnist_lenet,
    fashion_mnist_lenet,
    svhn_resnet_1,
)
from tda.rootpath import rootpath, db_path
from copy import deepcopy


base_configs = cartesian_product(
    {
        "embedding_type": [EmbeddingType.PersistentDiagram],
        "dataset_size": [500],
        "attack_type": [AttackType.PGD, AttackType.CW, AttackType.BOUNDARY],
        "attack_backend": [AttackBackend.FOOLBOX],
        "noise": [0.0],
        "n_jobs": [1],
        "all_epsilons": ["0.01;0.05;0.1"],
        "raw_graph_pca": [-1],
    }
)

binary = f"{rootpath}/tda/experiments/ours/our_binary.py"

all_experiments = list()

for (model, dataset, nb_epochs, best_threshold, threshold_strategy, sigmoidize,) in [
    # MNIST
    [
        mnist_lenet.name,
        "MNIST",
        50,
        "0:0.0:0.025_2:0.0:0.025_4:0.0:0.025_6:0.0:0.025",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        False,
    ],
    [
        mnist_lenet.name,
        "MNIST",
        50,
        "0:0.0:0.025_2:0.0:0.025_4:0.0:0.025_6:0.0:0.025",
        ThresholdStrategy.UnderoptimizedLargeFinal,
        False,
    ],
    [
        mnist_lenet.name,
        "MNIST",
        50,
        "0:0.975:1.0_2:0.975:1.0_4:0.975:1.0_6:0.975:1.0",
        ThresholdStrategy.UnderoptimizedLargeFinal,
        False,
    ],
    # FashionMNIST
    [
        fashion_mnist_lenet.name,
        "FashionMNIST",
        100,
        "0:0.0:0.05_2:0.0:0.05_4:0.0:0.05_6:0.0:0.05",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        False,
    ],
    [
        fashion_mnist_lenet.name,
        "FashionMNIST",
        100,
        "0:0.0:0.05_2:0.0:0.05_4:0.0:0.05_6:0.0:0.05",
        ThresholdStrategy.UnderoptimizedLargeFinal,
        False,
    ],
    [
        fashion_mnist_lenet.name,
        "FashionMNIST",
        100,
        "0:0.95:1.0_2:0.95:1.0_4:0.95:1.0_6:0.95:1.0",
        ThresholdStrategy.UnderoptimizedLargeFinal,
        False,
    ],
    # CIFAR10
    [
        cifar_resnet_1.name,
        "CIFAR10",
        100,
        "39:0.0:0.3_40:0.0:0.3_41:0.0:0.3_42:0.0:0.3_43:0.0:0.3",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        False,
    ],
    [
        cifar_resnet_1.name,
        "CIFAR10",
        100,
        "39:0.0:0.3_40:0.0:0.3_41:0.0:0.3_42:0.0:0.3_43:0.0:0.3",
        ThresholdStrategy.UnderoptimizedLargeFinal,
        False,
    ],
    [
        cifar_resnet_1.name,
        "CIFAR10",
        100,
        "39:0.7:1.0_40:0.7:1.0_41:0.7:1.0_42:0.7:1.0_43:0.7:1.0",
        ThresholdStrategy.UnderoptimizedLargeFinal,
        False,
    ],
    # SVHN
    [
        svhn_resnet_1.name,
        "CIFAR10",
        100,
        "39:0.0:0.275_40:0.0:0.275_41:0.0:0.275_42:0.0:0.275_43:0.0:0.275",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        False,
    ],
    [
        svhn_resnet_1.name,
        "CIFAR10",
        100,
        "39:0.0:0.275_40:0.0:0.275_41:0.0:0.275_42:0.0:0.275_43:0.0:0.275",
        ThresholdStrategy.UnderoptimizedLargeFinal,
        False,
    ],
    [
        svhn_resnet_1.name,
        "CIFAR10",
        100,
        "39:0.725:1.0_40:0.725:1.0_41:0.725:1.0_42:0.725:1.0_43:0.725:1.0",
        ThresholdStrategy.UnderoptimizedLargeFinal,
        False,
    ],
]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs
        config["thresholds"] = best_threshold
        config["threshold_strategy"] = threshold_strategy
        config["sigmoidize"] = sigmoidize

        if config["attack_type"] not in ["FGSM", "PGD"]:
            config["all_epsilons"] = "1.0"

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=8, db_path=db_path
)
