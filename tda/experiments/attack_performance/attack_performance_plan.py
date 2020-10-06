from copy import deepcopy

from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product

from tda.models.architectures import (
    mnist_lenet,
    fashion_mnist_lenet,
    svhn_lenet,
    cifar_lenet,
    cifar_toy_resnet,
    cifar_resnet_1,
)
from tda.rootpath import rootpath, db_path
from tda.dataset.adversarial_generation import AttackType, AttackBackend

base_configs = cartesian_product(
    {
        "dataset_size": [500],
        "attack_type": [AttackType.PGD, AttackType.CW],
        "attack_backend": [
            AttackBackend.FOOLBOX,
        ],
        "noise": [0.0],
        "all_epsilons": ["0.01;0.1;0.4"],
    }
)

binary = f"{rootpath}/tda/experiments/attack_performance/attacks_performance_binary.py"

all_experiments = list()

for model, dataset, nb_epochs in [
    # [mnist_lenet.name, "MNIST", 50],
    # [fashion_mnist_lenet.name, "FashionMNIST", 100],
    # [svhn_lenet.name, "SVHN", 300],
    # [cifar_lenet.name, "CIFAR10", 300],
    # [cifar_toy_resnet.name, "CIFAR10", 300],
    [cifar_resnet_1.name, "CIFAR10", 100],
]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs

        if not AttackType.require_epsilon(config["attack_type"]):
            config["all_epsilons"] = "1.0"

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
