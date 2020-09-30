from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product

from tda.dataset.adversarial_generation import AttackType
from tda.models import cifar_resnet_1
from tda.models.architectures import (
    mnist_mlp,
    mnist_lenet,
    fashion_mnist_lenet,
    fashion_mnist_mlp,
    svhn_lenet,
    svhn_lenet_bandw,
    svhn_lenet_bandw2,
    cifar_lenet,
)
from tda.rootpath import rootpath, db_path
from copy import deepcopy

base_configs = cartesian_product(
    {
        "attack_type": [AttackType.PGD],
        "noise": [0.0],
        "dataset_size": [500],
        "successful_adv": [1],
        "train_noise": [0.0],
        "all_epsilons": ["0.01;0.1;0.4"],
        "selected_layers": ["all"] + [str(x) for x in range(48)],
    }
)

binary = f"{rootpath}/tda/experiments/lid/lid_binary.py"

all_experiments = list()

for model, dataset, nb_epochs, perc_of_nn, batch_size in [
    [cifar_resnet_1.name, "CIFAR10", 100, 0.3, 100],
]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs
        config["perc_of_nn"] = perc_of_nn
        config["batch_size"] = batch_size

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
