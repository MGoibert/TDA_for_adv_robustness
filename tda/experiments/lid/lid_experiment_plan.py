from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product
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
        "attack_type": ["FGSM_art"],
        "noise": [0.0],
        "dataset_size": [500],
        "successful_adv": [1],
        "train_noise": [0.0],
        "all_epsilons": ["0.01;0.1;0.4"],
    }
)

binary = f"{rootpath}/tda/experiments/lid/lid_binary.py"

all_experiments = list()

for model, dataset, nb_epochs, perc_of_nn, batch_size in [
    [mnist_mlp.name, "MNIST", 50, 0.1, 250],  # Not benched
    [mnist_lenet.name, "MNIST", 50, 0.08, 250],
    [fashion_mnist_mlp.name, "FashionMNIST", 50, 0.1, 250],  # Not benched
    [fashion_mnist_lenet.name, "FashionMNIST", 200, 0.02, 250],
    [svhn_lenet.name, "SVHN", 250, 0.1, 250],
    [svhn_lenet_bandw2.name, "SVHN_BandW", 200, 0.1, 250],  # Not benched
    [cifar_lenet.name, "CIFAR10", 300, 0.3, 100],
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
    experiments=all_experiments, max_nb_processes=4, db_path=db_path
)
