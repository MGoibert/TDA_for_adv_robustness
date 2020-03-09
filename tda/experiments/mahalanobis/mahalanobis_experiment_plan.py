from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product
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
        "attack_type": ["FGSM", "DeepFool"],
        "dataset_size": [500],
        "number_of_samples_for_mu_sigma": [500],
        "preproc_epsilon": [1e-2],
        "noise": [0.0],
        "successful_adv": [1],
    }
)

binary = f"{rootpath}/tda/experiments/mahalanobis/mahalanobis_binary.py"

all_experiments = list()

for model, dataset, nb_epochs in [
    [mnist_mlp.name, "MNIST", 50],
    [mnist_lenet.name, "MNIST", 50],
    [fashion_mnist_mlp.name, "FashionMNIST", 50],
    [fashion_mnist_lenet.name, "FashionMNIST", 200],
    [svhn_lenet.name, "SVHN", 250],
    [cifar_lenet.name, "CIFAR10", 300],
]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
