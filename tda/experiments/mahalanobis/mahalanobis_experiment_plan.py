from copy import deepcopy

from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product

from tda.dataset.adversarial_generation import AttackType, AttackBackend
from tda.experiments.mahalanobis.mahalanobis_binary_v2 import CovarianceMethod
from tda.models.architectures import (
    mnist_lenet,
    fashion_mnist_lenet,
    cifar_resnet_1,
)
from tda.rootpath import rootpath, db_path

base_configs = cartesian_product(
    {
        "attack_type": [AttackType.PGD],
        "attack_backend": [AttackBackend.FOOLBOX],
        "dataset_size": [500],
        "number_of_samples_for_mu_sigma": [500],
        "preproc_epsilon": [1e-2],
        "noise": [0.0],
        "successful_adv": [1],
        "all_epsilons": ["0.1"],
        "covariance_method": [
            # CovarianceMethod.NAIVE,
            CovarianceMethod.NAIVE_SVD,
            # CovarianceMethod.LEDOIT_WOLF,
            # CovarianceMethod.GRAPHICAL_LASSO
        ]
    }
)

binary = f"{rootpath}/tda/experiments/mahalanobis/mahalanobis_binary_v2.py"

all_experiments = list()

for model, dataset, nb_epochs in [
    #[mnist_lenet.name, "MNIST", 50],
    #[fashion_mnist_lenet.name, "FashionMNIST", 100],
    [cifar_resnet_1.name, "CIFAR10", 100],

]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs
        config["selected_layers"] = "all"

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
