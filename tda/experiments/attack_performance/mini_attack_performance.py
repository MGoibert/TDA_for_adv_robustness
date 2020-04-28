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

experiment_plan = R3D3ExperimentPlan.from_cartesian_space(
    db_path=db_path,
    configs={
        'architecture': [
            mnist_mlp.name
        ],
        'epochs': [
            25
        ],
        'dataset': [
            "MNIST"
        ],
        'dataset_size': [
            500
        ],
        'attack_type': [
            "FGSM"
        ],
        'noise': [
            0.0
        ],
        'first_pruned_iter': [
            10
        ],
        'prune_percentile': [
            0.1
        ],
        'tot_prune_percentile': [
            0.8
        ],
        'n_jobs': [
            1
        ]
    },
    binary=f"{rootpath}/tda/experiments/attack_performance/attacks_performance_binary.py",
    max_nb_processes=1
)
