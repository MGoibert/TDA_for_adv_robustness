from r3d3 import R3D3Experiment

from tda.models.architectures import mnist_mlp, mnist_lenet
from tda.rootpath import rootpath, db_path

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'noise': [
            0.0
        ],
        'architecture': [
            mnist_lenet.name
        ],
        'epochs': [
            50
        ],
        'dataset': [
            "MNIST"
        ],
        'attack_type': [
            "FGSM"
        ],
        'num_iter': [
            -1
        ],
        'dataset_size': [500]
    },
    binary=f"{rootpath}/tda/experiments/attack_performance/attacks_performance_binary.py",
    max_nb_processes=1
)
