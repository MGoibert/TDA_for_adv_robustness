from r3d3 import R3D3Experiment

from tda.models.architectures import mnist_mlp, mnist_lenet
from tda.rootpath import rootpath, db_path
import numpy as np

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
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
        'dataset_size': [
          25
        ],
        'epsilon': [
            0.02, 0.1, 0.2, 0.4
        ],
        'noise': [
            0.1
        ],
        "batch_size": [
            500
        ],
        "number_of_nn": [
            100
        ]
    },
    binary=f"{rootpath}/tda/experiments/lid/lid_binary.py",
    max_nb_processes=1
)
