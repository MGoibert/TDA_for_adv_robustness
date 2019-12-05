from r3d3 import R3D3Experiment

from tda.models.architectures import mnist_mlp, mnist_lenet
from tda.rootpath import rootpath, db_path
import numpy as np

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'architecture': [
            mnist_mlp.name
        ],
        'epochs': [
            20
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
        'epsilon': np.linspace(0.02, 0.5, 5)
    },
    binary=f"{rootpath}/tda/experiments/mahalanobis/mahalanobis_binary.py",
    max_nb_processes=1
)
