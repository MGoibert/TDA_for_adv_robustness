import json
from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import mnist_mlp

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'dataset': [
          "MNIST"
        ],
        'architecture': [
          "simple_fcn_mnist"
        ],
        'attack_type': [
          "CW"
        ],
        'thresholds': [
            "0.8_0.8_0.0"
        ],
        'epochs': [
            50
        ],
        'desired_y': [
           -1
        ]
    },
    binary=f"{rootpath}/tda/experiments/visualization/visualization_binary.py",
    max_nb_processes=1
)
