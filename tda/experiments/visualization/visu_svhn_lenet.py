import json
from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import mnist_mlp

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'dataset': [
          "SVHN"
        ],
        'architecture': [
          "svhn_lenet"
        ],
        'attack_type': [
          "FGSM"
        ],
        'thresholds': [
            "0.5_0_0.25_0_0.1_0_0"
        ],
        'epochs': [
            200
        ],
        'attack_type': [
           "FGSM"
        ]
    },
    binary=f"{rootpath}/tda/experiments/visualization/visualization_binary.py",
    max_nb_processes=1
)
