from r3d3 import R3D3Experiment

from tda.models.architectures import mnist_mlp
from tda.rootpath import rootpath, db_path

experiment = R3D3Experiment(
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
        'attack_type': [
            "FGSM", "BIM", "CW", "DeepFool"
        ],
        'noise': [
            0.1
        ],
        "dataset_size": [
            500
        ],
        "batch_size": [
            100
        ],
        "perc_of_nn": [
            0.2
        ],
        "successful_adv": [
            1
        ],
        "train_noise": [
            0.0
        ]
    },
    binary=f"{rootpath}/tda/experiments/lid/lid_binary.py",
    max_nb_processes=1
)
