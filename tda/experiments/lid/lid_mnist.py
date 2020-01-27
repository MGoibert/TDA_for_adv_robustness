from r3d3 import R3D3Experiment

from tda.models.architectures import mnist_lenet
from tda.rootpath import rootpath, db_path

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
            "FGSM", "BIM", "CW", "DeepFool"
        ],
        'noise': [
            0.1
        ],
        "batch_size": [
            100
        ],
        "number_of_nn": [
            20
        ],
        "successful_adv": [
            1
        ],
        "nb_batches": [
            1
        ],
        "train_noise": [
            0.0
        ]
    },
    binary=f"{rootpath}/tda/experiments/lid/lid_binary.py",
    max_nb_processes=1
)
