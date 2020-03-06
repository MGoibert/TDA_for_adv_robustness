from r3d3 import R3D3ExperimentPlan

from tda.models.architectures import fashion_mnist_lenet
from tda.rootpath import rootpath, db_path

experiment_plan = R3D3ExperimentPlan.from_cartesian_space(
    db_path=db_path,
    configs={
        'architecture': [
            fashion_mnist_lenet.name
        ],
        'epochs': [
            200
        ],
        'dataset': [
            "FashionMNIST"
        ],
        'attack_type': [
            "FGSM", "BIM", "CW", "DeepFool"
        ],
        'noise': [
            0.0
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
