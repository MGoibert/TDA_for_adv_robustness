from r3d3 import R3D3Experiment

from tda.models.architectures import cifar_lenet
from tda.rootpath import rootpath, db_path

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'architecture': [
            cifar_lenet.name
        ],
        'epochs': [
            200
        ],
        'dataset': [
            "CIFAR10"
        ],
        'attack_type': [
            "FGSM", "BIM", "CW", "DeepFool"
        ],
        'dataset_size': [
          100
        ],
        'number_of_samples_for_mu_sigma': [
          500
        ],
        'preproc_epsilon': [
            1e-2
        ],
        'noise': [
            0.0
        ],
        'successful_adv': [
            1
        ]
    },
    binary=f"{rootpath}/tda/experiments/mahalanobis/mahalanobis_binary.py",
    max_nb_processes=1
)
