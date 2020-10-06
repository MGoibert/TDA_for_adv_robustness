from r3d3 import R3D3Experiment

from tda.models.architectures import fashion_mnist_lenet
from tda.rootpath import rootpath, db_path

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'architecture': [
            fashion_mnist_lenet.name
        ],
        'epochs': [
            50
        ],
        'dataset': [
            "FashionMNIST"
        ],
        'attack_type': [
            "FGSM"#, "BIM", "CW", "DeepFool"
        ],
        'dataset_size': [
          100
        ],
        'number_of_samples_for_mu_sigma': [
          500
        ],
        'preproc_epsilon': [
            1e-3
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
