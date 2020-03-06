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
        'dataset_size': [
          500
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
