from r3d3 import R3D3Experiment

from tda.models.architectures import svhn_lenet
from tda.rootpath import rootpath, db_path

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'noise': [
            0.0#, 0.02
        ],
        'architecture': [
            svhn_lenet.name
        ],
        'epochs': [
            1
        ],
        'dataset': [
            "SVHN"
        ],
        'attack_type': [
            "FGSM"#"CW", "DeepFool", "FGSM", "BIM"
        ],
        'num_iter': [
            25
        ]
    },
    binary=f"{rootpath}/tda/experiments/attack_performance/attacks_performance_binary.py",
    max_nb_processes=1
)
