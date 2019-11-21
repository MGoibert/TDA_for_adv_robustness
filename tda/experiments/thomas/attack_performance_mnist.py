from r3d3 import R3D3Experiment

from tda.models.architectures import mnist_mlp, mnist_lenet
from tda.rootpath import rootpath, db_path

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'noise': [
            0.0#, 0.02
        ],
        'architecture': [
            mnist_mlp.name,
            #mnist_lenet.name
        ],
        'epochs': [
            20
        ],
        'dataset': [
            "MNIST"
        ],
        'attack_type': [
            "CW", "DeepFool", "FGSM", "BIM"
        ],
        'num_iter': [
            25
        ]
    },
    binary=f"{rootpath}/tda/experiments/thomas/attacks_performance_binary.py",
    max_nb_processes=1
)
