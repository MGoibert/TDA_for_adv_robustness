from r3d3 import R3D3Experiment

from tda.models.architectures import mnist_mlp, mnist_lenet
from tda.rootpath import rootpath, db_path

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'noise': [
            0.0
        ],
        'architecture': [
            mnist_mlp.name
        ],
        'epochs': [
            20
        ],
        'dataset': [
            "MNIST"
        ],
        'attack_type': [
            "FGSM"
        ],
        'num_iter': [
            25
        ],
        'dataset_size': [
          30
        ],
        'thresholds': [
            '_'.join([str(20000) for _ in range(mnist_mlp.get_nb_graph_layers())])
        ]
    },
    binary=f"{rootpath}/tda/experiments/predictions_scores/scatter_binary.py",
    max_nb_processes=1
)
