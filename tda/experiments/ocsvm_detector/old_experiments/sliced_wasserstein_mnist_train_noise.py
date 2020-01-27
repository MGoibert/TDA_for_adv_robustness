import json
from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import mnist_mlp

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'embedding_type': [
            EmbeddingType.PersistentDiagram
            ],
        'kernel_type': [
            KernelType.SlicedWasserstein
        ],
        'dataset': [
          "MNIST"
        ],
        'attack_type': [
          "FGSM"
        ],
        'threshold': [
            '_'.join([str(20000) for _ in range(mnist_mlp.get_nb_graph_layers())])
        ],
        'noise': [
            0.0, 0.05
        ],
        'train_noise': [
            0.0, 0.05
        ],
        'epochs': [
            50
        ],
        'identical_train_samples': [
            1
        ]
    },
    binary=f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py",
    max_nb_processes=1
)
