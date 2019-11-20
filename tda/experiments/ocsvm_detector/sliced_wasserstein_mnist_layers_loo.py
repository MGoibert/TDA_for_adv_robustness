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
            '20000_20000_20000',
            'inf_20000_20000',
            '20000_inf_20000',
            '20000_20000_inf'
        ],
        'noise': [
            0.0
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
