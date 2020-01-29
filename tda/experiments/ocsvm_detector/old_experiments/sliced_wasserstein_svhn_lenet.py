import json
from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import svhn_lenet

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
          "SVHN"
        ],
        'architecture': [
          svhn_lenet.name
        ],
        'attack_type': [
          "FGSM"
        ],
        'threshold': [
            '0.5_0_0.25_0_0.1_0_0'
        ],
        'noise': [
            0.0, 0.02
        ],
        'epochs': [
            100
        ],
        'identical_train_samples': [
            0, 1
        ],
        'dataset_size': [
            30
        ]
    },
    binary=f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py",
    max_nb_processes=1
)