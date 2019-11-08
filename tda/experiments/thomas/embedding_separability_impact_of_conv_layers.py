from r3d3 import R3D3Experiment

from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import svhn_lenet
from tda.rootpath import rootpath, db_path

"""
The goal of this experiment
"""

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'embedding_type': [
            EmbeddingType.PersistentDiagram
        ],
        'kernel_type': [
            KernelType.SlicedWasserstein
        ],
        'dataset_size': [
            25
        ],
        'threshold': [
            "70000_1_100000_1_20000_0_0",
            "700000000_1_1000000000_1_20000_0_0"  # Skipping conv layers
        ],
        'noise': [
            0.0, 0.02
        ],
        'architecture': [
            svhn_lenet.name
        ],
        'epochs': [
            100
        ]
    },
    binary=f"{rootpath}/tda/experiments/thomas/embedding_separability_binary_gram.py",
    max_nb_processes=1
)
