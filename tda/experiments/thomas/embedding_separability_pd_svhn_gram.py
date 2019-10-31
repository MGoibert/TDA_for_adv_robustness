from r3d3 import R3D3Experiment

from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import svhn_lenet
from tda.rootpath import rootpath, db_path

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'embedding_type': [
            EmbeddingType.PersistentDiagram
            ],
        'kernel_type': [
            KernelType.SlicedWasserstein
        ],
        'thresholds': [
            "70000_1_100000_1_20000_0_0"
        ],
        'noise': [
            0.0, 1e-4, 0.01
        ],
        'architecture': [
            svhn_lenet.name
        ],
        'dataset': [
            "SVHN"
        ],
        'epochs': [
            100
        ],
        'dataset_size':[
            30
        ],
        'attack_type':[
            "DeepFool"
        ],
        'num_iter':[
            50
        ]
    },
    binary=f"{rootpath}/tda/experiments/thomas/embedding_separability_binary_gram.py",
    max_nb_processes=1
)
