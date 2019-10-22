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
            # "25000_1_1125000_1_2000000_5700000_25400000"
            "70000_1_100000_1_25000_70000_230000"
        ],
        'noise': [
            0.0
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
        ]
    },
    binary=f"{rootpath}/tda/experiments/thomas/embedding_separability_binary_gram.py",
    max_nb_processes=1
)
