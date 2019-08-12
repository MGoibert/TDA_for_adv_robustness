import json
from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'embedding_type': [
            EmbeddingType.WeisfeilerLehman
            ],
        'threshold': [
            5000,
            10000,
            25000,
            50000
        ],
        'height': [
            5
        ],
        'hash_size': [
            50
        ]
    },
    binary=f"{rootpath}/tda/experiments/thomas/embedding_separability_binary.py",
    max_nb_processes=1
)
