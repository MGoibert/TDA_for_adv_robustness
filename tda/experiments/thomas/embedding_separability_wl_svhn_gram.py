import json
from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType
from tda.embeddings.weisfeiler_lehman import NodeLabels
from tda.models.architectures import svhn_lenet

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'embedding_type': [
            EmbeddingType.WeisfeilerLehman
            ],
        'threshold': [
            20000
        ],
        'height': [
            5
        ],
        'hash_size': [
            50
        ],
        'node_labels': [
            NodeLabels.LAYERS
        ],
        'noise': [
            0.02
        ],
        'architecture': [
            svhn_lenet.name
        ],
        'dataset': [
            "SVHN"
        ],
        'epochs': [
            50
        ]

    },
    binary=f"{rootpath}/tda/experiments/thomas/embedding_separability_binary_gram.py",
    max_nb_processes=1
)
