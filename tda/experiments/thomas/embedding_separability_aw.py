import json
from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType
from tda.embeddings.weisfeiler_lehman import NodeLabels

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'embedding_type': [
            EmbeddingType.AnonymousWalk
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
            NodeLabels.NONE
        ],
        'steps': [
            1, 2, 3
        ]

    },
    binary=f"{rootpath}/tda/experiments/thomas/embedding_separability_binary.py",
    max_nb_processes=1
)