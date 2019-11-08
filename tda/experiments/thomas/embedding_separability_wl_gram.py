import json
from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType
from tda.embeddings.weisfeiler_lehman import NodeLabels
from tda.models.architectures import mnist_mlp, mnist_lenet

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'embedding_type': [
            EmbeddingType.WeisfeilerLehman
            ],
        'threshold': [
            '_'.join([str(20000) for _ in range(mnist_mlp.get_nb_graph_layers())])
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
        'epochs': [
            20
        ]
    },
    binary=f"{rootpath}/tda/experiments/thomas/embedding_separability_binary_gram.py",
    max_nb_processes=1
)
