from r3d3 import R3D3Experiment

from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import mnist_mlp, mnist_lenet
from tda.rootpath import rootpath, db_path

"""
The goal of this experiment is to assess how the performance of a model
is correlated with the 
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
        'threshold': [
            '_'.join([str(18000) for _ in range(mnist_lenet.get_nb_graph_layers())])
        ],
        'noise': [
            0.0, 0.02
        ],
        'architecture': [
            mnist_lenet.name
        ],
        'epochs': [
            5, 20, 30, 40, 50
        ],
        'dataset_size': [
            25
        ]
    },
    binary=f"{rootpath}/tda/experiments/thomas/embedding_separability_binary_gram.py",
    max_nb_processes=1
)
