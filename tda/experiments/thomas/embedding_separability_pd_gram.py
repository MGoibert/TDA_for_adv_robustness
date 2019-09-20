import json
from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType, KernelType
from tda.embeddings.weisfeiler_lehman import NodeLabels

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
            15000
        ],
        'noise': [
            0, 0.02
        ]
    },
    binary=f"{rootpath}/tda/experiments/thomas/embedding_separability_binary_gram.py",
    max_nb_processes=1
)