from r3d3 import R3D3Experiment

from tda.embeddings import EmbeddingType, KernelType
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
        'threshold': [
            15000, 20000
        ],
        'noise': [
            0.0, 0.02
        ]
    },
    binary=f"{rootpath}/tda/experiments/thomas/embedding_separability_binary_gram.py",
    max_nb_processes=1
)
