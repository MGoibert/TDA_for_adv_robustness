from r3d3 import R3D3ExperimentPlan

from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import fashion_mnist_mlp
from tda.rootpath import rootpath, db_path

experiment_plan = R3D3ExperimentPlan.from_cartesian_space(
    db_path=db_path,
    configs={
        'embedding_type': [
            EmbeddingType.PersistentDiagram
        ],
        'kernel_type': [
            KernelType.SlicedWasserstein
        ],
        'architecture': [
            fashion_mnist_mlp.name
        ],
        'epochs': [
            50
        ],
        'dataset': [
            "FashionMNIST"
        ],
        'dataset_size': [
            100
        ],
        'attack_type': [
           "FGSM", "BIM", "DeepFool", "CW"
        ],
        'thresholds': [
            '0.5_0.3_0.3_0.3_0.3'
        ],
        'noise': [
            0.0
        ],
        'n_jobs': [
            24
        ]
    },
    binary=f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py",
    max_nb_processes=1
)
