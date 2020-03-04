from r3d3 import R3D3Experiment

from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import fashion_mnist_mlp
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
        'architecture': [
            fashion_mnist_mlp.name
        ],
        'threshold_strategy': [
            ThresholdStrategy.ActivationValue,
            ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
            ThresholdStrategy.UnderoptimizedMagnitudeIncreaseV2
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
            "FGSM", "DeepFool"
        ],
        'thresholds': [
            '0.1_0.0_0.0_0.0_0.0_0.0'
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
