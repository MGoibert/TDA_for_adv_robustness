from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import mnist_mlp

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
            mnist_mlp.name
        ],
        'threshold_strategy': [
            ThresholdStrategy.ActivationValue
        ],
        'epochs': [
            50
        ],
        'dataset': [
            "MNIST"
        ],
        'dataset_size': [
            100
        ],
        'attack_type': [
            "FGSM", "BIM", "CW", "DeepFool"
        ],
        'thresholds': [
            '0.1_0.1_0'
        ],
        'noise': [
            0.0
        ],
        'n_jobs': [
            1
        ]
    },
    binary=f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py",
    max_nb_processes=1
)
