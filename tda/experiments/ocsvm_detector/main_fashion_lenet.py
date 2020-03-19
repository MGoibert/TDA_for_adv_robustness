from r3d3 import R3D3ExperimentPlan

from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import fashion_mnist_lenet
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
            fashion_mnist_lenet.name
        ],
        'epochs': [
            200
        ],
        'dataset': [
            "FashionMNIST"
        ],
        'dataset_size': [
            100
        ],
        'attack_type': [
            "FGSM"#, "BIM", "DeepFool", "CW"
        ],
        'thresholds': [
            "0:0.05_2:0.05_4:0.0_5:0.0"
        ],
        'threshold_strategy': [
            ThresholdStrategy.UnderoptimizedMagnitudeIncrease
        ],
        'noise': [
            0.0
        ],
        'n_jobs': [
            2
        ],
        'all_epsilons': [
            "0.01;0.1"
        ]
    },
    binary=f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py",
    max_nb_processes=1
)
