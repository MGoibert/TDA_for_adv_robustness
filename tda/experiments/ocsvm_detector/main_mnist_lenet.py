from r3d3 import R3D3ExperimentPlan

from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import mnist_lenet
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
            mnist_lenet.name
        ],
        'threshold_strategy': [
            ThresholdStrategy.UnderoptimizedMagnitudeIncrease
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
            "FGSM"  #  , "BIM", "DeepFool", "CW"
        ],
        'thresholds': [
            "0:0.05_2:0.05_4:0.05_5:0.0"
        ],
        'noise': [
            0.0
        ],
        'n_jobs': [
            24
        ],
        'num_iter': [
            50
        ],
        'all_epsilons': [
            "0.01;0.1"
        ],
        'sigmoidize': [
            False
        ]
    },
    binary=f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py",
    max_nb_processes=1
)
