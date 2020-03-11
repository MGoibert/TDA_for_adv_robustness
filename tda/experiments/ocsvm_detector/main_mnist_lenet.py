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
            ThresholdStrategy.ActivationValue
        ],
        'epochs': [
            50
        ],
        'dataset': [
            "MNIST"
        ],
        'dataset_size': [
            500
        ],
        'attack_type': [
            "FGSM"  #  , "BIM", "DeepFool", "CW"
        ],
        'thresholds': [
            '0;1;0_-1;0;0.1_1;2;0.1_2;3;0.025_3;4;0_5;6;0'
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
