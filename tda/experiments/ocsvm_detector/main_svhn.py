from r3d3 import R3D3Experiment, R3D3ExperimentPlan
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import svhn_lenet

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
            svhn_lenet.name
        ],
        'epochs': [
            250
        ],
        'dataset': [
            "SVHN"
        ],
        'dataset_size': [
            100
        ],
        'attack_type': [
            "FGSM" #, "BIM", "CW", "DeepFool"
        ],
        'thresholds': [
            "0:0.5_2:0.5_4:0.5_5:0_6:0"
        ],
        'threshold_strategy': [
            ThresholdStrategy.UnderoptimizedMagnitudeIncrease
        ],
        'noise': [
            0.0
        ],
        'n_jobs': [
            5
        ],
        'all_epsilons': [
            "0.01;0.1"
        ],
        'sigmoidize': [
            True
        ]
    },
    binary=f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py",
    max_nb_processes=1
)
