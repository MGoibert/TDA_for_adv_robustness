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
            "0:0.35_2:0.35_4:0.35_5:0_6:0"
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
        ],
        'sigmoidize': [
            False
        ]
    },
    binary=f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py",
    max_nb_processes=1
)
