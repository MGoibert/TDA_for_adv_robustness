from r3d3 import R3D3ExperimentPlan

from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import cifar_lenet
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
            cifar_lenet.name
        ],
        'epochs': [
            300
        ],
        'dataset': [
            "CIFAR10"
        ],
        'dataset_size': [
            100
        ],
        'attack_type': [
            "FGSM", "BIM", "CW", "DeepFool"
        ],
        'thresholds': [
            #'0.5_0.03_0.5_0.03_inf_inf_inf'
            '0.1_0.1_0.2_0.2_0.2'
        ],
        'threshold_strategy': [
            ThresholdStrategy.ActivationValue
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
