from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import mnist_mlp

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'embedding_type': [
            EmbeddingType.PersistentDiagram
        ],
        'kernel_type': [
            KernelType.SlicedWasserstein  # OldVersion
        ],
        'architecture': [
            mnist_mlp.name
        ],
        'epochs': [
            49
        ],
        'dataset': [
            "MNIST"
        ],
        'dataset_size': [
            30
        ],
        'attack_type': [
            "FGSM"  # "FGSM", "BIM", "CW", "DeepFool"
        ],
        'threshold': [
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
