from r3d3 import R3D3Experiment

from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import fashion_mnist_lenet
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
            "FGSM", "BIM", "DeepFool", "CW"
        ],
        'thresholds': [
            #'0.1_0.01_0.1_0_0_0'
            '0.5_0.03_0.3_0_0_0'
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
