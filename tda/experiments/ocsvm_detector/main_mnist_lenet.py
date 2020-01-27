from r3d3 import R3D3Experiment

from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import mnist_mlp
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
            mnist_mlp.name
        ],
        'epochs': [
            50
        ],
        'dataset': [
            "MNIST"
        ],
        'attack_type': [
            "FGSM", "BIM", "CW", "DeepFool"
        ],
        'threshold': [
            '0;1;0.05_-1;0;0.2_2;1;0.2_2;3;0.05_3;4;0_5;6;0'
        ],
        'noise': [
            0.0
        ],
        'identical_train_samples': [
            1
        ]
    },
    binary=f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py",
    max_nb_processes=1
)
