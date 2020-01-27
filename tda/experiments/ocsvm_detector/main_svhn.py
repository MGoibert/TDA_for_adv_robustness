from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import svhn_lenet

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
            svhn_lenet.name
        ],
        'epochs': [
            200
        ],
        'dataset': [
            "MNIST"
        ],
        'attack_type': [
            "FGSM", "BIM", "CW", "DeepFool"
        ],
        'threshold': [
            '0.4_0.03_0.4_0.03_0_0_0'
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
