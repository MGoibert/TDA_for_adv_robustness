from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import mnist_lenet

experiment = R3D3Experiment(
    db_path=db_path,
    configs={
        'embedding_type': [
            EmbeddingType.RawGraph
        ],
        'kernel_type': [
            KernelType.RBF
        ],
        'architecture': [
            mnist_lenet.name
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
            "FGSM", "BIM", "CW", "DeepFool"
        ],
        'threshold': [
            '0.2_0.2_0.2'
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