from r3d3 import R3D3Experiment

from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import svhn_lenet
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
        'thresholds': [
            #"70000_1_100000_1_20000_0_0"
            #"70000_1_220000_1_350000_1050000_4500000"
            #"225000_1_1250000_1_0_0_0"
            #"40000_1_30000_1_8000_0_0"
            "0.5_0_0.25_0_0.1_0_0"
        ],
        'architecture': [
            svhn_lenet.name
        ],
        'noise': [
            0.0
        ],
        'dataset': [
            "SVHN"
        ],
        'epochs': [
            201
        ],
        'dataset_size':[
            10
        ],
        'attack_type':[
            "FGSM"
        ],
        'num_iter':[
            20
        ],
        'train_noise':[
            0.0
        ]
    },
    binary=f"{rootpath}/tda/experiments/thomas/embedding_separability_binary_gram.py",
    max_nb_processes=1
)
