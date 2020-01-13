import json
from r3d3 import R3D3Experiment
from tda.rootpath import rootpath, db_path
from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import mnist_lenet

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
          mnist_lenet.name
        ],
        'dataset': [
          "MNIST"
        ],
        'attack_type': [
          "FGSM"
        ],
        'threshold': [
            '_'.join([str(0.99) for _ in range(mnist_lenet.get_nb_graph_layers())])
        ],
        'noise': [
            0.0
        ],
        'epochs': [
            50
        ],
        'identical_train_samples': [
            0, 1
        ]
    },
    binary=f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py",
    max_nb_processes=1
)
