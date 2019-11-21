from r3d3 import R3D3Experiment

from tda.embeddings import EmbeddingType, KernelType
from tda.models.architectures import mnist_mlp, mnist_lenet
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
        'threshold': [
            #'_'.join([str(18000) for _ in range(mnist_mlp.get_nb_graph_layers())]),
            # '_'.join([str(20000) for _ in range(mnist_mlp.get_nb_graph_layers())])
            "1_1_0"
        ],
        'noise': [
            0.0
        ],
        'dataset':[
            "MNIST"
        ],
        'architecture': [
            mnist_mlp.name,
            #mnist_lenet.name
        ],
        'epochs': [
            50
        ],
        'dataset_size':[
            25
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
