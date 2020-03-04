import json
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
        'dataset': [
          "SVHN"
        ],
        'architecture': [
          svhn_lenet.name
        ],
        'attack_type': [
          "FGSM"#, "BIM", "DeepFool", "CW"
        ],
        'threshold': [
            #'0.5_0_0.25_0_0.1_0_0',
            #'0.3_inf_0.25_inf_inf_inf_inf'
            #"0.3_0_0.25_0_0.1_0_0"
            "0.3_0_0.25_inf_inf_inf_0"
        ],
        'noise': [
            0.0
        ],
        'epochs': [
            200
        ],
        'identical_train_samples': [
            1
        ],
        'dataset_size': [
            4
        ],
        'num_iter':[
            20
        ]
    },
    binary=f"{rootpath}/tda/experiments/other_detector/supervised_detector_binary.py",
    max_nb_processes=1
)
