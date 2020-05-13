from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product

from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import (
    mnist_mlp,
    mnist_lenet,
    fashion_mnist_lenet,
    fashion_mnist_mlp,
    svhn_lenet,
    cifar_lenet,
)
from tda.rootpath import rootpath, db_path
from copy import deepcopy


base_configs = cartesian_product(
    {
        "embedding_type": [EmbeddingType.PersistentDiagram, EmbeddingType.RawGraph],
        "dataset_size": [500],
        "attack_type": ["FGSM"],
        "noise": [0.0],
        "n_jobs": [8],
        "all_epsilons": ["0.01;0.1;0.4"],
        "raw_graph_pca": [-1],
        "architecture": [mnist_lenet.name],
        "epochs": [50],
        "thresholds": ["0:0.025_2:0.025_4:0.025_5:0.025"],
        "sigmoidize": [True],
        "threshold_strategy": [
            ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
            ThresholdStrategy.UnderoptimizedLargeFinal,
            ThresholdStrategy.NoThreshold
            # ThresholdStrategy.ActivationValue
        ],
        "thresholds_are_low_pass": [
            True, False
        ]
    }
)

binary = f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py"

all_experiments = list()

for config in base_configs:

    if config["embedding_type"] == EmbeddingType.PersistentDiagram:
        config["kernel_type"] = KernelType.SlicedWasserstein
    elif config["embedding_type"] == EmbeddingType.RawGraph:
        config["kernel_type"] = KernelType.RBF

    all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=4, db_path=db_path
)
