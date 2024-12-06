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
        "embedding_type": [EmbeddingType.PersistentDiagram],
        "dataset_size": [500],
        "attack_type": ["FGSM"],
        "noise": [0.0],
        "n_jobs": [8],
        "all_epsilons": ["0.01"],
        "raw_graph_pca": [-1],
        "architecture": [mnist_lenet.name],
        "epochs": [50],
        "thresholds": [f"0:{str(x)}_2:{str(x)}_4:{str(x)}_5:{str(x)}" for x in [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3]],
        "sigmoidize": [True],
        "kernel_type": [KernelType.SlicedWasserstein],
        "threshold_strategy": [
            ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
            ThresholdStrategy.UnderoptimizedMagnitudeIncreaseComplement,
            ThresholdStrategy.UnderoptimizedRandom,
            ThresholdStrategy.UnderoptimizedRandom,
            ThresholdStrategy.UnderoptimizedRandom,
            ThresholdStrategy.UnderoptimizedRandom,
            ThresholdStrategy.UnderoptimizedRandom,
        ],
        "thresholds_are_low_pass": [True],
    }
)

binary = f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py"

all_experiments = list()

for config in base_configs:
    all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=4, db_path=db_path
)
