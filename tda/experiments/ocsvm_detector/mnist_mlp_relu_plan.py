from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product

from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import (
    mnist_mlp,
    mnist_mlp_relu
)
from tda.rootpath import rootpath, db_path
from copy import deepcopy

base_configs = cartesian_product(
    {
        "embedding_type": [EmbeddingType.PersistentDiagram],
        "kernel_type": [KernelType.SlicedWasserstein],
        "architecture": [mnist_mlp.name, mnist_mlp_relu.name],
        "dataset_size": [500],
        "attack_type": ["FGSM"],
        "noise": [0.0],
        "n_jobs": [8],
        "all_epsilons": ["0.01;0.1;0.4"],
        "dataset": ["MNIST"],
        "epochs": [50]
    }
)

binary = f"{rootpath}/tda/experiments/ocsvm_detector/ocsvm_detector_binary.py"

all_experiments = list()

for threshold, threshold_strategy, sigmoidize in [
    [
        "0:0.3_1:0.3_2:0.0",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        True
    ],
]:
    for config in base_configs:
        config = deepcopy(config)
        config["thresholds"] = threshold
        config["threshold_strategy"] = threshold_strategy
        config["sigmoidize"] = sigmoidize

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
