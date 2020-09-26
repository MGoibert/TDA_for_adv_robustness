from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product

from tda.dataset.adversarial_generation import AttackType, AttackBackend
from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import cifar_resnet_1, cifar_toy_resnet
from tda.rootpath import rootpath, db_path
from copy import deepcopy


base_configs = cartesian_product(
    {
        "embedding_type": [EmbeddingType.RawGraph],
        "dataset_size": [500],
        "attack_type": [AttackType.PGD, AttackType.CW],
        "attack_backend": [AttackBackend.FOOLBOX],
        "noise": [0.0],
        "n_jobs": [8],
        "all_epsilons": ["0.01;0.1;0.4"],
        "raw_graph_pca": [-1],
    }
)

binary = f"{rootpath}/tda/experiments/ours/our_binary.py"

all_experiments = list()

for (
    model,
    dataset,
    nb_epochs,
    best_threshold,
    threshold_strategy,
    sigmoidize_rawgraph,
) in [
    # [
    #    cifar_toy_resnet.name,
    #    "CIFAR10",
    #    300,
    #    "0.05",
    #    ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
    #    True
    # ],
    [
        cifar_resnet_1.name,
        "CIFAR10",
        100,
        "1:0.05_34:0.05_35:0.05_0:0.05",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        True,
    ],
    [
        cifar_resnet_1.name,
        "CIFAR10",
        100,
        "1:0.05_16:0.05_17:0.05_34:0.05_35:0.05_0:0.05",
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        True,
    ],
]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs
        config["thresholds"] = best_threshold
        config["threshold_strategy"] = threshold_strategy
        config["sigmoidize"] = sigmoidize_rawgraph

        if config["embedding_type"] == EmbeddingType.PersistentDiagram:
            config["kernel_type"] = KernelType.SlicedWasserstein
            config["sigmoidize"] = True
        elif config["embedding_type"] == EmbeddingType.RawGraph:
            config["kernel_type"] = KernelType.RBF

        if config["attack_type"] not in ["FGSM", "PGD"]:
            config["all_epsilons"] = "1.0"

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=1, db_path=db_path
)
