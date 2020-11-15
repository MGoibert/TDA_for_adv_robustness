from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product

from tda.dataset.adversarial_generation import AttackType, AttackBackend
from tda.embeddings import EmbeddingType, KernelType, ThresholdStrategy
from tda.models.architectures import mnist_lenet, cifar_resnet_1, cifar_toy_resnet, svhn_resnet_1
from tda.rootpath import rootpath, db_path
from copy import deepcopy


base_configs = cartesian_product(
    {   
        "embedding_type": [EmbeddingType.PersistentDiagram],
        "dataset_size": [500],
        "attack_type": [AttackType.PGD],#, AttackType.CW, AttackType.BOUNDARY],
        "attack_backend": [AttackBackend.FOOLBOX],
        "noise": [0.0],
        "n_jobs": [1],
        "all_epsilons": ["0.01;0.05;0.1"],
        "raw_graph_pca": [-1],
        "prune_percentile": [0.5],
        "tot_prune_percentile": [0.99],
        "first_pruned_iter": [100],
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
    [   
        cifar_resnet_1.name,
        "CIFAR10",
        99,
        '39:0.3_40:0.3_41:0.3_42:0.3_43:0.3',
        ThresholdStrategy.UnderoptimizedMagnitudeIncreaseV3,
        False,
    ],
    #[   
    #    mnist_lenet.name,
    #    "MNIST",
    #    100,
    #    '0:0,025_2:0,025_4:0,025_6:0,025',
    #    ThresholdStrategy.UnderoptimizedMagnitudeIncreaseV3,
    #    False,
    #],
]:
    for config in base_configs:
        config = deepcopy(config)
        config["architecture"] = model
        config["dataset"] = dataset
        config["epochs"] = nb_epochs
        config["thresholds"] = best_threshold
        config["threshold_strategy"] = threshold_strategy
        config["sigmoidize"] = False #sigmoidize_rawgraph

        #if config["embedding_type"] == EmbeddingType.PersistentDiagram:
        #    config["sigmoidize"] = True

        if config["attack_type"] not in ["FGSM", "PGD"]:
            config["all_epsilons"] = "1.0"

        all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
      experiments=all_experiments, max_nb_processes=1, db_path=db_path
)

