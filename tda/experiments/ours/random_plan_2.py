from copy import deepcopy

from r3d3.experiment import R3D3ExperimentPlan, R3D3Experiment
from r3d3.utils import cartesian_product

from tda.dataset.adversarial_generation import AttackType, AttackBackend
from tda.embeddings import EmbeddingType, ThresholdStrategy
from tda.models import mnist_lenet
from tda.models.architectures import cifar_resnet_1
from tda.rootpath import rootpath, db_path

base_configs = cartesian_product(
    {
        "embedding_type": [EmbeddingType.PersistentDiagram],
        "dataset_size": [500],
        "dataset": ["MNIST"],
        "attack_type": [AttackType.PGD],
        "attack_backend": [AttackBackend.FOOLBOX],
        "noise": [0.0],
        "n_jobs": [1],
        "threshold_strategy": [ThresholdStrategy.UnderoptimizedMagnitudeIncrease]+ 10 * [ThresholdStrategy.UnderoptimizedRandom],
        "all_epsilons": ["0.01"],
        "raw_graph_pca": [-1],
        "sigmoidize": [False],
    }
)

binary = f"{rootpath}/tda/experiments/ours/our_binary.py"

all_experiments = list()

for config in base_configs:

    config = deepcopy(config)
    l, u = (0.0, 0.025)

    if config["dataset"] == "CIFAR10":
        config["architecture"] = cifar_resnet_1.name
        config["epochs"] = 100
        config["thresholds"] = f"39:{l}:{u}_40:{l}:{u}_41:{l}:{u}_42:{l}:{u}_43:{l}:{u}"
    elif config["dataset"] == "MNIST":
        config["architecture"] = mnist_lenet.name
        config["epochs"] = 50
        config["thresholds"] = f"0:{l}:{u}_2:{l}:{u}_4:{l}:{u}_6:{l}:{u}"

    if not AttackType.require_epsilon(config["attack_type"]):
        config["all_epsilons"] = "1.0"

    all_experiments.append(R3D3Experiment(binary=binary, config=config))

experiment_plan = R3D3ExperimentPlan(
    experiments=all_experiments, max_nb_processes=24, db_path=db_path
)
