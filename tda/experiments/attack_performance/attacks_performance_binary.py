#!/usr/bin/env python
# coding: utf-8

import argparse
import time
import typing
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
from r3d3 import ExperimentDB

from tda.dataset.graph_dataset import get_sample_dataset, AttackBackend
from tda.models import Dataset, get_deep_model
from tda.models.architectures import Architecture
from tda.models.architectures import get_architecture, svhn_lenet
from tda.tda_logging import get_logger
from tda.rootpath import db_path, rootpath

start_time = time.time()

my_db = ExperimentDB(db_path=db_path)

logger = get_logger("GraphStats")


class Config(typing.NamedTuple):
    # Noise to consider for the noisy samples
    noise: float
    # Number of epochs for the model
    epochs: int
    # Dataset we consider (MNIST, SVHN)
    dataset: str
    # Name of the architecture
    architecture: str
    # Noise to be added during the training of the model
    train_noise: float
    # Size of the dataset used for the experiment
    dataset_size: int
    # Type of attack (FGSM, PGD, CW)
    attack_type: str
    # Backend for the attack
    attack_backend: str
    # Parameter used by DeepFool and CW
    num_iter: int
    # Pruning
    first_pruned_iter: int = 10
    prune_percentile: float = 0.0
    tot_prune_percentile: float = 0.0
    # Default parameters when running interactively for instance
    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0

    @property
    def result_path(self):
        directory = f"{rootpath}/results/{self.experiment_id}/{self.run_id}/"
        pathlib.Path(directory).mkdir(exist_ok=True, parents=True)
        return directory


def get_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", type=int, default=-1)
    parser.add_argument("--run_id", type=int, default=-1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="SVHN")
    parser.add_argument("--architecture", type=str, default=svhn_lenet.name)
    parser.add_argument("--train_noise", type=float, default=0.0)
    parser.add_argument("--dataset_size", type=int, default=50)
    parser.add_argument("--attack_type", type=str, default="FGSM")
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--first_pruned_iter", type=int, default=10)
    parser.add_argument("--prune_percentile", type=float, default=0.0)
    parser.add_argument("--tot_prune_percentile", type=float, default=0.0)
    parser.add_argument("--attack_backend", type=str, default=AttackBackend.ART)

    args, _ = parser.parse_known_args()

    return Config(**args.__dict__)


def compute_adv_accuracy(
    epsilon: float,
    noise: float,
    dataset: Dataset,
    architecture: Architecture,
    dataset_size: int = 100,
    attack_type: str = "FGSM",
    num_iter: int = 50,
) -> (float, typing.List):
    # Else we have to compute the dataset first

    dataset = get_sample_dataset(
        epsilon=epsilon,
        noise=noise,
        adv=epsilon > 0,
        dataset=dataset,
        train=False,
        succ_adv=False,
        archi=architecture,
        dataset_size=dataset_size,
        attack_type=attack_type,
        num_iter=num_iter,
        compute_graph=False,
        attack_backend=config.attack_backend,
    )

    # Since we set succ_adv to False, we should have
    # exactly dataset_size points
    assert len(dataset) == dataset_size

    corr = sum([1 for line in dataset if line.y == line.y_pred])

    some_images = [line.x for line in dataset if line.y != line.y_pred][:8]

    return corr / dataset_size, some_images


def get_all_accuracies(config: Config):

    if __name__ != "__main__":
        my_db.add_experiment(
            experiment_id=config.experiment_id,
            run_id=config.run_id,
            config=config._asdict(),
        )

    if config.attack_type in ["FGSM", "PGD"]:
        all_epsilons = list(sorted(np.linspace(0.0, 0.4, num=11)))
    else:
        all_epsilons = [0.0, 1]

    dataset = Dataset(name=config.dataset)

    architecture = get_deep_model(
        num_epochs=config.epochs,
        dataset=dataset,
        architecture=get_architecture(config.architecture),
        train_noise=config.train_noise,
        prune_percentile=config.prune_percentile,
        tot_prune_percentile=config.tot_prune_percentile,
        first_pruned_iter=config.first_pruned_iter,
    )

    accuracies = dict()

    for epsilon in all_epsilons:

        adversarial_acc, some_images = compute_adv_accuracy(
            epsilon=epsilon,
            noise=config.noise,
            dataset=dataset,
            architecture=architecture,
            dataset_size=config.dataset_size,
            attack_type=config.attack_type,
            num_iter=config.num_iter,
        )

        logger.info(f"Epsilon={epsilon}: acc={adversarial_acc}")
        accuracies[epsilon] = adversarial_acc

        with open(config.result_path + f"/images_eps_{epsilon}.pickle", "wb") as fw:
            pickle.dump(some_images, fw)

    return accuracies


def plot_and_save(config, accuracies):
    logger.info(accuracies)
    file_name = (
        config.result_path
        + str(config.dataset)
        + "_"
        + str(config.architecture)
        + str(config.epochs)
        + "_"
        + str(config.attack_type)
        + ".png"
    )
    logger.info(f"file name = {file_name}")
    plt.style.use("ggplot")
    plt.plot(list(accuracies.keys()), list(accuracies.values()), "o-", linewidth=1.5)
    plt.title("Standard and adversarial accuracies")
    plt.xlabel("Perturbation value")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig(file_name, dpi=800)
    plt.close()

    end_time = time.time()
    total_time = end_time - start_time

    my_db.update_experiment(
        experiment_id=config.experiment_id,
        run_id=config.run_id,
        metrics={"accuracies": accuracies, "total_time": total_time},
    )

    logger.info(f"Success in {total_time} seconds")


if __name__ == "__main__":
    config = get_config()
    accuracies = get_all_accuracies(config)
    plot_and_save(config, accuracies)
