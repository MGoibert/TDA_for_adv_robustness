#!/usr/bin/env python
# coding: utf-8

import argparse
import time
import typing
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from r3d3 import ExperimentDB

from tda.graph_dataset import process_sample
from tda.models import Dataset, get_deep_model
from tda.models.architectures import Architecture
from tda.models.architectures import get_architecture, svhn_lenet
from tda.logging import get_logger
from tda.rootpath import db_path, rootpath

start_time = time.time()
directory = f"{rootpath}/plots/"
pathlib.Path(directory).mkdir(exist_ok=True)

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
    # Type of attack (FGSM, BIM, CW)
    attack_type: str
    # Parameter used by DeepFool and CW
    num_iter: int
    # Default parameters when running interactively for instance
    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0


def get_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', type=int, default=-1)
    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="SVHN")
    parser.add_argument('--architecture', type=str, default=svhn_lenet.name)
    parser.add_argument('--train_noise', type=float, default=0.0)
    parser.add_argument('--dataset_size', type=int, default=50)
    parser.add_argument('--attack_type', type=str, default="FGSM")
    parser.add_argument('--num_iter', type=int, default=10)

    args, _ = parser.parse_known_args()

    return Config(**args.__dict__)


def compute_adv_accuracy(
        epsilon: float,
        noise: float,
        dataset: Dataset,
        architecture: Architecture,
        dataset_size: int = 100,
        attack_type: str = "FGSM",
        num_iter: int = 50
) -> float:
    # Else we have to compute the dataset first

    assert architecture.is_trained

    logger.info(f"I am going to generate a dataset of {dataset_size} points...")

    nb_samples = 0
    corr = 0

    while nb_samples < dataset_size:
        sample = dataset.test_and_val_dataset[nb_samples]

        x, y = process_sample(
            sample=sample,
            adversarial=epsilon > 0,
            noise=noise,
            epsilon=epsilon,
            model=architecture,
            num_classes=10,
            attack_type=attack_type,
            num_iter=num_iter
        )

        y_pred = architecture(x).argmax(dim=-1).item()

        if y == y_pred:
            corr += 1

        nb_samples += 1

    return corr / dataset_size


def get_all_accuracies(config: Config):

    if __name__ != "__main__":
        my_db.add_experiment(
            experiment_id=config.experiment_id,
            run_id=config.run_id,
            config=config._asdict()
        )

    if config.attack_type in ["FGSM", "BIM"]:
        all_epsilons = list(sorted(np.linspace(0.0, 0.4, num=11)))
    else:
        all_epsilons = [0.0, 1]

    dataset = Dataset(name=config.dataset)

    architecture = get_deep_model(
        num_epochs=config.epochs,
        dataset=dataset,
        architecture=get_architecture(config.architecture),
        train_noise=config.train_noise
    )

    accuracies = dict()

    for epsilon in all_epsilons:

        adversarial_acc = compute_adv_accuracy(
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

    return accuracies


def plot_and_save(config, accuracies):
    logger.info(accuracies)
    file_name = directory + str(config.dataset) + "_" + str(config.architecture) + str(config.epochs) + "_" + str(
        config.attack_type) + ".png"
    logger.info(f"file name = {file_name}")
    plt.style.use('ggplot')
    plt.plot(list(accuracies.keys()), list(accuracies.values()), "o-", linewidth=1.5)
    plt.title("Standard and adversarial accuracies")
    plt.xlabel("Perturbation value")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig(file_name, dpi=800)
    plt.close()

    my_db.update_experiment(
        experiment_id=config.experiment_id,
        run_id=config.run_id,
        metrics={
            "accuracies": accuracies
        }
    )

    end_time = time.time()

    logger.info(f"Success in {end_time - start_time} seconds")


if __name__ == "__main__":
    config = get_config()
    accuracies = get_all_accuracies(config)
    plot_and_save(config, accuracies)


