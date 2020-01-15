#!/usr/bin/env python
# coding: utf-8

import argparse
import time
import typing
from joblib import Parallel, delayed

import numpy as np
from r3d3.experiment_db import ExperimentDB
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

from tda.embeddings import get_embedding, EmbeddingType, \
    get_gram_matrix, KernelType
from tda.embeddings.weisfeiler_lehman import NodeLabels
from tda.graph_dataset import get_graph_dataset
from tda.models import get_deep_model, Dataset
from tda.models.architectures import mnist_mlp, get_architecture, Architecture
from tda.rootpath import db_path
from tda.thresholds import process_thresholds
from tda.logging import get_logger

logger = get_logger("Detector")
start_time = time.time()

my_db = ExperimentDB(db_path=db_path)


class Config(typing.NamedTuple):
    # Type of embedding to use
    embedding_type: str
    # Type of kernel to use on the embeddings
    kernel_type: str
    # High threshold for the edges of the activation graph
    thresholds: str
    # Parameters used only for Weisfeiler-Lehman embedding
    height: int
    hash_size: int
    node_labels: str
    steps: int
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
    # Should we ignore unsuccessful attacks or not
    successful_adv: int
    # Type of attack (FGSM, BIM, CW)
    attack_type: str
    # Parameter used by DeepFool and CW
    num_iter: int
    # Should we use the same images clean vs attack when training the detector
    identical_train_samples: int
    # Default parameters when running interactively for instance
    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0
    # Number of jobs to spawn
    n_jobs: int=1

def get_config() -> Config:
    parser = argparse.ArgumentParser(
        description='Transform a dataset in pail files to tf records.')
    parser.add_argument('--experiment_id', type=int, default=-1)
    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--embedding_type', type=str, default=EmbeddingType.PersistentDiagram)
    parser.add_argument('--kernel_type', type=str, default=KernelType.SlicedWasserstein)
    parser.add_argument('--thresholds', type=str, default='0')
    parser.add_argument('--height', type=int, default=1)
    parser.add_argument('--hash_size', type=int, default=100)
    parser.add_argument('--node_labels', type=str, default=NodeLabels.NONE)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dataset', type=str, default="MNIST")
    parser.add_argument('--architecture', type=str, default=mnist_mlp.name)
    parser.add_argument('--train_noise', type=float, default=0.0)
    parser.add_argument('--dataset_size', type=int, default=100)
    parser.add_argument('--successful_adv', type=int, default=1)
    parser.add_argument('--attack_type', type=str, default="FGSM")
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--identical_train_samples', type=int, default=1)

    args, _ = parser.parse_known_args()

    return Config(**args.__dict__)


def get_embeddings(
        config: Config,
        architecture: Architecture,
        noise: float,
        thresholds: typing.List[float],
        epsilon: float,
        dataset: Dataset,
        stats: typing.Dict,
        stats_inf: typing.Dict,
        start: int = 0
) -> typing.List:
    """
    Compute the embeddings used for the detection
    """

    logger.info(f"Adversarial test dataset for espilon = {epsilon} !!")

    my_embeddings = list()
    for line in get_graph_dataset(
            epsilon=epsilon,
            noise=noise,
            adv=epsilon > 0.0,
            architecture=architecture,
            dataset=dataset,
            dataset_size=config.dataset_size,
            thresholds=thresholds,
            only_successful_adversaries=config.successful_adv > 0,
            attack_type=config.attack_type,
            num_iter=config.num_iter,
            start=start):
        stats[epsilon].append(line.l2_norm)
        stats_inf[epsilon].append(line.linf_norm)
        my_embeddings.append(get_embedding(
            embedding_type=config.embedding_type,
            graph=line.graph,
            params={
                "hash_size": int(config.hash_size),
                "height": int(config.height),
                "node_labels": config.node_labels,
                "steps": config.steps
            }
        ))
    logger.info(
        f"Computed embeddings for (attack = {config.attack_type}, "
        f"eps={epsilon}, noise={config.noise}), "
        f"number of sample = {len(my_embeddings)}")
    logger.debug(
        f"Stats for diff btw clean and adv: "
        f"{np.quantile(stats[epsilon], 0.1), np.quantile(stats[epsilon], 0.25), np.median(stats[epsilon]), np.quantile(stats[epsilon], 0.75), np.quantile(stats[epsilon], 0.9)}")

    return my_embeddings


def get_all_embeddings(config: Config, epsilons: typing.List[float]=None):
    architecture = get_architecture(config.architecture)
    dataset = Dataset.get_or_create(name=config.dataset)

    architecture = get_deep_model(
        num_epochs=config.epochs,
        dataset=dataset,
        architecture=architecture,
        train_noise=config.train_noise
    )

    thresholds = process_thresholds(
        raw_thresholds=config.thresholds,
        dataset=dataset,
        architecture=architecture,
        dataset_size=5
    )

    if epsilons is None:
        if config.attack_type in ["FGSM", "BIM"]:
            epsilons = list([0.0, 0.025, 0.05, 0.1, 0.4])
        else:
            epsilons = [0.0, 1.0]

    start = 0

    stats = {
        epsilon: list()
        for epsilon in epsilons
    }
    stats_inf = {
        epsilon: list()
        for epsilon in epsilons
    }

    # Clean train
    logger.info(f"Clean train dataset !!")
    clean_embeddings_train = get_embeddings(
        config=config, dataset=dataset, noise=0.0,
        architecture=architecture, thresholds=thresholds,
        epsilon=0.0, start=0,
        stats=stats, stats_inf=stats_inf)
    if config.identical_train_samples < 0.5:
        start += config.dataset_size

    # Noisy train
    if config.noise > 0.0:
        logger.info(f"Noisy train dataset !!")
        noisy_embeddings_train = get_embeddings(
            config=config, dataset=dataset, noise=config.noise,
            architecture=architecture, thresholds=thresholds,
            epsilon=0.0, start=start,
            stats=stats, stats_inf=stats_inf)
    else:
        noisy_embeddings_train = list()
    start += config.dataset_size

    # Clean test
    logger.info(f"Clean test dataset !!")
    clean_embeddings_test = get_embeddings(
        config=config, dataset=dataset, noise=0.0,
        architecture=architecture, thresholds=thresholds,
        epsilon=0.0, start=start,
        stats=stats, stats_inf=stats_inf)
    start += config.dataset_size

    # Noisy test
    if config.noise > 0.0:
        logger.info(f"Noisy test dataset !!")
        noisy_embeddings_test = get_embeddings(
            config=config, dataset=dataset, noise=config.noise,
            architecture=architecture, thresholds=thresholds,
            epsilon=0.0, start=start,
            stats=stats, stats_inf=stats_inf)
    else:
        noisy_embeddings_test = list()
    start += config.dataset_size

    logger.info("Computing embeddings for epsilons %s using n_jobs=%i" % (
        epsilons, config.n_jobs))
    adv_embeddings = {}
    for epsilon in epsilons:
        adv_embeddings[epsilon] = get_embeddings(
        config=config, dataset=dataset, noise=0.0,
        architecture=architecture, thresholds=thresholds,
        epsilon=epsilon, start=start,
        stats=stats, stats_inf=stats_inf)
    # artifacts = Parallel(n_jobs=config.n_jobs)(delayed(get_embeddings)(
    #     config=config, dataset=dataset, noise=0.0,
    #     architecture=architecture, thresholds=thresholds,
    #     epsilon=epsilon, start=start,
    #     stats=stats, stats_inf=stats_inf) for epsilon in epsilons)
    # adv_embeddings = dict(zip(epsilons, artifacts))

    embedding_train = clean_embeddings_train + noisy_embeddings_train
    embedding_test = clean_embeddings_test + noisy_embeddings_test

    return embedding_train, embedding_test, adv_embeddings, thresholds, stats, stats_inf


def evaluate_embeddings(
        gram_train_matrices: typing.Dict,
        embeddings_train: typing.List,
        embeddings_test: typing.List,
        adv_embeddings: typing.List,
        param_space: typing.List,
        kernel_type: str
) -> float:
    """
    Compute the AUC for a given epsilon and returns also the scores
    of the best OneClass SVM
    """

    # embeddings = clean_embeddings + adv_embeddings[epsilon]
    best_auc = 0.0

    for i, param in enumerate(param_space):
        ocs = OneClassSVM(
            tol=1e-5,
            kernel="precomputed")

        # Training model
        start_time = time.time()
        logger.info(f"sum gram matrix train = {gram_train_matrices[i].sum()}")
        ocs.fit(gram_train_matrices[i])
        logger.info(f"Trained model in {time.time() - start_time} secs")

        # Testing model
        start_time = time.time()
        gram_test_and_bad = get_gram_matrix(
            kernel_type=kernel_type,
            embeddings_in=embeddings_test + adv_embeddings,
            embeddings_out=embeddings_train,
            params=param
        )
        logger.info(f"Computed Gram Test Matrix in {time.time() - start_time} secs")

        predictions = ocs.score_samples(gram_test_and_bad)

        labels = np.concatenate(
            (
                np.ones(len(embeddings_test)),
                np.zeros(len(adv_embeddings))
            )
        )

        roc_auc_val = roc_auc_score(y_true=labels, y_score=predictions)
        logger.info(f"AUC score for param = {param} : {roc_auc_val}")

        if roc_auc_val > best_auc:
            best_auc = roc_auc_val

    return best_auc


def evaluate_all_embeddings(
        gram_train_matrices: typing.Dict,
        embeddings_train: typing.List,
        embeddings_test: typing.List,
        adv_embeddings: typing.Dict[float, typing.List[float]],
        param_space: typing.List,
        config: Config) -> typing.Dict:
    """
    Evaluate embeddings all embeddings for all values of epsilon
    """
    epsilons = list(adv_embeddings.keys())
    logger.info("Evaluating embeddings for epsilons %s using n_jobs=%i" % (
        epsilons, config.n_jobs))
    artifacts = Parallel(n_jobs=config.n_jobs)(delayed(evaluate_embeddings)(
        gram_train_matrices=gram_train_matrices,
        embeddings_train=embeddings_train,
        embeddings_test=embeddings_test,
        adv_embeddings=adv_embeddings[epsilon],
        param_space=param_space,
        kernel_type=config.kernel_type) for epsilon in epsilons)
    assert len(epsilons) == len(artifacts)
    return dict(zip(epsilons, artifacts))


def run_experiment(config: Config):
    """
    Main entry point to run the experiment
    """

    logger.info(f"Starting experiment {config.experiment_id}_{config.run_id} !!")

    if __name__ != "__main__":
        my_db.add_experiment(
            experiment_id=config.experiment_id,
            run_id=config.run_id,
            config=config._asdict()
        )

    embedding_train, embedding_test, adv_embeddings, thresholds, stats, stats_inf = get_all_embeddings(config)

    if config.kernel_type == KernelType.RBF:
        param_space = [
            {'gamma': gamma}
            for gamma in np.logspace(-6, -3, 10)
        ]
    elif config.kernel_type == KernelType.SlicedWasserstein:
        param_space = [
            {'M': 20, 'sigma': 5 * 10 ** (-1)},
        ]
    else:
        raise NotImplementedError(f"Unknown kernel {config.kernel_type}")

    gram_train_matrices = {i: get_gram_matrix(
        kernel_type=config.kernel_type,
        embeddings_in=embedding_train,
        embeddings_out=None,
        params=param
    )
        for i, param in enumerate(param_space)
    }
    logger.info(f"Computed all Gram train matrices !")

    all_results = {
        epsilon: evaluate_embeddings(
            gram_train_matrices=gram_train_matrices,
            embeddings_train=embedding_train,
            embeddings_test=embedding_test,
            adv_embeddings=adv_embeddings[epsilon],
            param_space=param_space,
            kernel_type=config.kernel_type
        )
        for epsilon in adv_embeddings
    }

    logger.info(all_results)

    end_time = time.time()

    my_db.update_experiment(
        experiment_id=config.experiment_id,
        run_id=config.run_id,
        metrics={
            "separability_values": all_results,
            "effective_thresholds": {
                "_".join([str(v) for v in key]): thresholds[key]
                for key in thresholds
            },
            "running_time": end_time - start_time,
            "l2_diff": stats,
            "linf_diff": stats_inf
        }
    )

    logger.info(f"Done with experiment {config.experiment_id}_{config.run_id} !!")


if __name__ == "__main__":
    config = get_config()
    run_experiment(config)
