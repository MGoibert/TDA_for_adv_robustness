#!/usr/bin/env python
# coding: utf-8

import argparse
import time
import typing

import numpy as np
from joblib import delayed, Parallel
from r3d3.experiment_db import ExperimentDB
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM, SVC

from tda.embeddings import get_embedding, EmbeddingType, \
    get_gram_matrix, KernelType
from tda.embeddings.weisfeiler_lehman import NodeLabels
from tda.logging import get_logger
from tda.models import get_deep_model, Dataset
from tda.models.architectures import mnist_mlp, get_architecture
from tda.protocol import get_protocolar_datasets
from tda.rootpath import db_path
from tda.thresholds import process_thresholds

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
    # Default parameters when running interactively for instance
    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0


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

    args, _ = parser.parse_known_args()

    return Config(**args.__dict__)


def get_all_embeddings(config: Config):
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

    if config.attack_type in ["FGSM", "BIM"]:
        # all_epsilons = list([0.0, 0.025, 0.05, 0.1, 0.4])
        all_epsilons = np.linspace(1e-2, 1.0, 10)
        # all_epsilons = [0.0, 1.0]
    else:
        all_epsilons = [1.0]

    train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
        noise=config.noise,
        dataset=dataset,
        succ_adv=config.successful_adv > 0,
        archi=architecture,
        dataset_size=config.dataset_size,
        attack_type=config.attack_type,
        all_epsilons=all_epsilons
    )

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def embedding_getter(line_chunk):
        ret = list()
        for line in line_chunk:
            ret.append(get_embedding(
                embedding_type=config.embedding_type,
                line=line,
                params={
                    "hash_size": int(config.hash_size),
                    "height": int(config.height),
                    "node_labels": config.node_labels,
                    "steps": config.steps
                },
                architecture=architecture,
                thresholds=thresholds
            ))
        return ret

    nb_jobs = 24

    def process(input_dataset):
        my_chunks = chunks(input_dataset, len(input_dataset) // nb_jobs)
        ret = Parallel(n_jobs=nb_jobs)(delayed(embedding_getter)(chunk) for chunk in my_chunks)
        ret = [item for sublist in ret for item in sublist]
        return ret

    # Clean train
    logger.info(f"Clean train dataset !!")
    clean_embeddings_train = process(train_clean)

    # Clean test
    logger.info(f"Clean test dataset !!")
    clean_embeddings_test = process(test_clean)

    adv_embeddings_train = dict()
    adv_embeddings_test = dict()

    stats = dict()
    stats_inf = dict()

    for epsilon in all_epsilons:
        logger.info(f"Adversarial train dataset for espilon = {epsilon} !!")
        adv_embeddings_train[epsilon] = process(train_adv[epsilon])

        logger.info(f"Adversarial test dataset for espilon = {epsilon} !!")
        adv_embeddings_test[epsilon] = process(test_adv[epsilon])

        stats[epsilon] = [line.l2_norm for line in test_adv[epsilon]]
        stats_inf[epsilon] = [line.linf_norm for line in test_adv[epsilon]]

        logger.debug(
            f"Stats for diff btw clean and adv: "
            f"{np.quantile(stats[epsilon], 0.1), np.quantile(stats[epsilon], 0.25), np.median(stats[epsilon]), np.quantile(stats[epsilon], 0.75), np.quantile(stats[epsilon], 0.9)}")

    return clean_embeddings_train, clean_embeddings_test, adv_embeddings_train, adv_embeddings_test, thresholds, stats, stats_inf


def evaluate_embeddings(
        gram_train_matrices: typing.Dict,
        embeddings_train: typing.List,
        embeddings_test: typing.List,
        adv_embeddings_train: typing.List,
        adv_embeddings_test: typing.List,
        param_space: typing.List,
        kernel_type: str
) -> (float, float):
    """
    Compute the AUC for a given epsilon and returns also the scores
    of the best OneClass SVM
    """

    # embeddings = clean_embeddings + adv_embeddings[epsilon]
    best_auc = 0.0
    best_auc_supervised = 0.0

    for i, param in enumerate(param_space):

        #########################
        # Unsupervised Learning #
        #########################

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
            embeddings_in=embeddings_test + adv_embeddings_test,
            embeddings_out=embeddings_train,
            params=param
        )
        logger.info(f"Computed Gram Test Matrix in {time.time() - start_time} secs")

        predictions = ocs.score_samples(gram_test_and_bad)

        labels = np.concatenate(
            (
                np.ones(len(embeddings_test)),
                np.zeros(len(adv_embeddings_test))
            )
        )

        roc_auc_val = roc_auc_score(y_true=labels, y_score=predictions)
        logger.info(f"AUC score for param = {param} : {roc_auc_val}")

        if roc_auc_val > best_auc:
            best_auc = roc_auc_val

        #######################
        # Supervised Learning #
        #######################

        detector = SVC(
            verbose=0,
            tol=1e-9,
            max_iter=100000,
            kernel='precomputed'
        )

        labels_train = np.concatenate(
            (
                np.ones(len(embeddings_train)),
                np.zeros(len(adv_embeddings_train))
            )
        )

        gram_train = get_gram_matrix(
            kernel_type=kernel_type,
            embeddings_in=embeddings_train + adv_embeddings_train,
            embeddings_out=None,
            params=param
        )

        detector.fit(gram_train, labels_train)

        gram_test = get_gram_matrix(
            kernel_type=kernel_type,
            embeddings_in=embeddings_test + adv_embeddings_test,
            embeddings_out=embeddings_train + adv_embeddings_train,
            params=param
        )

        predictions = detector.decision_function(gram_test)

        roc_auc_val = roc_auc_score(y_true=labels, y_score=predictions)
        logger.info(f"Supervised AUC score for param = {param} : {roc_auc_val}")

        if roc_auc_val > best_auc_supervised:
            best_auc_supervised = roc_auc_val

    return best_auc, best_auc_supervised


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

    embedding_train, embedding_test, adv_embeddings_train, adv_embeddings_test, thresholds, stats, stats_inf = get_all_embeddings(
        config)

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
    logger.info(f"Computed all unsupervised Gram train matrices !")

    all_results = {
        epsilon: evaluate_embeddings(
            gram_train_matrices=gram_train_matrices,
            embeddings_train=embedding_train,
            embeddings_test=embedding_test,
            adv_embeddings_train=adv_embeddings_train[epsilon],
            adv_embeddings_test=adv_embeddings_test[epsilon],
            param_space=param_space,
            kernel_type=config.kernel_type
        )
        for epsilon in adv_embeddings_train
    }

    logger.info(all_results)

    end_time = time.time()

    my_db.update_experiment(
        experiment_id=config.experiment_id,
        run_id=config.run_id,
        metrics={
            "aucs": {key: all_results[key][0] for key in all_results},
            "aucs_supervised": {key: all_results[key][1] for key in all_results},
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
