#!/usr/bin/env python
# coding: utf-8

import argparse
import time
import typing
from multiprocessing import Pool

import numpy as np
from r3d3.experiment_db import ExperimentDB
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

from tda.embeddings import get_embedding, EmbeddingType, \
    get_gram_matrix, KernelType
from tda.embeddings.weisfeiler_lehman import NodeLabels
from tda.graph_dataset import get_dataset
from tda.models.architectures import mnist_mlp, get_architecture
from tda.rootpath import db_path
from tda.thresholds import process_thresholds
from tda.logging import get_logger

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


# save np.load and modify the default parameters of np.load
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def run_experiment(config: Config):
    logger = get_logger(f"{config.experiment_id}_{config.run_id}")

    if __name__ != "__main__":
        my_db.add_experiment(
            experiment_id=config.experiment_id,
            run_id=config.run_id,
            config=config._asdict()
        )

    architecture = get_architecture(config.architecture)

    if config.embedding_type == EmbeddingType.OriginalDataPoint:
        retain_data_point = True
    else:
        retain_data_point = False

    thresholds = process_thresholds(
        raw_thresholds=config.thresholds,
        dataset=config.dataset,
        architecture=architecture,
        epochs=config.epochs,
        dataset_size=5
    )

    stats = {}
    stats_inf = {}

    def get_embeddings(epsilon: float, noise: float, start: int = 0) -> typing.List:
        """
        Helper function to get list of embeddings
        """
        my_embeddings = list()
        for line in get_dataset(
                num_epochs=config.epochs,
                epsilon=epsilon,
                noise=noise,
                adv=epsilon > 0.0,
                retain_data_point=retain_data_point,
                architecture=architecture,
                source_dataset_name=config.dataset,
                dataset_size=config.dataset_size,
                thresholds=thresholds,
                only_successful_adversaries=config.successful_adv > 0,
                attack_type=config.attack_type,
                num_iter=config.num_iter,
                start=start,
                train_noise=config.train_noise
        ):
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
            f"Computed embeddings for (attack = {config.attack_type}, eps={epsilon}, noise={noise}), number of sample = {len(my_embeddings)}")
        return my_embeddings


    # Clean embeddings
    stats[0.0] = list()
    stats_inf[0.0] = list()

    start = 0

    # Clean train
    logger.info(f"Clean train dataset !!")
    clean_embeddings_train = get_embeddings(epsilon=0.0, noise=0.0, start=0)
    if config.identical_train_samples < 0.5:
        start += config.dataset_size
    #clean_embeddings_train = list()

    # Noisy train
    if config.noise > 0.0:
        logger.info(f"Noisy train dataset !!")
        noisy_embeddings_train = get_embeddings(epsilon=0.0, noise=config.noise, start=start)
    else:
        noisy_embeddings_train = list()
    start += config.dataset_size

    if config.kernel_type == KernelType.RBF:
        param_space = [
            {'gamma': gamma}
            for gamma in np.logspace(-6, -3, 10)
        ]
    #logger.info(f"kernel type = {args.kernel_type} and {args.kernel_type == KernelType.SlicedWasserstein}")
    #if args.kernel_type == KernelType.SlicedWasserstein:
    else:
        param_space = [
            #{'M': 20, 'sigma': 5 * 10 ** (-5)},
            #{'M': 20, 'sigma': 5 * 10 ** (-4)},
            #{'M': 20, 'sigma': 5 * 10 ** (-3)},
            #{'M': 20, 'sigma': 5 * 10 ** (-2)},
            {'M': 20, 'sigma': 5 * 10 ** (-1)},
        ]

    gram_train_matrices = {i: get_gram_matrix(
        kernel_type=config.kernel_type,
        embeddings_in=clean_embeddings_train + noisy_embeddings_train,
        embeddings_out=None,
        params=param
    )
        for i, param in enumerate(param_space)
    }
    logger.info(f"Computed all Gram train matrices !")

    # Clean test
    logger.info(f"Clean test dataset !!")
    clean_embeddings_test = get_embeddings(epsilon=0.0, noise=0.0, start=start)
    start += config.dataset_size
    #clean_embeddings_test = list()

    # Noisy test
    if config.noise > 0.0:
        logger.info(f"Noisy test dataset !!")
        noisy_embeddings_test = get_embeddings(epsilon=0.0, noise=config.noise, start=start)
    else:
        noisy_embeddings_test = list()
    start += config.dataset_size

    if config.attack_type in ["FGSM", "BIM"]:
        all_epsilons = list([0.0, 0.025, 0.05, 0.1, 0.4])
    else:
        all_epsilons = [0.0, 1]

    adv_embeddings = dict()
    for epsilon in all_epsilons[1:]:
        stats[epsilon] = list()
        stats_inf[epsilon] = list()
        logger.info(f"Adversarial test dataset for espilon = {epsilon} !!")
        adv_embeddings[epsilon] = get_embeddings(epsilon=epsilon, noise=0.0, start=start)
        logger.debug(
            f"Stats for diff btw clean and adv: {np.quantile(stats[epsilon], 0.1), np.quantile(stats[epsilon], 0.25), np.median(stats[epsilon]), np.quantile(stats[epsilon], 0.75), np.quantile(stats[epsilon], 0.9)}")



    def process_epsilon(epsilon: float) -> float:
        """
        Compute the AUC for a given epsilon and returns also the scores
        of the best OneClass SVM
        """
        if epsilon == 0.0:
            return 0.5

        # embeddings = clean_embeddings + adv_embeddings[epsilon]

        logger.info(f"Computing performance for epsilon={epsilon}")

        best_auc = 0.0

        for i, param in enumerate(param_space):
            ocs = OneClassSVM(
                tol=1e-5,
                kernel="precomputed")

            # Training model
            start_time = time.time()
            logger.info(f"sum ram matrix train = {gram_train_matrices[i].sum()}")
            ocs.fit(gram_train_matrices[i])
            logger.info(f"Trained model in {time.time() - start_time} secs")

            # Testing model
            start_time = time.time()
            gram_test_and_bad = get_gram_matrix(
                kernel_type=config.kernel_type,
                embeddings_in=clean_embeddings_test + noisy_embeddings_test + adv_embeddings[epsilon],
                embeddings_out=clean_embeddings_train + noisy_embeddings_train,
                params=param
            )
            logger.info(f"Computed Gram Test Matrix in {time.time() - start_time} secs")

            predictions = ocs.score_samples(gram_test_and_bad)

            labels = np.concatenate(
                (
                    np.ones(len(clean_embeddings_test)),
                    np.ones(len(noisy_embeddings_test)),
                    np.zeros(len(adv_embeddings[epsilon]))
                )
            )

            roc_auc_val = roc_auc_score(y_true=labels, y_score=predictions)
            logger.info(f"AUC score for param = {param} : {roc_auc_val}")

            if roc_auc_val > best_auc:
                best_auc = roc_auc_val

        return best_auc

    #with Pool(2) as p:
    #    all_results = p.map(process_epsilon, all_epsilons)

    all_results = [process_epsilon(epsilon) for epsilon in all_epsilons]

    all_results = dict(zip(all_epsilons, all_results))

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


if __name__ == "__main__":
    config = get_config()
    run_experiment(config)