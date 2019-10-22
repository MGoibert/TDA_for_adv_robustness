#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import time
import typing
from multiprocessing import Pool
from random import shuffle

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

start_time = time.time()

my_db = ExperimentDB(db_path=db_path)

################
# Parsing args #
################

parser = argparse.ArgumentParser(
    description='Transform a dataset in pail files to tf records.')
parser.add_argument('--experiment_id', type=int, default=-1)
parser.add_argument('--run_id', type=int, default=-1)
parser.add_argument('--embedding_type', type=str, default=EmbeddingType.WeisfeilerLehman)
parser.add_argument('--kernel_type', type=str, default=KernelType.RBF)
parser.add_argument('--thresholds', type=str, default='0')
parser.add_argument('--height', type=int, default=1)
parser.add_argument('--hash_size', type=int, default=100)
parser.add_argument('--node_labels', type=str, default=NodeLabels.NONE)
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--architecture', type=str, default=mnist_mlp.name)
parser.add_argument('--dataset_size', type=int, default=100)
parser.add_argument('--successful_adv', type=int, default=1)
parser.add_argument('--attack_type', type=str, default="FGSM")
parser.add_argument('--num_iter', type=int, default=10)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"[{args.experiment_id}_{args.run_id}]")

#####################
# Fetching datasets #
#####################

architecture = get_architecture(args.architecture)

if args.embedding_type == EmbeddingType.OriginalDataPoint:
    retain_data_point = True
else:
    retain_data_point = False

thresholds = [int(x) for x in args.thresholds.split("_")]
print(thresholds)
stats = {}


def get_embeddings(epsilon: float, noise: float) -> typing.List:
    """
    Helper function to get list of embeddings
    """
    my_embeddings = list()
    for line in get_dataset(
            num_epochs=args.epochs,
            epsilon=epsilon,
            noise=noise,
            adv=epsilon > 0.0,
            retain_data_point=retain_data_point,
            architecture=architecture,
            source_dataset_name=args.dataset,
            dataset_size=args.dataset_size,
            thresholds=thresholds,
            only_successful_adversaries=args.successful_adv > 0,
            attack_type=args.attack_type,
            num_iter=args.num_iter
        ):
        logger.info(f"Line = {line[:3]} and diff = {line[4]}")
        stats[epsilon].append(line[4])
        my_embeddings.append(get_embedding(
            embedding_type=args.embedding_type,
            graph=line[0],
            params={
                "hash_size": int(args.hash_size),
                "height": int(args.height),
                "node_labels": args.node_labels,
                "steps": args.steps
            }
        ))
    logger.info(f"Computed embeddings for (attack = {args.attack_type}, eps={epsilon}, noise={noise}), number of sample = {len(my_embeddings)}")
    return my_embeddings


# Clean embeddings
stats[0.0] = list()
clean_embeddings = get_embeddings(epsilon=0.0, noise=0.0)
#clean_embeddings += get_embeddings(epsilon=0.0, noise=args.noise)
shuffle(clean_embeddings)

if args.attack_type in ["FGSM", "BIM"]:
    all_epsilons = [0.0] + list(sorted(np.linspace(0.01, 0.075, num=5)))
else:
    all_epsilons = [0.0, 1]
#all_epsilons = [0.0] + list(sorted(np.logspace(-15, -15, 1)))

adv_embeddings = dict()
for epsilon in all_epsilons[1:]:
    stats[epsilon] = list()
    adv_embeddings[epsilon] = get_embeddings(epsilon=epsilon, noise=0.0)
    shuffle(adv_embeddings[epsilon])
    logger.info(f"Stats for diff btw clean and adv: {np.quantile(stats[epsilon], 0.1), np.quantile(stats[epsilon], 0.25), np.median(stats[epsilon]), np.quantile(stats[epsilon], 0.75), np.quantile(stats[epsilon], 0.9)}")

def process_epsilon(epsilon: float) -> float:
    if epsilon == 0.0:
        return 0.5

    # embeddings = clean_embeddings + adv_embeddings[epsilon]

    logger.info(f"Computing performance for epsilon={epsilon}")

    roc_values = list()

    if args.kernel_type == KernelType.RBF:
        param_space = [
            {'gamma': gamma}
            for gamma in np.logspace(-6, -3, 10)
        ]
    if args.kernel_type == KernelType.SlicedWasserstein:
        param_space = [
            {'M': 10, 'sigma': 5 * 10 ** (-5)}
        ]

    for param in param_space:
        ocs = OneClassSVM(
            tol=1e-5,
            kernel="precomputed")

        # Datasets used for the OneClassSVM

        # clean_data = [np.ndarray.flatten(np.array((e[0]))) for e in embeddings if e[4] == 0.0]
        train_data = clean_embeddings[:len(clean_embeddings) // 2]
        test_data = clean_embeddings[len(clean_embeddings) // 2:]

        # Training model
        start_time = time.time()
        gram_train = get_gram_matrix(
            args.kernel_type, train_data, train_data,
            param
        )
        logger.info(f"Computed Gram Matrix in {time.time()-start_time} secs")

        start_time = time.time()
        ocs.fit(gram_train)
        logger.info(f"Trained model in {time.time()-start_time} secs")

        # Testing model

        start_time = time.time()
        gram_test_and_bad = get_gram_matrix(
            args.kernel_type, test_data + adv_embeddings[epsilon], train_data,
            param
        )
        logger.info(f"Computed Gram Test Matrix in {time.time() - start_time} secs")

        predictions = ocs.score_samples(gram_test_and_bad)

        labels = np.concatenate((np.ones(len(test_data)), np.zeros(len(adv_embeddings[epsilon]))))

        roc_val = roc_auc_score(y_true=labels, y_score=predictions)
        roc_values.append(roc_val)

    return np.max(roc_values)


with Pool(2) as p:
    separability_values = p.map(process_epsilon, all_epsilons)


logger.info(separability_values)

end_time = time.time()

my_db.update_experiment(
    experiment_id=args.experiment_id,
    run_id=args.run_id,
    metrics={
        "separability_values": dict(zip(all_epsilons, separability_values)),
        "running_time": end_time - start_time
    }
)
