#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import typing
import numpy as np
from r3d3.experiment_db import ExperimentDB
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from multiprocessing import Pool

from tda.embeddings import get_embedding, EmbeddingType
from tda.graph_dataset import get_dataset
from tda.rootpath import db_path
from tda.embeddings.weisfeiler_lehman import NodeLabels

my_db = ExperimentDB(db_path=db_path)

################
# Parsing args #
################

parser = argparse.ArgumentParser(
        description='Transform a dataset in pail files to tf records.')
parser.add_argument('--experiment_id', type=int, default=-1)
parser.add_argument('--run_id', type=int, default=-1)
parser.add_argument('--embedding_type', type=str, default=EmbeddingType.WeisfeilerLehman)
parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--height', type=int, default=1)
parser.add_argument('--hash_size', type=int, default=100)
parser.add_argument('--node_labels', type=str, default=NodeLabels.NONE)
parser.add_argument('--steps', type=int, default=1)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"[{args.experiment_id}_{args.run_id}]")

#####################
# Fetching datasets #
#####################

if args.embedding_type == EmbeddingType.OriginalDataPoint:
    retain_data_point = True
else:
    retain_data_point = False

datasets = {0: get_dataset(
    num_epochs=20,
    epsilon=0.04,
    noise=0.0,
    adv=False,
    retain_data_point=retain_data_point
)
}

all_epsilons = list(sorted(np.linspace(0.01, 0.075, num=5)))

for epsilon in all_epsilons:
    logger.info(f"Computing dataset for epsilon={epsilon}")
    datasets[epsilon] = get_dataset(
        num_epochs=20,
        epsilon=epsilon,
        noise=0.0,
        adv=True,
        retain_data_point=retain_data_point
    )

all_epsilons = [0.0] + all_epsilons


def get_vector_from_diagram(dgm):
    """
    Simple tentative to get vector from persistent diagram
    (Top 20 lifespans)
    """
    return list(reversed(sorted([dp.death - dp.birth for dp in dgm][1:])))[:20]


separability_values = list()


def get_embeddings(epsilon: float) -> typing.List:
    logger.info(f"Computing embeddings for epsilon={epsilon}")
    embeddings = list()
    ds = datasets[epsilon]
    for idx in range(len(ds[:100])):
        embedding = get_embedding(
            embedding_type=args.embedding_type,
            graph=ds[idx][0],
            params={
                "threshold": int(args.threshold),
                "hash_size": int(args.hash_size),
                "height": int(args.height),
                "node_labels": args.node_labels,
                "steps": args.steps
            }
        )

        embeddings.append((embedding, ds[idx][1], ds[idx][2], ds[idx][3], epsilon))

    return embeddings


clean_embeddings = get_embeddings(0.0)


def process_epsilon(epsilon: float) -> float:

    if epsilon == 0.0:
        return 0.5

    embeddings = clean_embeddings + get_embeddings(epsilon)

    logger.info(f"Computing performance for epsilon={epsilon}")

    roc_values = list()

    for gamma in np.logspace(-6, -3, 10):
        ocs = OneClassSVM(
            tol=1e-5,
            gamma=gamma)

        clean_data = [np.ndarray.flatten(np.array((e[0]))) for e in embeddings if e[4] == 0.0]
        train_data = clean_data[:len(clean_data) // 2]
        test_data = clean_data[len(clean_data) // 2:]

        ocs.fit(train_data)

        bad_data = [np.ndarray.flatten(np.array((e[0]))) for e in embeddings if e[4] == epsilon]

        predictions = ocs.score_samples(test_data + bad_data)

        labels = np.concatenate((np.ones(len(test_data)), np.zeros(len(bad_data))))

        roc_val = roc_auc_score(y_true=labels, y_score=predictions)
        roc_values.append(roc_val)

    return np.max(roc_values)


with Pool(4) as p:
    separability_values = p.map(process_epsilon, all_epsilons)

logger.info(separability_values)

my_db.update_experiment(
    experiment_id=args.experiment_id,
    run_id=args.run_id,
    metrics={
        "separability_values": separability_values
    }
)
