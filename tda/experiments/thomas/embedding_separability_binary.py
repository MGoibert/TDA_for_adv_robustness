#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import numpy as np
from r3d3.experiment_db import ExperimentDB
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

from tda.embeddings import get_embedding
from tda.graph_dataset import get_dataset
from tda.rootpath import db_path

my_db = ExperimentDB(db_path=db_path)

################
# Parsing args #
################

parser = argparse.ArgumentParser(
        description='Transform a dataset in pail files to tf records.')
parser.add_argument('--experiment_id', type=int)
parser.add_argument('--run_id', type=int)
parser.add_argument('--embedding_type', type=str)
parser.add_argument('--threshold', type=int)
parser.add_argument('--height', type=int)
parser.add_argument('--hash_size', type=int)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"[{args.experiment_id}_{args.run_id}]")

#####################
# Fetching datasets #
#####################

datasets = {0: get_dataset(
    num_epochs=20,
    epsilon=0.04,
    noise=0.0,
    adv=False)
}

all_epsilons = list(sorted(np.linspace(0.01, 0.075, num=5)))

for epsilon in all_epsilons:
    logger.info(f"Computing dataset for epsilon={epsilon}")
    datasets[epsilon] = get_dataset(
        num_epochs=20,
        epsilon=epsilon,
        noise=0.0,
        adv=True
    )

all_epsilons = [0.0] + all_epsilons


def get_vector_from_diagram(dgm):
    """
    Simple tentative to get vector from persistent diagram
    (Top 20 lifespans)
    """
    return list(reversed(sorted([dp.death - dp.birth for dp in dgm][1:])))[:20]


embeddings = list()

for epsilon in all_epsilons:
    logger.info(f"Computing embeddings for epsilon={epsilon}")
    ds = datasets[epsilon]
    for idx in range(len(ds[:100])):

        embedding = get_embedding(
            embedding_type=args.embedding_type,
            graph=ds[idx][0],
            params={
                "threshold": int(args.threshold),
                "hash_size": int(args.hash_size),
                "height": int(args.height)
            }
        )

        embeddings.append((embedding, ds[idx][1], ds[idx][2], ds[idx][3], epsilon))


separability_values = list()

for epsilon in all_epsilons:

    logger.info(f"Computing performance for epsilon={epsilon}")

    if epsilon == 0.0:
        continue

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

    separability_values.append((epsilon, np.max(roc_values)))

logger.info(separability_values)

my_db.update_experiment(
    experiment_id=args.experiment_id,
    run_id=args.run_id,
    metrics=separability_values
)
