#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import time
import typing
from multiprocessing import Pool

import numpy as np
from r3d3.experiment_db import ExperimentDB
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.svm import OneClassSVM, SVC

from tda.embeddings import get_embedding, EmbeddingType, \
    get_gram_matrix, KernelType
from tda.embeddings.weisfeiler_lehman import NodeLabels
from tda.graph_dataset import get_graph_dataset
from tda.models.architectures import mnist_mlp, get_architecture
from tda.rootpath import db_path
from tda.thresholds import process_thresholds

start_time = time.time()

my_db = ExperimentDB(db_path=db_path)

################
# Parsing args #
################

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"[{args.experiment_id}_{args.run_id}]")

# save np.load and modify the default parameters of np.load
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#####################
# Fetching datasets #
#####################

architecture = get_architecture(args.architecture)

if args.embedding_type == EmbeddingType.OriginalDataPoint:
    retain_data_point = True
else:
    retain_data_point = False

thresholds = process_thresholds(
    raw_thresholds=args.thresholds,
    dataset=args.dataset,
    architecture=args.architecture,
    epochs=args.epochs
)

stats = {}

# f"stats/{dataset}_{architecture}_{str(epochs)}_epochs.npy"

def get_embeddings(epsilon: float, noise: float, start: int = 0, attack=args.attack_type) -> typing.List:
    """
    Helper function to get list of embeddings
    """
    my_embeddings = dict()
    for k in range(10): my_embeddings[k] = list()
    for line in get_graph_dataset(
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
            attack_type=attack,
            num_iter=args.num_iter,
            start=start,
            train_noise=args.train_noise,
            per_class=True
    ):
        stats[epsilon].append(line.l2_norm)
        my_embeddings[line.y].append(get_embedding(
            embedding_type=args.embedding_type,
            graph=line.graph,
            params={
                "hash_size": int(args.hash_size),
                "height": int(args.height),
                "node_labels": args.node_labels,
                "steps": args.steps
            }
        ))
    logger.info(
        f"Computed embeddings for (attack = {args.attack_type}, eps={epsilon}, noise={noise}), number of sample = {len(my_embeddings)}")
    return my_embeddings


# Clean embeddings
train_eps = 0.005
stats[0.0] = list()
stats[train_eps] = list()

start = 0

# Clean train
logger.info(f"Clean train dataset !!")
#clean_embeddings_train = get_embeddings(epsilon=0.0, noise=0.0, start=0)
#if args.identical_train_samples < 0.5:
#    start += args.dataset_size
clean_embeddings_train = dict()
for k in range(10):
    clean_embeddings_train[k] = list()

# Noisy train
if args.noise > 0.0:
    logger.info(f"Noisy train dataset !!")
    noisy_embeddings_train = get_embeddings(epsilon=0.0, noise=args.noise, start=start)
else:
    noisy_embeddings_train = dict()
    for k in range(10):
        noisy_embeddings_train[k] = list()
if args.identical_train_samples < 0.5:
    start += args.dataset_size

# Adv train dataset
logger.info(f"Adv train dataset !!")
adv_embeddings_train = get_embeddings(epsilon=train_eps, noise=0.0, start=start, attack="FGSM")
start += args.dataset_size

# Concatenate train datasets
train_list_dict = [clean_embeddings_train, noisy_embeddings_train, adv_embeddings_train]
train_dataset = {}
for key in clean_embeddings_train.keys():
    train_dataset[key] = [elem for d in train_list_dict for elem in d[key]]

labels_train = np.concatenate(
            (
                np.zeros(len(clean_embeddings_train[0] + noisy_embeddings_train[0] + adv_embeddings_train[0])),
                1*np.ones(len(clean_embeddings_train[1] + noisy_embeddings_train[1] + adv_embeddings_train[1])),
                2*np.ones(len(clean_embeddings_train[2] + noisy_embeddings_train[2] + adv_embeddings_train[2])),
                3*np.ones(len(clean_embeddings_train[3] + noisy_embeddings_train[3] + adv_embeddings_train[3])),
                4*np.ones(len(clean_embeddings_train[4] + noisy_embeddings_train[4] + adv_embeddings_train[4])),
                5*np.ones(len(clean_embeddings_train[5] + noisy_embeddings_train[5] + adv_embeddings_train[5])),
                6*np.ones(len(clean_embeddings_train[6] + noisy_embeddings_train[6] + adv_embeddings_train[6])),
                7*np.ones(len(clean_embeddings_train[7] + noisy_embeddings_train[7] + adv_embeddings_train[7])),
                8*np.ones(len(clean_embeddings_train[8] + noisy_embeddings_train[8] + adv_embeddings_train[8])),
                9*np.ones(len(clean_embeddings_train[9] + noisy_embeddings_train[9] + adv_embeddings_train[9]))
            )
        )

if args.kernel_type == KernelType.RBF:
    param_space = [
        {'gamma': gamma}
        for gamma in np.logspace(-6, -3, 10)
    ]
#logger.info(f"kernel type = {args.kernel_type} and {args.kernel_type == KernelType.SlicedWasserstein}")
#if args.kernel_type == KernelType.SlicedWasserstein:
else:
    param_space = [
        {'M': 10, 'sigma': 5 * 10 ** (-5)}
    ]

gram_train_matrices = {i: get_gram_matrix(
    kernel_type=args.kernel_type,
    embeddings_in=[elem for l in train_dataset.values() for elem in l],
    embeddings_out=None,
    params=param
)
    for i, param in enumerate(param_space)
}
logger.info(f"Computed all Gram train matrices !")

# Clean test
logger.info(f"Clean test dataset !!")
#clean_embeddings_test = get_embeddings(epsilon=0.0, noise=0.0, start=start)
#start += args.dataset_size
clean_embeddings_test = dict()
for k in range(10):
    clean_embeddings_test[k] = list()

# Noisy test
if args.noise > 0.0:
    logger.info(f"Noisy test dataset !!")
    noisy_embeddings_test = get_embeddings(epsilon=0.0, noise=args.noise, start=start)
else:
    noisy_embeddings_test = dict()
    for k in range(10):
        noisy_embeddings_test[k] = list()
start += args.dataset_size

if args.attack_type in ["FGSM", "BIM"]:
    all_epsilons = list([0.0, 0.005])#, 0.01, 0.02, 0.05, 0.1])
else:
    all_epsilons = [0.0, 1]

adv_embeddings = dict()
for epsilon in all_epsilons[1:]:
    stats[epsilon] = list()
    logger.info(f"Adversarial test dataset for espilon = {epsilon} !!")
    adv_embeddings[epsilon] = get_embeddings(epsilon=epsilon, noise=0.0, start=start)
    logger.info(
        f"Stats for diff btw clean and adv: {np.quantile(stats[epsilon], 0.1), np.quantile(stats[epsilon], 0.25), np.median(stats[epsilon]), np.quantile(stats[epsilon], 0.75), np.quantile(stats[epsilon], 0.9)}")

logger.info(f"Clean = {clean_embeddings_test}, noisy = {noisy_embeddings_test} et adv = {adv_embeddings[epsilon]}")
# Concatenate test datasets
test_dataset = dict()
for epsilon in all_epsilons[1:]:
    logger.info(f"eps = {epsilon}")
    test_dataset[epsilon] = dict()
    test_list_dict = [clean_embeddings_test, noisy_embeddings_test, adv_embeddings[epsilon]]
    for key in clean_embeddings_test.keys():
        logger.info(f"key = {key}")
        test_dataset[epsilon][key] = [elem for d in test_list_dict for elem in d[key]]


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
        svc = SVC(
                probability=True,
                kernel="precomputed")

        # Training model
        start_time = time.time()
        svc.fit(gram_train_matrices[i], labels_train)
        logger.info(f"Trained model in {time.time() - start_time} secs")

        # Testing model
        start_time = time.time()
        gram_test_and_bad = get_gram_matrix(
            kernel_type=args.kernel_type,
            embeddings_in=[elem for l in test_dataset[epsilon].values() for elem in l],
            embeddings_out=[elem for l in train_dataset.values() for elem in l],
            params=param
        )
        logger.info(f"Computed Gram Test Matrix in {time.time() - start_time} secs")

        predictions = svc.predict_proba(gram_test_and_bad)
        logger.info(f"predictions : {predictions}")

        labels = np.concatenate(
            (
                np.zeros(len(clean_embeddings_test[0] + noisy_embeddings_test[0] + adv_embeddings[epsilon][0])),
                1*np.ones(len(clean_embeddings_test[1] + noisy_embeddings_test[1] + adv_embeddings[epsilon][1])),
                2*np.ones(len(clean_embeddings_test[2] + noisy_embeddings_test[2] + adv_embeddings[epsilon][2])),
                3*np.ones(len(clean_embeddings_test[3] + noisy_embeddings_test[3] + adv_embeddings[epsilon][3])),
                4*np.ones(len(clean_embeddings_test[4] + noisy_embeddings_test[4] + adv_embeddings[epsilon][4])),
                5*np.ones(len(clean_embeddings_test[5] + noisy_embeddings_test[5] + adv_embeddings[epsilon][5])),
                6*np.ones(len(clean_embeddings_test[6] + noisy_embeddings_test[6] + adv_embeddings[epsilon][6])),
                7*np.ones(len(clean_embeddings_test[7] + noisy_embeddings_test[7] + adv_embeddings[epsilon][7])),
                8*np.ones(len(clean_embeddings_test[8] + noisy_embeddings_test[8] + adv_embeddings[epsilon][8])),
                9*np.ones(len(clean_embeddings_test[9] + noisy_embeddings_test[9] + adv_embeddings[epsilon][9]))
            )
        )
        score = svc.score(gram_test_and_bad, labels)
        logger.info(f"Score {epsilon}: {score}")
        logger.info(f"Confusion matrix {epsilon} = {confusion_matrix(labels, svc.predict(gram_test_and_bad))}")

        #roc_auc_val = roc_auc_score(y_true=labels, y_score=predictions)

        #if roc_auc_val > best_auc:
        #    best_auc = roc_auc_val

    return score


with Pool(2) as p:
    all_results = p.map(process_epsilon, all_epsilons)

all_results = dict(zip(all_epsilons, all_results))

logger.info(all_results)

end_time = time.time()

my_db.update_experiment(
    experiment_id=args.experiment_id,
    run_id=args.run_id,
    metrics={
        "Supervised": "Yes",
        "separability_values": all_results,
        "effective_thresholds": thresholds,
        "running_time": end_time - start_time
    }
)
