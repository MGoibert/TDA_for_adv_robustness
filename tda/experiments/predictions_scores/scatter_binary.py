#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import time
import typing

import matplotlib.pyplot as plt
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
parser.add_argument('--thresholds', type=str)
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
parser.add_argument('--do_plot', type=int, default=0)

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

thresholds = process_thresholds(
    raw_thresholds=args.thresholds,
    dataset=args.dataset,
    architecture=args.architecture,
    epochs=args.epochs
)

stats = {}

corrects_i = list()


def get_embeddings(epsilon: float, noise: float, start: int = 0) -> typing.List:
    """
    Helper function to get list of embeddings
    """
    global corrects_i
    my_embeddings = list()

    if epsilon != 0.0:
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
                num_iter=args.num_iter,
                start=start,
                train_noise=args.train_noise
        ):
            logger.info(f"Sample_id = {line.sample_id} ; l2_norm = {line.l2_norm}")
            stats[epsilon].append(line.l2_norm)
            corrects_i.append(line.sample_id)
            my_embeddings.append(get_embedding(
                embedding_type=args.embedding_type,
                graph=line.graph,
                params={
                    "hash_size": int(args.hash_size),
                    "height": int(args.height),
                    "node_labels": args.node_labels,
                    "steps": args.steps
                }
            ))
    else:
        for correct_i in corrects_i:
            logger.info(f"correct_i = {correct_i}")
            for line in get_dataset(
                    num_epochs=args.epochs,
                    epsilon=epsilon,
                    noise=noise,
                    adv=epsilon > 0.0,
                    retain_data_point=retain_data_point,
                    architecture=architecture,
                    source_dataset_name=args.dataset,
                    dataset_size=1,
                    thresholds=thresholds,
                    only_successful_adversaries=args.successful_adv > 0,
                    attack_type=args.attack_type,
                    num_iter=args.num_iter,
                    start=correct_i
            ):
                logger.info(f"Sample_id = {line.sample_id} ; l2_norm = {line.l2_norm}")
                stats[epsilon].append(line.l2_norm)
                my_embeddings.append(get_embedding(
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

if args.attack_type in ["FGSM", "BIM"]:
    all_epsilons = list(sorted(np.linspace(0.02, 0.02, num=1)))
else:
    all_epsilons = [1]
logger.info(f"All epsilons = {all_epsilons}")

adv_embeddings = dict()
for epsilon in all_epsilons:
    stats[epsilon] = list()
    adv_embeddings[epsilon] = get_embeddings(epsilon=epsilon, noise=0.0)
    logger.info(
        f"Stats for diff btw clean and adv: {np.quantile(stats[epsilon], 0.1), np.quantile(stats[epsilon], 0.25), np.median(stats[epsilon]), np.quantile(stats[epsilon], 0.75), np.quantile(stats[epsilon], 0.9)}")
    logger.info(f"corrects_i for eps = {epsilon} = {corrects_i}")

stats[0.0] = list()
clean_embeddings = get_embeddings(epsilon=0.0, noise=0.0)
noisy_embeddings = get_embeddings(epsilon=0.0, noise=args.noise)


def process_epsilon(epsilon: float) -> list:
    if epsilon == 0.0:
        return 0.5
    roc_values = list()
    logger.info(f"Computing performance for epsilon={epsilon}")

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

        train_data = clean_embeddings[:len(clean_embeddings) // 2]  # + clean_embeddings2[:len(clean_embeddings) // 2]
        test_data = clean_embeddings[len(clean_embeddings) // 2:]  # + clean_embeddings2[len(clean_embeddings) // 2:]

        # Training model
        start_time = time.time()
        gram_train = get_gram_matrix(
            args.kernel_type, train_data, train_data,
            param
        )
        logger.info(f"Computed Gram Matrix in {time.time() - start_time} secs")

        start_time = time.time()
        ocs.fit(gram_train)
        logger.info(f"Trained model in {time.time() - start_time} secs")

        # Testing model
        start_time = time.time()
        gram_test_and_bad = get_gram_matrix(
            args.kernel_type, test_data + noisy_embeddings + adv_embeddings[epsilon], train_data,
            param
        )
        logger.info(f"Computed Gram Test Matrix in {time.time() - start_time} secs")

        predictions_train = ocs.score_samples(gram_train)
        predictions = ocs.score_samples(gram_test_and_bad)
        pred_clean = list(predictions_train) + list(predictions[:len(test_data)])
        pred_noisy = predictions[len(test_data):len(test_data) + len(noisy_embeddings)]
        pred_adv = predictions[len(test_data) + len(noisy_embeddings):]
        logger.info(
            f"Mean pred clean = {np.mean(pred_clean)}, noisy = {np.mean(pred_noisy)} and adv = {np.mean(pred_adv)}")

        predictions2 = ocs.score_samples(gram_test_and_bad)
        labels = np.concatenate((np.ones(len(test_data)), np.zeros(len(noisy_embeddings + adv_embeddings[epsilon]))))
        roc_val = roc_auc_score(y_true=labels, y_score=predictions2)
        roc_values.append(roc_val)
        logger.info(f"AUC score = {np.max(roc_values)}")

    return [pred_clean, pred_noisy, pred_adv]


binary_path = os.path.dirname(os.path.realpath(__file__))
directory = f"{binary_path}/plots"

if not os.path.exists(directory):
    os.mkdir(directory)
attack_param = all_epsilons if args.attack_type in ["FGSM", "BIM"] else args.num_iter

all_results = dict()

for epsilon in all_epsilons:
    pred_clean, pred_noisy, pred_adv = process_epsilon(epsilon)

    all_results[str(epsilon).replace(".", "_")] = {
        "pred_clean": list(pred_clean),
        "pred_noisy": list(pred_noisy),
        "pred_adv": list(pred_adv)
    }

    if args.do_plot > 0:
        plt.scatter(pred_adv, pred_noisy)
        plt.plot(pred_adv, pred_adv, 'r-')
        plt.plot(pred_noisy, pred_noisy, 'r-')
        plt.xlabel("Adv prediction eps = " + str(epsilon))
        plt.ylabel("Noisy prediction")
        plt.title("Adv vs noisy score for eps = " + str(epsilon))
        plt.savefig(
            directory + "/scatter_adv_noisy_" + args.dataset + "_" + args.architecture + "_" + args.attack_type + "_" + str(
                attack_param) + ".png", dpi=800)
        plt.close()

        plt.scatter(pred_adv, np.array(pred_clean))
        plt.plot(pred_adv, pred_adv, 'r-')
        plt.plot(np.array(pred_clean), np.array(pred_clean), 'r-')
        plt.xlabel("Adv prediction eps = " + str(epsilon))
        plt.ylabel("Clean prediction")
        plt.title("Adv vs clean score for eps = " + str(epsilon))
        plt.savefig(
            directory + "/scatter_adv_clean_" + args.dataset + "_" + args.architecture + "_" + args.attack_type + "_" + str(
                attack_param) + ".png", dpi=800)
        plt.close()

        plt.scatter(pred_noisy, np.array(pred_clean))
        plt.plot(pred_noisy, pred_noisy, 'r-')
        plt.plot(np.array(pred_clean), np.array(pred_clean), 'r-')
        plt.xlabel("Noisy prediction")
        plt.ylabel("Clean prediction")
        plt.title("Noisy vs clean score for eps = " + str(epsilon))
        plt.savefig(
            directory + "/scatter_noisy_clean_" + args.dataset + "_" + args.architecture + "_" + args.attack_type + "_" + str(
                attack_param) + ".png", dpi=800)
        plt.close()

my_db.update_experiment(
    experiment_id=args.experiment_id,
    run_id=args.run_id,
    metrics={
        "predictions": all_results
    }
)