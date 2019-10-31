#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import time
import typing
import numpy as np

from tda.graph import Graph
from tda.graph_dataset import get_dataset, compute_adv_accuracy
from tda.models.architectures import mnist_mlp, get_architecture, svhn_lenet

from igraph import Graph as IGraph
from networkx.algorithms.centrality import betweenness_centrality, eigenvector_centrality
from networkx.algorithms.centrality.katz import katz_centrality

start_time = time.time()

################
# Parsing args #
################

parser = argparse.ArgumentParser()
parser.add_argument('--thresholds', type=str, default="0_0_0_0_0_0_0")
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--dataset', type=str, default="SVHN")
parser.add_argument('--architecture', type=str, default=svhn_lenet.name)
parser.add_argument('--dataset_size', type=int, default=100)
parser.add_argument('--attack_type', type=str, default="FGSM")
parser.add_argument('--num_iter', type=int, default=10)

args, _ = parser.parse_known_args()

logger = logging.getLogger("GraphStats")

if args.attack_type in ["FGSM", "BIM"]:
    all_epsilons = [0.0] + list(sorted(np.linspace(0.01, 0.1, num=4)))
else:
    all_epsilons = [0.0, 1]

architecture = get_architecture(args.architecture)
accuracies = dict()

for epsilon in all_epsilons:

    adversarial_acc = compute_adv_accuracy(
        epsilon=epsilon,
        noise=args.noise,
        num_epochs=args.epochs,
        source_dataset_name=args.dataset,
        architecture=architecture,
        dataset_size=args.dataset_size,
        attack_type=args.attack_type,
        num_iter=args.num_iter

    )

    accuracies[epsilon] = adversarial_acc

logging.info(accuracies)

end_time = time.time()

logging.info(f"Success in {end_time - start_time} seconds")
