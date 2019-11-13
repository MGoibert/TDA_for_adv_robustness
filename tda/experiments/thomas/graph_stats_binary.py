#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import time
import typing
import numpy as np
import matplotlib.pyplot as plt

from tda.graph import Graph
from tda.graph_dataset import get_dataset
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
parser.add_argument('--train_noise', type=float, default=0.0)
parser.add_argument('--dataset_size', type=int, default=100)
parser.add_argument('--num_iter', type=int, default=10)
parser.add_argument('--visualize_adj_mat', type=float, default=0)

args, _ = parser.parse_known_args()

logger = logging.getLogger("GraphStats")

#####################
# Fetching datasets #
#####################

architecture = get_architecture(args.architecture)

thresholds = [int(x) for x in args.thresholds.split("_")]

def get_stats(epsilon: float, noise: float, attack_type: str = "FGSM") -> typing.List:
    """
    Helper function to get list of embeddings
    """

    weights_per_layer = dict()
    print("eps =", epsilon)

    for line in get_dataset(
            num_epochs=args.epochs,
            epsilon=epsilon,
            noise=noise,
            adv=epsilon > 0.0,
            retain_data_point=False,
            architecture=architecture,
            source_dataset_name=args.dataset,
            dataset_size=args.dataset_size,
            thresholds=thresholds,
            only_successful_adversaries=False,
            attack_type=attack_type,
            num_iter=args.num_iter,
            train_noise=args.train_noise
        ):

        graph: Graph = line[0]
        logger.info(f"The data point: y = {line[1]}, y_pred = {line[2]} and adv = {line[3]} and the attack = {attack_type}")
        adjacency_matrix = graph.get_adjacency_matrix()
        #print(np.shape(adjacency_matrix))
        #print(adjacency_matrix[0,0])
        #print("Nan =", np.isnan(adjacency_matrix).sum())
        #print(adjacency_matrix.min(), adjacency_matrix.max())
        #plt.matshow(adjacency_matrix, vmin=0, vmax=750000, cmap='Greys')
        #filename = "/Users/m.goibert/Downloads/test_adj_matrix_" + str(epsilon) +  "_" + str(attack_type) + ".png"
        #plt.savefig(filename, dpi=800)
        #plt.show()

        for i, layer_matrix in enumerate(graph._edge_list):
            if i in weights_per_layer:
                weights_per_layer[i] = np.concatenate([weights_per_layer[i], layer_matrix])
            else:
                weights_per_layer[i] = layer_matrix

    all_weights = list()

    for i in weights_per_layer:
        m = weights_per_layer[i]
        nonzero_m = m[np.where(m > 0)].reshape(-1, 1)
        all_weights.append(nonzero_m)

        q10 = np.quantile(nonzero_m, 0.1)
        q25 = np.quantile(nonzero_m, 0.25)
        q50 = np.quantile(nonzero_m, 0.5)
        q75 = np.quantile(nonzero_m, 0.75)
        q80 = np.quantile(nonzero_m, 0.8)
        q90 = np.quantile(nonzero_m, 0.9)
        q95 = np.quantile(nonzero_m, 0.95)
        q99 = np.quantile(nonzero_m, 0.99)
        qmax = max(nonzero_m)
        print(f"Layer {i} weights med = {q50} [0.10 = {q10}; 0.25 = {q25}; 0.75 = {q75}; 0.8 = {q80}; 0.9 = {q90}; 0.95 = {q95}; 0.99 = {q99}; max = {qmax}]")

    all_weights2 = np.concatenate(all_weights, axis=0)
    q10 = np.quantile(all_weights2, 0.1)
    q50 = np.quantile(all_weights2, 0.5)
    q90 = np.quantile(all_weights2, 0.9)
    q95 = np.quantile(all_weights2, 0.95)
    q99 = np.quantile(all_weights2, 0.99)
    print(f"All weights {q50} [{q10}; {q90} {q95} {q99}]")

    return all_weights, adjacency_matrix

weights = get_stats(epsilon=0.0, noise=0.0)
logger.info(f"weights = {weights}")

if args.visualize_adj_mat > 0.5:
    # FGSM part
    a = get_stats(epsilon=0.0, noise=0.0)
    b = get_stats(epsilon=0.02, noise=0.0, attack_type="FGSM")
    diff_mat = a - b
    print("Extreme values =", diff_mat.min(), diff_mat.max())
    print("Quantile =", np.quantile(diff_mat, 0.001), np.quantile(diff_mat, 0.01), np.quantile(diff_mat, 0.05), np.quantile(diff_mat, 0.1), np.quantile(diff_mat, 0.2), np.quantile(diff_mat, 0.5), np.quantile(diff_mat, 0.8), np.quantile(diff_mat, 0.9), np.quantile(diff_mat, 0.95), np.quantile(diff_mat, 0.99), np.quantile(diff_mat, 0.999))
    print("Quantile v2 =", np.quantile(diff_mat[np.where(diff_mat != 0)], 0.01), np.quantile(diff_mat[np.where(diff_mat != 0)], 0.05), np.quantile(diff_mat[np.where(diff_mat != 0)], 0.1), np.quantile(diff_mat[np.where(diff_mat != 0)], 0.2), np.quantile(diff_mat[np.where(diff_mat != 0)], 0.8), np.quantile(diff_mat[np.where(diff_mat != 0)], 0.9), np.quantile(diff_mat[np.where(diff_mat != 0)], 0.95), np.quantile(diff_mat[np.where(diff_mat != 0)], 0.99))
    plt.matshow(diff_mat, vmin=-50000, vmax=50000, cmap='Greys')
    plt.savefig("/Users/m.goibert/Downloads/diff_adj_matrix" + "_FGSM_" ".png", dpi=800)
    plt.show()

    c = get_stats(epsilon=0.2, noise=0.0, attack_type="DeepFool")
    diff_mat2 = a - c
    print("Extreme values =", diff_mat2.min(), diff_mat2.max())
    print("Quantile =", np.quantile(diff_mat2, 0.001), np.quantile(diff_mat2, 0.01), np.quantile(diff_mat2, 0.05), np.quantile(diff_mat2, 0.1), np.quantile(diff_mat2, 0.2), np.quantile(diff_mat2, 0.5), np.quantile(diff_mat2, 0.8), np.quantile(diff_mat2, 0.9), np.quantile(diff_mat2, 0.95), np.quantile(diff_mat2, 0.99), np.quantile(diff_mat2, 0.999))
    print("Quantile v2 =", np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.01), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.05), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.1), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.2), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.8), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.9), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.95), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.99))
    plt.matshow(diff_mat2, vmin=-50000, vmax=50000, cmap='Greys')
    plt.savefig("/Users/m.goibert/Downloads/diff_adj_matrix" + "_DeepFool_" ".png", dpi=800)
    plt.show()

    diff_mat_att = np.abs(b - c)
    plt.matshow(diff_mat_att, vmin=0, vmax=0.1, cmap='Greys')
    plt.savefig("/Users/m.goibert/Downloads/diff_adj_matrix" + "_FGSMvsDeepFool_" ".png", dpi=800)
    plt.show()

end_time = time.time()

logger.info(f"Success in {end_time-start_time} seconds")
