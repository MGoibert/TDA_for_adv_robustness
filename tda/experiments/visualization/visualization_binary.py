#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import time
import typing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import seaborn as sns
import os
import pathlib

from tda.graph import Graph
from tda.graph_dataset import get_dataset
from tda.models.architectures import mnist_mlp, get_architecture, svhn_lenet
from tda.models.architectures import get_architecture, svhn_lenet
from tda.thresholds import process_thresholds

# from igraph import Graph as IGraph
# from networkx.algorithms.centrality import betweenness_centrality, eigenvector_centrality
# from networkx.algorithms.centrality.katz import katz_centrality

start_time = time.time()

################
# Parsing args #
################

parser = argparse.ArgumentParser()
parser.add_argument('--thresholds', type=str, default="0")
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--dataset', type=str, default="SVHN")
parser.add_argument('--architecture', type=str, default=svhn_lenet.name)
parser.add_argument('--train_noise', type=float, default=0.0)
parser.add_argument('--num_iter', type=int, default=10)
parser.add_argument('--attack_type', type=str, default="FGSM")
parser.add_argument('--desired_y', type=int, default=-1)

args, _ = parser.parse_known_args()

logger = logging.getLogger("Visu")

#####################
# Fetching datasets #
#####################

architecture = get_architecture(args.architecture)

if args.thresholds == "0":
    thresholds = list(np.zeros(10))
else:
    thresholds = process_thresholds(
    raw_thresholds=args.thresholds,
    dataset=args.dataset,
    architecture=args.architecture,
    epochs=args.epochs
    )

plt.style.use('seaborn-dark')

binary_path = os.path.dirname(os.path.realpath(__file__))
binary_path_split = pathlib.Path(binary_path)
directory = str(pathlib.Path(*binary_path_split.parts[:-3])) + "/plots/visualization/" + str(args.architecture)
if not os.path.exists(directory):
    os.makedirs(directory)

def get_stats(epsilon: float, noise: float, attack_type: str = "FGSM", start: int = 0, desired_y: int = -1) -> (typing.List, np.matrix):
    """
    Helper function to get list of embeddings
    """
    logger.info(f"Computing weights stats")

    weights_per_layer = dict()
    print("eps =", epsilon)

    wrong_ex = True

    while wrong_ex:
        for line in get_dataset(
                num_epochs=args.epochs,
                epsilon=epsilon,
                noise=noise,
                adv=epsilon > 0.0,
                retain_data_point=False,
                architecture=architecture,
                source_dataset_name=args.dataset,
                dataset_size=1,
                thresholds=thresholds,
                only_successful_adversaries=True,
                attack_type=attack_type,
                num_iter=args.num_iter,
                train_noise=args.train_noise,
                start=start
            ):

            graph: Graph = line.graph
            if (line.y == desired_y) or desired_y == -1:
                wrong_ex = False
                logger.info(f"The data point: y = {line.y}, y_pred = {line.y_pred} and adv = {line.y_adv} and the attack = {attack_type}")
                logger.info(f"Perturbation: L2 = {line.l2_norm} and Linf = {line.linf_norm}")
                adjacency_matrix = graph.get_adjacency_matrix()
                print(np.shape(adjacency_matrix))
                print(adjacency_matrix.min(), adjacency_matrix.max())

                for i, layer_matrix in enumerate(graph._edge_list):
                    if i in weights_per_layer:
                        weights_per_layer[i] = np.concatenate([weights_per_layer[i], layer_matrix])
                    else:
                        weights_per_layer[i] = layer_matrix
            else:
                logger.info(f"not desired y ({line.y})")
                start = start + 1
                continue

    all_weights = list()

    for i in weights_per_layer:
        m = weights_per_layer[i]
        nonzero_m = m[np.where(m > 0)].reshape(-1, 1)
        logger.info(f"Number of edges > 0 in layer {i}: {len(nonzero_m)}")
        all_weights.append(nonzero_m)

        qmin = min(nonzero_m)
        q10 = np.quantile(nonzero_m, 0.1)
        q25 = np.quantile(nonzero_m, 0.25)
        q50 = np.quantile(nonzero_m, 0.5)
        q75 = np.quantile(nonzero_m, 0.75)
        q80 = np.quantile(nonzero_m, 0.8)
        q90 = np.quantile(nonzero_m, 0.9)
        q95 = np.quantile(nonzero_m, 0.95)
        q99 = np.quantile(nonzero_m, 0.99)
        qmax = max(nonzero_m)
        print(f"Layer {i} weights [min = {qmin}; 0.10 = {q10}; 0.25 = {q25}; 0.5 = {q50}; 0.75 = {q75}; 0.8 = {q80}; 0.9 = {q90}; 0.95 = {q95}; 0.99 = {q99}; max = {qmax}]")

    all_weights2 = np.concatenate(all_weights, axis=0)
    aq10 = np.quantile(all_weights2, 0.1)
    aq50 = np.quantile(all_weights2, 0.5)
    aq90 = np.quantile(all_weights2, 0.9)
    aq95 = np.quantile(all_weights2, 0.95)
    aq99 = np.quantile(all_weights2, 0.99)
    print(f"All weights {aq50} [{aq10}; {aq90} {aq95} {aq99}]")

    return all_weights, adjacency_matrix, line.y, line.y_pred, line.sample_id, line.x, line.linf_norm

epsilon = 0.05
b_, b, by, by_pred, sample_id, bx, linf_pert = get_stats(epsilon=epsilon, noise=0.0, attack_type=args.attack_type, desired_y=args.desired_y)
a_, a, ay, ay_pred, _, ax, _ = get_stats(epsilon=0.0, noise=0.0, start=sample_id)
c_, c, cy, cy_pred, _, cx, _ = get_stats(epsilon=0.0, noise=0.0, start=sample_id, desired_y=by_pred)
n_, n, ny, ny_pred, _, nx, nlinf_pert = get_stats(epsilon=0.0, noise=epsilon, start=sample_id)
qmin = np.quantile(np.concatenate(a_, axis=0), 0.2)
qmax = np.quantile(np.concatenate(a_, axis=0), 0.8)
cqmin = np.quantile(np.concatenate(c_, axis=0), 0.2)
cqmax = np.quantile(np.concatenate(c_, axis=0), 0.8)
logger.info(f"qmin = {qmin} and qmax = {qmax}")

# Clean adj matrix
p1 = plt.matshow(a, 
    #vmin=20500, vmax=30000, 
    cmap='viridis_r',
    norm=LogNorm(vmin=qmin, vmax=qmax),
    )
plt.colorbar(p1)
filename = directory + "/adj_matrix_y=" + str(ay) + "_clean" + ".png"
plt.savefig(filename, dpi=800)
plt.close()

# Adv adj matrix
p2 = plt.matshow(b, 
    #vmin=20500, vmax=30000, 
    cmap='viridis_r',
    norm=LogNorm(vmin=qmin, vmax=qmax),
    )
plt.colorbar(p2)
filename = directory + "/adj_matrix_y=" + str(ay) + "_adv_ypred=" + str(by_pred) + "_" + str(epsilon) +  "_" + str(args.attack_type) + ".png"
plt.savefig(filename, dpi=800)
plt.close()

# Clean adj matrix of class the one adv predicted
p3 = plt.matshow(c, 
    #vmin=20500, vmax=30000, 
    cmap='viridis_r',
    norm=LogNorm(vmin=qmin, vmax=qmax),
    )
plt.colorbar(p3)
filename = directory + "/adj_matrix_y=" + str(ay) + "_clean" + ".png"
plt.savefig(filename, dpi=800)
plt.close()

# Diff of adj matrices between clean and adv
diff_mat2 = (a - b)
print("Extreme values =", diff_mat2.min(), diff_mat2.max())
print("Quantile =", np.quantile(diff_mat2, 0.001), np.quantile(diff_mat2, 0.01), np.quantile(diff_mat2, 0.05), np.quantile(diff_mat2, 0.1), np.quantile(diff_mat2, 0.2), np.quantile(diff_mat2, 0.5), np.quantile(diff_mat2, 0.8), np.quantile(diff_mat2, 0.9), np.quantile(diff_mat2, 0.95), np.quantile(diff_mat2, 0.99), np.quantile(diff_mat2, 0.999))
print("Quantile v2 =", np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.01), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.05), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.1), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.2), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.8), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.9), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.95), np.quantile(diff_mat2[np.where(diff_mat2 != 0)], 0.99))
p4 = plt.matshow(diff_mat2,
    vmin=-qmax, vmax=qmax,
    cmap='viridis_r'
    #norm=SymLogNorm(linthresh=15000)
    )
plt.colorbar(p4)
filename1 = directory + "/diff_adj_matrix_y=" + str(ay) + "_adv_pred=" + str(by_pred) + "_clean_vs_" + str(args.attack_type) + ".png"
plt.savefig(filename1, dpi=800)
plt.close()

# Diff of adj matrices between clean and adv
diff_mat3 = (c - b)
print("Extreme values =", diff_mat3.min(), diff_mat3.max())
print("Quantile =", np.quantile(diff_mat3, 0.001), np.quantile(diff_mat3, 0.01), np.quantile(diff_mat3, 0.05), np.quantile(diff_mat3, 0.1), np.quantile(diff_mat3, 0.2), np.quantile(diff_mat3, 0.5), np.quantile(diff_mat3, 0.8), np.quantile(diff_mat3, 0.9), np.quantile(diff_mat3, 0.95), np.quantile(diff_mat3, 0.99), np.quantile(diff_mat3, 0.999))
print("Quantile v2 =", np.quantile(diff_mat3[np.where(diff_mat3 != 0)], 0.01), np.quantile(diff_mat3[np.where(diff_mat3 != 0)], 0.05), np.quantile(diff_mat3[np.where(diff_mat3 != 0)], 0.1), np.quantile(diff_mat3[np.where(diff_mat3 != 0)], 0.2), np.quantile(diff_mat3[np.where(diff_mat3 != 0)], 0.8), np.quantile(diff_mat3[np.where(diff_mat3 != 0)], 0.9), np.quantile(diff_mat3[np.where(diff_mat3 != 0)], 0.95), np.quantile(diff_mat3[np.where(diff_mat3 != 0)], 0.99))
p5 = plt.matshow(diff_mat3,
    vmin=-qmax, vmax=qmax,
    cmap='viridis_r'
    #norm=SymLogNorm(linthresh=15000)
    )
plt.colorbar(p5)
filename1 = directory + "/diff_adj_matrix_y=" + str(cy) + "_adv_pred=" + str(by_pred) + "_clean_vs_" + str(args.attack_type) + ".png"
plt.savefig(filename1, dpi=800)
plt.close()

# Diff of adj matrices of clean and clean of predicted class
diff_mat4 = (a - c)
p6 = plt.matshow(diff_mat4,
    vmin=-qmax, vmax=qmax,
    cmap='viridis_r'
    #norm=SymLogNorm(linthresh=15000)
    )
plt.colorbar(p6)
filename1 = directory + "/diff_adj_matrix_y=" + str(ay) + "_clean_adv_pred=" + str(cy) + ".png"
plt.savefig(filename1, dpi=800)
plt.close()

for layer in range(len(a_)):
    plt.hist(a_[layer], range=(np.quantile(a_[layer], 0.1), np.quantile(a_[layer], 0.9)), bins=50, density=False, alpha=0.4, label="Clean")
    plt.hist(b_[layer], range=(np.quantile(b_[layer], 0.1), np.quantile(b_[layer], 0.9)), bins=50, density=False, alpha=0.4, label="Adv")
    plt.savefig(directory + "/weight_distrib_layer_y=" + str(ay) + "_adv_pred=" + str(by_pred) + "_clean_vs_" + str(args.attack_type) + "_layer" + str(layer) + ".png", dpi=800)
    plt.close()

plt.subplot(1,3,1)
plt.imshow(ax.squeeze(0).detach().numpy(), cmap="gray")
plt.title(f"Clean {ay}")
plt.subplot(1,3,2)
plt.imshow(bx.squeeze(0).detach().numpy(), cmap="gray")
plt.title(f"Adv {ay} -> {by_pred}, Linf pert = {np.round(linf_pert,3)}")
plt.subplot(1,3,3)
plt.imshow(nx.squeeze(0).detach().numpy(), cmap="gray")
plt.title(f"Noisy {ny}, Linf pert = {np.round(nlinf_pert,3)}")
plt.savefig(directory + "/images_clean" + str(ay_pred) + "_vs_" + str(args.attack_type) + str(by_pred) + "eps=" + str(epsilon) + ".png", dpi=800)

end_time = time.time()

logger.info(f"Success in {end_time-start_time} seconds")
