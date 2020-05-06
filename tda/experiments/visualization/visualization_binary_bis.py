#!/usr/bin/env python
# coding: utf-8

import argparse
import io
import time
import re
import traceback
import typing
import pathlib
import pickle

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from joblib import delayed, Parallel
from r3d3.experiment_db import ExperimentDB
from sklearn.decomposition import PCA

from tda.embeddings import get_embedding, EmbeddingType, KernelType, ThresholdStrategy
from tda.embeddings.raw_graph import identify_active_indices, featurize_vectors
from tda.embeddings.weisfeiler_lehman import NodeLabels
from tda.models import get_deep_model, Dataset
from tda.models.architectures import mnist_mlp, get_architecture
from tda.protocol import get_protocolar_datasets, evaluate_embeddings
from tda.rootpath import db_path
from tda.tda_logging import get_logger
from tda.threshold_underoptimized_edges import (
    process_thresholds_underopt,
    thresholdize_underopt_v2,
)
from tda.thresholds import process_thresholds
from tda.graph_stats import get_stats
from tda.graph_dataset import get_sample_dataset
from tda.rootpath import db_path, rootpath

logger = get_logger("Viz Bis")
start_time = time.time()

my_db = ExperimentDB(db_path=db_path)


class Config(typing.NamedTuple):
    # Type of embedding to use
    embedding_type: str
    # Type of kernel to use on the embeddings
    kernel_type: str
    # High threshold for the edges of the activation graph
    thresholds: str
    # Which thresholding strategy should we use
    threshold_strategy: str
    # Are the threshold low pass or not
    thresholds_are_low_pass: bool
    # Underoptimized threshold or normal threshold?
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
    # PCA Parameter for RawGraph (-1 = No PCA)
    raw_graph_pca: int
    l2_norm_quantile: bool = True
    sigmoidize: bool = False
    # Pruning
    first_pruned_iter : int = 10
    prune_percentile : float = 0.0
    tot_prune_percentile : float = 0.0
    # Default parameters when running interactively for instance
    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0
    # Number of processes to spawn
    n_jobs: int = 1

    all_epsilons: typing.List[float] = None

    @property
    def viz_path(self):
        directory = f"{rootpath}/viz/{self.experiment_id}/"
        pathlib.Path(directory).mkdir(exist_ok=True, parents=True)
        return directory

def str2bool(value):
    if value in [True, "True", 'true']:
        return True
    else:
        return False


def get_config() -> Config:
    parser = argparse.ArgumentParser(
        description="Transform a dataset in pail files to tf records."
    )
    parser.add_argument("--experiment_id", type=int, default=-1)
    parser.add_argument("--run_id", type=int, default=-1)
    parser.add_argument(
        "--embedding_type", type=str, default=EmbeddingType.PersistentDiagram
    )
    parser.add_argument("--kernel_type", type=str, default=KernelType.SlicedWasserstein)
    parser.add_argument("--thresholds", type=str, default="0:0.0_2:0.1_4:0.0_5:0.0") # "0:0.05_2:0.05_4:0.05_5:0.0"
    parser.add_argument(
        "--threshold_strategy", type=str, default=ThresholdStrategy.UnderoptimizedMagnitudeIncrease
    )
    parser.add_argument("--height", type=int, default=1)
    parser.add_argument("--hash_size", type=int, default=100)
    parser.add_argument("--node_labels", type=str, default=NodeLabels.NONE)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--architecture", type=str, default=mnist_mlp.name)
    parser.add_argument("--train_noise", type=float, default=0.0)
    parser.add_argument("--dataset_size", type=int, default=100)
    parser.add_argument("--successful_adv", type=int, default=1)
    parser.add_argument("--raw_graph_pca", type=int, default=-1)
    parser.add_argument("--attack_type", type=str, default="FGSM")
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--all_epsilons", type=str, default=None)
    parser.add_argument("--l2_norm_quantile", type=bool, default=True)
    parser.add_argument("--sigmoidize", type=str2bool, default=False)
    parser.add_argument("--thresholds_are_low_pass", type=bool, default=True)
    parser.add_argument("--first_pruned_iter", type=int, default=10)
    parser.add_argument("--prune_percentile", type=float, default=0.0)
    parser.add_argument("--tot_prune_percentile", type=float, default=0.0)

    args, _ = parser.parse_known_args()

    if args.all_epsilons is not None:
        args.all_epsilons = list(map(float, str(args.all_epsilons).split(";")))
    return Config(**args.__dict__)


#######################
# Fetching the points #
#######################

base_dataset_size = 500

def all_info(config: Config):
    architecture = get_architecture(config.architecture)
    dataset = Dataset.get_or_create(name=config.dataset)
    architecture = get_deep_model(
        num_epochs=config.epochs,
        dataset=dataset,
        architecture=architecture,
        train_noise=config.train_noise,
        prune_percentile=config.prune_percentile,
        tot_prune_percentile=config.tot_prune_percentile,
        first_pruned_iter=config.first_pruned_iter,
    )
    if config.sigmoidize:
        logger.info(f"Using inter-class regularization (sigmoid)")
        all_weights = get_stats(
            dataset=dataset, architecture=architecture, dataset_size=100
        )
    else:
        all_weights = None

    thresholds = None
    edges_to_keep = None

    if config.threshold_strategy == ThresholdStrategy.ActivationValue:
        thresholds = process_thresholds(
            raw_thresholds=config.thresholds,
            dataset=dataset,
            architecture=architecture,
            dataset_size=100,
        )
    elif config.threshold_strategy == ThresholdStrategy.QuantilePerGraphLayer:
        thresholds = config.thresholds.split("_")
        thresholds = [val.split(";") for val in thresholds]
        thresholds = {
            (int(start), int(end)): float(val) for (start, end, val) in thresholds
        }
        logger.info(f"Using thresholds per graph {thresholds}")
    elif config.threshold_strategy in [
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        ThresholdStrategy.UnderoptimizedLargeFinal,
        ThresholdStrategy.UnderoptimizedMagnitudeIncreaseNbEdges,
        ThresholdStrategy.UnderoptimizedLargeFinalNbEdges
    ]:
        edges_to_keep = process_thresholds_underopt(
            raw_thresholds=config.thresholds,
            architecture=architecture,
            method=config.threshold_strategy,
        )
        logger.info(f"Number of edges to keep = {[len(edges_to_keep[k]) for k in edges_to_keep.keys()]}")
    elif config.threshold_strategy in [
        ThresholdStrategy.UnderoptimizedMagnitudeIncreaseV2,
        ThresholdStrategy.UnderoptimizedLargeFinalV2,
    ]:
        thresholdize_underopt_v2(
            raw_thresholds=config.thresholds,
            architecture=architecture,
            method=config.threshold_strategy,
        )

    if config.attack_type not in ["FGSM", "BIM"]:
        all_epsilons = [1.0]
    elif config.all_epsilons is None:
        all_epsilons = [0.01, 0.05, 0.1, 0.4, 1.0]
        # all_epsilons = [0.01]
    else:
        all_epsilons = config.all_epsilons

    return dataset, architecture, all_epsilons, all_weights, thresholds, edges_to_keep

def get_the_points(config: Config, label1, label2, val_idx,
    dataset, architecture, all_epsilons, all_weights, thresholds, edges_to_keep):
    adv_label1_before = dict()
    for epsilon in all_epsilons:
        adv_label1_before[epsilon] = get_sample_dataset(
            adv=True,
            noise=0.0,
            dataset=dataset,
            train=False,
            succ_adv=config.successful_adv,
            archi=architecture,
            attack_type=config.attack_type,
            epsilon=epsilon,
            num_iter=100,
            dataset_size=base_dataset_size,
            offset=0,
            compute_graph=True,
        )

    # Keep only the correct adv
    if label1 != None:
        adv_label1 = dict()
        sample_id_adv = dict()
        for epsilon in all_epsilons:
            adv_label1[epsilon] = list()
            for adv_input in adv_label1_before[epsilon]:
                logger.info(f"y = {adv_input.y} and pred = {adv_input.y_pred}")
                if adv_input.y == label1 and adv_input.y_pred != adv_input.y:
                    logger.info(f"Accepting point")
                    adv_label1[epsilon].append(adv_input)
                if len(adv_label1[epsilon]) == config.dataset_size:
                    break
            sample_id_adv[epsilon] = [adv.sample_id for adv in adv_label1[epsilon]]
            logger.info(f"Sample id = {sample_id_adv[epsilon]} and len adv = {len(adv_label1[epsilon])}")

        clean_label1 = dict()
        for epsilon in all_epsilons:
            clean_label1[epsilon] = list()
            for sample_id in sample_id_adv[epsilon]:
                clean_label1[epsilon] += get_sample_dataset(
                    adv=False,
                    noise=0.0,
                    dataset=dataset,
                    train=False,
                    succ_adv=config.successful_adv,
                    archi=architecture,
                    attack_type=config.attack_type,
                    epsilon=epsilon,
                    num_iter=100,
                    dataset_size=1,
                    offset=sample_id,
                    compute_graph=True,
                )
            logger.info(f"len clean = {len(clean_label1[epsilon])}")

    else:
        adv_label1 = adv_label1_before
        clean_label1 = dict()
        for epsilon in all_epsilons:
            clean_label1[epsilon] = get_sample_dataset(
                adv=False,
                noise=0.0,
                dataset=dataset,
                train=False,
                succ_adv=config.successful_adv,
                archi=architecture,
                attack_type=config.attack_type,
                epsilon=epsilon,
                num_iter=100,
                dataset_size=base_dataset_size,
                offset=0,
                compute_graph=True,
            )


    clean_label2 = list()
    clean_label2_ = list()
    if label2 == "auto":
        label2 = adv_label1[epsilon][val_idx].y_pred
    clean_label2_ += get_sample_dataset(
                adv=False,
                noise=0.0,
                dataset=dataset,
                train=False,
                succ_adv=config.successful_adv,
                archi=architecture,
                attack_type=config.attack_type,
                epsilon=epsilon,
                num_iter=100,
                dataset_size=100,
                offset=0,
                compute_graph=True,
            )
    logger.info(f"Label2 = {label2}")
    for clean_input in clean_label2_:
        if (clean_input.y == label2) and (clean_input.y_pred == label2):
            clean_label2.append(clean_input)
        if len(clean_label2) == config.dataset_size:
            break

    return all_epsilons, clean_label1, adv_label1, clean_label2


# Create the embeddings based on the datasets generated
def generate_embeddings(config, datas,
    dataset, architecture, all_epsilons, all_weights, thresholds, edges_to_keep):
    ret = list()
    for data in datas:
        ret.append(
            get_embedding(
                embedding_type=config.embedding_type,
                line=data,
                params={
                    "hash_size": int(config.hash_size),
                    "height": int(config.height),
                    "node_labels": config.node_labels,
                    "steps": config.steps,
                    "raw_graph_pca": config.raw_graph_pca,
                },
                architecture=architecture,
                dataset=dataset,
                thresholds=thresholds,
                edges_to_keep=edges_to_keep,
                threshold_strategy=config.threshold_strategy,
                all_weights_for_sigmoid=all_weights,
                thresholds_are_low_pass=config.thresholds_are_low_pass,
            )
        )
    return ret

def generate_new_pos(Gnodes, config):
    new_pos = dict()
    if config.dataset == "MNIST" and config.architecture == "mnist_lenet":
        nb_nodes_layer = [784, 5760, 1440, 1280, 320, 50, 10]
    elif config.dataset == "SVHN" and config.architecture == "svhn_lenet":
        nb_nodes_layer = [3072, 4704, 1176, 1600, 400, 120, 84, 10]
    cum_nodes_layer = [0] + list(np.cumsum(nb_nodes_layer))
    min_per_layer = list()
    for i in range(len(nb_nodes_layer)):
        list_layer = [node for node in Gnodes if node < cum_nodes_layer[i+1] and node >= cum_nodes_layer[i]]
        if len(list_layer) > 0:
            min_per_layer.append(min(list_layer))
        else:
            min_per_layer.append(0)
    for k in Gnodes:
        for i in range(len(nb_nodes_layer)):
            if k >= cum_nodes_layer[i] and k < cum_nodes_layer[i+1]:
                new_pos[k] = [i*200, k-min_per_layer[i]]

    return new_pos

def plot_graph(data, ax, config, all_weights, edges_to_keep):
    if data.graph is None:
        graph = Graph.from_architecture_and_data_point(
            architecture=architecture, x=data.x.double()
        )
    else:
        graph = data.graph

    if all_weights is not None:
        graph.sigmoidize(all_weights=all_weights)
    if config.threshold_strategy == ThresholdStrategy.ActivationValue:
        logger.info(f"Activ value")
        graph.thresholdize(thresholds=thresholds, low_pass=config.thresholds_are_low_pass)
    elif config.threshold_strategy in [
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        ThresholdStrategy.UnderoptimizedLargeFinal,
        ThresholdStrategy.UnderoptimizedMagnitudeIncreaseNbEdges,
        ThresholdStrategy.UnderoptimizedLargeFinalNbEdges
    ]:
        logger.info(f"Underopt edges and edges to keep len = {[len(edges_to_keep[k]) for k in edges_to_keep.keys()]}")
        graph.thresholdize_underopt(edges_to_keep)
    elif config.threshold_strategy == ThresholdStrategy.QuantilePerGraphLayer:
        logger.info(f"Quantile per graph")
        graph.thresholdize_per_graph(
            thresholds=thresholds, low_pass=config.thresholds_are_low_pass
        )

    G = graph.to_nx_graph()
    G.remove_nodes_from(list(nx.isolates(G)))
    Gedges = G.edges()
    Gnodes = list(G.nodes)
    nb_edges = G.number_of_edges()
    weight_edges = np.round(G.size(weight='weight'))
    logger.info(f"Number of edges = {nb_edges}")
    Gweights = [G[u][v]['weight']/(1*1e0) for u,v in Gedges]
    new_pos = generate_new_pos(Gnodes, config)

    return nx.draw(G, new_pos, node_size=0.1, node_color="r", edges=Gedges, width=Gweights, alpha=0.5, ax=ax), weight_edges, nb_edges

def plot_inputs_and_diagrams(config, all_epsilons, label1, label2,
    clean_label1, adv_label1, clean_label2,
    clean_label1_embedding, adv_label1_embedding, clean_label2_embedding,
    idx, all_weights, edges_to_keep):

    def replace_inf(list_tuple, replacing_val="auto"):
        if replacing_val == "auto":
            replacing_val = max([elem[1] if elem[1] < np.inf else 0 for elem in list_tuple])
        b = [(elem[0], replacing_val) if elem[1]==np.inf else (elem[0], elem[1]) for elem in list_tuple]
        return b

    def min_and_max(list_tuple):
        min0 = min([elem[0] if elem[0] < np.inf else 0 for elem in list_tuple])
        min1 = min([elem[1] if elem[1] < np.inf else 0 for elem in list_tuple])
        max0 = max([elem[0] if elem[0] < np.inf else 0 for elem in list_tuple])
        max1 = max([elem[1] if elem[1] < np.inf else 0 for elem in list_tuple])
        return [min(min0, min1), max(max0, max1)], [min(min0, min1), max(max0, max1)]

    def resize_x(x):
        if x.size()[0] == 1:
            y = x.squeeze(0)
        else:
            y = x.permute(1, 2, 0)
        return y.detach().numpy()

    for epsilon in all_epsilons:
        file_name = (
            config.viz_path
            + str(config.architecture)
            + str(config.epochs)
            + "_"
            + str(config.attack_type)
            + "_"
            + "label"+str(label1)
            + "_"
            + str(epsilon)
            + "_"
            + config.thresholds
            + ".png"
        )
        logger.info(f"file name = {file_name}")

        plt.style.use("ggplot")
        plt.figure(figsize=(20,10))
        plt.suptitle(f"{config.architecture} with threshold {config.thresholds}")
        ax1 = plt.subplot2grid((3,4), (0,0))
        ax1.set_title(f"Clean {clean_label1[epsilon][idx].y} (pred {clean_label1[epsilon][idx].y_pred})")
        ax1.axis('off')
        ax1.grid(False)
        ax1.imshow(resize_x(clean_label1[epsilon][idx].x), cmap="gray")
        ax2 = plt.subplot2grid((3,4), (0,1))
        ax2.set_title(f"Adv (pred {adv_label1[epsilon][idx].y_pred}, eps {epsilon})")
        ax2.axis('off')
        ax2.grid(False)
        ax2.imshow(resize_x(adv_label1[epsilon][idx].x), cmap="gray")
        ax3 = plt.subplot2grid((3,4), (0,2))
        ax3.set_title(f"Other clean {clean_label1[epsilon][idx+1].y}")
        ax3.axis('off')
        ax3.grid(False)
        ax3.imshow(resize_x(clean_label1[epsilon][idx+1].x), cmap="gray")
        ax4 = plt.subplot2grid((3,4), (0,3))
        ax4.set_title(f"Clean {clean_label2[idx].y}")
        ax4.axis('off')
        ax4.grid(False)
        ax4.imshow(resize_x(clean_label2[idx].x), cmap="gray")
        ax5 = plt.subplot2grid((3,4), (1,0))
        ax5.plot(*min_and_max(clean_label1_embedding[epsilon][idx]), "-")
        ax5.scatter(*zip(*replace_inf(clean_label1_embedding[epsilon][idx])), s=4)
        ax6 = plt.subplot2grid((3,4), (1,1))
        ax6.plot(*min_and_max(adv_label1_embedding[epsilon][idx]), "-")
        ax6.scatter(*zip(*replace_inf(adv_label1_embedding[epsilon][idx])), s=4)
        ax7 = plt.subplot2grid((3,4), (1,2))
        ax7.plot(*min_and_max(clean_label1_embedding[epsilon][idx+1]), "-")
        ax7.scatter(*zip(*replace_inf(clean_label1_embedding[epsilon][idx+1])), s=4)
        ax8 = plt.subplot2grid((3,4), (1,3))
        ax8.plot(*min_and_max(clean_label2_embedding[idx]), "-")
        ax8.scatter(*zip(*replace_inf(clean_label2_embedding[idx])), s=4)
        ax9 = plt.subplot2grid((3,4), (2,0))
        p, w, e = plot_graph(clean_label1[epsilon][idx], ax9, config, all_weights, edges_to_keep)
        ax9.text(0.5,-0.1, f"Nb edges = {e} and weight = {w}", size=12, ha="center", 
                transform=ax9.transAxes)
        p
        ax10 = plt.subplot2grid((3,4), (2,1))
        p, w, e = plot_graph(adv_label1[epsilon][idx], ax10, config, all_weights, edges_to_keep)
        ax10.text(0.5,-0.1, f"Nb edges = {e} and weight = {w}", size=12, ha="center", 
                transform=ax10.transAxes)
        p
        ax11 = plt.subplot2grid((3,4), (2,2))
        p, w, e = plot_graph(clean_label1[epsilon][idx+1], ax11, config, all_weights, edges_to_keep)
        ax11.text(0.5,-0.1, f"Nb edges = {e} and weight = {w}", size=12, ha="center", 
                transform=ax11.transAxes)
        p
        ax12 = plt.subplot2grid((3,4), (2,3))
        p, w, e = plot_graph(clean_label2[idx], ax12, config, all_weights, edges_to_keep)
        ax12.text(0.5,-0.1, f"Nb edges = {e} and weight = {w}", size=12, ha="center", 
                transform=ax12.transAxes)
        p

        plt.savefig(file_name, dpi=800)
        plt.close()

def compute_nb_edges(data, weight, config, all_weights, edges_to_keep, thresholds):
    if data.graph is None:
        graph = Graph.from_architecture_and_data_point(
            architecture=architecture, x=data.x.double()
        )
    else:
        graph = data.graph

    if config.threshold_strategy == ThresholdStrategy.ActivationValue:
        logger.info(f"Activ value")
        graph.thresholdize(thresholds=thresholds, low_pass=config.thresholds_are_low_pass)
    elif config.threshold_strategy in [
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        ThresholdStrategy.UnderoptimizedLargeFinal,
        ThresholdStrategy.UnderoptimizedMagnitudeIncreaseNbEdges,
        ThresholdStrategy.UnderoptimizedLargeFinalNbEdges,
    ]:
        logger.info(f"Underopt edges and edges to keep len = {[len(edges_to_keep[k]) for k in edges_to_keep.keys()]}")
        graph.thresholdize_underopt(edges_to_keep)
    elif config.threshold_strategy == ThresholdStrategy.QuantilePerGraphLayer:
        logger.info(f"Quantile per graph")
        graph.thresholdize_per_graph(
            thresholds=thresholds, low_pass=config.thresholds_are_low_pass
        )

    G = graph.to_nx_graph()
    G.remove_nodes_from(list(nx.isolates(G)))
    if weight == False:
        nb_edges = G.number_of_edges()
    elif weight == True:
        nb_edges = G.size(weight='weight')

    return nb_edges

def hist_nb_edges(clean_label1, adv_label1, label1, weight,
    config, all_epsilons, all_weights, edges_to_keep, thresholds):
    logger.info(f"label his_nb_edgest = {label1}")
    nb_edges = dict()
    nb_edges["clean"] = dict()
    nb_edges["adv"] = dict()
    for epsilon in all_epsilons:
        file_name = (
            config.viz_path
            + str(config.architecture)
            + str(config.epochs)
            + "_"
            + str(config.attack_type)
            + "_"
            + "hist"
            + "_"
            + "label"+str(label1)
            + "_"
            + str(epsilon)
            + "_"
            + config.thresholds
            + ".pkl"
        )
        logger.info(f"file name = {file_name}")
        nb_edges["clean"][epsilon] = list()
        nb_edges["adv"][epsilon] = list()
        for clean_input in clean_label1[epsilon]:
            nb_edges["clean"][epsilon].append(compute_nb_edges(clean_input, weight, config, all_weights, edges_to_keep, thresholds))
        for adv_input in adv_label1[epsilon]:
            nb_edges["adv"][epsilon].append(compute_nb_edges(adv_input, weight, config, all_weights, edges_to_keep, thresholds))
    with open(file_name, 'wb') as f:
        pickle.dump(nb_edges, f, pickle.HIGHEST_PROTOCOL)

def launch_hist(config: Config, label1, weight=True, label2=0, val_idx=0):
    logger.info(f"label hist = {label1}")
    dataset, architecture, all_epsilons, all_weights, thresholds, edges_to_keep = all_info(config)

    all_epsilons, clean_label1, adv_label1, clean_label2 = get_the_points(config, label1, label2, val_idx,
        dataset, architecture, all_epsilons, all_weights, thresholds, edges_to_keep)

    hist_nb_edges(clean_label1, adv_label1, label1, weight,
    config, all_epsilons, all_weights, edges_to_keep, thresholds)

    state = "Done Hist"
    return state



def launch_viz(config: Config, label1=1, label2="auto", val_idx=0):
    dataset, architecture, all_epsilons, all_weights, thresholds, edges_to_keep = all_info(config)

    all_epsilons, clean_label1, adv_label1, clean_label2 = get_the_points(config, label1, label2, val_idx,
        dataset, architecture, all_epsilons, all_weights, thresholds, edges_to_keep)

    clean_label1_embedding = dict()
    adv_label1_embedding = dict()
    for epsilon in all_epsilons:
        clean_label1_embedding[epsilon] = generate_embeddings(config, clean_label1[epsilon],
            dataset, architecture, all_epsilons, all_weights, thresholds, edges_to_keep)
        adv_label1_embedding[epsilon] = generate_embeddings(config, adv_label1[epsilon],
            dataset, architecture, all_epsilons, all_weights, thresholds, edges_to_keep)
    clean_label2_embedding = generate_embeddings(config, clean_label2,
        dataset, architecture, all_epsilons, all_weights, thresholds, edges_to_keep)

    plot_inputs_and_diagrams(config, all_epsilons, label1, label2,
    clean_label1, adv_label1, clean_label2,
    clean_label1_embedding, adv_label1_embedding, clean_label2_embedding,
    val_idx, all_weights, edges_to_keep)

    state = "Done Viz"
    return state


if __name__ == "__main__":
    my_config = get_config()
    s = launch_viz(my_config, label1=2, label2="auto", val_idx=1) 
    #s = launch_hist(my_config, label1=None, weight=True) 
    logger.info(s)
    
