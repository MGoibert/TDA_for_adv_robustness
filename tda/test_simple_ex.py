import argparse
import io
import time
import traceback
from typing import NamedTuple, List
import torch
import networkx as nx

import numpy as np
from joblib import delayed, Parallel
from sklearn.decomposition import PCA

from tda.dataset.adversarial_generation import AttackType, AttackBackend
from tda.embeddings import (
    get_embedding,
    EmbeddingType,
    KernelType,
    ThresholdStrategy,
    Embedding,
)
from tda.embeddings.raw_graph import identify_active_indices, featurize_vectors
from tda.graph_stats import get_quantiles_helpers
from tda.models import get_deep_model, Dataset
from tda.models.architectures import mnist_mlp, get_architecture
from tda.protocol import get_protocolar_datasets, evaluate_embeddings
from tda.tda_logging import get_logger
from tda.threshold_underoptimized_edges import process_thresholds_underopt
from tda.rootpath import rootpath
from tda.embeddings.raw_graph import to_sparse_vector
from tda.graph import Graph
from tda.dataset.graph_dataset import DatasetLine
from tda.embeddings.persistent_diagrams import (
    sliced_wasserstein_kernel,
    compute_dgm_from_graph,
)
import pathlib
from itertools import cycle, islice
import matplotlib.pyplot as plt
from operator import itemgetter
from numpy import inf

logger = get_logger("Simple example")
start_time = time.time()


class Config(NamedTuple):
    # Type of embedding to use
    embedding_type: str
    # Type of kernel to use on the embeddings
    kernel_type: str
    # High threshold for the edges of the activation graph
    thresholds: str
    # Which thresholding strategy should we use
    threshold_strategy: str
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
    # Type of attack (FGSM, PGD, CW)
    attack_type: str
    # Type of attack (CUSTOM, ART, FOOLBOX)
    attack_backend: str
    # Parameter used by DeepFool and CW
    num_iter: int
    # PCA Parameter for RawGraph (-1 = No PCA)
    raw_graph_pca: int
    # Whether to use pre-saved adversarial examples or not
    transfered_attacks: bool = False
    l2_norm_quantile: bool = True
    sigmoidize: bool = False
    # Pruning
    first_pruned_iter: int = 10
    prune_percentile: float = 0.0
    tot_prune_percentile: float = 0.0
    # Default parameters when running interactively for instance
    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0
    # Number of processes to spawn
    n_jobs: int = 1

    all_epsilons: List[float] = None

    @property
    def result_path(self):
        directory = f"{rootpath}/plots/toy_viz/{self.experiment_id}/"
        pathlib.Path(directory).mkdir(exist_ok=True, parents=True)
        return directory


def str2bool(value):
    if value in [True, "True", "true"]:
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
    parser.add_argument("--thresholds", type=str, default="0")
    parser.add_argument(
        "--threshold_strategy",
        type=str,
        default=ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
    )
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--architecture", type=str, default=mnist_mlp.name)
    parser.add_argument("--train_noise", type=float, default=0.0)
    parser.add_argument("--dataset_size", type=int, default=500)
    parser.add_argument("--successful_adv", type=int, default=1)
    parser.add_argument("--raw_graph_pca", type=int, default=-1)
    parser.add_argument("--attack_type", type=str, default=AttackType.FGSM)
    parser.add_argument("--attack_backend", type=str, default=AttackBackend.FOOLBOX)
    parser.add_argument("--transfered_attacks", type=str2bool, default=False)
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--all_epsilons", type=str, default=None)
    parser.add_argument("--l2_norm_quantile", type=bool, default=True)
    parser.add_argument("--first_pruned_iter", type=int, default=10)
    parser.add_argument("--prune_percentile", type=float, default=0.0)
    parser.add_argument("--tot_prune_percentile", type=float, default=0.0)

    args, _ = parser.parse_known_args()

    if args.all_epsilons is not None:
        args.all_epsilons = list(map(float, str(args.all_epsilons).split(";")))

    if args.embedding_type == EmbeddingType.PersistentDiagram:
        args.kernel_type = KernelType.SlicedWasserstein
        args.sigmoidize = False
    elif args.embedding_type == EmbeddingType.RawGraph:
        args.kernel_type = KernelType.RBF
        args.sigmoidize = True

    logger.info(args.__dict__)

    return Config(**args.__dict__)



# Plan !

#DONE 1) Get the dataset, the trained model, and the adversarial datasets
#DONE 2) Plot the dataset
#DONE 3) Get the attack perf
#DONE 4) Plot the adversaries
#DONE 5) Plot decision boundary
# 6) Get model parameters
# 7) Persistent diagram and Raw Graph

plt.style.use('ggplot') 

def get_all_inputs(config: Config, myeps=None):
    # Dataset and architecture
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

    if config.threshold_strategy in [
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        ThresholdStrategy.UnderoptimizedLargeFinal,
        ThresholdStrategy.UnderoptimizedRandom,
    ]:
        edges_to_keep = process_thresholds_underopt(
            raw_thresholds=config.thresholds,
            architecture=architecture,
            method=config.threshold_strategy,
        )
        architecture.threshold_layers(edges_to_keep)

    if config.attack_type not in ["FGSM", "PGD"]:
        all_epsilons = [1.0]
    elif (config.all_epsilons is None) and (myeps is None):
        all_epsilons = [0.05]
    elif myeps != None:
        all_epsilons = myeps
    else:
        all_epsilons = config.all_epsilons
    logger.info(f"all epsilons = {all_epsilons} !!")

    # Get the protocolar datasets --> clean and adv datasets
    train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
        noise=config.noise,
        dataset=dataset,
        succ_adv=False,#config.successful_adv > 0,
        archi=architecture,
        dataset_size=config.dataset_size,
        attack_type=config.attack_type,
        attack_backend=config.attack_backend,
        all_epsilons=all_epsilons,
        compute_graph=False,
        transfered_attacks=config.transfered_attacks,
    )

    return architecture, train_clean, test_clean, train_adv, test_adv

def config_for_plot(config, train_clean, train_adv):
    mydata = list()
    color_variable = list()
    for line in train_clean:
        mydata.append(line.x)
        color_variable.append(line.y)
    mydata_adv = list()
    marker_variable = list()
    color_bis_variable = list()
    for line in train_adv[list(train_adv.keys())[0]]:
        mydata_adv.append(line.x)
        color_bis_variable.append(line.y)
        marker_variable.append(line.y_pred)
    return mydata, color_variable, mydata_adv, color_bis_variable, marker_variable

def plot_dataset(config, data, color_variable,
    data_bis=None, marker_variable=None, color_bis_variable=None,
    setup=None):
    if setup == None:
        setup_message = ""
    else:
        setup_message = "_"+str(setup)
    file_name = (
        config.result_path
        + str(config.dataset)
        + "_"
        + str(config.architecture)
        + str(config.epochs)
        + setup_message
        + ".png"
    )
    logger.info(f"file name for plot = {file_name}")

    mycolor = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(color_variable) + 1))))
    data1 = list(map(itemgetter(0), map(itemgetter(0), data)))
    data2 = list(map(itemgetter(1), map(itemgetter(0), data)))
    plt.scatter(data1, data2, s=15, color=mycolor[color_variable])
    if data_bis is not None:
        data_bis1 = list(map(itemgetter(0), map(itemgetter(0), data_bis)))
        data_bis2 = list(map(itemgetter(1), map(itemgetter(0), data_bis)))
        for i in range(len(data_bis1)):
            if marker_variable[i] > 0.5:
                plt.scatter(data_bis1[i], data_bis2[i], s=15, color=mycolor[color_bis_variable[i]],
                    marker="x", linewidths=1)
            else:
                plt.scatter(data_bis1[i], data_bis2[i], s=15, color=mycolor[color_bis_variable[i]],
                    marker="+", linewidths=1)
    plt.title("Toy dataset")
    plt.savefig(file_name, dpi=150)
    plt.close()

def plot_decision_boundary(config, model, our_data, color_variable):
    file_name = (
        config.result_path
        + str(config.dataset)
        + "_"
        + str(config.architecture)
        + str(config.epochs)
        + "_decision_boundary"
        + ".png"
    )
    data = list()
    for x in np.linspace(0.0, 1.0, 75):
        for y in np.linspace(0.0, 1.0, 75):
            data.append(torch.tensor([[x,y]]))

    for data_ in data:
        y_pred = torch.argmax(model(data_))
        if y_pred > 0.5:
            col = '#ff7f00'
        else:
            col = '#377eb8'
        plt.scatter(data_[0][0], data_[0][1], s=10,  marker="s", color=col, alpha=0.1)
    mycolor = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(color_variable) + 1))))
    data1 = list(map(itemgetter(0), map(itemgetter(0), our_data)))
    data2 = list(map(itemgetter(1), map(itemgetter(0), our_data)))
    plt.scatter(data1, data2, s=15, color=mycolor[color_variable])
    plt.title("Decision boundary")
    plt.savefig(file_name, dpi=150)
    plt.close()

def get_graphs_dgms(config, dataset, architecture, target=None, nb=2000, clean=True):
    line_list = list()
    for elem in dataset:
        line_list.append(elem)
        if (target != None) and (elem.y == target) and (clean*(elem.y_pred == target) or (not clean)*(elem.y_pred != target)):
            line_list.append(elem)
            if len(line_list) >= nb:
                break
        elif (target == None) and (clean*(elem.y == elem.y_pred) or (not clean)*(elem.y != elem.y_pred)):
            #logger.info(f"elem in dataset (adv = {elem.y_adv}) --> y = {elem.y} and pred = {elem.y_pred}")
            line_list.append(elem)
            if len(line_list) >= nb:
                break

    graph_list = list()
    dgm_list = list()
    for line in line_list:
        graph = Graph.from_architecture_and_data_point(
                architecture=architecture, x=line.x.double()
            )
        graph_list.append(graph)
        dgm_list.append(compute_dgm_from_graph(graph))

    return line_list, graph_list, dgm_list

def plot_graph(config, lines_clean, lines_adv, graphs_clean, graphs_adv):
    #plt.style.use('dark_background')
    plt.style.use('seaborn')
    file_name = (
        config.result_path
        + str(config.dataset)
        + "_"
        + str(config.architecture)
        + str(config.epochs)
        + "_graph"
        + ".png"
    )

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,16))
    ax = axes.flatten()

    for i in range(9):
        if i < 6:
            adj = graphs_clean[i].get_adjacency_matrix()
            subplot_title = f'clean {lines_clean[i].y} (pred {lines_clean[i].y_pred}) and x = {lines_clean[i].x.detach().numpy()}'
        else:
            adj = graphs_adv[i-6].get_adjacency_matrix()
            subplot_title = f'adv {lines_adv[i-6].y} (pred {lines_adv[i-6].y_pred}) and x = {lines_adv[i-6].x.detach().numpy()}'
            #adj = adj.to_dense()

        G = nx.from_scipy_sparse_matrix(adj)
        # Position for mlp3
        pos = {0:(-1,4), 1:(-1,-4), 22:(1,4), 23:(1,-4)}
        for j in range(2,22):
            pos[j]=(0,j-12+0.5)

        # Position for mlp2
        #pos={0:(-1,1), 1:(-1,-1),
        #2:(0,3.5), 3:(0,2.5), 4:(0,1.5), 5:(0,0.5), 6:(0,-0.5), 7:(0,-1.5), 8:(0,-2.5), 9:(0,-3.5),
        #10:(1,1), 11:(1,-1)}  
        nx.set_node_attributes(G, pos, 'coord') 
        widths = nx.get_edge_attributes(G, 'weight')
        nodelist = G.nodes()
        nx.draw_networkx_nodes(G,pos,
                           nodelist=nodelist,
                           node_size=1000,
                           node_color='black',
                           alpha=0.7,
                           ax=ax[i])
        nx.draw_networkx_edges(G,pos,
                           edgelist = widths.keys(),
                           width=[val/10**5 for val in list(widths.values())],
                           edge_color='navy',#'lightblue',
                           alpha=0.8,
                           ax=ax[i])
        #names = {0:"x1", 1:"x2", 2:"n1", 3:"n2", 4:"n3", 5:"n4", 6:"n5", 7:"n6", 8:"n7", 9:"n8", 10:"B", 11:"O"}
        #dict(zip(nodelist,nodelist))
        nx.draw_networkx_labels(G, pos=pos,
                            labels=dict(zip(nodelist,nodelist)),#names,
                            font_color='white',
                            ax=ax[i])
        ax[i].title.set_text(subplot_title)
        ax[i].grid(False)
    plt.savefig(file_name, dpi=300)
    plt.close()

def plot_dgms(config, lines_clean, lines_adv, dgms_clean, dgms_adv):
    plt.style.use('seaborn')
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.architecture)
            + str(config.epochs)
            + "_dgm"
            + ".png"
        )

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,16))
    ax = axes.flatten()

    def get_dgm_item(dgms, i):
        dgm_ = [(elem0/10**5, elem1/10**5) for (elem0, elem1) in dgms[i]]
        dgm_ = [(elem0, 0) if elem1 == inf else (elem0, elem1) for (elem0, elem1) in dgm_]
        return dgm_

    for i in range(9):
        if i < 6:
            dgm_ = [(elem0/10**5, elem1/10**5) for (elem0, elem1) in dgms_clean[i]]
            dgm_ = [(elem0, 0) if elem1 == inf else (elem0, elem1) for (elem0, elem1) in dgm_]
            subplot_title = f'clean {lines_clean[i].y} (pred {lines_clean[i].y_pred}) and nb pts = {len(dgm_)}'# and x = {lines_clean[i].x.detach().numpy()}'
        else:
            dgm_ = [(elem0/10**5, elem1/10**5) for (elem0, elem1) in dgms_adv[i-6]]
            dgm_ = [(elem0, 0) if elem1 == inf else (elem0, elem1) for (elem0, elem1) in dgm_]
            subplot_title = f'adv {lines_adv[i-6].y} (pred {lines_adv[i-6].y_pred}) and nb pts = {len(dgm_)}'#' and x = {lines_adv[i-6].x.detach().numpy()}'

        ax[i].scatter(list(map(itemgetter(0), dgm_)), list(map(itemgetter(1), dgm_)), s=25, alpha=0.5)
        ax[i].plot([-30,0],[-30,0], "-", color="black")
        ax[i].set_xlim([-30,0.1])
        ax[i].set_ylim([-30,0.1])
        ax[i].title.set_text(subplot_title)
    plt.savefig(file_name, dpi=200)
    plt.close()

    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.architecture)
            + str(config.epochs)
            + "_dgm2"
            + ".png"
        )
    plt.scatter(list(map(itemgetter(0), get_dgm_item(dgms_clean, 0))),
        list(map(itemgetter(1), get_dgm_item(dgms_clean, 0))), s=25, color="orange", alpha=0.5)
    plt.scatter(list(map(itemgetter(0), get_dgm_item(dgms_clean, 3))),
        list(map(itemgetter(1), get_dgm_item(dgms_clean, 3))), s=25, color="blue", alpha=0.5)
    plt.scatter(list(map(itemgetter(0), get_dgm_item(dgms_adv, 0))),
        list(map(itemgetter(1), get_dgm_item(dgms_adv, 0))), s=25, color="gray", alpha=0.5)
    #plt.scatter(list(map(itemgetter(0), get_dgm_item(dgms_adv, 2))),
    #    list(map(itemgetter(1), get_dgm_item(dgms_adv, 2))), s=75, color="green")
    plt.plot([-100,0],[-100,0], "-", color="black")
    plt.xlim(-30,0.1)
    plt.ylim(-30,0.1)
    plt.title("Comparision of dgms")
    plt.savefig(file_name, dpi=200)
    plt.close()


def get_correct_dgms(lines, graphs, dgms, target, target_pred=None):
    lines_ = list()
    graphs_ = list()
    dgms_ = list()
    for i in range(len(lines)):
        if (lines[i].y == target) and (target_pred == None or lines[i].y_pred == target_pred):
            lines_.append(lines[i])
            graphs_.append(graphs[i])
            dgms_.append(dgms[i])
    return lines_, graphs_, dgms_



def nb_pts_dgms(dgms):  
    nb_pts = [len(dgm) for dgm in dgms]
    logger.info(f"len dgm for nb_pts = {len(nb_pts)}")
    return nb_pts

def plot_nb_pts_dgms(config, nb1, nb2=None, nb3=None):
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.architecture)
            + str(config.epochs)
            + "_hist_dgms"
            + ".png"
        )

    plt.hist(nb1, color="lightskyblue", alpha=0.5)
    if nb2 != None:
        plt.hist(nb2, color="steelblue", alpha=0.5)
    if nb3 != None:
        plt.hist(nb3, color="firebrick", alpha=0.5)
    plt.title("Number of points in dgms")
    plt.savefig(file_name, dpi=200)
    plt.close()

def stat_nb_pts_dgms(config, nb1, nb2=None, nb3=None):
    m1 = np.mean(nb1)
    yerr1 = np.array([m1-np.quantile(nb1,0.1), np.quantile(nb1,0.9)-m1])
    if nb2 == None and nb3 == None:
        return m1, yerr1
    elif (nb2 != None) and (nb3 == None):
        m2 = np.mean(nb2)
        yerr2 = np.array([m2-np.quantile(nb2,0.1), np.quantile(nb2,0.9)-m2])
        return m1, yerr1, m2, yerr2
    elif nb3 != None:
        m3 = np.mean(nb3)
        yerr3 = np.array([m3-np.quantile(nb3,0.1), np.quantile(nb3,0.9)-m3])
        return m1, yerr1, m2, yerr2, m3, yerr3

def plot_errorbar_pts_dmgs(config, eps, m1s, yerr1s, m2s=None, yerr2s=None, m3s=None, yerr3s=None):
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.architecture)
            + str(config.epochs)
            + "_errorbar_dgms"
            + ".png"
        )
    plt.figure(figsize=(12,5))
    plt.errorbar(eps, m1s, yerr=yerr1s, fmt="-", markersize=20, linewidth=3,
        elinewidth=3, label="Clean") # color="lightskyblue",
    if m2s != None:
        plt.errorbar(eps, m2s, yerr=yerr2s, fmt="o-", markersize=20, linewidth=3,
        elinewidth=3, label="Adv") # color="firebrick",
    if m3s != None:
        plt.errorbar(eps, m3s, yerr=yerr3s, fmt="o-", color="steelblue", label="Clean bis")
    plt.legend(fontsize=25)
    plt.title(r'Number of points in PD as a function $\varepsilon$', fontsize=28)
    plt.xlabel(r'Perturbation strenght ($\varepsilon$)', fontsize=25)
    plt.ylabel("Number of points", fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(file_name, dpi=200)
    plt.close()


def run_experiment2(config: Config):

    #### 1) Get everything
    #architecture, train_clean, test_clean, train_adv, test_adv = get_all_inputs(config)

    #### 2) Plot dataset
    #mydata, color_variable, mydata_adv, color_bis_variable, marker_variable = config_for_plot(config, train_clean, train_adv)
    #plot_dataset(config, mydata, color_variable, setup=None)

    #### 3) Get adversarial accuracy
    #import tda.experiments.attack_performance.attacks_performance_binary as att_perf
    #accuracies = att_perf.get_all_accuracies(config, with_image=False)
    #att_perf.plot_and_save(config, accuracies)

    #### 4) Plot adversaries
    #plot_dataset(config, mydata, color_variable, mydata_adv,
    #    marker_variable, color_bis_variable, setup="adv")

    #### 5) Decision boundary
    #plot_decision_boundary(config, architecture, mydata, color_variable)

    #### 6) Model parameters
    #for name, param in architecture.named_parameters():
    #    if param.requires_grad:
    #        logger.info(f"{name}: {param.data}")

    #### 7) Graph and PD of some inputs
    #lines_clean1, graphs_clean1, dgms_clean1 = get_graphs_dgms(config,
    #    train_clean, architecture, target=None, clean=True)
    #lines_clean2, graphs_clean2, dgms_clean2 = get_graphs_dgms(config,
    #    train_clean, architecture, target=None, clean=True)
    #lines_adv, graphs_adv, dgms_adv = get_graphs_dgms(config,
    #    train_adv[list(train_adv.keys())[0]], architecture, target=None, clean=False)
    #i = 0
    #plot_graph(config, lines_clean1[i:i+3]+lines_clean2[i:i+3], lines_adv[i:i+3],
    #    graphs_clean1[i:i+3]+graphs_clean2[i:i+3], graphs_adv[i:i+3])
    #plot_dgms(config, lines_clean1_[i:i+3]+lines_clean2_[i:i+3], lines_adv_[i:i+3],
    #    dgms_clean1_[i:i+3]+dgms_clean2_[i:i+3], dgms_adv_[i:i+3])


    #### 8) Graph of number of points in dgms for specific class
    #l1, g1, d1 = get_correct_dgms(lines_clean1, graphs_clean1, dgms_clean1, 7)
    #l2, g2, d2 = get_correct_dgms(lines_clean1, graphs_clean1, dgms_clean1, 6)
    #l3, g3, d3 = get_correct_dgms(lines_adv, graphs_adv, dgms_adv, 7, target_pred=6)
    #nb1 = nb_pts_dgms(d1)
    #nb2 = nb_pts_dgms(d2)
    #nb3 = nb_pts_dgms(d3)
    #plot_nb_pts_dgms(config, nb1, nb2=nb2, nb3=nb3)

    #### 9) Graph of number of points in dgms for all class
    #nb1 = nb_pts_dgms(dgms_clean1)
    #nb3 = nb_pts_dgms(dgms_adv)
    #plot_nb_pts_dgms(config, nb1, nb2=None, nb3=nb3)

    #### 10) Nb pts as a function of one epsilon
    #eps = [0.05]
    #m1, yerr1, m2, yerr2 = stat_nb_pts_dgms(config, nb1, nb2=nb3, nb3=None)
    #plot_errorbar_pts_dmgs(config, eps, [m1], np.asarray(yerr1).reshape(2,-1),
    #    [m2], np.asarray(yerr2).reshape(2,-1))


    #### 11) Nb pts as a function of several eps
    myeps = [0.01, 0.05, 0.1, 0.25, 0.4]
    architecture, train_clean, test_clean, train_adv, test_adv = get_all_inputs(config, myeps)
    lines_c, graphs_c, dgms_c = get_graphs_dgms(config, train_clean+test_clean, architecture,
        target=None, clean=True)
    m1s = list()
    m2s = list()
    yerr1s = list()
    yerr2s = list()
    for eps in myeps:
        nb_c = nb_pts_dgms(dgms_c)
        lines_a, graphs_a, dgms_a = get_graphs_dgms(config,
            train_adv[eps]+test_adv[eps],
            architecture, target=None, clean=False)
        nb_a = nb_pts_dgms(dgms_a)
        m1, yerr1, m2, yerr2 = stat_nb_pts_dgms(config, nb_c, nb2=nb_a)
        m1s.append(m1)
        m2s.append(m2)
        yerr1s.append(yerr1)
        yerr2s.append(yerr2)
    yerr1s = np.asarray(yerr1s).reshape(-1,2).T
    yerr2s = np.asarray(yerr2s).reshape(-1,2).T
    plot_errorbar_pts_dmgs(config, myeps, m1s, yerr1s, m2s, yerr2s)






if __name__ == "__main__":
    my_config = get_config()
    try:
        run_experiment2(my_config)
    except Exception as e:
        my_trace = io.StringIO()
        traceback.print_exc(file=my_trace)

        logger.error(my_trace.getvalue())