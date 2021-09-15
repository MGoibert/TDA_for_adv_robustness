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
from tda.dataset.graph_dataset import DatasetLine, process_sample
from tda.embeddings.persistent_diagrams import (
    sliced_wasserstein_kernel,
    compute_dgm_from_graph,
)
from tda.devices import device
import pathlib
from itertools import cycle, islice
import matplotlib.pyplot as plt
from operator import itemgetter
from numpy import inf
from scipy.sparse import coo_matrix

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

plt.style.use('seaborn-dark') 

def get_protocolar_datasets2(
        dataset,
        archi,
        all_epsilons,
        offset=0,
        dataset_size=20,
        attack_type="FGSM",
        num_iter=20,
        attack_backend=AttackBackend.FOOLBOX):

    source_dataset = dataset.test_and_val_dataset
    source_dataset_size = len(source_dataset)
    current_sample_id = offset
    compute_graph = False
    final_dataset_clean = list()
    final_dataset_adv = dict()
    batch_size = dataset_size

    while len(final_dataset_clean)<dataset_size and current_sample_id < source_dataset_size:
        samples = None
        processed_samples = None
        y_pred = None

        batch = source_dataset[current_sample_id:current_sample_id+batch_size]
        if isinstance(batch[0], DatasetLine):
            x = torch.cat([torch.unsqueeze(s.x, 0) for s in batch], 0).to(device)
            y = np.array([s.y for s in batch])
            logger.info(f"shape of x = {x.shape}")
        else:
            x = torch.cat([torch.unsqueeze(s[0], 0) for s in batch], 0).to(device)
            y = np.array([s[1] for s in batch])
        samples = (x, y)

        processed_samples_clean = process_sample(
                sample=samples,
                adversarial=False,
                noise=0.0,
                epsilon=0.0,
                model=archi,
                attack_type=attack_type,
                num_iter=num_iter,
                attack_backend=attack_backend,
            )

        current_sample_id += batch_size
        assert (samples[1] == processed_samples_clean[1]).all()
        y_pred = archi(processed_samples_clean[0]).argmax(dim=-1).cpu().numpy()

        for i in range(len(processed_samples_clean[1])):
            x = torch.unsqueeze(processed_samples_clean[0][i], 0).double()
            graph = (
                Graph.from_architecture_and_data_point(architecture=archi, x=x)
                if compute_graph
                else None
            )
            final_dataset_clean.append(
                DatasetLine(
                    graph=graph,
                    x=x,
                    y=processed_samples_clean[1][i],
                    y_pred=y_pred[i],
                    y_adv=False,
                    l2_norm=None,
                    linf_norm=None,
                    sample_id=current_sample_id,
                )
            )

        for epsilon in all_epsilons:
            logger.info(f"processing eps={epsilon}")
            final_dataset_adv[epsilon] = list()
            processed_samples_adv = process_sample(
                sample=samples,
                adversarial=True,
                noise=0.0,
                epsilon=epsilon,
                model=archi,
                attack_type=attack_type,
                num_iter=num_iter,
                attack_backend=attack_backend,
            )

            current_sample_id += batch_size
            assert (samples[1] == processed_samples_adv[1]).all()
            y_pred = archi(processed_samples_adv[0]).argmax(dim=-1).cpu().numpy()

            for i in range(len(processed_samples_adv[1])):
                x = torch.unsqueeze(processed_samples_adv[0][i], 0).double()
                graph = (
                    Graph.from_architecture_and_data_point(architecture=archi, x=x)
                    if compute_graph
                    else None
                )
                final_dataset_adv[epsilon].append(
                DatasetLine(
                    graph=graph,
                    x=x,
                    y=processed_samples_adv[1][i],
                    y_pred=y_pred[i],
                    y_adv=True,
                    l2_norm=None,
                    linf_norm=None,
                    sample_id=current_sample_id,
                )
            )

    return final_dataset_clean, final_dataset_adv

    

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
    #train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
    #    noise=config.noise,
    #    dataset=dataset,
    #    succ_adv=False,#config.successful_adv > 0,
    #    archi=architecture,
    #    dataset_size=config.dataset_size,
    #    attack_type=config.attack_type,
    #    attack_backend=config.attack_backend,
    #    all_epsilons=all_epsilons,
    #    compute_graph=False,
    #    transfered_attacks=config.transfered_attacks,
    #)

    # Get the protocolar datasets --> clean and adv datasets
    test_clean, test_adv = get_protocolar_datasets2(
        dataset=dataset,
        archi=architecture,
        all_epsilons=all_epsilons,
        offset=0,
        dataset_size=config.dataset_size,
        attack_type=config.attack_type,
        num_iter=config.num_iter,
        attack_backend=config.attack_backend,
    )
    train_clean=None
    train_adv=None

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

def get_graphs_dgms(config, dataset, architecture, target=None, target_adv=None, nb=2000, clean=True):
    line_list = list()
    for elem in dataset:
        if (target != None) and (target != False) and (elem.y == target) and (clean*(elem.y_pred == target) or (not clean)*(elem.y_pred == target_adv)):
            line_list.append(elem)
            if len(line_list) >= nb:
                break
        elif (target == None) and (clean*(elem.y == elem.y_pred) or (not clean)*(elem.y != elem.y_pred)):
            #logger.info(f"elem in dataset (adv = {elem.y_adv}) --> y = {elem.y} and pred = {elem.y_pred}")
            line_list.append(elem)
            if len(line_list) >= nb:
                break
        elif (target == False) :
            #logger.info(f"Here")
            line_list.append(elem)
            if len(line_list) >= nb:
                break

    graph_list = list()
    dgm_list = None #list()
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

def plot_kde_dgms(config, lines_clean, lines_adv, dgms_clean, dgms_adv):
    plt.style.use('seaborn')
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.architecture)
            + str(config.epochs)
            + "_kde_dgm"
            + ".png"
        )

    birth_date = [dgm[0]/10**5 for dgm in dgms_clean]+[dgm[0]/10**5 for dgm in dgms_adv]
    death_date = [dgm[1]/10**5 for dgm in dgms_clean]+[dgm[1]/10**5 for dgm in dgms_adv]
    min_ = np.min(birth_date) - 1
    status = ["Clean"]*len(dgms_clean)+["Adv"]*len(dgms_adv)
    df = pd.DataFrame({"bd":birth_date, "dd":death_date, "Status":status})

    sns.displot(df, x="bd", y="dd", hue="Status", kind="kde", thresh=0.2, levels=10, alpha=0.7)
    plt.plot([min_,0],[min_,0],"-", color="black")
    plt.xlim(min_,1)
    plt.ylim(min_,1)
    plt.savefig(file_name, dpi=350)
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

#####
#####
# Reducing the dim of each layer for viz
#####
#####

def from_old_to_new(old_indx, old_shape=4, new_shape=2):
    old_row = old_indx//old_shape
    old_col = old_indx%old_shape
    new_row = old_row//(old_shape/new_shape)
    new_col = old_col//(old_shape/new_shape)
    new_indx = int(new_row*new_shape + new_col)
    return new_indx

def new_to_old(new_row, new_col=None, old_shape=4, new_shape=2, previous_cumsum_shape=0):
    if new_col != None:
        #logger.info(f"new to old for conv layer")
        old_row = list(range(int(new_row*(old_shape/new_shape)), int((new_row+1)*(old_shape/new_shape))))
        old_col = list(range(int(new_col*(old_shape/new_shape)), int((new_col+1)*(old_shape/new_shape))))
        old_row_col_list_ = [[i,j] for i in old_row for j in old_col]
        old_indx = [int(old_row*old_shape + old_col + previous_cumsum_shape) for (old_row, old_col) in old_row_col_list_]
    else:
        #logger.info(f"new to old for linear layer")
        new_indx = new_row
        old_indx_ = list( range( int(new_indx*(old_shape/new_shape)), int((new_indx+1)*(old_shape/new_shape))))
        old_indx = [int(oi_ + previous_cumsum_shape) for oi_ in old_indx_]
    return old_indx

def new_to_old_linear(new_indx, old_shape=50, new_shape=25, previous_cumsum_shape=0):
    old_indx_ = list( range( int(new_indx*(old_shape/new_shape)), int((new_indx+1)*(old_shape/new_shape))))
    old_indx = [int(oi_ + previous_cumsum_shape) for oi_ in old_indx_]
    return old_indx

def from_indx_to_rowcol(indx, shape):
    row = indx//shape
    col = indx%shape
    return [row, col]

def mnist_viz(old_sparse_mat, new_dim_total_=[16, 10, 20, 10, 10]):
    old_sparse_mat = old_sparse_mat.todense().T
    channels = [1, 10, 10, 20, 20, 1, 1, 1]

    old_shape_ = [28, 24, 24, 8, 8, 50, 50, 10]
    old_dim_ = [28*28, 24*24*10, 24*24*10, 8*8*20, 8*8*20, 50, 50, 10]

    new_shape_ = [0]*8
    new_dim_ = [0]*8
    for layer in range(5):
        if layer == 0:
            theshape_ = int(np.sqrt(new_dim_total_[layer]/channels[0]))
            new_shape_[0] = theshape_
            new_dim_[0] = (theshape_**2)*channels[0]
        elif layer == 1:
            theshape_ = int(np.sqrt(new_dim_total_[layer]/channels[1]))
            new_shape_[1:3] = [theshape_]*2
            new_dim_[1:3] = [(theshape_**2)*channels[1]]*2
        elif layer == 2:
            theshape_ = int(np.sqrt(new_dim_total_[layer]/channels[3]))
            new_shape_[3:5] = [theshape_]*2
            new_dim_[3:5] = [(theshape_**2)*channels[3]]*2
        elif layer == 3:
            theshape_ = int(new_dim_total_[layer]/channels[5])
            new_shape_[5:7] = [theshape_]*2
            new_dim_[5:7] = [theshape_*channels[5]]*2
        elif layer == 4:
            theshape_ = int(new_dim_total_[layer]/channels[7])
            new_shape_[7] = theshape_
            new_dim_[7] = theshape_*channels[7]
    logger.info(f"Total dim = {new_dim_total_}, new_shape_={new_shape_} and new_dim_={new_dim_}")

    old_dim_cum_ = [0] + list(np.cumsum(old_dim_))
    new_dim_cum_ = [0] + list(np.cumsum(new_dim_))
    new_dim = np.sum(new_dim_)

    lim_sup = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    lim_inf = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    #rescaling_layer = [1*10**4, 1*10**4, 4*10**4, 4*10**4, 1*10**5, 1*10**5, 1*10**6, 1*10**6]
    #lim_sup = [38, np.inf, 35, np.inf, 25, np.inf, 30, np.inf]
    #lim_inf = [0, -np.inf, 0, -np.inf, 0, -np.inf, 0, -np.inf]
    rescaling_layer = [10**2, np.inf, 10**3/2, np.inf, 5*10**3, np.inf, 5*10**4, np.inf]

    new_mat = np.zeros((np.sum(new_dim_total_),np.sum(new_dim_total_)))
    for layer_ in range(4):
        all_edges_layer = list()
        logger.info(f"layer {layer_}")
        for i in range(new_dim_[layer_]):
            ri, ci = from_indx_to_rowcol(i, shape=new_shape_[layer_])
            nto_i = new_to_old(ri, ci,
                    old_shape=old_shape_[layer_], new_shape=new_shape_[layer_],
                    previous_cumsum_shape=old_dim_cum_[layer_])
            new_i = i+new_dim_cum_[layer_]
            if layer_ >= 2:
                new_i = new_i - new_dim_[2]

            for j in range(new_dim_[layer_+1]):
                rj, cj = from_indx_to_rowcol(j, shape=new_shape_[layer_+1])
                nto_j = new_to_old(rj, cj,
                    old_shape=old_shape_[layer_+1], new_shape=new_shape_[layer_+1],
                    previous_cumsum_shape=old_dim_cum_[layer_+1])
                new_j = j+new_dim_cum_[layer_+1]

                if layer_ >= 1:
                    new_j = new_j - new_dim_[2]
                if layer_ >= 3:
                    new_j = new_j - new_dim_[4]

                val__ = [old_sparse_mat[n,m] for n in nto_i for m in nto_j ]
                val_ = np.array(val__)#[np.nonzero(val__)]
                val = np.mean(val_)/rescaling_layer[layer_] if len(val_) > 0 else 0
                if val < lim_sup[layer_] and val > lim_inf[layer_]:
                    val = 0
                else:
                    all_edges_layer.append(val)
                new_mat[new_i,new_j] = val
        if len(all_edges_layer)>0:
            logger.info(f"Layer {layer_}: nb={len(all_edges_layer)}, min={np.min(all_edges_layer)}, mean={np.mean(all_edges_layer)}, max={np.max(all_edges_layer)}")

    layer_ = 4
    all_edges_layer = list()
    for i in range(new_dim_[layer_]):
        ri, ci = from_indx_to_rowcol(i, shape=new_shape_[layer_])
        nto_i = new_to_old(ri, ci,
                old_shape=old_shape_[layer_], new_shape=new_shape_[layer_],
                previous_cumsum_shape=old_dim_cum_[layer_])
        new_i = i+new_dim_cum_[layer_]
        new_i = new_i - new_dim_[2] - new_dim_[4]
        for j in range(new_dim_[layer_+1]):
            nto_j = new_to_old_linear(j,
                old_shape=old_shape_[layer_+1],
                new_shape=new_shape_[layer_+1],
                previous_cumsum_shape=old_dim_cum_[layer_+1])
            new_j = j+new_dim_cum_[layer_+1]
            new_j = new_j - new_dim_[2] - new_dim_[4]
            val__ = [old_sparse_mat[n,m] for n in nto_i for m in nto_j ]
            val_ = np.array(val__)#[np.nonzero(val__)]
            val = np.mean(val_)/rescaling_layer[layer_] if len(val_) > 0 else 0
            if val < lim_sup[layer_] and val > lim_inf[layer_]:
                val = 0
            else:
                all_edges_layer.append(val)
            new_mat[new_i,new_j] = val
    if len(all_edges_layer)>0:
            logger.info(f"Layer {layer_}: nb={len(all_edges_layer)}, min={np.min(all_edges_layer)}, mean={np.mean(all_edges_layer)}, max={np.max(all_edges_layer)}")

    
    new_output = list()
    old_output = list()
    for layer_ in [5, 6]:
        all_edges_layer = list()
        for i in range(new_dim_[layer_]):
            nto_i = new_to_old_linear(i,
                old_shape=old_shape_[layer_],
                new_shape=new_shape_[layer_],
                previous_cumsum_shape=old_dim_cum_[layer_])
            new_i = i+new_dim_cum_[layer_]
            new_i = new_i - new_dim_[2] - new_dim_[4]
            if layer_ >= 6:
                new_i = new_i - new_dim_[6]
            for j in range(new_dim_[layer_+1]):
                nto_j = new_to_old_linear(j,
                    old_shape=old_shape_[layer_+1],
                    new_shape=new_shape_[layer_+1],
                    previous_cumsum_shape=old_dim_cum_[layer_+1])
                new_j = j+new_dim_cum_[layer_+1]
                new_j = new_j - new_dim_[2] - new_dim_[4]
                if layer_ >= 5:
                    new_j = new_j - new_dim_[6]
                val__ = [old_sparse_mat[n,m] for n in nto_i for m in nto_j]
                val_ = np.array(val__)#[np.nonzero(val__)]
                val = np.mean(val_)/rescaling_layer[layer_] if len(val_) > 0 else 0
                if val < lim_sup[layer_] and val > lim_inf[layer_]:
                    val = 0
                else:
                    all_edges_layer.append(val)
                new_mat[new_i,new_j] = val
        if len(all_edges_layer)>0:
            logger.info(f"Layer {layer_}: nb={len(all_edges_layer)}, min={np.min(all_edges_layer)}, mean={np.mean(all_edges_layer)}, max={np.max(all_edges_layer)}")

    new_mat = coo_matrix(new_mat)
    logger.info(f"Nb elem = {len(new_mat.data)}")

    return new_mat








def mnist_viz2(old_sparse_mat, new_dim=[16, 10, 20, 10, 10]):
    old_sparse_mat = old_sparse_mat.todense()
    channels = [1, 10, 20, 1, 1]
    old_shape = [28, 24, 8, 50, 10]
    old_dim = [28*28, 24*24*10, 8*8*20, 50, 10]

    old_mat_indices = {(0,1):(range(0,784), range(784,6544)),
                       (1,2):(range(6544,12304), range(12304,13584)),
                       (2,3):(range(13584,14864), range(14864,14914)),
                       (3,4):(range(14914,14964), range(14964,14974))}

    new_shape = 5*[0]
    for layer in range(len(new_dim)):
        if layer < 3:
            new_shape[layer] = int(np.sqrt(new_dim[layer]/channels[layer]))
        else:
            new_shape[layer] = new_dim[layer]/channels[layer]
    new_dim_cum = np.cumsum(new_dim)
    new_mat_indices = {(0,1):(range(0, new_dim_cum[0]), range(new_dim_cum[0], new_dim_cum[1])),
                       (1,2):(range(new_dim_cum[0], new_dim_cum[1]), range(new_dim_cum[1], new_dim_cum[2])),
                       (2,3):(range(new_dim_cum[1], new_dim_cum[2]), range(new_dim_cum[2], new_dim_cum[3])),
                       (3,4):(range(new_dim_cum[2], new_dim_cum[3]), range(new_dim_cum[3], new_dim_cum[4]))}
    logger.info(f"new_mat_indices = {new_mat_indices}")
    new_mat = np.zeros((np.sum(new_dim),np.sum(new_dim)))

    lim_sup = [-np.inf, -np.inf, -np.inf, -np.inf]
    lim_inf = [np.inf, np.inf, np.inf, np.inf]
    rescaling_layer = [10**2, 10**3/2, 5*10**3, 5*10**4]


    for (layer0, layer1), (new_neurons0, new_neurons1) in new_mat_indices.items():
        logger.info(f"layers = {(layer0, layer1)}")
        all_edges_layer = list()
        for i, indx_i in enumerate(new_neurons0):
            if layer0 < 3:
                row_i, col_i = from_indx_to_rowcol(i, shape=new_shape[layer0])
            else:
                row_i = i
                col_i = None
            old_indx_i = new_to_old(row_i, col_i,
                old_shape=old_shape[layer0], new_shape=new_shape[layer0],
                previous_cumsum_shape=np.min(old_mat_indices[(layer0, layer1)][0]))
            #logger.info(f"i={i}, indx_i={indx_i}")
            for j, indx_j in enumerate(new_neurons1):
                if layer1 < 3:
                    row_j, col_j = from_indx_to_rowcol(j, shape=new_shape[layer1])
                else:
                    row_j = j
                    col_j = None
                old_indx_j = new_to_old(row_j, col_j,
                    old_shape=old_shape[layer1], new_shape=new_shape[layer1],
                    previous_cumsum_shape=np.min(old_mat_indices[(layer0, layer1)][1]))
                #logger.info(f"j={j}, indx_j={indx_j}: old_indx_j = {old_indx_j}")

                list_val = np.array([old_sparse_mat[n,m] for n in old_indx_i for m in old_indx_j ])
                val = np.mean(list_val)/rescaling_layer[layer0] if len(list_val) > 0 else 0
                if val < lim_sup[layer0] and val > lim_inf[layer0]:
                    val = 0
                else:
                    all_edges_layer.append(val)

                new_mat[indx_i,indx_j] = val
        logger.info(f"Layers {(layer0, layer1)}: {len(all_edges_layer)} edges, mean={np.mean(all_edges_layer)}, max={np.max(all_edges_layer)}")

    return coo_matrix(new_mat)

def rescaling_weights(mat, shape=[16, 10, 20, 10, 10], min_=None, max_=None):
    mat = mat.todense()
    if min_ == None:
        min_ = np.min(mat)
        max_ = np.max(mat)
        mat2 = (mat-min_)/(max_-min_)
        return coo_matrix(mat2), min_, max_
    else:
        mat2 = (mat-min_)/(max_-min_)
        return coo_matrix(mat2)


def plot_mnist_viz(config, new_mat, status, y, y_pred, shape=[16, 10, 20, 10, 10], eps=0.0):
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.architecture)
            + str(config.epochs)
            + "_reduced_dim_viz"
            + "_" + str(status)
            + "_eps_" + str(eps)
            + ".png"
        )

    G = nx.from_scipy_sparse_matrix(new_mat)
    #shape = [7*7, 6*6*10, 2*2*20, 50, 10]
    #shape = [4*4, 2*2*10, 1*1*20, 50, 10]
    #shape = [4*4, 1*1*10, 1*1*20, 10, 10]
    cum_shape = np.cumsum(shape)
    pos = {}
    for i in range(cum_shape[-1]):
        if i < cum_shape[0]:
            pos[i]=(0,10*(shape[0]-i)/shape[0]) #shape[0]-
        elif i < cum_shape[1]:
            pos[i]=(1,10*(shape[1]-(i-cum_shape[0]))/shape[1]) # shape[1]-
        elif i < cum_shape[2]:
            pos[i]=(2,10*(shape[2]-(i-cum_shape[1]))/shape[2])
        elif i < cum_shape[3]:
            pos[i]=(3,10*(shape[3]-(i-cum_shape[2]))/shape[3])
        elif i < cum_shape[4]:
            pos[i]=(4,10*((i-cum_shape[3]))/shape[4])

    if eps != 0.0:
        adv_message = f"eps = {eps} "
    else:
        adv_message = f""
    nx.set_node_attributes(G, pos, 'coord') 
    widths = nx.get_edge_attributes(G, 'weight')
    nodelist = G.nodes()
    nx.draw_networkx_nodes(G, pos,
                            nodelist=nodelist,
                            node_size=3,
                            node_color='black',
                            alpha=0.7)
    nx.draw_networkx_edges(G, pos,
                            edgelist = widths.keys(),
                            width=[5*val for val in list(widths.values())],
                            edge_color='darkred',#'lightblue',
                            alpha=0.6)
    plt.title(f"{status} {y} ({adv_message}pred. {y_pred})")
    labels = ['Input', 'Conv. 1', 'Conv. 2', 'Linear', 'Output']
    x = [0, 1, 2, 3, 4]
    plt.xticks(x, labels)
    plt.rcParams["axes.grid"] = False
    plt.savefig(file_name, dpi=500)
    plt.close()

def plot_image(config, line, eps=0.0):
    status = line.y_adv
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.architecture)
            + str(config.epochs)
            + "_reduced_dim_viz_image"
            + "_" + str(status)
            + "_eps_" + str(eps)
            + ".png"
        )
    if eps != 0.0:
        adv_message = f"eps = {eps} "
    else:
        adv_message = f""
    x = line.x.view(28,28,1).cpu().detach().numpy()
    plt.imshow(x, cmap="gray")
    plt.title(f"y = {line.y} ({adv_message}pred {line.y_pred})")
    plt.savefig(file_name, dpi=350)
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
    #myeps = [0.1]
    #architecture, train_clean, test_clean, train_adv, test_adv = get_all_inputs(config, myeps)
    #lines_c, graphs_c, dgms_c = get_graphs_dgms(config, train_clean+test_clean, architecture,
    #    target=None, clean=True)
    #a = 100
    #i = 0
    #g_old = graphs_c[i].get_adjacency_matrix().todense()[:6544-1,:6544-1]
    #g_new = np.zeros((2,3))
    #m1s = list()
    #m2s = list()
    #yerr1s = list()
    #yerr2s = list()
    #for eps in myeps:
    #    nb_c = nb_pts_dgms(dgms_c)
    #    lines_a, graphs_a, dgms_a = get_graphs_dgms(config,
    #        train_adv[eps]+test_adv[eps],
    #        architecture, target=None, clean=False)
    #    nb_a = nb_pts_dgms(dgms_a)
    #    m1, yerr1, m2, yerr2 = stat_nb_pts_dgms(config, nb_c, nb2=nb_a)
    #    m1s.append(m1)
    #    m2s.append(m2)
    #    yerr1s.append(yerr1)
    #    yerr2s.append(yerr2)
    #yerr1s = np.asarray(yerr1s).reshape(-1,2).T
    #yerr2s = np.asarray(yerr2s).reshape(-1,2).T
    #plot_errorbar_pts_dmgs(config, myeps, m1s, yerr1s, m2s, yerr2s)

    ##### Test
    myeps = [0.1, 0.4]
    #nb_sample = 3
    nb_sample = 5

    architecture, train_clean, test_clean, train_adv, test_adv = get_all_inputs(config, myeps)

    lines_c, graphs_c, dgms_c = get_graphs_dgms(config,
        test_clean, architecture,
        target=False, clean=True, nb=10)
    logger.info(f"How many graphs ? {len(graphs_c)} and lines ? {len(lines_c)}")

    shape = [16, 10, 20, 10, 10]
    plot_image(config, lines_c[nb_sample])

    g_old_c = graphs_c[nb_sample].get_adjacency_matrix()
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    logger.info(f"shapes = {graphs_c[nb_sample]._get_shapes()}")
    plt.style.use('ggplot') 
    plt.imshow(g_old_c.todense(), norm=norm)
    plt.savefig("/Users/m.goibert/Documents/Criteo/P2_TDA_Detection/TDA_for_adv_robustness/plots/toy_viz/-1/mat_view.png", dpi=350)
    plt.close()
    g_new_c = mnist_viz2(g_old_c, shape)
    g_new_c, min_, max_ = rescaling_weights(g_new_c, shape)
    plot_mnist_viz(config, g_new_c, status="Clean", y=lines_c[nb_sample].y, y_pred=lines_c[nb_sample].y_pred, shape=shape)

    for i, eps in enumerate(myeps):
        logger.info(f"eps={eps}, len dataset={len(test_adv[list(test_adv.keys())[i]])}")
        lines_a, graphs_a, dgms_a = get_graphs_dgms(config,
        list(test_adv[list(test_adv.keys())[i]]), architecture,
        target=False, clean=False, nb=10)

        logger.info(f"len new : {len(lines_a)}")
        plot_image(config, lines_a[nb_sample], eps=eps)
        g_old_a = graphs_a[nb_sample].get_adjacency_matrix()
        g_new_a = mnist_viz2(g_old_a)
        g_new_a = rescaling_weights(g_new_a, shape, min_, max_)
        plot_mnist_viz(config, g_new_a, status="Adversarial", y=lines_a[nb_sample].y, y_pred=lines_a[nb_sample].y_pred, shape=shape, eps=eps)


def run_experiment3(config: Config):   
    myeps = [0.1]
    nb_sample = 5

    architecture, train_clean, test_clean, train_adv, test_adv = get_all_inputs(config, myeps)

    lines_c, graphs_c, dgms_c = get_graphs_dgms(config,
        test_clean, architecture, target=7, target_adv=6, clean=True, nb=5)
    lines_a, graphs_a, dgms_a = get_graphs_dgms(config,
        list(test_adv[list(test_adv.keys())[0]]), architecture, target=7, target_adv=6, clean=False, nb=5)

    plot_kde_dgms(config, lines_c, lines_a, dgms_c, dgms_a)

if __name__ == "__main__":
    my_config = get_config()
    try:
        run_experiment3(my_config)
    except Exception as e:
        my_trace = io.StringIO()
        traceback.print_exc(file=my_trace)

        logger.error(my_trace.getvalue())
