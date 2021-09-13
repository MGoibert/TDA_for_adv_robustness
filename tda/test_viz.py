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
import matplotlib
matplotlib.use("Agg")
from operator import itemgetter
from numpy import inf
from scipy.sparse import coo_matrix
import mlflow

logger = get_logger("Viz test")
start_time = time.time()

mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in")
mlflow.set_experiment("tda_adv_detection")

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
    ns : int = -1

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
    parser.add_argument("--ns", type=int, default=-1)

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


def get_graphs_dgms(config, dataset, architecture, target=None, nb=2000, clean=True):
    line_list = list()
    for elem in dataset:
        line_list.append(elem)
        if (target != None) and (target != False) and (elem.y == target) and (clean*(elem.y_pred == target) or (not clean)*(elem.y_pred != target)):
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
    dgm_list = list()
    for line in line_list:
        graph = Graph.from_architecture_and_data_point(
                architecture=architecture, x=line.x.double()
            )
        graph_list.append(graph)
        dgm_list.append(compute_dgm_from_graph(graph))

    return line_list, graph_list, dgm_list


def plot_graph_from_adj_mat(config, adj_mat, message_file="", message_title=""):
    file_name = (
            config.result_path
            + str(config.architecture)
            + str(config.epochs)
            + "_graph"
            + message_file
            + ".png"
        )
    
    G = nx.from_numpy_matrix(adj_mat, create_using=nx.Graph())
    pos = {}
    for i in range(26):
        if i < 9:
            pos[i] = (0, i)
        elif i < 9+8:
            pos[i] = (2, 0.5+(i-9))
        elif i < 9+8+4:
            pos[i] = (4, 2.5+(i-9-8))
        elif i < 9+8+4+3:
            pos[i] = (6, 3+(i-9-8-4))
        elif i < 9+8+4+3+2:
            pos[i] = (8, 3.5+(i-9-8-4-3))

    edge_list = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] != 0]
    edge_labels = nx.get_edge_attributes(G,'weight')
    edge_labels = {x:np.round(y/(5*1e5),0) for x,y in edge_labels.items() if y!=0}
    
    plt.figure(figsize=(8,8))
    plt.title(f"{message_title}")
    nx.draw_networkx_nodes(G, pos, label=list(G.nodes()))
    nx.draw_networkx_labels(G,pos, {a:a for a in list(G.nodes())},font_size=11,color="grey")
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=list(edge_labels.values()), alpha=0.7)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(file_name, dpi=250)
    plt.close()

def plot_image(config, line, message_file="", message_title=""):
    status = line.y_adv
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.epochs)
            + "_viz_image"
            + message_file
            + ".png"
        )

    x = line.x.view(3,3).cpu().detach().numpy()
    logger.info(f"{message_title} and pred = {line.y_pred}")
    plt.imshow(x, cmap="viridis", vmin=0, vmax=1)
    plt.title(f"y = {line.y} ({message_title} pred {line.y_pred})")
    plt.savefig(file_name, dpi=350)
    plt.close()

def plot_adj_mat(config, adj_mat, message_file=""):
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.epochs)
            + "_adj_mat"
            + message_file
            + ".png"
        )
    adj_mat = np.round(adj_mat/(1*1e5),2)
    fig, ax = plt.subplots()
    ax.imshow(adj_mat, cmap=plt.cm.Blues)
    for i in range(26):
        for j in range(26):
            c = adj_mat[i,j]
            ax.text(i, j, str(c), va='center', ha='center', fontsize=2)
    plt.savefig(file_name, dpi=350)
    plt.close()

def plot_dgms(config, dgm_clean, dgm_adv, ns):
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.epochs)
            + "_dgms" + "_" + str(ns)
            + ".png"
        )
 
    dgmc_ = [(elem0/10**5, elem1/10**5) for (elem0, elem1) in dgm_clean]
    dgmc_ = [(elem0, 0) if elem1 == inf else (elem0, elem1) for (elem0, elem1) in dgmc_]
    dgma_ = [(elem0/10**5, elem1/10**5) for (elem0, elem1) in dgm_adv]
    dgma_ = [(elem0, 0) if elem1 == inf else (elem0, elem1) for (elem0, elem1) in dgma_]
    list_dgms = [dgmc_, dgma_]
    logger.info(f"dgm clean = {dgmc_} and adv = {dgma_}")

    fig, ax = plt.subplots(figsize=(8,8))

    legend = ["Clean", "Adv"]
    for i, col_ in enumerate(legend):
        ax.scatter(list(map(itemgetter(0), list_dgms[i])), list(map(itemgetter(1), list_dgms[i])), s=10, alpha=0.5, label=col_)
    
    ax.plot([-60,0],[-60,0], "-", color="black")
    ax.set_xlim([-60,1])
    ax.set_ylim([-60,1])
    ax.legend()
    plt.savefig(file_name, dpi=350)
    plt.close()

def run_experiment(config: Config):

    ##### Test
    myeps = [0.1]

    if config.ns >= 0:
        ns = config.ns
    else:
        ns = -1
    status = True

    architecture, train_clean, test_clean, train_adv, test_adv = get_all_inputs(config, myeps)

    lines_c, graphs_c, dgms_c = get_graphs_dgms(config,
            test_clean, architecture,
            target=False, clean=True, nb=500)
    lines_a, graphs_a, dgms_a = get_graphs_dgms(config,
            list(test_adv[list(test_adv.keys())[0]]), architecture,
            target=False, clean=False, nb=500)

    if config.ns < 0:
        while status:
            ns += 1
            y_clean = lines_c[ns].y
            y_pred_clean = lines_c[ns].y_pred
            y_pred_adv = lines_a[ns].y_pred
            logger.info(f"ns = {ns} --> clean {y_clean} (pred {y_pred_clean}) and adv {y_pred_adv}")
            if (y_clean == y_pred_clean) and (y_clean != y_pred_adv):
                status = False

    adj_mat_clean = graphs_c[ns].get_adjacency_matrix()
    adj_mat_adv = graphs_a[ns].get_adjacency_matrix()
    adj_mat_clean_ = adj_mat_clean.todense()
    adj_mat_adv_ = adj_mat_adv.todense()
    
    param_dict = {"ns": ns, "dataset_size":config.dataset_size, "y":lines_c[ns].y, "clean_pred":lines_c[ns].y_pred, "adv_pred":lines_a[ns].y_pred}
    for key in param_dict:
        mlflow.log_param(key, param_dict[key])
    #for i in range(26):
    #    for j in range(26):
    #        logger.info(f"i={i} and j={j}")
    #        mlflow.log_metric(f"clean {i}-{j}", adj_mat_clean_[i,j])
    #        mlflow.log_metric(f"adv {i}-{j}", adj_mat_adv_[i,j])

        

    plot_graph_from_adj_mat(config, adj_mat_clean.todense(), message_file=f"_clean_{ns}")
    plot_graph_from_adj_mat(config, adj_mat_adv.todense(), message_file=f"_adv_e_{myeps[0]}_{ns}")

    plot_image(config, lines_c[ns], message_file=f"_clean_{ns}", message_title="clean")
    plot_image(config, lines_a[ns], message_file=f"_adv_{ns}", message_title=f"adv {myeps[0]}")
    
    plot_adj_mat(config, adj_mat_clean_, message_file=f"_clean_{ns}")
    plot_adj_mat(config, adj_mat_adv_, message_file=f"_adv_{ns}")

    plot_dgms(config, dgms_c[ns], dgms_a[ns], ns)


if __name__ == "__main__":
    my_config = get_config()
    try:
        run_experiment(my_config)
    except Exception as e:
        my_trace = io.StringIO()
        traceback.print_exc(file=my_trace)

        logger.error(my_trace.getvalue())
