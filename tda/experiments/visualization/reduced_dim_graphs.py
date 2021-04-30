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

logger = get_logger("Reduced dimension visualization")
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
        directory = f"{rootpath}/tda/experiments/visualization/viz_plots/{self.experiment_id}/"
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

    parser.add_argument("--experiment_id", type=int, default=np.round(int(time.time()),4))
    parser.add_argument("--run_id", type=int, default=-1)
    parser.add_argument("--embedding_type", type=str, default=EmbeddingType.PersistentDiagram)
    parser.add_argument("--thresholds", type=str, default="0:0:0.025_2:0:0.025_4:0:0.025_6:0:0.025")
    parser.add_argument("--threshold_strategy", type=str, default=ThresholdStrategy.UnderoptimizedMagnitudeIncrease)
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




######
######
# Utils fonctions to get datasets, archi, etc
######
######


plt.style.use('seaborn-dark') 

def get_protocolar_datasets_viz(
        dataset,
        archi,
        all_epsilons,
        offset=0,
        succ_adv=True,
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

    for epsilon in all_epsilons:
        final_dataset_adv[epsilon] = list()

    while len(final_dataset_clean)<dataset_size and current_sample_id < source_dataset_size:
        samples = None
        processed_samples = None
        y_pred = None

        batch = source_dataset[current_sample_id:current_sample_id+batch_size]
        if isinstance(batch[0], DatasetLine):
            x = torch.cat([torch.unsqueeze(s.x, 0) for s in batch], 0).to(device)
            y = np.array([s.y for s in batch])
        else:
            x = torch.cat([torch.unsqueeze(s[0], 0) for s in batch], 0).to(device)
            y = np.array([s[1] for s in batch])
        samples = (x, y)

        processed_samples_adv = dict()

        for indx_epsilon, epsilon in enumerate(all_epsilons):
            processed_samples_adv[epsilon] = process_sample(
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
            assert (samples[1] == processed_samples_adv[epsilon][1]).all()
            y_pred_adv = archi(processed_samples_adv[epsilon][0]).argmax(dim=-1).cpu().numpy()

            if succ_adv:
                if indx_epsilon == 0:
                    valid_attacks = np.where(samples[1] != y_pred_adv)[0]
            else:
                valid_attacks = np.array(range(len(samples[1])))

            if len(valid_attacks) == 0:
                processed_samples = None
                samples = None
                continue
            else:
                processed_samples_adv[epsilon] = (
                    processed_samples_adv[epsilon][0][valid_attacks],
                    processed_samples_adv[epsilon][1][valid_attacks],
                )
                samples = (
                    samples[0][valid_attacks],
                    samples[1][valid_attacks],
                )

            for i in range(len(processed_samples_adv[epsilon][1])):
                x = torch.unsqueeze(processed_samples_adv[epsilon][0][i], 0).double()
                graph = None
                final_dataset_adv[epsilon].append(
                DatasetLine(
                    graph=graph,
                    x=x,
                    y=processed_samples_adv[epsilon][1][i],
                    y_pred=y_pred_adv[i],
                    y_adv=True,
                    l2_norm=None,
                    linf_norm=None,
                    sample_id=current_sample_id,
                )
            )

        if len(valid_attacks) == 0:
            processed_samples = None
            samples = None
            continue

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

        y_pred_clean = archi(processed_samples_clean[0]).argmax(dim=-1).cpu().numpy()

        for i in range(len(processed_samples_clean[1])):
            x = torch.unsqueeze(processed_samples_clean[0][i], 0).double()
            graph = None
            final_dataset_clean.append(
                DatasetLine(
                    graph=graph,
                    x=x,
                    y=processed_samples_clean[1][i],
                    y_pred=y_pred_clean[i],
                    y_adv=False,
                    l2_norm=None,
                    linf_norm=None,
                    sample_id=current_sample_id,
                )
            )

    return final_dataset_clean, final_dataset_adv

def get_all_inputs(config: Config, epsilons=[0.1]):
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
    elif (epsilons is None):
        all_epsilons = [0.1]
    else:
        all_epsilons = epsilons

    test_clean, test_adv = get_protocolar_datasets_viz(
        dataset=dataset,
        archi=architecture,
        all_epsilons=all_epsilons,
        offset=0,
        succ_adv=config.successful_adv,
        dataset_size=config.dataset_size,
        attack_type=config.attack_type,
        num_iter=config.num_iter,
        attack_backend=config.attack_backend,
    )

    return architecture, test_clean, test_adv

def generate_graphs(config, data, architecture, offset_max=10, offset_min=0, target=None):
    graph_list = list()
    line_list = list()
    for line in data[offset_min:offset_max]:
        if target == None:
            graph = Graph.from_architecture_and_data_point(architecture=architecture, x=line.x.double())
            graph_list.append(graph)
            line_list.append(line)
        else:
            if line.y == target:
                graph = Graph.from_architecture_and_data_point(architecture=architecture, x=line.x.double())
                graph_list.append(graph)
                line_list.append(line)

    return graph_list, line_list


######
######
# Generate reduced graphs
######
######


###

def from_old_to_new(old_indx, old_shape=4, new_shape=2):
    old_row = old_indx//old_shape
    old_col = old_indx%old_shape
    new_row = old_row//(old_shape/new_shape)
    new_col = old_col//(old_shape/new_shape)
    new_indx = int(new_row*new_shape + new_col)
    return new_indx

def new_to_old(new_row, new_col=None, old_shape=4, new_shape=2, previous_cumsum_shape=0):
    if new_col != None:
        old_row = list(range(int(new_row*(old_shape/new_shape)), int((new_row+1)*(old_shape/new_shape))))
        old_col = list(range(int(new_col*(old_shape/new_shape)), int((new_col+1)*(old_shape/new_shape))))
        old_row_col_list_ = [[i,j] for i in old_row for j in old_col]
        old_indx = [int(old_row*old_shape + old_col + previous_cumsum_shape) for (old_row, old_col) in old_row_col_list_]
    else:
        new_indx = new_row
        old_indx_ = list( range( int(new_indx*(old_shape/new_shape)), int((new_indx+1)*(old_shape/new_shape))))
        old_indx = [int(oi_ + previous_cumsum_shape) for oi_ in old_indx_]
    return old_indx

def from_indx_to_rowcol(indx, shape):
    row = indx//shape
    col = indx%shape
    return [row, col]

###


###

def reduced_graph_viz(
    old_sparse_mat,
    new_dim=[16, 10, 10, 20, 20, 10, 10],
    aggregation_method="mean",
    rescaling=[1,1,1,1,1,1],
    lim_sup=[-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    lim_inf=[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    ):

    ### Old
    old_sparse_mat = old_sparse_mat.todense()
    channels = [1, 10, 10, 20, 20, 1, 1]
    old_shape = [28, 24, 12, 8, 4, 50, 10]
    old_dim = [28*28, 24*24*10, 12*12*10, 8*8*20, 4*4*20, 50, 10]
    old_mat_indices = {(0,1):(range(0,784), range(784,6544)),
                       (1,2):(range(784,6544), range(6544,7984)),
                       (2,3):(range(6544,7984), range(7984,9264)),
                       (3,4):(range(7984,9264), range(9264,9584)),
                       (4,5):(range(9264,9584), range(9584,9634)),
                       (5,6):(range(9634,9684), range(9684,9694))}

    ### New
    new_shape = 7*[0]
    for layer in range(len(new_dim)):
        if layer < 5:
            new_shape[layer] = int(np.sqrt(new_dim[layer]/channels[layer]))
        else:
            new_shape[layer] = new_dim[layer]/channels[layer]
    new_dim_cum = np.cumsum(new_dim)
    new_mat_indices = {(0,1):(range(0, new_dim_cum[0]), range(new_dim_cum[0], new_dim_cum[1])),
                       (1,2):(range(new_dim_cum[0], new_dim_cum[1]), range(new_dim_cum[1], new_dim_cum[2])),
                       (2,3):(range(new_dim_cum[1], new_dim_cum[2]), range(new_dim_cum[2], new_dim_cum[3])),
                       (3,4):(range(new_dim_cum[2], new_dim_cum[3]), range(new_dim_cum[3], new_dim_cum[4])),
                       (4,5):(range(new_dim_cum[3], new_dim_cum[4]), range(new_dim_cum[4], new_dim_cum[5])),
                       (5,6):(range(new_dim_cum[4], new_dim_cum[5]), range(new_dim_cum[5], new_dim_cum[6]))}
    new_mat = np.zeros((np.sum(new_dim),np.sum(new_dim)))

    #lim_sup = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    #lim_inf = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    #escaling_layer = [10**0, 10**10, 10**2/44, 10**10, 10**2/2, 4.3*10**3]


    for (layer0, layer1), (new_neurons0, new_neurons1) in new_mat_indices.items():
        all_edges_layer = list()
        for i, indx_i in enumerate(new_neurons0):
            if layer0 < 5:
                row_i, col_i = from_indx_to_rowcol(i, shape=new_shape[layer0])
            else:
                row_i = i
                col_i = None
            old_indx_i = new_to_old(row_i, col_i,
                old_shape=old_shape[layer0], new_shape=new_shape[layer0],
                previous_cumsum_shape=np.min(old_mat_indices[(layer0, layer1)][0]))
            for j, indx_j in enumerate(new_neurons1):
                if layer1 < 5:
                    row_j, col_j = from_indx_to_rowcol(j, shape=new_shape[layer1])
                else:
                    row_j = j
                    col_j = None
                old_indx_j = new_to_old(row_j, col_j,
                    old_shape=old_shape[layer1], new_shape=new_shape[layer1],
                    previous_cumsum_shape=np.min(old_mat_indices[(layer0, layer1)][1]))

                list_val = np.array([old_sparse_mat[n,m] for n in old_indx_i for m in old_indx_j ])

                if aggregation_method == "mean":
                    val = np.mean(list_val) if len(list_val) > 0 else 0
                elif aggregation_method == "min":
                    val = np.min(np.nonzero(list_val)[0]) if len(np.nonzero(list_val)[0]) > 0 else 0
                elif aggregation_method == "max":
                    val = np.max(list_val) if len(list_val) > 0 else 0
                val = rescaling[layer0]*val

                if val < lim_sup[layer0] and val > lim_inf[layer0]:
                    val = 0
                else:
                    all_edges_layer.append(val)

                new_mat[indx_i,indx_j] = val
        q10 = np.round(np.quantile(np.nonzero(all_edges_layer), 0.1), 2) if len(np.nonzero(all_edges_layer)) > 0 else 0
        q90 = np.round(np.quantile(np.nonzero(all_edges_layer), 0.9), 2) if len(np.nonzero(all_edges_layer)) > 0 else 0
        m = np.round(np.mean(np.nonzero(all_edges_layer)), 2) if len(np.nonzero(all_edges_layer)) > 0 else 0
        M = np.round(np.max(np.nonzero(all_edges_layer)), 2) if len(np.nonzero(all_edges_layer)) > 0 else 0
        m_ = np.round(np.min(np.nonzero(all_edges_layer)), 2) if len(np.nonzero(all_edges_layer)) > 0 else 0
        logger.info(f"Layers {(layer0, layer1)}: {len(all_edges_layer)} edges; min={m_} / q10={q10} / mean={m} / q90={q90} / max={M}")


    new_dim_final = [16, 10, 20, 10, 10]
    new_dim_final_cum = np.cumsum(new_dim_final)
    new_mat_final = np.zeros((np.sum(new_dim_final),np.sum(new_dim_final)))
    for i in range(new_mat_final.shape[0]):
        for j in range(new_mat_final.shape[1]):
            if i < new_dim_final_cum[0] and j < new_dim_final_cum[1]:
                new_mat_final[i,j] = new_mat[i,j]
            elif i < new_dim_final_cum[1] and j < new_dim_final_cum[2]:
                new_mat_final[i,j] =new_mat[i+new_dim[1],j+new_dim[1]]
            elif i < new_dim_final_cum[2] and j < new_dim_final_cum[3]:
                new_mat_final[i,j] =new_mat[i+new_dim[1]+new_dim[3],j+new_dim[1]+new_dim[3]]
            elif i < new_dim_final_cum[3] and j < new_dim_final_cum[4]:
                new_mat_final[i,j] =new_mat[i+new_dim[1]+new_dim[3],j+new_dim[1]+new_dim[3]]

    return coo_matrix(new_mat_final)

def plot_mnist_viz(config, graph, line, shape=[16, 10, 20, 10, 10], eps=0.0):
    status = "Clean" if line.y_adv < 0.5 else "Adv"
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.architecture)
            + str(config.epochs)
            + "_graph"
            + "_" + str(status)
            + "_eps_" + str(eps)
            + ".png"
        )

    G = nx.from_scipy_sparse_matrix(graph)
    cum_shape = np.cumsum(shape)
    mylen = len(cum_shape)
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
            pos[i]=(4,10*(shape[4]-(i-cum_shape[3]))/shape[4])
        if mylen > 5:
            if i < cum_shape[5] and i >= cum_shape[4]:
                pos[i]=(5,10*(shape[5]-(i-cum_shape[4]))/shape[5])
            elif i < cum_shape[6] and i >= cum_shape[5]:
                pos[i]=(6,10*((i-cum_shape[5]))/shape[6])

    if eps != 0.0:
        adv_message = f"eps = {eps}; "
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
    col_normal = ['darkred' if val<0.5 else 'steelblue' for val in list(widths.values())]
    col_diff = ['darkred' if val<0 else 'steelblue' for val in list(widths.values())]
    if status == "Diff":
        mycol = col_diff
    else:
        mycol = col_normal
    nx.draw_networkx_edges(G, pos,
                            edgelist = widths.keys(),
                            width=[5*val if val>0.5 else 10*val for val in list(widths.values())],
                            edge_color=mycol,
                            alpha=0.6)
    labels = ['Input', 'Conv. 1', 'Conv. 2', 'Linear', 'Output']
    x = [0, 1, 2, 3, 4]
    plt.xticks(x, labels, rotation="vertical")
    plt.rcParams["axes.grid"] = False
    plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    plt.close()


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


def distrib(mat, shape=[16, 10, 20, 10, 10]):
    mat = mat.todense()
    mydistrib = dict()
    cumshape = [0] + list(np.cumsum(shape))
    for layer0 in range(len(shape)-1):
        mydistrib[layer0] = list()
        for i in range(cumshape[layer0], cumshape[layer0+1]):
            for j in range(cumshape[layer0+1], cumshape[layer0+2]):
                if mat[i,j] != 0:
                    mydistrib[layer0].append(mat[i,j])
    return mydistrib

def plot_distribs(config, epsilons, *args):
    from matplotlib import cm
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.architecture)
            + str(config.epochs)
            + "_distrib"
            + ".png"
        )

    colors_ = ["cornflowerblue", "lightcoral", "crimson", "brown", "darkred"]
    labels_ = ["Clean"] + [f"Eps = {ep}" for ep in epsilons]

    plt.figure(1, figsize=(16, 3))
    for key_layer in args[0]:
        if key_layer == 0:
            subplot_ = 141
        elif key_layer == 1:
            subplot_ = 142
        elif key_layer == 2:
            subplot_ = 143
        elif key_layer == 3:
            subplot_ = 144
        plt.subplot(subplot_)
        distrib_for_layer = list()
        for i, mydict in enumerate(args):
            distrib_for_layer.append(mydict[key_layer])
        plt.hist(distrib_for_layer, bins=8, alpha=1, label=labels_[:len(epsilons)+1], color=colors_[:len(epsilons)+1])
        plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)


def plot_image(config, line, eps=0.0):
    status = "Clean" if line.y_adv < 0.5 else "Adv"
    file_name = (
            config.result_path
            + str(config.dataset)
            + "_"
            + str(config.architecture)
            + str(config.epochs)
            + "_image"
            + "_" + str(status)
            + "_eps_" + str(eps)
            + ".png"
        )
    if eps != 0.0:
        adv_message = f"eps = {eps}; "
    else:
        adv_message = f""
    x = line.x.view(28,28,1).cpu().detach().numpy()
    plt.imshow(x, cmap="gray")
    plt.title(f"y = {line.y} ({adv_message}pred {line.y_pred})", fontsize=25)
    plt.tick_params(axis='both',which='both',bottom=False,top=False,labelbottom=False,labelleft=False)
    plt.tight_layout()
    plt.savefig(file_name, dpi=200)
    plt.close()

def generate_all_viz(config, architecture,
    inputs, nb_sample, epsilon=0,
    rescaling=[1,1,1,1,1,1],
    aggregation_method="mean",
    lim_sup=[-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    lim_inf=[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    offset_min=0,
    offset_max=10,
    min_=None,
    max_=None):

    graphs_, inputs_ = generate_graphs(config, inputs, architecture, offset_max=offset_max, offset_min=offset_min)
    adj_mat_ = graphs_[nb_sample].get_adjacency_matrix()

    plot_image(config, inputs_[nb_sample], eps=epsilon)

    reduced_graph = reduced_graph_viz(adj_mat_, rescaling=rescaling, aggregation_method=aggregation_method, lim_sup=lim_sup, lim_inf=lim_inf)
    if min_ == None:
        rescaled_graph, min_, max_ = rescaling_weights(reduced_graph)
    else:
        rescaled_graph = rescaling_weights(reduced_graph, min_=min_, max_=max_)
    distrib_ = distrib(rescaled_graph)

    plot_mnist_viz(config, rescaled_graph, inputs_[nb_sample])

    return distrib_


def run_experiment(config: Config):

    # Parameters
    nb_sample = 2
    logger.info(f"nb sample: {nb_sample == 2}")

    aggregation_method = "mean"
    offset_min = 20
    offset_max = 40
    epsilons = [0.1]

    if nb_sample == 0:
        scale = [1e-2, 1e-20, 8.1*1e-3, 1e-20, 3.6*1e-4, 3*1e-5]
        lim_sup = [75, -np.inf, 90, -np.inf, 90, 60]
        lim_inf = [10, np.inf, 8, np.inf, 8, 8]
    elif nb_sample == 2:
        logger.info(f"here")
        scale = [1.25*1e-2, 1e-20, 2.2*1e-3, 1e-20, 1.7*1e-4, 1.4*1e-5]
        lim_sup = [75, -np.inf, 90, -np.inf, 90, 60]
        lim_inf = [10, np.inf, 8, np.inf, 8, 8]



    architecture, clean_inputs, adv_inputs = get_all_inputs(config, epsilons=epsilons)

    logger.info(f"Clean viz !")
    distrib_clean = generate_all_viz(
                config,
                architecture,
                clean_inputs,
                nb_sample,
                epsilon=0,
                rescaling=scale,
                aggregation_method=aggregation_method,
                lim_sup=lim_sup,
                lim_inf=lim_inf,
                offset_min=offset_min,
                offset_max=offset_max,
                min_=None,
                max_=None)

    distrib_adv = list()
    for epsilon in epsilons:
        logger.info(f"Adv viz for espilon = {epsilon} !")
        distrib_adv.append(generate_all_viz(
                config,
                architecture,
                adv_inputs[epsilon],
                nb_sample,
                epsilon=epsilon,
                rescaling=scale,
                aggregation_method=aggregation_method,
                lim_sup=lim_sup,
                lim_inf=lim_inf,
                offset_min=offset_min,
                offset_max=offset_max,
                min_=None,
                max_=None))

    plot_distribs(config, epsilons, distrib_clean, *distrib_adv)

    


if __name__ == "__main__":
    my_config = get_config()
    try:
        run_experiment(my_config)
    except Exception as e:
        my_trace = io.StringIO()
        traceback.print_exc(file=my_trace)

        logger.error(my_trace.getvalue())