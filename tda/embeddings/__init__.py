from typing import List, Optional, Dict
import pickle

import fwg
import time
import numpy as np
from tda.rootpath import rootpath
from tda.graph import Graph
from tda.embeddings.anonymous_walk import AnonymousWalks
from tda.embeddings.weisfeiler_lehman import get_wl_embedding
from tda.embeddings.persistent_diagrams import (
    sliced_wasserstein_kernel,
    sliced_wasserstein_distance_old_version,
    compute_dgm_from_graph,
    compute_dgm_from_graph_ripser,
)
from tda.embeddings.raw_graph import to_sparse_vector
from tda.graph_dataset import DatasetLine
from tda.models import Architecture, Dataset
from tda.tda_logging import get_logger
from joblib import Parallel, delayed

logger = get_logger("Embeddings")


class EmbeddingType(object):
    AnonymousWalk = "AnonymousWalk"
    WeisfeilerLehman = "WeisfeilerLehman"
    PersistentDiagram = "PersistentDiagram"
    PersistentDiagramReverse = "PersistentDiagramReverse"
    LastLayerSortedLogits = "LastLayerSortedLogits"
    PersistentDiagramRipser = "PersistentDiagramRipser"
    RawGraph = "RawGraph"


class KernelType(object):
    Euclidean = "Euclidean"
    RBF = "RBF"
    SlicedWasserstein = "SlicedWasserstein"
    SlicedWassersteinOldVersion = "SlicedWassersteinOldVersion"


class ThresholdStrategy(object):
    NoThreshold = "NoThreshold"
    ActivationValue = "ActivationValue"
    UnderoptimizedMagnitudeIncrease = "UnderoptimizedMagnitudeIncrease"
    UnderoptimizedMagnitudeIncreaseComplement = "UnderoptimizedMagnitudeIncreaseComplement"
    UnderoptimizedLargeFinal = "UnderoptimizedLargeFinal"
    UnderoptimizedRandom = "UnderoptimizedRandom"
    QuantilePerGraphLayer = "QuantilePerGraphLayer"


def get_embedding(
    embedding_type: str,
    line: DatasetLine,
    architecture: Architecture,
    edges_to_keep,
    thresholds: Dict,
    threshold_strategy: str,
    params: Dict = dict(),
    all_weights_for_sigmoid=None,
    thresholds_are_low_pass: bool = True,
):

    if line.graph is None:
        graph = Graph.from_architecture_and_data_point(
            architecture=architecture, x=line.x.double()
        )
    else:
        graph = line.graph

    if all_weights_for_sigmoid is not None:
        graph.sigmoidize(all_weights=all_weights_for_sigmoid)
    if threshold_strategy == ThresholdStrategy.ActivationValue:
        graph.thresholdize(thresholds=thresholds, low_pass=thresholds_are_low_pass)
    elif threshold_strategy in [
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        ThresholdStrategy.UnderoptimizedLargeFinal,
        ThresholdStrategy.UnderoptimizedRandom,
        ThresholdStrategy.UnderoptimizedMagnitudeIncreaseComplement
    ]:
        graph.thresholdize_underopt(edges_to_keep)
    elif threshold_strategy == ThresholdStrategy.QuantilePerGraphLayer:
        graph.thresholdize_per_graph(
            thresholds=thresholds, low_pass=thresholds_are_low_pass
        )

    if embedding_type == EmbeddingType.AnonymousWalk:
        walk = AnonymousWalks(G=graph.to_nx_graph())
        embedding = walk.embed(steps=params["steps"], method="sampling", verbose=False)[
            0
        ]
        return embedding
    elif embedding_type == EmbeddingType.WeisfeilerLehman:
        return get_wl_embedding(
            graph=graph,
            height=params["height"],
            hash_size=params["hash_size"],
            node_labels=params["node_labels"],
        ).todense()
    elif embedding_type == EmbeddingType.PersistentDiagram:
        return compute_dgm_from_graph(graph)
    elif embedding_type == EmbeddingType.PersistentDiagramReverse:
        return compute_dgm_from_graph(graph, negate=False)
    elif embedding_type == EmbeddingType.PersistentDiagramRipser:
        return compute_dgm_from_graph_ripser(graph)
    elif embedding_type == EmbeddingType.LastLayerSortedLogits:
        return sorted(graph.final_logits)
    elif embedding_type == EmbeddingType.RawGraph:
        return to_sparse_vector(graph.get_adjacency_matrix())
    else:
        raise NotImplementedError(embedding_type)


def get_gram_matrix_legacy(
    kernel_type: str,
    embeddings_in: List,
    embeddings_out: Optional[List] = None,
    params: Dict = dict(),
):
    """
    Compute the gram matrix of the given embeddings
    using a specific kernel

    :param kernel_type: Kernel to be used (str)
    :param embeddings_in: Embeddings in
    :param embeddings_out: Embeddings out
        if None, we use the same as embeddings_in and use the fact
        the matrix is symmetric to speed up the process
    :param params: Parameters for the kernel
    :return:
    """

    n = len(embeddings_in)

    # If no embeddings out is specified, we use embeddings_in
    if not embeddings_out:
        embeddings_out = embeddings_in
        gram = np.zeros([n, n])
        sym = True
    else:
        m = len(embeddings_out)
        gram = np.zeros([n, m])
        sym = False

    for i in range(n):

        if sym:
            lim = i + 1
        else:
            lim = m

        for j in range(lim):
            if kernel_type == KernelType.Euclidean:
                gram[i, j] = np.transpose(embeddings_in[i]) @ embeddings_out[j]
            elif kernel_type == KernelType.RBF:
                gram[i, j] = np.exp(
                    -np.linalg.norm(embeddings_in[i] - embeddings_out[j])
                    / 2
                    * params["gamma"] ** 2
                )
            elif kernel_type == KernelType.SlicedWasserstein:
                gram[i, j] = sliced_wasserstein_kernel(
                    embeddings_in[i], embeddings_out[j], M=params["M"]
                )
            else:
                raise NotImplementedError(f"Unknown kernel {kernel_type}")
            if sym:
                gram[j, i] = gram[i, j]
    return gram


def get_gram_matrix(
    kernel_type: str,
    embeddings_in: List,
    embeddings_out: Optional[List] = None,
    params: List = [dict()],
    n_jobs: int = 1,
    verbatim=False,
    save=False,
):
    """
    Compute the gram matrix of the given embeddings
    using a specific kernel

    :param kernel_type: Kernel to be used (str)
    :param embeddings_in: Embeddings in
    :param embeddings_out: Embeddings out
        if None, we use the same as embeddings_in and use the fact
        the matrix is symmetric to speed up the process
    :param params: Parameters for the kernel
    :return:
    """

    # If no embeddings out is specified, we use embeddings_in
    embeddings_out = embeddings_out or embeddings_in

    n = len(embeddings_in)
    m = len(embeddings_out)

    logger.info(f"Computing Gram matrix {n} x {m} (params {params})...")

    if kernel_type == KernelType.SlicedWassersteinOldVersion:
        logger.info("Old (incorrect) version for SW kernel !!")
        start = time.time()
        distance_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                # if verbatim:
                #    logger.info(f"Row {i} and col {j}")
                distance_matrix[i, j] = sliced_wasserstein_distance_old_version(
                    embeddings_in[i],
                    embeddings_out[j],
                    M=20,
                    verbatim=f"row{i}_col{j}",
                    save=save,
                )
        grams = [
            np.exp(-distance_matrix / (2 * a_param["sigma"] ** 2)) for a_param in params
        ]
        # grams = [distance_matrix for a_param in params]
        logger.info(f"Computed {n} x {m} gram matrix in {time.time()-start} secs")
        return grams

    if kernel_type == KernelType.SlicedWasserstein:
        logger.info("Using FWG !!!")
        start = time.time()
        distance_matrix = np.reshape(fwg.fwd(embeddings_in, embeddings_out, 50), (n, m))
        grams = [
            np.exp(-distance_matrix / (2 * a_param["sigma"] ** 2)) for a_param in params
        ]
        logger.info(f"Computed {n} x {m} gram matrix in {time.time()-start} secs")
        return grams

    def compute_gram_chunk(my_slices):
        ret = list()
        for i, j in my_slices:
            if kernel_type == KernelType.Euclidean:
                ret.append(
                    np.transpose(np.array(embeddings_in[i]))
                    @ np.array(embeddings_out[j])
                )
            elif kernel_type == KernelType.RBF:
                ret.append(
                    np.linalg.norm(
                        np.array(embeddings_in[i]) - np.array(embeddings_out[j])
                    )
                )
            else:
                raise NotImplementedError(f"Unknown kernel {kernel_type}")
        return ret

    p = Parallel(n_jobs=n_jobs)

    all_indices = [(i, j) for i in range(n) for j in range(m)]

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    my_chunks = chunks(all_indices, max([len(all_indices) // n_jobs, 1]))

    gram = p([delayed(compute_gram_chunk)(chunk) for chunk in my_chunks])
    gram = [item for sublist in gram for item in sublist]
    gram = np.reshape(gram, (n, m))

    if kernel_type == KernelType.RBF:
        return [np.exp(-gram / 2 * a_param["gamma"] ** 2) for a_param in params]
    elif kernel_type == KernelType.Euclidean:
        return [gram]
