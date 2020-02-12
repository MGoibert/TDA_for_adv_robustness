from typing import List, Optional, Dict

import fwg
import time
import numpy as np
from tda.graph import Graph
from tda.embeddings.anonymous_walk import AnonymousWalks
from tda.embeddings.weisfeiler_lehman import get_wl_embedding
from tda.embeddings.persistent_diagrams import (sliced_wasserstein_kernel,
                                                sliced_wasserstein_distance_old_version,
                                                compute_dgm_from_graph,
                                                compute_dgm_from_graph_ripser)
from tda.embeddings.raw_graph import to_sparse_vector
from tda.graph_dataset import DatasetLine
from tda.models import Architecture
from tda.logging import get_logger
from joblib import Parallel, delayed

logger = get_logger("Embeddings")


class EmbeddingType(object):
    AnonymousWalk = "AnonymousWalk"
    WeisfeilerLehman = "WeisfeilerLehman"
    PersistentDiagram = "PersistentDiagram"
    PersistentDiagramTop100 = "PersistentDiagramTop100"
    PersistentDiagramTopLifetimes = "PersistentDiagramTopLifetimes"
    LastLayerSortedLogits = "LastLayerSortedLogits"
    PersistentDiagramRipser = "PersistentDiagramRipser"
    RawGraph = "RawGraph"
    RawGraphWithPCA = "RawGraphWithPCA"

    
class KernelType(object):
    Euclidean = "Euclidean"
    RBF = "RBF"
    SlicedWasserstein = "SlicedWasserstein",
    SlicedWassersteinOldVersion = "SlicedWassersteinOldVersion"


def get_embedding(
        embedding_type: str,
        line: DatasetLine,
        architecture: Architecture,
        thresholds: Dict,
        params: Dict = dict()
):
    graph = Graph.from_architecture_and_data_point(
        architecture=architecture,
        x=line.x.double(),
        thresholds=thresholds
    )

    if embedding_type == EmbeddingType.AnonymousWalk:
        walk = AnonymousWalks(G=graph.to_nx_graph())
        embedding = walk.embed(
            steps=params['steps'],
            method="sampling",
            verbose=False)[0]
        return embedding
    elif embedding_type == EmbeddingType.WeisfeilerLehman:
        return get_wl_embedding(
            graph=graph,
            height=params['height'],
            hash_size=params['hash_size'],
            node_labels=params["node_labels"]
        ).todense()
    elif embedding_type == EmbeddingType.PersistentDiagram:
        return compute_dgm_from_graph(graph)
    elif embedding_type == EmbeddingType.PersistentDiagramTop100:
        dgm = compute_dgm_from_graph(graph)
        return sorted(dgm, key=lambda x: x[1]-x[0])[-100:]
    elif embedding_type == EmbeddingType.PersistentDiagramTopLifetimes:
        dgm = compute_dgm_from_graph(graph)
        lifetimes = [pt[1]-pt[0] for pt in dgm]
        return sorted(lifetimes)[-10:]
    elif embedding_type == EmbeddingType.PersistentDiagramRipser:
        return compute_dgm_from_graph_ripser(graph)
    elif embedding_type == EmbeddingType.LastLayerSortedLogits:
        return sorted(graph.final_logits)
    elif embedding_type in [EmbeddingType.RawGraph, EmbeddingType.RawGraphWithPCA]:
        return to_sparse_vector(graph.get_adjacency_matrix())
    else:
        raise NotImplementedError(embedding_type)


def get_gram_matrix_legacy(
        kernel_type: str,
        embeddings_in: List,
        embeddings_out: Optional[List] = None,
        params: Dict = dict()
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
                gram[i, j] = np.exp(-np.linalg.norm(
                    embeddings_in[i] - embeddings_out[j]
                ) / 2 * params['gamma'] ** 2)
            elif kernel_type == KernelType.SlicedWasserstein:
                sw = sliced_wasserstein_kernel(
                    embeddings_in[i],
                    embeddings_out[j],
                    M=params['M'])
                gram[i, j] = np.exp(-sw / (2 * params['sigma'] ** 2))
            else:
                raise NotImplementedError(
                    f"Unknown kernel {kernel_type}"
                )
            if sym:
                gram[j, i] = gram[i, j]
    return gram


def get_gram_matrix(
        kernel_type: str,
        embeddings_in: List,
        embeddings_out: Optional[List] = None,
        params: List = [dict()],
        n_jobs: int=1,
        verbatim: bool = False
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
        distance_matrix = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                if verbatim:
                    logger.info(f"Row {i} and col {j}")
                distance_matrix[i,j] = sliced_wasserstein_distance_old_version(
                    embeddings_in[i],
                    embeddings_out[j],
                    M=20,
                    verbatim=verbatim,
                    row=i)
        grams = [np.exp(- distance_matrix / (2 * a_param['sigma'] ** 2)) for a_param in params]
        #grams = [distance_matrix for a_param in params]
        logger.info(f"Computed {n} x {m} gram matrix in {time.time()-start} secs")
        return grams

    if kernel_type == KernelType.SlicedWasserstein:
        logger.info("Using FWG !!!")
        start = time.time()
        distance_matrix = np.reshape(fwg.fwd(
            embeddings_in,
            embeddings_out,
            50
        ), (n, m))
        grams = [np.exp(- distance_matrix / (2 * a_param['sigma'] ** 2)) for a_param in params]
        logger.info(f"Computed {n} x {m} gram matrix in {time.time()-start} secs")
        return grams

    def compute_gram_chunk(my_slices):
        ret = list()
        for i, j in my_slices:
            if kernel_type == KernelType.Euclidean:
                ret.append(np.transpose(np.array(embeddings_in[i])) @ np.array(embeddings_out[j]))
            elif kernel_type == KernelType.RBF:
                ret.append(np.linalg.norm(
                    np.array(embeddings_in[i]) - np.array(embeddings_out[j])
                ))
            else:
                raise NotImplementedError(
                    f"Unknown kernel {kernel_type}"
                )
        return ret

    p = Parallel(n_jobs=n_jobs)

    all_indices = [(i, j) for i in range(n) for j in range(m)]

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    my_chunks = chunks(all_indices, max([len(all_indices) // n_jobs, 1]))

    gram = p([delayed(compute_gram_chunk)(chunk) for chunk in my_chunks])
    gram = [item for sublist in gram for item in sublist]
    gram = np.reshape(gram, (n, m))

    if kernel_type == KernelType.RBF:
        return [
            np.exp(-gram / 2 * a_param['gamma'] ** 2)
            for a_param in params]
    elif kernel_type == KernelType.Euclidean:
        return [gram]
