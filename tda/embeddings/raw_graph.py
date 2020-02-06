import typing
import numpy as np


def to_sparse_vector(sparse_matrix):
    return sparse_matrix.reshape((
        1, sparse_matrix.shape[0] * sparse_matrix.shape[1]))


def identify_active_indices(sparse_vectors: typing.List):
    """
    Identifying set of edge indices that are active at least once in the set.
    This should be called on the training set

    :param sparse_vectors: list of sparse adjacency matrices reshaped as sparse vectors
    :return:
    """
    indices = set()
    for sparse_vector in sparse_vectors:
        indices.update(set(sparse_vector.col))
    return sorted(list(indices))


def featurize_vectors(
        sparse_vectors: typing.List,
        indices: typing.Optional[typing.List]
):
    """
    Create feature vectors for the raw graphs

    :param sparse_vectors: list of sparse adjacency matrices reshaped as sparse vectors
    :param indices: list of indices active at least once (should come from the training set)
    :return:
    """
    if indices is not None:
        return [
            np.squeeze(np.array(sparse_vector.tocsc()[:, indices].todense()))
            for sparse_vector in sparse_vectors
        ]
    else:
        return [
            np.squeeze(np.array(sparse_vector.todense()))
            for sparse_vector in sparse_vectors
        ]
