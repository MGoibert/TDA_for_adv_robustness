import pytest
from scipy.sparse import coo_matrix
from tda.embeddings.raw_graph import (
    identify_active_indices,
    featurize_vectors,
    to_sparse_vector,
)


@pytest.mark.parametrize("use_indices, expected_length", [(True, 4), (False, 100)])
def test_featurization(use_indices, expected_length):
    # Train graphs
    adj_mat_1 = coo_matrix(([34, 32, 33], ([1, 2, 4], [4, 5, 6])), shape=(10, 10))
    adj_mat_2 = coo_matrix(([24, 21, 28], ([8, 2, 4], [4, 5, 6])), shape=(10, 10))

    # Test graphs
    adj_mat_3 = coo_matrix(([24, 39, 28], ([3, 2, 4], [4, 5, 6])), shape=(10, 10))
    adj_mat_4 = coo_matrix(([42, 90, 28], ([3, 5, 7], [4, 5, 6])), shape=(10, 10))

    train_set = [to_sparse_vector(adj_mat_1), to_sparse_vector(adj_mat_2)]

    test_set = [to_sparse_vector(adj_mat_3), to_sparse_vector(adj_mat_4)]

    indices = identify_active_indices(train_set) if use_indices else None
    print(indices)

    train_features = featurize_vectors(train_set, indices)
    print(train_features[0], train_features[1])
    assert len(train_features[0]) == expected_length

    test_features = featurize_vectors(test_set, indices)
    print(test_features[0], test_features[1])
    assert len(test_features[0]) == expected_length
