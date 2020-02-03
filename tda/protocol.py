import typing
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM, SVC

from tda.embeddings import get_gram_matrix
from tda.graph_dataset import get_sample_dataset
from tda.models import Architecture, Dataset
from tda.logging import get_logger

logger = get_logger("C3PO")


def get_protocolar_datasets(
        noise: float,
        dataset: Dataset,
        succ_adv: bool,
        archi: Architecture,
        dataset_size: int,
        attack_type: str,
        all_epsilons: typing.List
):
    logger.info("I will produce for you the protocolar datasets !")

    train_clean = get_sample_dataset(
        adv=False,
        epsilon=0.0,
        noise=0.0,
        dataset=dataset,
        train=False,
        succ_adv=succ_adv,
        archi=archi,
        dataset_size=dataset_size // 2,
        offset=0
    )

    if False:  # noise > 0.0:
        train_clean += get_sample_dataset(
            adv=False,
            epsilon=0.0,
            noise=noise,
            dataset=dataset,
            train=False,
            succ_adv=succ_adv,
            archi=archi,
            dataset_size=dataset_size // 2,
            offset=0
        )

    test_clean = get_sample_dataset(
        adv=False,
        epsilon=0.0,
        noise=0.0,
        dataset=dataset,
        train=False,
        succ_adv=succ_adv,
        archi=archi,
        dataset_size=dataset_size // 2,
        offset=dataset_size // 2
    )

    if False:  # noise > 0.0:
        test_clean += get_sample_dataset(
            adv=False,
            epsilon=0.0,
            noise=noise,
            dataset=dataset,
            train=False,
            succ_adv=succ_adv,
            archi=archi,
            dataset_size=dataset_size // 2,
            offset=dataset_size // 2
        )

    train_adv = dict()
    test_adv = dict()

    for epsilon in all_epsilons:
        adv = get_sample_dataset(
            adv=True,
            noise=0.0,
            dataset=dataset,
            train=False,
            succ_adv=succ_adv,
            archi=archi,
            attack_type=attack_type,
            epsilon=epsilon,
            num_iter=50,
            dataset_size=dataset_size,
            offset=dataset_size
        )

        train_adv[epsilon], test_adv[epsilon] = train_test_split(adv, test_size=0.5, random_state=37)

    return train_clean, test_clean, train_adv, test_adv


def evaluate_embeddings(
        embeddings_train: typing.List,
        embeddings_test: typing.List,
        all_adv_embeddings_train: typing.Dict,
        all_adv_embeddings_test: typing.Dict,
        param_space: typing.List,
        kernel_type: str
) -> (float, float):
    """
    Compute the AUC for a given epsilon and returns also the scores
    of the best OneClass SVM
    """

    logger.info(f"I will evaluate your embeddings with {kernel_type} kernel !")
    logger.info(f"Found {len(embeddings_train)} clean embeddings for train")
    logger.info(f"Found {len(embeddings_test)} clean embeddings for test")

    gram_train_matrices = get_gram_matrix(
            kernel_type=kernel_type,
            embeddings_in=embeddings_train,
            embeddings_out=None,
            params=param_space
    )

    logger.info(f"Computed all unsupervised Gram train matrices !")

    aucs = dict()
    aucs_supervised = dict()

    for key in all_adv_embeddings_train:

        best_auc = 0.0
        best_auc_supervised = 0.0
        best_param = None
        best_param_supervised = None

        adv_embeddings_test = all_adv_embeddings_test[key]
        adv_embeddings_train = all_adv_embeddings_train[key]

        start_time = time.time()
        gram_test_and_bad = get_gram_matrix(
            kernel_type=kernel_type,
            embeddings_in=list(embeddings_test) + list(adv_embeddings_test),
            embeddings_out=list(embeddings_train),
            params=param_space
        )
        logger.info(f"Computed Gram Test Matrix in {time.time() - start_time} secs")

        gram_train_supervised = get_gram_matrix(
            kernel_type=kernel_type,
            embeddings_in=list(embeddings_train) + list(adv_embeddings_train),
            embeddings_out=None,
            params=param_space
        )

        gram_test_supervised = get_gram_matrix(
            kernel_type=kernel_type,
            embeddings_in=list(embeddings_test) + list(adv_embeddings_test),
            embeddings_out=list(embeddings_train) + list(adv_embeddings_train),
            params=param_space
        )

        for i, param in enumerate(param_space):

            #########################
            # Unsupervised Learning #
            #########################

            ocs = OneClassSVM(
                tol=1e-5,
                kernel="precomputed")

            # Training model
            start_time = time.time()
            logger.info(f"sum gram matrix train = {gram_train_matrices[i].sum()}")
            ocs.fit(gram_train_matrices[i])
            logger.info(f"Trained model in {time.time() - start_time} secs")

            # Testing model
            predictions = ocs.score_samples(gram_test_and_bad[i])

            labels = np.concatenate(
                (
                    np.ones(len(embeddings_test)),
                    np.zeros(len(adv_embeddings_test))
                )
            )

            roc_auc_val = roc_auc_score(y_true=labels, y_score=predictions)
            logger.info(f"AUC score for param = {param} : {roc_auc_val}")

            if roc_auc_val > best_auc:
                best_auc = roc_auc_val
                best_param = param

            #######################
            # Supervised Learning #
            #######################

            detector = SVC(
                verbose=0,
                tol=1e-9,
                max_iter=100000,
                kernel='precomputed'
            )

            labels_train = np.concatenate(
                (
                    np.ones(len(embeddings_train)),
                    np.zeros(len(adv_embeddings_train))
                )
            )

            detector.fit(gram_train_supervised[i], labels_train)

            predictions = detector.decision_function(gram_test_supervised[i])

            roc_auc_val = roc_auc_score(y_true=labels, y_score=predictions)
            logger.info(f"Supervised AUC score for param = {param} : {roc_auc_val}")

            if roc_auc_val > best_auc_supervised:
                best_auc_supervised = roc_auc_val
                best_param_supervised = param

        aucs[key] = best_auc
        aucs_supervised[key] = best_auc_supervised

        logger.info(f"Best param unsupervised {best_param}")
        logger.info(f"Best param supervised {best_param_supervised}")

        logger.info(f"Best auc unsupervised {best_auc}")
        logger.info(f"Best auc supervised {best_auc_supervised}")

    return aucs, aucs_supervised
