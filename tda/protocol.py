import typing
import time
import random

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM, SVC
from sklearn.utils import check_random_state

from tda.embeddings import get_gram_matrix
from tda.graph_dataset import get_sample_dataset
from tda.models import Architecture, Dataset
from tda.tda_logging import get_logger

logger = get_logger("C3PO")


def get_protocolar_datasets(
    noise: float,
    dataset: Dataset,
    succ_adv: bool,
    archi: Architecture,
    dataset_size: int,
    attack_type: str,
    all_epsilons: typing.List,
    compute_graph: bool = False,
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
        dataset_size=dataset_size // 2,  # 8,
        offset=0,
        compute_graph=compute_graph,
    )

    if noise > 0.0:
        train_clean += get_sample_dataset(
            adv=False,
            epsilon=0.0,
            noise=noise,
            dataset=dataset,
            train=False,
            succ_adv=succ_adv,
            archi=archi,
            dataset_size=dataset_size // 2,  # 8,
            offset=0,
            compute_graph=compute_graph,
        )

    test_clean = get_sample_dataset(
        adv=False,
        epsilon=0.0,
        noise=0.0,
        dataset=dataset,
        train=False,
        succ_adv=succ_adv,
        archi=archi,
        dataset_size=dataset_size // 2,  # 8,
        offset=dataset_size // 2,  # 8,
        compute_graph=compute_graph,
    )

    if noise > 0.0:
        test_clean += get_sample_dataset(
            adv=False,
            epsilon=0.0,
            noise=noise,
            dataset=dataset,
            train=False,
            succ_adv=succ_adv,
            archi=archi,
            dataset_size=dataset_size // 2,  # 8,
            offset=dataset_size // 2,  # 8,
            compute_graph=compute_graph,
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
            num_iter=100,
            dataset_size=dataset_size,
            offset=dataset_size,
            compute_graph=compute_graph,
        )

        train_adv[epsilon], test_adv[epsilon] = train_test_split(
            adv, test_size=0.5, random_state=37
        )

    return train_clean, test_clean, train_adv, test_adv


def score_with_confidence(
    scorer,
    y_true,
    y_pred,
    n_bootstraps=100,
    bootstrap_size=None,
    random_state=None,
    **kwargs,
):
    """
    Adapted from Olivier Grisel's https://stackoverflow.com/a/19132400/1080358
    """
    rng = check_random_state(random_state)
    n_samples = len(y_true)
    if bootstrap_size is None:
        bootstrap_size = max(100, n_samples // 2)
    n_bootstraps = min(n_samples, n_bootstraps)
    logger.debug(
        "Running %d bootstraps of size %d each" % (n_bootstraps, bootstrap_size)
    )

    true_score = scorer(y_true, y_pred, **kwargs)

    b_scores = []
    while len(b_scores) < n_bootstraps:
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, n_samples, bootstrap_size)
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for the scores
            # to be defined: reject the sample
            continue

        score = scorer(y_true[indices], y_pred[indices], **kwargs)
        b_scores.append(score)

    c10 = 2 * true_score - np.percentile(b_scores, 90)
    c90 = 2 * true_score - np.percentile(b_scores, 10)

    return c10, true_score, c90


def evaluate_embeddings(
    embeddings_train: typing.List,
    embeddings_test: typing.List,
    all_adv_embeddings_train: typing.Dict,
    all_adv_embeddings_test: typing.Dict,
    param_space: typing.List,
    kernel_type: str,
    index_l2_norm: typing.List = None
) -> (float, float, float):
    """
    Compute the AUC for a given epsilon and returns also the scores
    of the best OneClass SVM
    """

    np.random.seed(42)
    random.seed(111)

    best_auc_l2_norm = None

    logger.info(f"I will evaluate your embeddings with {kernel_type} kernel !")
    logger.info(f"Found {len(embeddings_train)} clean embeddings for train")
    logger.info(f"Found {len(embeddings_test)} clean embeddings for test")

    gram_train_matrices = get_gram_matrix(
        kernel_type=kernel_type,
        embeddings_in=embeddings_train,
        embeddings_out=None,
        params=param_space,
    )
    # with open('/Users/m.goibert/Documents/temp/gram_mat/gram_mat_train.pickle', 'wb') as f:
    #    pickle.dump(gram_train_matrices, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Computed all unsupervised Gram train matrices !")

    aucs = dict()
    aucs_supervised = dict()

    for key in all_adv_embeddings_train:

        best_auc = 0.0
        best_auc_supervised = 0.0
        best_auc_c10 = 0.0
        best_auc_c90 = 0.0
        best_auc_supervised_c10 = 0.0
        best_auc_supervised_c90 = 0.0
        best_param = None
        best_nu_param = None
        best_param_supervised = None

        adv_embeddings_test = all_adv_embeddings_test[key]
        adv_embeddings_train = all_adv_embeddings_train[key]

        start_time = time.time()
        gram_test_and_bad = get_gram_matrix(
            kernel_type=kernel_type,
            embeddings_in=list(embeddings_test) + list(adv_embeddings_test),
            embeddings_out=list(embeddings_train),
            params=param_space,
            verbatim="clean_test_and_adv",
            save=True,
        )
        # with open('/Users/m.goibert/Documents/temp/gram_mat/gram_mat_test_unsupervised_'+str(key)+'.pickle', 'wb') as f:
        #    pickle.dump(gram_test_and_bad, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Computed Gram Test Matrix in {time.time() - start_time} secs")

        gram_train_supervised = get_gram_matrix(
            kernel_type=kernel_type,
            embeddings_in=list(embeddings_train) + list(adv_embeddings_train),
            embeddings_out=None,
            params=param_space,
        )

        gram_test_supervised = get_gram_matrix(
            kernel_type=kernel_type,
            embeddings_in=list(embeddings_test) + list(adv_embeddings_test),
            embeddings_out=list(embeddings_train) + list(adv_embeddings_train),
            params=param_space,
        )

        labels = np.concatenate(
            (np.ones(len(embeddings_test)), np.zeros(len(adv_embeddings_test)))
        )

        for i, param in enumerate(param_space):

            #########################
            # Unsupervised Learning #
            #########################

            for nu in np.linspace(0.1, 0.9, 9):
                ocs = OneClassSVM(tol=1e-5, nu=nu, kernel="precomputed")

                # Training model
                start_time = time.time()
                logger.info(f"sum gram matrix train = {gram_train_matrices[i].sum()}")
                ocs.fit(gram_train_matrices[i])
                logger.info(f"Trained model in {time.time() - start_time} secs")

                # Testing model
                predictions = ocs.score_samples(gram_test_and_bad[i])
                pred_clean = predictions[: len(embeddings_test)]
                pred_adv = predictions[len(embeddings_test) :]

                # with open('/Users/m.goibert/Documents/temp/gram_mat/predict_'+str(key)+'_param='+str(i)+'.pickle', 'wb') as f:
                #    pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)

                auc_c10, auc_true, auc_c90 = score_with_confidence(
                    roc_auc_score, y_true=labels, y_pred=predictions
                )
                logger.info(f"[nu={nu}] AUC score for param = {param} : {auc_true}")

                if auc_true > best_auc:
                    best_auc = auc_true
                    best_auc_c10 = auc_c10
                    best_auc_c90 = auc_c90
                    best_nu_param = nu
                    best_param = param
                    # For separating into l2 norm buckets
                    if index_l2_norm is not None:
                        pred_adv_l2_norm = [
                            pred_adv[index_l2_norm == i + 1]
                            for i in range(len(np.unique(index_l2_norm)))
                        ]
                        lab_l2_norm = [
                            np.concatenate(
                                (np.ones(len(embeddings_test)), np.zeros(len(pred)))
                            )
                            for pred in pred_adv_l2_norm
                        ]
                        best_auc_l2_norm = [
                            roc_auc_score(
                                y_true=lab_l2_norm[i],
                                y_score=list(pred_clean) + list(pred_adv_l2_norm[i]),
                            )
                            for i in range(len(np.unique(index_l2_norm)))
                        ]
                    else:
                        best_auc_l2_norm = None

            #######################
            # Supervised Learning #
            #######################

            detector = SVC(verbose=0, tol=1e-9, max_iter=100000, kernel="precomputed")

            labels_train = np.concatenate(
                (np.ones(len(embeddings_train)), np.zeros(len(adv_embeddings_train)))
            )

            detector.fit(gram_train_supervised[i], labels_train)

            predictions = detector.decision_function(gram_test_supervised[i])

            auc_c10, auc_true, auc_c90 = score_with_confidence(
                roc_auc_score, y_true=labels, y_pred=predictions
            )
            logger.info(f"Supervised AUC score for param = {param} : {auc_true}")

            if auc_true > best_auc_supervised:
                best_auc_supervised = auc_true
                best_auc_supervised_c10 = auc_c10
                best_auc_supervised_c90 = auc_c90
                best_param_supervised = param

        aucs[key] = (best_auc_c10, best_auc, best_auc_c90)
        aucs_supervised[key] = (
            best_auc_supervised_c10,
            best_auc_supervised,
            best_auc_supervised_c90,
        )

        logger.info(f"Best param unsupervised {best_param}")
        logger.info(f"Best nu param unsupervised {best_nu_param}")
        logger.info(f"Best param supervised {best_param_supervised}")

        logger.info(f"Best auc unsupervised {best_auc}")
        logger.info(f"Best auc supervised {best_auc_supervised}")

        if index_l2_norm is not None:
            logger.info(f"Best auc l2 norm = {best_auc_l2_norm}")

    return aucs, aucs_supervised, best_auc_l2_norm
