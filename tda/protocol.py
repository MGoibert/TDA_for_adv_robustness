import typing
import time
import random
import tempfile

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM, SVC
from sklearn.utils import check_random_state

from tda.dataset.adversarial_generation import AttackBackend
from tda.embeddings import get_gram_matrix
from tda.dataset.graph_dataset import get_sample_dataset
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
    attack_backend: str = AttackBackend.FOOLBOX,
    compute_graph: bool = False,
    transfered_attacks: bool = False,
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
        attack_type=attack_type,
        attack_backend=attack_backend,
        dataset_size=dataset_size // 2,  # 8,
        offset=0,
        compute_graph=compute_graph,
        transfered_attacks=transfered_attacks,
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
            attack_type=attack_type,
            attack_backend=attack_backend,
            dataset_size=dataset_size // 2,  # 8,
            offset=0,
            compute_graph=compute_graph,
            transfered_attacks=transfered_attacks,
        )

    test_clean = get_sample_dataset(
        adv=False,
        epsilon=0.0,
        noise=0.0,
        dataset=dataset,
        train=False,
        succ_adv=succ_adv,
        archi=archi,
        attack_type=attack_type,
        attack_backend=attack_backend,
        dataset_size=dataset_size // 2,  # 8,
        offset=dataset_size // 2,  # 8,
        compute_graph=compute_graph,
        transfered_attacks=transfered_attacks,
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
            attack_type=attack_type,
            attack_backend=attack_backend,
            dataset_size=dataset_size // 2,  # 8,
            offset=dataset_size // 2,  # 8,
            compute_graph=compute_graph,
            transfered_attacks=transfered_attacks,
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
            attack_backend=attack_backend,
            epsilon=epsilon,
            num_iter=100,
            dataset_size=dataset_size,
            offset=dataset_size,
            compute_graph=compute_graph,
            transfered_attacks=transfered_attacks,
        )

        train_adv[epsilon], test_adv[epsilon] = train_test_split(
            adv, test_size=0.5, random_state=37
        )

    return train_clean, test_clean, train_adv, test_adv


##############
# Evaluation #
##############


class Metric(typing.NamedTuple):
    upper_bound: float
    value: float
    lower_bound: float

    def is_better_than(self, other_metric: "Metric"):
        return self.value > other_metric.value


worst_metric = Metric(upper_bound=-np.infty, value=-np.infty, lower_bound=-np.infty)


def score_with_confidence(
    scorer: typing.Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 100,
    bootstrap_size: int = None,
    random_state: int = None,
    **kwargs,
) -> Metric:
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

    return Metric(lower_bound=c10, value=true_score, upper_bound=c90)


def evaluate_embeddings(
    embeddings_train: typing.List,
    embeddings_test: typing.List,
    all_adv_embeddings_train: typing.Dict,
    all_adv_embeddings_test: typing.Dict,
    param_space: typing.List,
    kernel_type: str,
    stats_for_l2_norm_buckets: typing.Dict = dict(),
) -> typing.Dict:
    """
    Compute the AUC for a given epsilon and returns also the scores
    of the best OneClass SVM
    """

    np.random.seed(42)
    random.seed(111)

    logger.info(f"I will evaluate your embeddings with {kernel_type} kernel !")
    logger.info(f"Found {len(embeddings_train)} clean embeddings for train")
    logger.info(f"Found {len(embeddings_test)} clean embeddings for test")

    gram_train_matrices = get_gram_matrix(
        kernel_type=kernel_type,
        embeddings_in=embeddings_train,
        embeddings_out=None,
        params=param_space,
    )

    logger.info(f"Computed all unsupervised Gram train matrices !")

    scorers = {"auc": roc_auc_score}

    all_metrics = dict()
    all_metrics_supervised = dict()
    all_predictions = dict()
    all_predictions_supervised = dict()
    aucs_l2_norm = dict()

    param_curve = dict()
    param_curve_supervised = dict()

    all_results_unsup_df = list()
    all_results_sup_df = list()

    for key in all_adv_embeddings_train:

        best_metrics = dict()
        best_metrics_supervised = dict()
        best_predictions = (None, None)
        best_predictions_supervised = (None, None)

        best_param = None
        best_nu_param = None
        best_param_supervised = None

        if key in stats_for_l2_norm_buckets:
            # Separate datasets as a function of L2 norms for CW or DeepFool
            bins = [
                np.quantile(stats_for_l2_norm_buckets[key], q)
                for q in np.arange(0, 1, 0.2)
            ]
            index_for_bins = np.digitize(stats_for_l2_norm_buckets[key], bins)
            logger.info(f"Quantile for L2 norm = {bins}")
        else:
            bins = list()
            index_for_bins = None

        adv_embeddings_test = all_adv_embeddings_test[key]
        adv_embeddings_train = all_adv_embeddings_train[key]

        start_time = time.time()
        gram_test_and_bad = get_gram_matrix(
            kernel_type=kernel_type,
            embeddings_in=list(embeddings_test) + list(adv_embeddings_test),
            embeddings_out=list(embeddings_train),
            params=param_space,
        )
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
                predictions = ocs.decision_function(gram_test_and_bad[i])
                pred_clean = predictions[: len(embeddings_test)]
                pred_adv = predictions[len(embeddings_test) :]

                metrics = {
                    name: score_with_confidence(
                        scorers[name], y_true=labels, y_pred=predictions
                    )
                    for name in scorers
                }
                logger.info(
                    f"[nu={nu}] AUC score for param = {param} : {metrics['auc'].value}"
                )

                all_results_unsup_df.append((nu, param, metrics["auc"].value))

                param_curve[
                    "_".join([str(key), str(nu), str(param.get("gamma", 0))])
                ] = metrics["auc"].value

                if metrics["auc"].is_better_than(best_metrics.get("auc", worst_metric)):
                    best_metrics = metrics
                    best_nu_param = nu
                    best_param = param
                    best_predictions = (list(pred_clean), list(pred_adv))
                    # For separating into l2 norm buckets
                    # (If bins is not empty)
                    if index_for_bins is not None:
                        for bin_index in list(sorted(np.unique(index_for_bins))):
                            pred_for_bin = pred_adv[index_for_bins == bin_index]
                            lab_l2_norm = np.concatenate(
                                (
                                    np.ones(len(embeddings_test)),
                                    np.zeros(len(pred_for_bin)),
                                )
                            )
                            aucs_l2_norm[
                                (bins + ["inf"])[bin_index]
                            ] = score_with_confidence(
                                roc_auc_score,
                                y_true=lab_l2_norm,
                                y_pred=np.array(list(pred_clean) + list(pred_for_bin)),
                            )

            #######################
            # Supervised Learning #
            #######################

            detector = SVC(verbose=0, tol=1e-9, max_iter=100000, kernel="precomputed")

            labels_train = np.concatenate(
                (np.ones(len(embeddings_train)), np.zeros(len(adv_embeddings_train)))
            )

            detector.fit(gram_train_supervised[i], labels_train)

            predictions = detector.decision_function(gram_test_supervised[i])

            pred_clean = predictions[: len(embeddings_test)]
            pred_adv = predictions[len(embeddings_test) :]

            metrics = {
                name: score_with_confidence(
                    scorers[name], y_true=labels, y_pred=predictions
                )
                for name in scorers
            }
            logger.info(
                f"Supervised AUC score for param = {param} : {metrics['auc'].value}"
            )

            all_results_sup_df.append((param, metrics["auc"].value))

            param_curve_supervised[
                "_".join([str(key), str(param.get("gamma", 0))])
            ] = metrics["auc"].value

            if metrics["auc"].is_better_than(
                best_metrics_supervised.get("auc", worst_metric)
            ):
                best_metrics_supervised = metrics
                best_param_supervised = param
                best_predictions_supervised = (list(pred_clean), list(pred_adv))

        all_metrics[key] = {key: best_metrics[key]._asdict() for key in best_metrics}
        all_metrics_supervised[key] = {
            key: best_metrics_supervised[key]._asdict()
            for key in best_metrics_supervised
        }
        all_predictions[key] = best_predictions
        all_predictions_supervised[key] = best_predictions_supervised

        logger.info(f"Best param unsupervised {best_param}")
        logger.info(f"Best nu param unsupervised {best_nu_param}")
        logger.info(f"Best param supervised {best_param_supervised}")

        logger.info(f"Best metrics unsupervised {best_metrics}")
        logger.info(f"Best metrics supervised {best_metrics_supervised}")

        if len(aucs_l2_norm) > 0:
            logger.info(f"Best auc l2 norm = {aucs_l2_norm}")

    evaluation_results = {
        "unsupervised_metrics": all_metrics,
        "supervised_metrics": all_metrics_supervised,
        "unsupervised_predictions": all_predictions,
        "supervised_predictions": all_predictions_supervised,
        "param_curve": param_curve,
        "aucs_l2_norm": aucs_l2_norm if len(aucs_l2_norm) > 0 else "None",
    }

    all_results_unsup_df = pd.DataFrame(
        all_results_unsup_df, columns=["nu", "param", "auc"]
    )
    all_results_sup_df = pd.DataFrame(all_results_sup_df, columns=["param", "auc"])

    tmpdir = tempfile.mkdtemp()

    with open(f"{tmpdir}/unsupervised.html", "w") as f:
        all_results_unsup_df.to_html(f)
        mlflow.log_artifact(f"{tmpdir}/unsupervised.html", "metrics_solver")
    with open(f"{tmpdir}/unsupervised.csv", "w") as f:
        all_results_unsup_df.to_csv(f)
        mlflow.log_artifact(f"{tmpdir}/unsupervised.csv", "metrics_solver")
    with open(f"{tmpdir}/supervised.html", "w") as f:
        all_results_sup_df.to_html(f)
        mlflow.log_artifact(f"{tmpdir}/supervised.html", "metrics_solver")
    with open(f"{tmpdir}/supervised.csv", "w") as f:
        all_results_sup_df.to_csv(f)
        mlflow.log_artifact(f"{tmpdir}/supervised.csv", "metrics_solver")

    return evaluation_results
