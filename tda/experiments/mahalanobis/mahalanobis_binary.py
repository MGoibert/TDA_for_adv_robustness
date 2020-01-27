#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time
import typing

import numpy as np
import pandas as pd
import torch
from r3d3.experiment_db import ExperimentDB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

from tda.graph_dataset import get_sample_dataset
from tda.logging import get_logger
from tda.models import mnist_mlp, Dataset, get_deep_model
from tda.models.architectures import get_architecture, Architecture
from tda.rootpath import db_path
from tda.devices import device

logger = get_logger("Mahalanobis")

start_time = time.time()

plot_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

my_db = ExperimentDB(db_path=db_path)


class Config(typing.NamedTuple):
    # Number of epochs for the model
    epochs: int
    # Dataset we consider (MNIST, SVHN)
    dataset: str
    # Name of the architecture
    architecture: str
    # Size of the dataset used for the experiment
    dataset_size: int
    # Type of attack (FGSM, BIM, CW)
    attack_type: str
    # Epsilon for the preprocessing step (see the paper)
    preproc_epsilon: float
    # Noise to consider for the noisy samples
    noise: float
    # Number of sample per class to estimate mu_class and sigma_class
    number_of_samples_for_mu_sigma: int
    # Should we filter out non successful_adversaries
    successful_adv: int

    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0


def get_config() -> Config:
    parser = argparse.ArgumentParser(
        description='Transform a dataset in pail files to tf records.')
    parser.add_argument('--experiment_id', type=int)
    parser.add_argument('--run_id', type=int)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dataset', type=str, default="MNIST")
    parser.add_argument('--architecture', type=str, default=mnist_mlp.name)
    parser.add_argument('--dataset_size', type=int, default=100)
    parser.add_argument('--number_of_samples_for_mu_sigma', type=int, default=100)
    parser.add_argument('--attack_type', type=str, default="FGSM")
    parser.add_argument('--preproc_epsilon', type=float, default=0.0)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--successful_adv', type=int, default=0)

    args, _ = parser.parse_known_args()

    return Config(**args.__dict__)


def compute_means_and_sigmas_inv(
        config: Config,
        dataset: Dataset,
        architecture: Architecture
):
    assert architecture.is_trained

    ##########################################
    # Step 1: Compute all mu_clazz and sigma #
    #         for each layer                 #
    ##########################################

    nb_sample_for_classes = 0

    logger.info(f"I am going to go through a dataset of {config.number_of_samples_for_mu_sigma} points...")

    features_per_class = dict()
    mean_per_class = dict()
    sigma_per_class = dict()

    # As proposed in the paper, we use the train samples here
    for x, y in dataset.train_dataset:

        if nb_sample_for_classes >= config.number_of_samples_for_mu_sigma:
            break

        m_features = architecture.forward(
            x=x,
            store_for_graph=False,
            output="all_inner"
        )

        for layer_idx in m_features:
            feat = m_features[layer_idx].reshape(1, -1)
            if layer_idx not in features_per_class:
                features_per_class[layer_idx] = dict()
            features_per_class[layer_idx][y] = \
                features_per_class[layer_idx].get(y, list()) + [feat.cpu().detach().numpy()]

        if nb_sample_for_classes % 50 == 0:
            logger.info(f"Done {nb_sample_for_classes} / {config.number_of_samples_for_mu_sigma}")

        nb_sample_for_classes += 1

    all_feature_indices = sorted(list(features_per_class.keys()))
    logger.info(f"All indices for features are {all_feature_indices}")
    all_classes = sorted(list(features_per_class[all_feature_indices[0]].keys()))
    logger.info(f"All classes are {all_classes}")

    for layer_idx in all_feature_indices:
        mean_per_class[layer_idx] = dict()
        sigma_per_class[layer_idx] = None
        counters_per_class = 0

        for clazz in all_classes:

            # Shortcut for the name
            arr = features_per_class[layer_idx][clazz]

            # Computing mu_clazz
            mu_clazz = sum(arr) / len(arr)
            mean_per_class[layer_idx][clazz] = mu_clazz

            # Computing sigma_clazz
            sigma_clazz = sum([np.transpose(v - mu_clazz) @ (v - mu_clazz) for v in arr])
            if sigma_per_class[layer_idx] is None:
                sigma_per_class[layer_idx] = sigma_clazz
            else:
                sigma_per_class[layer_idx] += sigma_clazz
            counters_per_class += len(arr)

        sigma_per_class[layer_idx] = sigma_per_class[layer_idx] / counters_per_class

    logger.info("Computing inverse of sigmas...")
    sigma_per_class_inv = dict()
    for layer_idx in sigma_per_class:
        logger.info(f"Processing sigma for layer {layer_idx} (shape is {sigma_per_class[layer_idx].shape})")
        sigma_per_class_inv[layer_idx] = np.linalg.pinv(sigma_per_class[layer_idx], hermitian=True)

    logger.info("Done.")

    ################################################################
    # Step 2: (OPT) Evaluate classifier based on confidence scores #
    ################################################################

    i = 0
    corr = 0
    while i < config.dataset_size:
        x, y = dataset.test_and_val_dataset[i]
        all_inner_activations = architecture.forward(
            x=x,
            store_for_graph=False,
            output="all_inner"
        )

        # Assuming only one link to the softmax layer in the model
        softmax_layer_idx = architecture.get_pre_softmax_idx()
        m_features = all_inner_activations[softmax_layer_idx].reshape(1, -1).cpu().detach().numpy()

        best_score = np.inf
        best_class = -1

        for clazz in all_classes:
            mu_clazz = mean_per_class[softmax_layer_idx][clazz]
            sigma_clazz_inv = sigma_per_class_inv[softmax_layer_idx]

            score_clazz = (m_features - mu_clazz) @ sigma_clazz_inv @ np.transpose(m_features - mu_clazz)

            if score_clazz < best_score:
                best_score = score_clazz
                best_class = clazz

        if best_class == y:
            corr += 1

        i += 1

    logger.info(f"Accuracy on test set = {corr / config.dataset_size}")

    return mean_per_class, sigma_per_class_inv


def get_feature_datasets(
        config: Config,
        dataset: Dataset,
        architecture: Architecture,
        mean_per_class: typing.Dict,
        sigma_per_class_inv: typing.Dict,
        epsilon: float,
        num_iter: int = 50
):
    assert architecture.is_trained

    all_feature_indices = sorted(list(mean_per_class.keys()))
    logger.info(f"All indices for features are {all_feature_indices}")
    all_classes = sorted(list(mean_per_class[all_feature_indices[0]].keys()))
    logger.info(f"All classes are {all_classes}")

    logger.info(f"Evaluating epsilon={epsilon} / num_iter={num_iter}")

    def create_dataset(
            adv: bool
    ) -> pd.DataFrame:
        ret = list()

        for i, dataset_line in enumerate(get_sample_dataset(
                adv=adv,
                dataset=dataset,
                archi=architecture,
                dataset_size=config.dataset_size,
                succ_adv=config.successful_adv > 0.5,
                attack_type=config.attack_type,
                num_iter=num_iter,
                epsilon=epsilon,
                noise=config.noise,
                train=False
        )):
            logger.debug(f"{i}/{config.dataset_size} (adv={adv})")

            x = dataset_line.x

            m_features = architecture.forward(
                x=x,
                store_for_graph=False,
                output="all_inner"
            )

            scores = list()

            for layer_idx in all_feature_indices:

                best_score = np.inf
                sigma_l_inv = sigma_per_class_inv[layer_idx]
                mu_l = None

                for clazz in all_classes:
                    mu_clazz = mean_per_class[layer_idx][clazz]
                    gap = m_features[layer_idx].reshape(1, -1).cpu().detach().numpy() - mu_clazz
                    score_clazz = gap @ sigma_l_inv @ np.transpose(gap)

                    if score_clazz < best_score:
                        best_score = score_clazz[0, 0]
                        mu_l = mu_clazz

                # OPT: Add perturbation on x to reduce its score
                # (mainly useful for out-of-distribution ?)
                if config.preproc_epsilon > 0:
                    # Let's forget about the past (attack, clamping)
                    # and make x a leaf with require grad
                    x = x.detach()
                    x.requires_grad = True

                    inv_sigma_tensor = torch.from_numpy(sigma_l_inv).to(device)
                    mu_tensor = torch.from_numpy(mu_l).to(device)

                    # Adding perturbation on x
                    f = architecture.forward(
                        x=x,
                        store_for_graph=False,
                        output="all_inner"
                    )[layer_idx].reshape(1, -1)

                    live_score = (f - mu_tensor) @ inv_sigma_tensor @ (f - mu_tensor).T

                    assert np.isclose(live_score.cpu().detach().numpy(), best_score)

                    architecture.zero_grad()
                    live_score.backward()
                    assert x.grad is not None
                    xhat = x - config.preproc_epsilon * x.grad.data.sign()
                    xhat = torch.clamp(xhat, -0.5, 0.5)

                    # Computing new score
                    fhat = architecture.forward(
                        x=xhat,
                        store_for_graph=False,
                        output="all_inner"
                    )[layer_idx].reshape(1, -1)
                    new_score = (fhat - mu_tensor) @ inv_sigma_tensor @ (fhat - mu_tensor).T

                    logger.debug(f"Added perturbation to x {live_score.detach().numpy()[0, 0]} "
                                 f"-> {new_score.detach().numpy()[0, 0]}")

                    # Now we have to do a second pass of optimization
                    best_score = np.inf
                    fhat = fhat.detach().numpy()

                    for clazz in all_classes:
                        mu_clazz = mean_per_class[layer_idx][clazz]
                        gap = fhat - mu_clazz
                        score_clazz = gap @ sigma_l_inv @ np.transpose(gap)

                        if score_clazz < best_score:
                            best_score = score_clazz[0, 0]

                scores.append(best_score)

            ret.append(scores + [int(adv), dataset_line.l2_norm, dataset_line.linf_norm])
            i += 1

        return pd.DataFrame(ret,
                            columns=[f"layer_{idx}" for idx in all_feature_indices] + ["label", "l2_norm", "linf_norm"])

    non_adv_dataset = create_dataset(adv=False)
    logger.info("Generated clean dataset for detector !")

    adv_dataset = create_dataset(adv=True)
    logger.info("Generated test dataset for detector !")

    non_adv_dataset_train, non_adv_dataset_test = train_test_split(non_adv_dataset, test_size=0.5)
    adv_dataset_train, adv_dataset_test = train_test_split(adv_dataset, test_size=0.5)

    detector_train_dataset = pd.concat([non_adv_dataset_train, adv_dataset_train])
    detector_test_dataset = pd.concat([non_adv_dataset_test, adv_dataset_test])

    return detector_train_dataset, detector_test_dataset


def evaluate_detector(
        detector_train_dataset, detector_test_dataset
):
    detector = LogisticRegression(
        fit_intercept=True,
        verbose=0,
        tol=1e-9,
        max_iter=100000,
        solver='lbfgs'
    )

    detector.fit(X=detector_train_dataset.iloc[:, :-3], y=detector_train_dataset.iloc[:, -3])
    coefs = list(detector.coef_.flatten())
    logger.info(f"Coefs of detector {coefs}")

    test_predictions = detector.predict_proba(X=detector_test_dataset.iloc[:, :-3])[:, 1]
    auc = roc_auc_score(y_true=detector_test_dataset.iloc[:, -3], y_score=test_predictions)
    logger.info(f"AUC is {auc}")

    return auc, coefs


def evaluate_detector_unsupervised(
        detector_train_dataset, detector_test_dataset
):
    detector = OneClassSVM(
        tol=1e-5,
        kernel='rbf',
        nu=0.1
    )

    detector.fit(X=detector_train_dataset.iloc[:, :-3])

    test_predictions = detector.decision_function(X=detector_test_dataset.iloc[:, :-3])
    auc = roc_auc_score(y_true=detector_test_dataset.iloc[:, -3], y_score=test_predictions)
    logger.info(f"AUC is {auc}")

    return auc


def run_experiment(config: Config):
    logger.info(f"Starting experiment {config.experiment_id}_{config.run_id}")

    if __name__ != "__main__":
        my_db.add_experiment(
            experiment_id=config.experiment_id,
            run_id=config.run_id,
            config=config._asdict()
        )

    dataset = Dataset(name=config.dataset)

    logger.info(f"Getting deep model...")
    architecture: Architecture = get_deep_model(
        num_epochs=config.epochs,
        dataset=dataset,
        architecture=get_architecture(config.architecture),
        train_noise=0.0
    )

    aucs = dict()
    aucs_unsupervised = dict()
    coefs = dict()

    mean_per_class, sigma_per_class_inv = compute_means_and_sigmas_inv(
        config=config,
        dataset=dataset,
        architecture=architecture
    )

    if config.attack_type in ["FGSM", "BIM"]:
        all_epsilons = [0.01, 0.025, 0.05, 0.1, 0.4]
    else:
        all_epsilons = [1.0]  # Not used for DeepFool and CW

    for epsilon in all_epsilons:
        ds_train, ds_test = get_feature_datasets(
            config=config,
            epsilon=epsilon,
            dataset=dataset,
            architecture=architecture,
            mean_per_class=mean_per_class,
            sigma_per_class_inv=sigma_per_class_inv,
            num_iter=50  # Fixed for DeepFool and CW
        )

        auc, coef = evaluate_detector(ds_train, ds_test)
        auc_unsupervised = evaluate_detector_unsupervised(ds_train, ds_test)

        aucs[epsilon] = auc
        coefs[epsilon] = coef
        aucs_unsupervised[epsilon] = auc_unsupervised

    logger.info(f"All AUCS are {aucs}")

    my_db.update_experiment(
        experiment_id=config.experiment_id,
        run_id=config.run_id,
        metrics={
            "time": time.time() - start_time,
            "aucs": aucs,
            "aucs_unsupervised": aucs_unsupervised,
            "coefs": coefs
        }
    )


if __name__ == "__main__":
    config = get_config()
    run_experiment(config)
