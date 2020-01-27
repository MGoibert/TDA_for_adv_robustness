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
from sklearn.metrics.pairwise import euclidean_distances

from tda.graph_dataset import process_sample, get_graph_dataset, get_sample_dataset
from tda.models import Dataset, get_deep_model, mnist_lenet
from tda.models.architectures import SoftMaxLayer
from tda.models.architectures import get_architecture, Architecture
from tda.rootpath import db_path
from tda.logging import get_logger

logger = get_logger("LID")

start_time = time.time()

plot_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

my_db = ExperimentDB(db_path=db_path)


class Config(typing.NamedTuple):
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
    # Number of batches used in total
    nb_batches: int
    # Number of nearest neighbors for the estimation of the LID
    number_of_nn: int
    # Batch size for the estimation of the LID
    batch_size: int
    # Type of attack (FGSM, BIM, CW)
    attack_type: str
    # Parameter used by DeepFool and CW
    num_iter: int
    # Should we filter out non successful_adversaries
    successful_adv: int
    # Default parameters when running interactively for instance
    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0


def get_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', type=int, default=-1)
    parser.add_argument('--run_id', type=int, default=-1)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dataset', type=str, default="MNIST")
    parser.add_argument('--architecture', type=str, default=mnist_lenet.name)
    parser.add_argument('--nb_batches', type=int, default=10)
    parser.add_argument('--attack_type', type=str, default="FGSM")
    parser.add_argument('--num_iter', type=int, default=5)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--train_noise', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--number_of_nn', type=int, default=20)

    args, _ = parser.parse_known_args()

    return Config(**args.__dict__)


def get_feature_datasets(
        config: Config,
        epsilon: float,
        dataset: Dataset,
        archi: Architecture,
        num_iter: int = 50
):
    logger.info(f"Evaluating epsilon={epsilon}")

    all_lids_norm = list()
    all_lids_adv = list()
    all_lids_noisy = list()

    l2_noisy = list()
    linf_noisy = list()

    l2_adv = list()
    linf_adv = list()

    # Generating adv sample dataset
    adv_dataset = get_sample_dataset(
        adv=True,
        dataset=dataset,
        archi=archi,
        dataset_size=config.batch_size * config.nb_batches,
        succ_adv=config.successful_adv > 0.5,
        attack_type=config.attack_type,
        num_iter=num_iter,
        epsilon=epsilon,
        noise=config.noise,
        train=False
    )

    # Adv dataset

    for batch_idx in range(config.nb_batches):

        raw_batch_adv = adv_dataset[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size]

        #####################
        # Adversarial Batch #
        #####################

        b_adv = [line.x for line in raw_batch_adv]
        l2_adv += [line.l2_norm for line in raw_batch_adv]
        linf_adv += [line.linf_norm for line in raw_batch_adv]

        ###################################
        # Normal batch on the same points #
        ###################################

        batch_indices = [line.sample_id for line in raw_batch_adv]
        logger.debug(f"Batch indices {batch_indices}")
        b_norm = [dataset.test_and_val_dataset[idx][0] for idx in batch_indices]

        ##########################################
        # Creating noisy batch from normal batch #
        ##########################################

        b_noisy = list()

        for x_norm in b_norm:
            x_noisy, _ = process_sample(
                sample=(x_norm, 0),
                adversarial=False,
                noise=config.noise,
                epsilon=None,
                model=None,
                num_classes=None,
                attack_type=None
            )
            b_noisy.append(x_noisy)
            l2_noisy.append(np.linalg.norm(torch.abs((x_norm.double() - x_noisy.double()).flatten()).detach().numpy(), 2))
            linf_noisy.append(np.linalg.norm(torch.abs((x_norm.double() - x_noisy.double()).flatten()).detach().numpy(), np.inf))

        b_norm = torch.cat([x.unsqueeze(0) for x in b_norm])
        b_adv = torch.cat([x.unsqueeze(0) for x in b_adv])
        b_noisy = torch.cat([x.unsqueeze(0) for x in b_noisy])

        #########################
        # Computing activations #
        #########################

        a_norm = archi.forward(b_norm, output="all_inner")
        a_adv = archi.forward(b_adv, output="all_inner")
        a_noisy = archi.forward(b_noisy, output="all_inner")

        lids_norm = np.zeros((config.batch_size, len(archi.layers) - 1))
        lids_adv = np.zeros((config.batch_size, len(archi.layers) - 1))
        lids_noisy = np.zeros((config.batch_size, len(archi.layers) - 1))

        for layer_idx in archi.layer_visit_order:
            if layer_idx == -1:
                # Skipping input
                continue
            if isinstance(archi.layers[layer_idx], SoftMaxLayer):
                # Skipping softmax
                continue
            a_norm_layer = a_norm[layer_idx].reshape(config.batch_size, -1).cpu().detach().numpy()
            a_adv_layer = a_adv[layer_idx].reshape(config.batch_size, -1).cpu().detach().numpy()
            a_noisy_layer = a_noisy[layer_idx].reshape(config.batch_size, -1).cpu().detach().numpy()

            d_norm = euclidean_distances(a_norm_layer, a_norm_layer)
            d_adv = euclidean_distances(a_adv_layer, a_adv_layer)
            d_noisy = euclidean_distances(a_noisy_layer, a_noisy_layer)

            for sample_idx in range(config.batch_size):
                z_norm = d_norm[sample_idx]
                z_norm = z_norm[z_norm.argsort()[1:config.number_of_nn + 1]]

                lid_norm = -1 / (sum([np.log(x / z_norm[-1]) for x in z_norm]) / config.number_of_nn)

                z_adv = d_adv[sample_idx]
                z_adv = z_adv[z_adv.argsort()[1:config.number_of_nn + 1]]

                lid_adv = -1 / (sum([np.log(x / z_adv[-1]) for x in z_adv]) / config.number_of_nn)

                z_noisy = d_noisy[sample_idx]
                z_noisy = z_noisy[z_noisy.argsort()[1:config.number_of_nn + 1]]

                lid_noisy = -1 / (sum([np.log(x / z_noisy[-1]) for x in z_noisy]) / config.number_of_nn)

                lids_norm[sample_idx, layer_idx] = lid_norm
                lids_adv[sample_idx, layer_idx] = lid_adv
                lids_noisy[sample_idx, layer_idx] = lid_noisy

        all_lids_norm.append(lids_norm)
        all_lids_adv.append(lids_adv)
        all_lids_noisy.append(lids_noisy)

    all_lids_norm = np.concatenate(all_lids_norm, axis=0)
    all_lids_adv = np.concatenate(all_lids_adv, axis=0)
    all_lids_noisy = np.concatenate(all_lids_noisy, axis=0)

    N = len(all_lids_norm)
    logger.info(f"Computed LIDs for {N} points")

    train_features = np.concatenate([
        all_lids_norm[:N // 2],
        all_lids_adv[:N // 2]
    ])

    train_labels = np.concatenate([
        np.zeros((N // 2, 1)),
        np.ones((N // 2, 1))
    ])

    logger.info(f"Shape of train ds for LR is {np.shape(train_features)}")
    logger.info(f"Shape of train labels for LR is {np.shape(train_labels)}")

    train_ds = np.concatenate([train_features, train_labels], axis=1)
    train_ds = pd.DataFrame(train_ds,
                            columns=[f"x_{idx}" for idx in range(np.shape(train_features)[1])] + ["label"])

    test_features = np.concatenate([
        all_lids_norm[N // 2:],
        all_lids_adv[N // 2:]
    ])

    test_labels = np.concatenate([
        np.zeros((N // 2, 1)),
        np.ones((N // 2, 1))
    ])

    logger.info(f"Shape of train ds for LR is {np.shape(test_features)}")
    logger.info(f"Shape of train labels for LR is {np.shape(test_labels)}")

    test_ds = np.concatenate([test_features, test_labels], axis=1)
    test_ds = pd.DataFrame(test_ds,
                           columns=[f"x_{idx}" for idx in range(np.shape(test_features)[1])] + ["label"])

    return train_ds, test_ds


def evaluate_detector(
        train_ds, test_ds
):
    detector = LogisticRegression(
        fit_intercept=True,
        verbose=1,
        tol=1e-5,
        max_iter=1000,
        solver='lbfgs'
    )

    detector.fit(X=train_ds.iloc[:, :-1], y=train_ds.iloc[:, -1])
    coefs = list(detector.coef_.flatten())
    logger.info(f"Coefs of detector {coefs}")

    test_predictions = detector.predict_proba(X=test_ds.iloc[:, :-1])[:, 1]
    auc = roc_auc_score(y_true=test_ds.iloc[:, -1], y_score=test_predictions)
    logger.info(f"AUC is {auc}")

    return auc, coefs


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
    archi: Architecture = get_deep_model(
        num_epochs=config.epochs,
        dataset=dataset,
        architecture=get_architecture(config.architecture),
        train_noise=config.train_noise
    )

    datasets = dict()

    for epsilon in [0.01, 0.025, 0.05, 0.1, 0.4]:
        ds_train, ds_test = get_feature_datasets(
            config=config,
            epsilon=epsilon,
            dataset=dataset,
            archi=archi
        )
        datasets[epsilon] = (ds_train, ds_test)

    all_aucs = dict()
    all_coefs = dict()

    for epsilon in datasets:
        train_ds, test_ds = datasets[epsilon]

        auc, coefs = evaluate_detector(train_ds, test_ds)

        all_aucs[epsilon] = auc
        all_coefs[epsilon] = coefs

    logger.info(all_aucs)

    my_db.update_experiment(
        experiment_id=config.experiment_id,
        run_id=config.run_id,
        metrics={
            "time": time.time() - start_time,
            "classifier_coefs": all_coefs,
            "auc": all_aucs
        }
    )

    logger.info(f"Done with experiment {config.experiment_id}_{config.run_id} !")

    return all_aucs, all_coefs


if __name__ == "__main__":
    config = get_config()
    run_experiment(config)
