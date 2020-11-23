#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time
import traceback
import io
import gc
import re
from typing import Dict, List, NamedTuple, Set, Optional

import numpy as np
import torch

from tda.devices import device
from tda.embeddings import KernelType
from tda.dataset.graph_dataset import DatasetLine
from tda.tda_logging import get_logger
from tda.models import mnist_mlp, Dataset, get_deep_model
from tda.models.architectures import get_architecture, Architecture
from tda.protocol import get_protocolar_datasets, evaluate_embeddings

from tda.covariance import (
    CovarianceStreamComputer,
    NaiveCovarianceStreamComputer,
    LedoitWolfComputer,
    NaiveSVDCovarianceStreamComputer,
    GraphicalLassoComputer,
)

logger = get_logger("Mahalanobis")

start_time = time.time()

plot_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)


# Custom types for better readability
LayerIndex = int
ClassIndex = int


class Config(NamedTuple):
    # Number of epochs for the model
    epochs: int
    # Dataset we consider (MNIST, SVHN)
    dataset: str
    # Name of the architecture
    architecture: str
    # Size of the dataset used for the experiment
    dataset_size: int
    # Type of attack (FGSM, PGD, CW)
    attack_type: str
    # Epsilon for the preprocessing step (see the paper)
    preproc_epsilon: float
    # Selected layers
    selected_layers: Optional[Set[int]]
    # Noise to consider for the noisy samples
    noise: float
    # Number of sample per class to estimate mu_class and sigma_class
    number_of_samples_for_mu_sigma: int
    # Should we filter out non successful_adversaries
    successful_adv: int
    # Method for estimating covariance and precision matrices
    covariance_method: str
    # Transfered attacks
    transfered_attacks: bool = False
    # Pruning
    first_pruned_iter: int = 10
    prune_percentile: float = 0.0
    tot_prune_percentile: float = 0.0
    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0

    all_epsilons: List[float] = None


def str2bool(value):
    if value in [True, "True", "true"]:
        return True
    else:
        return False


def get_config() -> Config:
    parser = argparse.ArgumentParser(
        description="Transform a dataset in pail files to tf records."
    )
    parser.add_argument("--experiment_id", type=int)
    parser.add_argument("--run_id", type=int)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--architecture", type=str, default=mnist_mlp.name)
    parser.add_argument("--dataset_size", type=int, default=500)
    parser.add_argument("--number_of_samples_for_mu_sigma", type=int, default=100)
    parser.add_argument("--attack_type", type=str, default="FGSM")
    parser.add_argument("--preproc_epsilon", type=float, default=0.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--successful_adv", type=int, default=0)
    parser.add_argument("--transfered_attacks", type=str2bool, default=False)
    parser.add_argument("--all_epsilons", type=str)
    parser.add_argument("--first_pruned_iter", type=int, default=10)
    parser.add_argument("--prune_percentile", type=float, default=0.0)
    parser.add_argument("--tot_prune_percentile", type=float, default=0.0)
    parser.add_argument("--selected_layers", type=str, default="all")
    parser.add_argument(
        "--covariance_method", type=str, default=CovarianceMethod.NAIVE_SVD
    )

    args, _ = parser.parse_known_args()

    if args.all_epsilons is not None:
        args.all_epsilons = list(map(float, str(args.all_epsilons).split(";")))

    if args.selected_layers != "all":
        args.selected_layers = set([int(s) for s in args.selected_layers.split(";")])
    else:
        args.selected_layers = None

    return Config(**args.__dict__)


class CovarianceMethod(object):
    NAIVE = "NAIVE"
    NAIVE_SVD = "NAIVE_SVD"
    LEDOIT_WOLF = "LEDOIT_WOLF"
    GRAPHICAL_LASSO = "GRAPHICAL_LASSO"


def get_covariance_estimator(layer_idx: int, config: Config):
    if config.covariance_method == CovarianceMethod.NAIVE:
        return NaiveCovarianceStreamComputer(
            layer_idx=layer_idx,
            min_count_for_sigma=config.number_of_samples_for_mu_sigma // 4,
        )
    elif config.covariance_method == CovarianceMethod.LEDOIT_WOLF:
        return LedoitWolfComputer()
    elif config.covariance_method == CovarianceMethod.NAIVE_SVD:
        return NaiveSVDCovarianceStreamComputer()
    elif config.covariance_method == CovarianceMethod.GRAPHICAL_LASSO:
        return GraphicalLassoComputer()
    else:
        raise NotImplementedError(
            f"Unknown CovarianceMethod {config.covariance_method}"
        )


def compute_means_and_sigmas_inv(
    config: Config, dataset: Dataset, architecture: Architecture
):
    assert architecture.is_trained

    ##########################################
    # Step 1: Compute all mu_clazz and sigma #
    #         for each layer                 #
    ##########################################

    nb_sample_for_classes = 0

    logger.info(
        f"I am going to go through a dataset of {config.number_of_samples_for_mu_sigma} points..."
    )
    logger.info(f"Config is {config}")

    stream_computers: Dict[LayerIndex, CovarianceStreamComputer] = dict()
    all_classes: Set[ClassIndex] = set()

    if config.selected_layers is not None:
        config.selected_layers.add(architecture.get_pre_softmax_idx())

    # As proposed in the paper, we use the train samples here
    for x, y in dataset.train_dataset:

        if nb_sample_for_classes >= config.number_of_samples_for_mu_sigma:
            break

        m_features = architecture.forward(
            x=x, store_for_graph=False, output="all_inner"
        )

        all_classes.add(y)

        for layer_idx in m_features:
            if config.selected_layers is None or layer_idx in config.selected_layers:

                feat = m_features[layer_idx].reshape(1, -1)

                if layer_idx not in stream_computers:
                    stream_computers[layer_idx] = get_covariance_estimator(
                        layer_idx=layer_idx, config=config
                    )
                stream_computers[layer_idx].append(
                    x=feat.cpu().detach().numpy(), clazz=y
                )

        if nb_sample_for_classes % 50 == 0:
            logger.info(
                f"Done {nb_sample_for_classes} / {config.number_of_samples_for_mu_sigma}"
            )
        nb_sample_for_classes += 1

    mean_per_class: Dict[LayerIndex, Dict[ClassIndex, np.ndarray]] = dict()
    precision_root_per_layer: Dict[LayerIndex, np.ndarray] = dict()

    layer_indices = list(stream_computers.keys())

    for layer_idx in layer_indices:
        precision_root_per_layer[layer_idx] = stream_computers[
            layer_idx
        ].precision_root()

        mean_per_class[layer_idx] = {
            clazz: stream_computers[layer_idx].mean_per_class(clazz)
            for clazz in all_classes
        }

        del stream_computers[layer_idx]
        gc.collect()

    logger.info(f"All classes are {all_classes}")
    logger.info(f"All layers are {stream_computers.keys()}")

    ################################################################
    # Step 2: (OPT) Evaluate classifier based on confidence scores #
    ################################################################

    i = 0
    corr = 0
    while i < config.dataset_size:
        x, y = dataset.test_and_val_dataset[i]
        all_inner_activations = architecture.forward(
            x=x, store_for_graph=False, output="all_inner"
        )

        # Assuming only one link to the softmax layer in the model
        softmax_layer_idx = architecture.get_pre_softmax_idx()
        m_features = (
            all_inner_activations[softmax_layer_idx]
            .reshape(1, -1)
            .cpu()
            .detach()
            .numpy()
        )

        best_score = np.inf
        best_class = -1

        for clazz in all_classes:
            mu_clazz = mean_per_class[softmax_layer_idx][clazz]
            precision_root = precision_root_per_layer[softmax_layer_idx]

            score_clazz = (
                (m_features - mu_clazz)
                @ np.transpose(precision_root)
                @ precision_root
                @ np.transpose(m_features - mu_clazz)
            )

            if score_clazz < best_score:
                best_score = score_clazz
                best_class = clazz

        if best_class == y:
            corr += 1

        i += 1

    gaussian_accuracy = corr / config.dataset_size

    logger.info(f"Accuracy on test set = {gaussian_accuracy}")

    return mean_per_class, precision_root_per_layer, gaussian_accuracy


def get_feature_datasets(
    config: Config,
    dataset: Dataset,
    architecture: Architecture,
    mean_per_class: Dict,
    precision_root_per_layer: Dict,
    epsilons: List[float],
):
    assert architecture.is_trained

    all_feature_indices = sorted(list(mean_per_class.keys()))
    logger.info(f"All indices for features are {all_feature_indices}")
    all_classes = sorted(list(mean_per_class[all_feature_indices[0]].keys()))
    logger.info(f"All classes are {all_classes}")

    train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
        noise=config.noise,
        dataset=dataset,
        succ_adv=config.successful_adv > 0,
        archi=architecture,
        dataset_size=config.dataset_size,
        attack_type=config.attack_type,
        all_epsilons=epsilons,
        transfered_attacks=config.transfered_attacks,
    )

    def create_dataset(input_dataset: List[DatasetLine]) -> List:
        ret = list()

        for i, dataset_line in enumerate(input_dataset):
            logger.debug(f"Processing {i}/{len(input_dataset)}")

            x = dataset_line.x

            m_features = architecture.forward(
                x=x, store_for_graph=False, output="all_inner"
            )

            scores = list()

            for layer_idx in all_feature_indices:

                best_score = np.inf
                precision_root = precision_root_per_layer[layer_idx]
                mu_l = None

                for clazz in all_classes:
                    mu_clazz = mean_per_class[layer_idx][clazz]
                    gap = (
                        m_features[layer_idx].reshape(1, -1).cpu().detach().numpy()
                        - mu_clazz
                    )
                    score_clazz = (
                        gap
                        @ np.transpose(precision_root)
                        @ precision_root
                        @ np.transpose(gap)
                    )

                    if score_clazz < best_score:
                        best_score = score_clazz[0, 0]
                        mu_l = mu_clazz

                if mu_l is None:
                    logger.error(f"mu_l is still None for layer {layer_idx}")
                    logger.error(f"Best score is {best_score}")
                    logger.error(
                        f"Means per class for this layer are {mean_per_class[layer_idx]}"
                    )
                    logger.error(f"Input dataline is {dataset_line}")
                    raise RuntimeError("mu_l is None")

                # OPT: Add perturbation on x to reduce its score
                # (mainly useful for out-of-distribution ?)
                if config.preproc_epsilon > 0:
                    # Let's forget about the past (attack, clamping)
                    # and make x a leaf with require grad
                    x = x.detach()
                    x.requires_grad = True

                    precision_root_tensor = torch.from_numpy(precision_root).to(device)
                    mu_tensor = torch.from_numpy(mu_l).to(device)

                    # Adding perturbation on x
                    f = architecture.forward(
                        x=x, store_for_graph=False, output="all_inner"
                    )[layer_idx].reshape(1, -1)

                    live_score = ((f - mu_tensor) @ precision_root_tensor.T) @ (
                        precision_root_tensor @ (f - mu_tensor).T
                    )

                    if not np.isclose(
                        live_score.cpu().detach().numpy(), best_score, atol=1e-3
                    ):
                        debug_messages = list()

                        distance = np.linalg.norm(
                            live_score.cpu().detach().numpy() - best_score, 2
                        )

                        logger.warn(
                            f"Live score {live_score.cpu().detach().numpy()}"
                            f" and best_score {best_score} are different (dist={distance})\n\n"
                            + "\n".join(debug_messages)
                        )

                    architecture.zero_grad()
                    live_score.backward()
                    assert x.grad is not None
                    xhat = x - config.preproc_epsilon * x.grad.data.sign()
                    xhat = torch.clamp(xhat, -0.5, 0.5)

                    # Computing new score
                    fhat = architecture.forward(
                        x=xhat, store_for_graph=False, output="all_inner"
                    )[layer_idx].reshape(1, -1)
                    new_score = ((fhat - mu_tensor) @ precision_root_tensor.T) @ (
                        precision_root_tensor @ (fhat - mu_tensor).T
                    )

                    logger.debug(
                        f"Added perturbation to x {live_score.cpu().detach().numpy()[0, 0]} "
                        f"-> {new_score.cpu().detach().numpy()[0, 0]}"
                    )

                    # Now we have to do a second pass of optimization
                    best_score = np.inf
                    fhat = fhat.cpu().detach().numpy()

                    for clazz in all_classes:
                        mu_clazz = mean_per_class[layer_idx][clazz]
                        gap = fhat - mu_clazz
                        score_clazz = (gap @ np.transpose(precision_root)) @ (
                            precision_root @ np.transpose(gap)
                        )

                        if score_clazz < best_score:
                            best_score = score_clazz[0, 0]

                scores.append(best_score)

            ret.append(scores)
            i += 1

        return ret

    embeddings_train = create_dataset(train_clean)
    embeddings_test = create_dataset(test_clean)

    adv_embedding_train = {
        epsilon: create_dataset(train_adv[epsilon]) for epsilon in epsilons
    }

    adv_embedding_test = {
        epsilon: create_dataset(test_adv[epsilon]) for epsilon in epsilons
    }

    stats = {
        epsilon: [line.l2_norm for line in test_adv[epsilon]] for epsilon in epsilons
    }

    return (
        embeddings_train,
        embeddings_test,
        adv_embedding_train,
        adv_embedding_test,
        stats,
    )


def run_experiment(config: Config):
    logger.info(f"Starting experiment {config.experiment_id}_{config.run_id}")

    dataset = Dataset(name=config.dataset)

    logger.info(f"Getting deep model...")
    architecture: Architecture = get_deep_model(
        num_epochs=config.epochs,
        dataset=dataset,
        architecture=get_architecture(config.architecture),
        train_noise=0.0,
        prune_percentile=config.prune_percentile,
        tot_prune_percentile=config.tot_prune_percentile,
        first_pruned_iter=config.first_pruned_iter,
    )

    (
        mean_per_class,
        precision_root_per_layer,
        gaussian_accuracy,
    ) = compute_means_and_sigmas_inv(
        config=config, dataset=dataset, architecture=architecture
    )

    if config.attack_type not in ["FGSM", "PGD"]:
        all_epsilons = [1.0]
    elif config.all_epsilons is None:
        all_epsilons = [0.01, 0.05, 0.1, 0.4, 1.0]
        # all_epsilons = [0.01]
    else:
        all_epsilons = config.all_epsilons

    (
        embeddings_train,
        embeddings_test,
        adv_embedding_train,
        adv_embedding_test,
        stats,
    ) = get_feature_datasets(
        config=config,
        epsilons=all_epsilons,
        dataset=dataset,
        architecture=architecture,
        mean_per_class=mean_per_class,
        precision_root_per_layer=precision_root_per_layer,
    )

    if config.attack_type in ["DeepFool", "CW"]:
        stats_for_l2_norm_buckets = stats
    else:
        stats_for_l2_norm_buckets = dict()

    evaluation_results = evaluate_embeddings(
        embeddings_train=list(embeddings_train),
        embeddings_test=list(embeddings_test),
        all_adv_embeddings_train=adv_embedding_train,
        all_adv_embeddings_test=adv_embedding_test,
        param_space=[{"gamma": gamma} for gamma in np.logspace(-6, 3, 50)],
        kernel_type=KernelType.RBF,
        stats_for_l2_norm_buckets=stats_for_l2_norm_buckets,
    )

    logger.info(evaluation_results)

    metrics = {
        "name": "Mahalanobis",
        "time": time.time() - start_time,
        "gaussian_accuracy": gaussian_accuracy,
        **evaluation_results,
    }

    logger.info(metrics)

    return metrics


if __name__ == "__main__":
    my_config = get_config()
    try:
        run_experiment(my_config)
    except Exception as e:
        my_trace = io.StringIO()
        traceback.print_exc(file=my_trace)

        logger.error(my_trace.getvalue())
