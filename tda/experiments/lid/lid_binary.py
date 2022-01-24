#!/usr/bin/env python
# coding: utf-8

import argparse
import io
import os
import re
import time
import traceback
from typing import NamedTuple, Optional, Set, List

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
import mlflow

from tda.dataset.adversarial_generation import AttackType
from tda.dataset.graph_dataset import DatasetLine
from tda.embeddings import KernelType
from tda.models import Dataset, get_deep_model, mnist_lenet
from tda.models.layers import SoftMaxLayer
from tda.models.architectures import get_architecture, Architecture
from tda.protocol import get_protocolar_datasets, evaluate_embeddings
from tda.tda_logging import get_logger

mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in")
mlflow.set_experiment("tda_adv_detection")

logger = get_logger("LID")

start_time = time.time()

plot_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)


class Config(NamedTuple):
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
    # Dataset size
    dataset_size: int
    # Percentage of nearest neighbors in a batch for the estimation of the LID
    perc_of_nn: int
    # Batch size for the estimation of the LID
    batch_size: int
    # Selected layers
    selected_layers: Optional[Set[int]]
    # Type of attack (FGSM, PGD, CW)
    attack_type: str
    attack_backend : str
    # Should we filter out non successful_adversaries
    successful_adv: int
    # Transfered attacks
    transfered_attacks: bool = False
    # Pruning
    first_pruned_iter: int = 10
    prune_percentile: float = 0.0
    tot_prune_percentile: float = 0.0
    experiment_id: float = -1
    run_id: float = -1
    all_epsilons: List[float] = None


def str2bool(value):
    if value in [True, "True", "true"]:
        return True
    else:
        return False


def get_config() -> Config:
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--architecture", type=str, default=mnist_lenet.name)
    parser.add_argument("--attack_type", type=str, default="FGSM")
    parser.add_argument("--attack_backend", type=str, default="FOOLBOX")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--train_noise", type=float, default=0.0)
    parser.add_argument("--dataset_size", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--perc_of_nn", type=float, default=0.2)
    parser.add_argument("--successful_adv", type=int, default=1)
    parser.add_argument("--transfered_attacks", type=str2bool, default=False)
    parser.add_argument("--all_epsilons", type=str)
    parser.add_argument("--first_pruned_iter", type=int, default=10)
    parser.add_argument("--prune_percentile", type=float, default=0.0)
    parser.add_argument("--tot_prune_percentile", type=float, default=0.0)
    parser.add_argument("--selected_layers", type=str, default="all")
    parser.add_argument("--experiment_id", type=float, default=-1)
    parser.add_argument("--run_id", type=float, default=-1)

    args, _ = parser.parse_known_args()

    if args.all_epsilons is not None:
        args.all_epsilons = list(map(float, str(args.all_epsilons).split(";")))

    if args.selected_layers != "all":
        args.selected_layers = set([int(s) for s in args.selected_layers.split(";")])
    else:
        args.selected_layers = None

    return Config(**args.__dict__)


def create_lid_dataset(
    config: Config,
    archi: Architecture,
    input_dataset: List[DatasetLine],
    clean_dataset: List[DatasetLine],
) -> (List, List, List):
    logger.debug(f"Dataset size is {len(input_dataset)}")

    nb_batches = int(np.ceil(len(input_dataset) / config.batch_size))

    logger.debug(f"Number of batches is {nb_batches}")

    all_lids = list()
    l2_norms = list()
    linf_norms = list()

    for batch_idx in range(nb_batches):

        raw_batch = input_dataset[
            batch_idx * config.batch_size : (batch_idx + 1) * config.batch_size
        ]
        raw_batch_clean = clean_dataset[
            batch_idx * config.batch_size : (batch_idx + 1) * config.batch_size
        ]

        actual_batch_size = len(
            raw_batch
        )  # Can be < config.batch_size (for the last batch)
        actual_batch_size_clean = len(
            raw_batch_clean
        )  # Can be < config.batch_size (for the last batch)
        number_of_nn = int(actual_batch_size * config.perc_of_nn)

        #########################
        # Computing activations #
        #########################

        activations = archi.forward(
            torch.cat([line.x.unsqueeze(0) for line in raw_batch]), output="all_inner"
        )
        activations_clean = archi.forward(
            torch.cat([line.x.unsqueeze(0) for line in raw_batch_clean]),
            output="all_inner",
        )

        lids = np.zeros((actual_batch_size, len(archi.layers)))
        valid_lid_columns = list()

        for layer_idx in archi.layer_visit_order:
            if (
                config.selected_layers is not None
                and layer_idx not in config.selected_layers
            ):
                # Skipping layer since not in selected_layers
                continue
            if layer_idx == -1:
                # Skipping input
                continue
            if isinstance(archi.layers[layer_idx], SoftMaxLayer):
                # Skipping softmax
                continue

            valid_lid_columns.append(layer_idx)

            activations_layer = (
                activations[layer_idx]
                .reshape(actual_batch_size, -1)
                .cpu()
                .detach()
                .numpy()
            )
            activations_layer_clean = (
                activations_clean[layer_idx]
                .reshape(actual_batch_size_clean, -1)
                .cpu()
                .detach()
                .numpy()
            )

            try:
                distances = euclidean_distances(
                    activations_layer, activations_layer_clean
                )
            except ValueError as exc:

                debug = " ; ".join(
                    [
                        f"{layer_idx} -> {activations[layer_idx].shape}"
                        for layer_idx in archi.layer_visit_order
                    ]
                )
                debug_clean = " ; ".join(
                    [
                        f"{layer_idx} -> {activations_clean[layer_idx].shape}"
                        for layer_idx in archi.layer_visit_order
                    ]
                )

                raise RuntimeError(
                    f"Unable to compute distances between activations_layer ({activations_layer.shape})"
                    f" and activations_layer_clean ({activations_layer_clean.shape})"
                    f" (layer_idx = {layer_idx}) ({debug} || {debug_clean})"
                ) from exc

            for sample_idx in range(actual_batch_size):
                z = distances[sample_idx]
                z = z[z.argsort()[1 : number_of_nn + 1]]

                lid = -1 / (sum([np.log(x / z[-1]) for x in z]) / number_of_nn)

                lids[sample_idx, layer_idx] = lid

        lids = lids[:, valid_lid_columns]
        all_lids.append(lids)
        l2_norms += [line.l2_norm for line in raw_batch]
        linf_norms += [line.linf_norm for line in raw_batch]

    all_lids = np.concatenate(all_lids, axis=0)

    return all_lids, l2_norms, linf_norms


def get_feature_datasets(
    config: Config, epsilons: List[float], dataset: Dataset, archi: Architecture
):
    logger.info(f"Evaluating epsilon={epsilons}")
    if config.transfered_attacks:
        logger.info(f"Generating datasets on the trensferred architecture")
        trsf_archi = archi
        trsf_archi.epochs += 1
        train_clean_, test_clean_, train_adv_, test_adv_ = get_protocolar_datasets(
            noise=config.noise,
            dataset=dataset,
            succ_adv=config.successful_adv > 0,
            archi=trsf_archi,
            dataset_size=config.dataset_size,
            attack_type=config.attack_type,
            attack_backend=config.attack_backend,
            all_epsilons=epsilons,
            transfered_attacks=config.transfered_attacks,
        )
        archi.epochs = archi.epochs - 1
        logger.info(
            f"After generating transferred attacks, archi epochs = {archi.epochs}"
        )
    train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
        noise=config.noise,
        dataset=dataset,
        succ_adv=config.successful_adv > 0,
        archi=archi,
        dataset_size=config.dataset_size,  # 2 * config.batch_size * config.nb_batches,  # Train + Test
        attack_type=config.attack_type,
        attack_backend=config.attack_backend,
        all_epsilons=epsilons,
        transfered_attacks=config.transfered_attacks,
    )

    embeddings_train = create_lid_dataset(config, archi, train_clean, train_clean)[0]
    embeddings_test = create_lid_dataset(config, archi, test_clean, test_clean)[0]

    adv_embedding_train = {
        epsilon: create_lid_dataset(config, archi, train_adv[epsilon], train_clean)[0]
        for epsilon in epsilons
    }

    adv_embedding_test = {
        epsilon: create_lid_dataset(config, archi, test_adv[epsilon], test_clean)[0]
        for epsilon in epsilons
    }

    stats = {
        epsilon: [line.l2_norm for line in test_adv[epsilon]] for epsilon in epsilons
    }

    logger.info(f"Generated {len(embeddings_train)} clean embeddings for train")
    logger.info(f"Generated {len(embeddings_test)} clean embeddings for test")

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
    logger.info(f"LID: get archi = {get_architecture(config.architecture)}")
    archi: Architecture = get_deep_model(
        num_epochs=config.epochs,
        dataset=dataset,
        architecture=get_architecture(config.architecture),
        train_noise=config.train_noise,
        prune_percentile=config.prune_percentile,
        tot_prune_percentile=config.tot_prune_percentile,
        first_pruned_iter=config.first_pruned_iter,
    )

    if config.attack_type not in ["FGSM", "PGD", "FEATUREADVERSARIES"]:
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
        config=config, epsilons=all_epsilons, dataset=dataset, archi=archi
    )

    if config.attack_type in ["DeepFool", "CW", AttackType.BOUNDARY]:
        stats_for_l2_norm_buckets = stats
    else:
        stats_for_l2_norm_buckets = dict()

    evaluation_results = evaluate_embeddings(
        embeddings_train=list(embeddings_train),
        embeddings_test=list(embeddings_test),
        all_adv_embeddings_train=adv_embedding_train,
        all_adv_embeddings_test=adv_embedding_test,
        param_space=[{"gamma": gamma} for gamma in np.logspace(-3, 3, 6)],
        kernel_type=KernelType.RBF,
        stats_for_l2_norm_buckets=stats_for_l2_norm_buckets,
    )

    logger.info(evaluation_results)

    metrics = {"name": "LID", "time": time.time() - start_time, **evaluation_results}

    logger.info(f"Done with experiment {config.experiment_id}_{config.run_id} !")
    logger.info(metrics)

    for method in ["unsupervised", "supervised"]:
        res = evaluation_results[f"{method}_metrics"]
        for eps in res:
            res_eps = res[eps]
            for metric in res_eps:
                res_eps_met = res_eps[metric]
                for typ in res_eps_met:
                    mlflow.log_metric(f"{method}_{eps}_{metric}_{typ}", res_eps_met[typ])

    return metrics


if __name__ == "__main__":
    with mlflow.start_run(run_name="LID binary"):
        my_config = get_config()
        try:
            run_experiment(my_config)
        except Exception as e:
            my_trace = io.StringIO()
            traceback.print_exc(file=my_trace)

            logger.error(my_trace.getvalue())
