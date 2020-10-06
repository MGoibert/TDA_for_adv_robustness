#!/usr/bin/env python
# coding: utf-8

import argparse
import io
import time
import re
import traceback
from typing import NamedTuple, List

import numpy as np
from joblib import delayed, Parallel
from r3d3.experiment_db import ExperimentDB
from sklearn.decomposition import PCA

from tda.dataset.adversarial_generation import AttackType, AttackBackend
from tda.embeddings import (
    get_embedding,
    EmbeddingType,
    KernelType,
    ThresholdStrategy,
    Embedding,
)
from tda.embeddings.raw_graph import identify_active_indices, featurize_vectors
from tda.models import get_deep_model, Dataset
from tda.models.architectures import mnist_mlp, get_architecture
from tda.protocol import get_protocolar_datasets, evaluate_embeddings
from tda.rootpath import db_path
from tda.tda_logging import get_logger
from tda.threshold_underoptimized_edges import process_thresholds_underopt
from tda.thresholds import process_thresholds
from tda.graph_stats import get_quantiles_helpers

from tda.dataset.adversarial_generation import AttackType, AttackBackend

logger = get_logger("Detector")
start_time = time.time()

my_db = ExperimentDB(db_path=db_path)


class Config(NamedTuple):
    # Type of embedding to use
    embedding_type: str
    # Type of kernel to use on the embeddings
    kernel_type: str
    # High threshold for the edges of the activation graph
    thresholds: str
    # Which thresholding strategy should we use
    threshold_strategy: str
    # Are the threshold low pass or not
    thresholds_are_low_pass: bool
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
    # Size of the dataset used for the experiment
    dataset_size: int
    # Should we ignore unsuccessful attacks or not
    successful_adv: int
    # Type of attack (FGSM, PGD, CW)
    attack_type: str
    # Type of attack (CUSTOM, ART, FOOLBOX)
    attack_backend: str
    # Parameter used by DeepFool and CW
    num_iter: int
    # PCA Parameter for RawGraph (-1 = No PCA)
    raw_graph_pca: int
    # Whether to use pre-saved adversarial examples or not
    transfered_attacks: bool = False
    l2_norm_quantile: bool = True
    sigmoidize: bool = False
    # Pruning
    first_pruned_iter: int = 10
    prune_percentile: float = 0.0
    tot_prune_percentile: float = 0.0
    # Default parameters when running interactively for instance
    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0
    # Number of processes to spawn
    n_jobs: int = 1

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
    parser.add_argument("--experiment_id", type=int, default=-1)
    parser.add_argument("--run_id", type=int, default=-1)
    parser.add_argument(
        "--embedding_type", type=str, default=EmbeddingType.PersistentDiagram
    )
    parser.add_argument("--thresholds", type=str, default="0")
    parser.add_argument(
        "--threshold_strategy",
        type=str,
        default=ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
    )
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--architecture", type=str, default=mnist_mlp.name)
    parser.add_argument("--train_noise", type=float, default=0.0)
    parser.add_argument("--dataset_size", type=int, default=100)
    parser.add_argument("--successful_adv", type=int, default=1)
    parser.add_argument("--raw_graph_pca", type=int, default=-1)
    parser.add_argument("--attack_type", type=str, default=AttackType.FGSM)
    parser.add_argument("--attack_backend", type=str, default=AttackBackend.FOOLBOX)
    parser.add_argument("--transfered_attacks", type=str2bool, default=False)
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--all_epsilons", type=str, default=None)
    parser.add_argument("--l2_norm_quantile", type=bool, default=True)
    parser.add_argument("--sigmoidize", type=str2bool, default=False)
    parser.add_argument("--thresholds_are_low_pass", type=str2bool, default=True)
    parser.add_argument("--first_pruned_iter", type=int, default=10)
    parser.add_argument("--prune_percentile", type=float, default=0.0)
    parser.add_argument("--tot_prune_percentile", type=float, default=0.0)

    args, _ = parser.parse_known_args()

    if args.all_epsilons is not None:
        args.all_epsilons = list(map(float, str(args.all_epsilons).split(";")))

    if args.embedding_type == EmbeddingType.PersistentDiagram:
        args.kernel_type = KernelType.SlicedWasserstein
    elif args.embedding_type == EmbeddingType.RawGraph:
        args.kernel_type = KernelType.RBF

    logger.info(args.__dict__)

    return Config(**args.__dict__)


def get_all_embeddings(config: Config):
    detailed_times = dict()

    architecture = get_architecture(config.architecture)
    dataset = Dataset.get_or_create(name=config.dataset)

    architecture = get_deep_model(
        num_epochs=config.epochs,
        dataset=dataset,
        architecture=architecture,
        train_noise=config.train_noise,
        prune_percentile=config.prune_percentile,
        tot_prune_percentile=config.tot_prune_percentile,
        first_pruned_iter=config.first_pruned_iter,
    )
    if config.sigmoidize:
        logger.info(f"Using inter-class regularization (sigmoid)")
        start_time = time.time()
        quantiles_helpers = get_quantiles_helpers(
            dataset=dataset, architecture=architecture, dataset_size=100
        )
        detailed_times["stats"] = time.time() - start_time

    else:
        quantiles_helpers = None

    thresholds = None
    edges_to_keep = None

    if config.threshold_strategy == ThresholdStrategy.ActivationValue:
        thresholds = process_thresholds(
            raw_thresholds=config.thresholds,
            dataset=dataset,
            architecture=architecture,
            dataset_size=100,
        )
    elif config.threshold_strategy == ThresholdStrategy.QuantilePerGraphLayer:
        thresholds = config.thresholds.split("_")
        thresholds = [val.split(";") for val in thresholds]
        thresholds = {
            (int(start), int(end)): float(val) for (start, end, val) in thresholds
        }
        logger.info(f"Using thresholds per graph {thresholds}")
    elif config.threshold_strategy in [
        ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        ThresholdStrategy.UnderoptimizedMagnitudeIncreaseV3,
        ThresholdStrategy.UnderoptimizedLargeFinal,
        ThresholdStrategy.UnderoptimizedRandom,
        ThresholdStrategy.UnderoptimizedMagnitudeIncreaseComplement,
    ]:
        edges_to_keep = process_thresholds_underopt(
            raw_thresholds=config.thresholds,
            architecture=architecture,
            method=config.threshold_strategy,
            thresholds_are_low_pass=config.thresholds_are_low_pass,
        )

    if config.attack_type not in ["FGSM", "PGD"]:
        all_epsilons = [1.0]
    elif config.all_epsilons is None:
        all_epsilons = [0.01, 0.05, 0.1, 0.4, 1.0]
        # all_epsilons = [0.01]
    else:
        all_epsilons = config.all_epsilons

    start_time = time.time()
    train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
        noise=config.noise,
        dataset=dataset,
        succ_adv=config.successful_adv > 0,
        archi=architecture,
        dataset_size=config.dataset_size,
        attack_type=config.attack_type,
        attack_backend=config.attack_backend,
        all_epsilons=all_epsilons,
        compute_graph=False,
        transfered_attacks=config.transfered_attacks,
    )
    detailed_times["protocolar_datasets"] = time.time() - start_time

    if config.threshold_strategy == ThresholdStrategy.UnderoptimizedMagnitudeIncreaseV3:
        architecture.threshold_layers(edges_to_keep)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""

        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def embedding_getter(line_chunk) -> List[Embedding]:
        ret = list()
        c = 0
        for line in line_chunk:
            ret.append(
                get_embedding(
                    embedding_type=config.embedding_type,
                    line=line,
                    architecture=architecture,
                    thresholds=thresholds,
                    edges_to_keep=edges_to_keep,
                    threshold_strategy=config.threshold_strategy,
                    quantiles_helpers_for_sigmoid=quantiles_helpers,
                    thresholds_are_low_pass=config.thresholds_are_low_pass,
                )
            )
            c += 1
        return ret

    stats_inside_embeddings = dict()

    def process(input_dataset) -> List[Embedding]:

        my_chunks = chunks(input_dataset, len(input_dataset) // config.n_jobs)

        ret = Parallel(n_jobs=config.n_jobs)(
            delayed(embedding_getter)(chunk) for chunk in my_chunks
        )
        ret = [item for sublist in ret for item in sublist]

        # Extracting stats
        for embedding in ret:
            for key in embedding.time_taken:
                stats_inside_embeddings[key] = (
                    stats_inside_embeddings.get(key, 0) + embedding.time_taken[key]
                )

        return [embedding.value for embedding in ret]

    start_time = time.time()

    # Clean train
    clean_embeddings_train = process(train_clean)
    logger.info(f"Clean train dataset " f"({len(clean_embeddings_train)} points) !!")

    # Clean test
    clean_embeddings_test = process(test_clean)
    logger.info(f"Clean test dataset " f"({len(clean_embeddings_test)} points) !!")

    adv_embeddings_train = dict()
    adv_embeddings_test = dict()

    stats = dict()
    stats_inf = dict()

    for epsilon in all_epsilons:
        adv_embeddings_train[epsilon] = process(train_adv[epsilon])
        logger.info(
            f"Adversarial train dataset for espilon = {epsilon}"
            f"  ({len(adv_embeddings_train[epsilon])} points) !"
        )

        adv_embeddings_test[epsilon] = process(test_adv[epsilon])
        logger.info(
            f"Adversarial test dataset for espilon = {epsilon} "
            f"({len(adv_embeddings_test[epsilon])} points)  !"
        )

        stats[epsilon] = [line.l2_norm for line in test_adv[epsilon]]
        stats_inf[epsilon] = [line.linf_norm for line in test_adv[epsilon]]

        logger.debug(
            f"Stats for diff btw clean and adv: "
            f"{np.quantile(stats[epsilon], 0.1)}, "
            f"{np.quantile(stats[epsilon], 0.25)}, "
            f"{np.median(stats[epsilon])}, "
            f"{np.quantile(stats[epsilon], 0.75)}, "
            f"{np.quantile(stats[epsilon], 0.9)}"
        )

    if config.embedding_type == EmbeddingType.RawGraph:
        raw_graph_indices = identify_active_indices(clean_embeddings_train)

        clean_embeddings_train = featurize_vectors(
            clean_embeddings_train, raw_graph_indices
        )
        clean_embeddings_test = featurize_vectors(
            clean_embeddings_test, raw_graph_indices
        )

        if config.raw_graph_pca > 0:
            logger.info("Fitting PCA...")
            pca = PCA(
                n_components=config.raw_graph_pca,
                random_state=int(config.experiment_id),
            )
            clean_embeddings_train = pca.fit_transform(clean_embeddings_train)
            logger.info("Done fitting PCA...")
            clean_embeddings_test = pca.transform(clean_embeddings_test)

        for epsilon in all_epsilons:
            adv_embeddings_train[epsilon] = featurize_vectors(
                adv_embeddings_train[epsilon], raw_graph_indices
            )
            adv_embeddings_test[epsilon] = featurize_vectors(
                adv_embeddings_test[epsilon], raw_graph_indices
            )

            if config.raw_graph_pca > 0:
                adv_embeddings_train[epsilon] = pca.transform(
                    adv_embeddings_train[epsilon]
                )
                adv_embeddings_test[epsilon] = pca.transform(
                    adv_embeddings_test[epsilon]
                )

    detailed_times["embeddings"] = time.time() - start_time

    for key in stats_inside_embeddings:
        detailed_times[key] = stats_inside_embeddings[key]

    return (
        clean_embeddings_train,
        clean_embeddings_test,
        adv_embeddings_train,
        adv_embeddings_test,
        thresholds,
        stats,
        stats_inf,
        detailed_times,
    )


def run_experiment(config: Config):
    """
    Main entry point to run the experiment
    """

    logger.info(f"Starting experiment {config.experiment_id}_{config.run_id} !!")

    if __name__ != "__main__":
        my_db.add_experiment(
            experiment_id=config.experiment_id,
            run_id=config.run_id,
            config=config._asdict(),
        )

    (
        embedding_train,
        embedding_test,
        adv_embeddings_train,
        adv_embeddings_test,
        thresholds,
        stats,
        stats_inf,
        detailed_times,
    ) = get_all_embeddings(config)

    if config.kernel_type == KernelType.RBF:
        param_space = [{"gamma": gamma} for gamma in np.logspace(-6, -3, 10)]
    elif config.kernel_type in [KernelType.SlicedWasserstein]:
        param_space = [{"M": 20, "sigma": sigma} for sigma in np.logspace(-3, 3, 7)]
    else:
        raise NotImplementedError(f"Unknown kernel {config.kernel_type}")

    if config.attack_type in ["DeepFool", "CW", AttackType.BOUNDARY]:
        stats_for_l2_norm_buckets = stats
    else:
        stats_for_l2_norm_buckets = dict()

    evaluation_results = evaluate_embeddings(
        embeddings_train=embedding_train,
        embeddings_test=embedding_test,
        all_adv_embeddings_train=adv_embeddings_train,
        all_adv_embeddings_test=adv_embeddings_test,
        param_space=param_space,
        kernel_type=config.kernel_type,
        stats_for_l2_norm_buckets=stats_for_l2_norm_buckets,
    )

    logger.info(f"Threshold final = {thresholds}")
    logger.info(
        f"Results --> Unsup = {evaluation_results['unsupervised_metrics']} and sup = {evaluation_results['supervised_metrics']}"
    )

    metrics = {
        "name": "Graph",
        "time": time.time() - start_time,
        "detailed_times": detailed_times,
        "l2_diff": stats,
        "linf_diff": stats_inf,
        **evaluation_results,
    }

    if thresholds is not None:
        metrics["effective_thresholds"] = (
            {"_".join([str(v) for v in key]): thresholds[key] for key in thresholds},
        )

    my_db.update_experiment(
        experiment_id=config.experiment_id, run_id=config.run_id, metrics=metrics
    )

    logger.info(f"Done with experiment {config.experiment_id}_{config.run_id} !!")


if __name__ == "__main__":
    my_config = get_config()
    try:
        run_experiment(my_config)
    except Exception as e:
        my_trace = io.StringIO()
        traceback.print_exc(file=my_trace)

        logger.error(my_trace.getvalue())

        my_db.update_experiment(
            experiment_id=my_config.experiment_id,
            run_id=my_config.run_id,
            metrics={"ERROR": re.escape(my_trace.getvalue())},
        )
