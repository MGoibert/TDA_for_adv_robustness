#!/usr/bin/env python
# coding: utf-8

import argparse
import io
import time
import re
import traceback
import typing

import numpy as np
from joblib import delayed, Parallel
from r3d3.experiment_db import ExperimentDB
from sklearn.decomposition import PCA

from tda.embeddings import get_embedding, EmbeddingType, KernelType, ThresholdStrategy
from tda.embeddings.raw_graph import identify_active_indices, featurize_vectors
from tda.embeddings.weisfeiler_lehman import NodeLabels
from tda.models import get_deep_model, Dataset
from tda.models.architectures import mnist_mlp, get_architecture
from tda.protocol import get_protocolar_datasets, evaluate_embeddings
from tda.rootpath import db_path
from tda.tda_logging import get_logger
from tda.threshold_underoptimized_edges import (
    process_thresholds_underopt,
    thresholdize_underopt_v2,
)
from tda.thresholds import process_thresholds
from tda.graph_stats import get_stats

logger = get_logger("Detector")
start_time = time.time()

my_db = ExperimentDB(db_path=db_path)


class Config(typing.NamedTuple):
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
    # Underoptimized threshold or normal threshold?
    # Parameters used only for Weisfeiler-Lehman embedding
    height: int
    hash_size: int
    node_labels: str
    steps: int
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
    # Type of attack (FGSM, BIM, CW)
    attack_type: str
    # Parameter used by DeepFool and CW
    num_iter: int
    # PCA Parameter for RawGraph (-1 = No PCA)
    raw_graph_pca: int
    l2_norm_quantile: bool = True
    sigmoidize: bool = False
    # Default parameters when running interactively for instance
    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0
    # Number of processes to spawn
    n_jobs: int = 1

    all_epsilons: typing.List[float] = None


def get_config() -> Config:
    parser = argparse.ArgumentParser(
        description="Transform a dataset in pail files to tf records."
    )
    parser.add_argument("--experiment_id", type=int, default=-1)
    parser.add_argument("--run_id", type=int, default=-1)
    parser.add_argument(
        "--embedding_type", type=str, default=EmbeddingType.PersistentDiagram
    )
    parser.add_argument("--kernel_type", type=str, default=KernelType.SlicedWasserstein)
    parser.add_argument("--thresholds", type=str, default="0")
    parser.add_argument(
        "--threshold_strategy", type=str, default=ThresholdStrategy.ActivationValue
    )
    parser.add_argument("--height", type=int, default=1)
    parser.add_argument("--hash_size", type=int, default=100)
    parser.add_argument("--node_labels", type=str, default=NodeLabels.NONE)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--architecture", type=str, default=mnist_mlp.name)
    parser.add_argument("--train_noise", type=float, default=0.0)
    parser.add_argument("--dataset_size", type=int, default=100)
    parser.add_argument("--successful_adv", type=int, default=1)
    parser.add_argument("--raw_graph_pca", type=int, default=-1)
    parser.add_argument("--attack_type", type=str, default="FGSM")
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--all_epsilons", type=str, default=None)
    parser.add_argument("--l2_norm_quantile", type=bool, default=True)
    parser.add_argument("--sigmoidize", type=bool, default=False)
    parser.add_argument("--thresholds_are_low_pass", type=bool, default=True)

    args, _ = parser.parse_known_args()

    if args.all_epsilons is not None:
        args.all_epsilons = list(map(float, str(args.all_epsilons).split(";")))
    return Config(**args.__dict__)


def get_all_embeddings(config: Config):
    architecture = get_architecture(config.architecture)
    dataset = Dataset.get_or_create(name=config.dataset)

    architecture = get_deep_model(
        num_epochs=config.epochs,
        dataset=dataset,
        architecture=architecture,
        train_noise=config.train_noise,
    )
    if config.sigmoidize:
        logger.info(f"Using inter-class regularization (sigmoid)")
        all_weights = get_stats(
            dataset=dataset, architecture=architecture, dataset_size=100
        )
    else:
        all_weights = None

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
        ThresholdStrategy.UnderoptimizedLargeFinal,
    ]:
        edges_to_keep = process_thresholds_underopt(
            raw_thresholds=config.thresholds,
            architecture=architecture,
            method=config.threshold_strategy,
        )
    elif config.threshold_strategy in [
        ThresholdStrategy.UnderoptimizedMagnitudeIncreaseV2,
        ThresholdStrategy.UnderoptimizedLargeFinalV2,
    ]:
        thresholdize_underopt_v2(
            raw_thresholds=config.thresholds,
            architecture=architecture,
            method=config.threshold_strategy,
        )

    if config.attack_type not in ["FGSM", "BIM"]:
        all_epsilons = [1.0]
    elif config.all_epsilons is None:
        all_epsilons = [0.01, 0.05, 0.1, 0.4, 1.0]
        # all_epsilons = [0.01]
    else:
        all_epsilons = config.all_epsilons

    train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
        noise=config.noise,
        dataset=dataset,
        succ_adv=config.successful_adv > 0,
        archi=architecture,
        dataset_size=config.dataset_size,
        attack_type=config.attack_type,
        all_epsilons=all_epsilons,
        compute_graph=False,
    )

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def embedding_getter(line_chunk, save=False):
        ret = list()
        c = 0
        for line in line_chunk:
            save2 = save + "_" + str(c) if save else False
            ret.append(
                get_embedding(
                    embedding_type=config.embedding_type,
                    line=line,
                    params={
                        "hash_size": int(config.hash_size),
                        "height": int(config.height),
                        "node_labels": config.node_labels,
                        "steps": config.steps,
                        "raw_graph_pca": config.raw_graph_pca,
                    },
                    architecture=architecture,
                    dataset=dataset,
                    thresholds=thresholds,
                    edges_to_keep=edges_to_keep,
                    threshold_strategy=config.threshold_strategy,
                    save=save2,
                    all_weights_for_sigmoid=all_weights,
                    thresholds_are_low_pass=config.thresholds_are_low_pass,
                )
            )
            c += 1
        return ret

    def process(input_dataset, save=False):
        my_chunks = chunks(input_dataset, len(input_dataset) // config.n_jobs)
        ret = Parallel(n_jobs=config.n_jobs)(
            delayed(embedding_getter)(chunk, save) for chunk in my_chunks
        )
        ret = [item for sublist in ret for item in sublist]
        return ret

    # Clean train
    clean_embeddings_train = process(train_clean, save="clean_train")
    logger.info(f"Clean train dataset " f"({len(clean_embeddings_train)} points) !!")

    # Clean test
    clean_embeddings_test = process(test_clean, save="clean_test")
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

        adv_embeddings_test[epsilon] = process(test_adv[epsilon], save="adv_test")
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

    return (
        clean_embeddings_train,
        clean_embeddings_test,
        adv_embeddings_train,
        adv_embeddings_test,
        thresholds,
        stats,
        stats_inf
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
        stats_inf
    ) = get_all_embeddings(config)
    # with open('/Users/m.goibert/Documents/temp/gram_mat/dgm_clean_train.pickle', 'wb') as f:
    #            pickle.dump(embedding_train, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('/Users/m.goibert/Documents/temp/gram_mat/dgm_clean_test.pickle', 'wb') as f:
    #            pickle.dump(embedding_test, f, protocol=pickle.HIGHEST_PROTOCOL)
    # eps_to_save = 0.1
    # with open('/Users/m.goibert/Documents/temp/gram_mat/dgm_adv_'+str(eps_to_save)+'.pickle', 'wb') as f:
    #            pickle.dump(adv_embeddings_test[eps_to_save], f, protocol=pickle.HIGHEST_PROTOCOL)

    if config.kernel_type == KernelType.RBF:
        param_space = [{"gamma": gamma} for gamma in np.logspace(-6, -3, 10)]
    elif config.kernel_type in [
        KernelType.SlicedWasserstein,
        KernelType.SlicedWassersteinOldVersion,
    ]:
        param_space = [{"M": 20, "sigma": sigma} for sigma in np.logspace(-3, 3, 7)]
    else:
        raise NotImplementedError(f"Unknown kernel {config.kernel_type}")

    if config.attack_type in ["DeepFool", "CW"]:
        stats_for_l2_norm_buckets = stats
    else:
        stats_for_l2_norm_buckets = dict()

    aucs_unsupervised, aucs_supervised, auc_l2_norm = evaluate_embeddings(
        embeddings_train=embedding_train,
        embeddings_test=embedding_test,
        all_adv_embeddings_train=adv_embeddings_train,
        all_adv_embeddings_test=adv_embeddings_test,
        param_space=param_space,
        kernel_type=config.kernel_type,
        stats_for_l2_norm_buckets=stats_for_l2_norm_buckets,
    )

    if auc_l2_norm is not None:
        logger.info(f"aucs_l2_norm = {auc_l2_norm}")

    logger.info(aucs_unsupervised)
    logger.info(aucs_supervised)

    end_time = time.time()

    metrics = {
        "name": "Graph",
        "aucs_supervised": aucs_supervised,
        "aucs_unsupervised": aucs_unsupervised,
        "aucs_l2_norm": auc_l2_norm or "None",
        "time": end_time - start_time,
        "l2_diff": stats,
        "linf_diff": stats_inf,
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
