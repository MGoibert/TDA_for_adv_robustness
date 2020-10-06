import os
import time
import typing
from functools import reduce

import numpy as np
import torch
from numpy.random import Generator, PCG64

from tda.embeddings import ThresholdStrategy
from tda.models import Architecture
from tda.tda_logging import get_logger

from tda.models.architectures import mnist_mlp, get_architecture
from tda.dataset.adversarial_generation import AttackType, AttackBackend
from tda.embeddings import get_embedding, EmbeddingType, KernelType, ThresholdStrategy
from tda.embeddings.raw_graph import identify_active_indices, featurize_vectors
from tda.models import get_deep_model, Dataset
from tda.models.architectures import mnist_mlp, svhn_lenet, mnist_lenet, get_architecture
from tda.protocol import get_protocolar_datasets, evaluate_embeddings
from tda.rootpath import db_path
from tda.tda_logging import get_logger
from tda.threshold_underoptimized_edges import process_thresholds_underopt
from tda.thresholds import process_thresholds
from tda.graph_stats import get_quantiles_helpers

logger = get_logger("Test for thresholds")

archi = mnist_lenet.name
architecture = get_architecture(archi)
dataset = Dataset.get_or_create(name="MNIST")
epochs = 50
thresholds = "0:0.0001_2:0.0001_4:0_6:0"
attack_type = "FGSM"
all_epsilons = [0.1]

architecture = get_deep_model(
        num_epochs=epochs,
        dataset=dataset,
        architecture=architecture,
        train_noise=0.0,
        prune_percentile=0.0,
        tot_prune_percentile=0.0,
        first_pruned_iter=1,
    )

def method1(architecture, dataset, epochs, thresholds, attack_type, all_epsilons):

    # Quantiles for sigmoidize
    quantiles_helpers = get_quantiles_helpers(
        dataset=dataset, architecture=architecture, dataset_size=100
    )

    # Protocolar dataset
    train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
            noise=0.0,
            dataset=dataset,
            succ_adv=1 > 0,
            archi=architecture,
            dataset_size=5,
            attack_type=attack_type,
            attack_backend=AttackBackend.FOOLBOX,
            all_epsilons=all_epsilons,
            compute_graph=False,
            transfered_attacks=False,
        )
    logger.info(f"train_clean size = {len(train_clean)}")

    # Heart of the pb
    edges_to_keep = process_thresholds_underopt(
        raw_thresholds=thresholds,
        architecture=architecture,
        method=ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        thresholds_are_low_pass=True,
    )
    logger.info(f"edges to keep = {[elem for i, elem in enumerate(edges_to_keep[6]) if i < 20]}")


    start_time_m1 = time.time()
    logger.info(f"Start time")

    # Persistent diagram
    embedding = get_embedding(
        embedding_type=EmbeddingType.PersistentDiagram,
        line=train_clean[0],
        architecture=architecture,
        thresholds=None,
        edges_to_keep=edges_to_keep,
        threshold_strategy=ThresholdStrategy.UnderoptimizedMagnitudeIncrease,
        quantiles_helpers_for_sigmoid=quantiles_helpers,
        thresholds_are_low_pass=True,
        )

    #logger.info(f"embedding = {embedding}")

    end_time_m1 = time.time()

    logger.info(f"time of method 1 = {end_time_m1-start_time_m1} sec")

    return embedding, end_time_m1-start_time_m1


def method2(architecture, dataset, epochs, thresholds, attack_type, all_epsilons):

    # Quantiles for sigmoidize
    quantiles_helpers = get_quantiles_helpers(
        dataset=dataset, architecture=architecture, dataset_size=100
    )

    # Protocolar dataset
    train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
            noise=0.0,
            dataset=dataset,
            succ_adv=1 > 0,
            archi=architecture,
            dataset_size=5,
            attack_type=attack_type,
            attack_backend=AttackBackend.FOOLBOX,
            all_epsilons=all_epsilons,
            compute_graph=False,
            transfered_attacks=False,
        )
    logger.info(f"train_clean size = {len(train_clean)}")

    # Heart of the pb
    edges_to_keep = process_thresholds_underopt(
        raw_thresholds=thresholds,
        architecture=architecture,
        method=ThresholdStrategy.UnderoptimizedMagnitudeIncreaseV3,
        thresholds_are_low_pass=True,
    )

    logger.info(f"edges to keep = {[elem for i, elem in enumerate(edges_to_keep[6]) if i < 20]}")

    start_time_m2 = time.time()
    logger.info(f"Start time")

    # Thresholding the architecture !
    architecture.threshold_layers(edges_to_keep)

    # Persistent diagram
    embedding = get_embedding(
        embedding_type=EmbeddingType.PersistentDiagram,
        line=train_clean[0],
        architecture=architecture,
        thresholds=None,
        edges_to_keep=edges_to_keep,
        threshold_strategy=ThresholdStrategy.UnderoptimizedMagnitudeIncreaseV3,
        quantiles_helpers_for_sigmoid=quantiles_helpers,
        thresholds_are_low_pass=True,
        )

    #logger.info(f"embedding = {embedding}")

    end_time_m2 = time.time()

    logger.info(f"time of method 2 = {end_time_m2-start_time_m2} sec")

    return embedding, end_time_m2-start_time_m2

logger.info(f"\n \n STARTING METHOD 1 \n \n")
em1, time1 = method1(architecture, dataset, epochs, thresholds, attack_type, all_epsilons)

logger.info(f"\n \n STARTING METHOD 2 \n \n")
em2, time2 = method2(architecture, dataset, epochs, thresholds, attack_type, all_epsilons)

logger.info(f"Time: 1 = {time1} and 2 = {time2}")
if em1 == em2:
    logger.info(f"The two embeddings are equal !")
    if time2 < time1:
        logger.info(f"And the new method is faster !!")



