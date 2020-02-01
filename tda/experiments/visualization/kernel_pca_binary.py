#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import time
import typing
from multiprocessing import Pool
import os
import pathlib
import copy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from r3d3.experiment_db import ExperimentDB
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.svm import OneClassSVM, SVC
from sklearn.decomposition import KernelPCA

from tda.embeddings import get_embedding, EmbeddingType, \
    get_gram_matrix, KernelType
from tda.embeddings.weisfeiler_lehman import NodeLabels
from tda.graph_dataset import get_graph_dataset
from tda.models.architectures import mnist_mlp, get_architecture, Architecture
from tda.rootpath import db_path
from tda.thresholds import process_thresholds
from tda.models import get_deep_model, Dataset
from tda.logging import get_logger

#logging.basicConfig(level=logging.INFO)
logger = get_logger("Visualization PCA")
start_time = time.time()

my_db = ExperimentDB(db_path=db_path)

################
# Parsing args #
################

class Config(typing.NamedTuple):
    # Type of embedding to use
    embedding_type: str
    # Type of kernel to use on the embeddings
    kernel_type: str
    # High threshold for the edges of the activation graph
    thresholds: str
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
    # Type of attack (FGSM, BIM, DeepFool, CW and All)
    attack_type: str
    # Parameter used by DeepFool and CW
    num_iter: int
    # Should we use the same images clean vs attack when training the detector
    identical_train_samples: int
    # Default parameters when running interactively for instance
    # Used to store the results in the DB
    experiment_id: int = int(time.time())
    run_id: int = 0

def get_config() -> Config:
    parser = argparse.ArgumentParser(
        description='Transform a dataset in pail files to tf records.')
    parser.add_argument('--experiment_id', type=int, default=-1)
    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--embedding_type', type=str, default=EmbeddingType.PersistentDiagram)
    parser.add_argument('--kernel_type', type=str, default=KernelType.SlicedWasserstein)
    parser.add_argument('--thresholds', type=str, default='0')
    parser.add_argument('--height', type=int, default=1)
    parser.add_argument('--hash_size', type=int, default=100)
    parser.add_argument('--node_labels', type=str, default=NodeLabels.NONE)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dataset', type=str, default="MNIST")
    parser.add_argument('--architecture', type=str, default=mnist_mlp.name)
    parser.add_argument('--train_noise', type=float, default=0.0)
    parser.add_argument('--dataset_size', type=int, default=100)
    parser.add_argument('--successful_adv', type=int, default=1)
    parser.add_argument('--attack_type', type=str, default="FGSM")
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--identical_train_samples', type=int, default=1)

    args, _ = parser.parse_known_args()
    return Config(**args.__dict__)

# save np.load and modify the default parameters of np.load
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#####################
# Fetching datasets #
#####################

#if args.embedding_type == EmbeddingType.OriginalDataPoint:
#    retain_data_point = True
#else:
#    retain_data_point = False

def get_embeddings(
        config: Config,
        architecture: Architecture,
        noise: float,
        thresholds: typing.List[float],
        epsilon: float,
        dataset: Dataset,
        start: int = 0,
        train: bool = True,
        attack: str = "FGSM"
) -> typing.List:
    """
    Compute the embeddings used for the detection
    """

    my_embeddings = list()
    for line in get_graph_dataset(
            epsilon=epsilon,
            noise=noise,
            adv=epsilon > 0.0,
            architecture=architecture,
            dataset=dataset,
            dataset_size=config.dataset_size,
            thresholds=thresholds,
            only_successful_adversaries=config.successful_adv > 0,
            attack_type=attack,
            num_iter=config.num_iter,
            start=start,
            train=train
    ):
        my_embeddings.append(get_embedding(
            embedding_type=config.embedding_type,
            graph=line.graph,
            params={
                "hash_size": int(config.hash_size),
                "height": int(config.height),
                "node_labels": config.node_labels,
                "steps": config.steps
            }
        ))
    logger.info(
        f"Computed embeddings for (attack = {config.attack_type}, "
        f"eps={epsilon}, noise={noise}), "
        f"number of sample = {len(my_embeddings)}")
    return my_embeddings


def get_all_embeddings(config: Config):
    architecture = get_architecture(config.architecture)
    dataset = Dataset.get_or_create(name=config.dataset)

    architecture = get_deep_model(
        num_epochs=config.epochs,
        dataset=dataset,
        architecture=architecture,
        train_noise=config.train_noise
    )

    thresholds = process_thresholds(
        raw_thresholds=config.thresholds,
        dataset=dataset,
        architecture=architecture,
        dataset_size=500 if config.dataset == "MNIST" else 300
    )

    if config.attack_type in ["FGSM", "BIM", "All"]:
        all_epsilons = list([0.025, 0.05, 0.1])
    else:
        all_epsilons = [1.0]
    all_noises = list([0.025, 0.05, 0.1])

    start = 0

    # Clean examples
    logger.info(f"Clean dataset !!")
    clean_embeddings = get_embeddings(
        config=config, dataset=dataset, noise=0.0,
        architecture=architecture, thresholds=thresholds,
        epsilon=0.0, start=start,
        train=False)
    start += config.dataset_size

    # Noisy examples
    logger.info(f"Noisy dataset !!")
    noisy_embeddings = list()
    for noise in all_noises:
        noisy_embeddings += [item for item in get_embeddings(
        config=config, dataset=dataset, noise=noise,
        architecture=architecture, thresholds=thresholds,
        epsilon=0.0, start=start,
        train=False)]
    start += config.dataset_size

    # Adv examples
    adv_embeddings = dict()
    adv_embeddings_all = list()
    if config.attack_type != "All":
        logger.info(f"Not all attacks")
        for epsilon in all_epsilons[:]:
            logger.info(f"Adversarial dataset for espilon = {epsilon} !!")
            adv_embeddings[epsilon] = get_embeddings(
                                        config=config, dataset=dataset, noise=0.0,
                                        architecture=architecture, thresholds=thresholds,
                                        epsilon=epsilon, start=start,
                                        train=False, attack=config.attack_type)
            adv_embeddings_all += [item for item in adv_embeddings[epsilon]]
    else:
        logger.info(f"All attacks")
        for att in ["FGSM", "BIM", "DeepFool", "CW"]:
            adv_embeddings[att] = dict()
            if att in ["FGSM", "BIM"]:
                for epsilon in all_epsilons:
                    logger.info(f"Attack type = {att} and epsilon = {epsilon}")
                    adv_embeddings[att][epsilon] = get_embeddings(config=config, dataset=dataset,
                                                                epsilon=epsilon, noise=0.0, architecture=architecture,
                                                                start=start, thresholds=thresholds,
                                                                train=False, attack=att)
                    adv_embeddings_all += [item for item in adv_embeddings[att][epsilon]]
            else:
                adv_embeddings[att][1] = get_embeddings(config=config, dataset=dataset,
                                                        epsilon=1, noise=0.0, architecture=architecture,
                                                        thresholds=thresholds, start=start,
                                                        train=False, attack=att)
                adv_embeddings_all += [item for item in adv_embeddings[att][1]]

    return clean_embeddings, noisy_embeddings, adv_embeddings, adv_embeddings_all, thresholds, all_noises, all_epsilons

def process(config, embedding, fit=True, kpca0=None, embedding_init=None):
    """
    Return data points transformed by Kernel PCA
    """
    if config.kernel_type == KernelType.RBF:
        param_space = [
            {'gamma': gamma}
            for gamma in np.logspace(-6, -3, 10)
        ]
    elif config.kernel_type == KernelType.SlicedWasserstein:
        param_space = [
            {'M': 20, 'sigma': 5 * 10 ** (-1)},
        ]
    else:
        raise NotImplementedError(f"Unknown kernel {config.kernel_type}")

    for i, param in enumerate(param_space):
        if fit == True:
            logger.info(f"In fit")
            gram_matrix = get_gram_matrix(
            kernel_type=config.kernel_type,
            embeddings_in=embedding,
            embeddings_out=None,
            params=param
            )
            kpca = KernelPCA(2, kernel="precomputed")
            transform_input = kpca.fit_transform(gram_matrix)
        else:
            logger.info(f"Not in fit")
            other_gram_matrix = get_gram_matrix(
            kernel_type=config.kernel_type,
            embeddings_in=embedding,
            embeddings_out=embedding_init,
            params=param
            )
            transform_input = kpca0.transform(other_gram_matrix)
        logger.info(f"Trained model in {time.time() - start_time} secs")        

    if fit == True:
        return transform_input, kpca
    else:
        return transform_input

def plot_kernel_pca(config):

    binary_path = os.path.dirname(os.path.realpath(__file__))
    binary_path_split = pathlib.Path(binary_path)
    directory = str(pathlib.Path(*binary_path_split.parts[:-3])) + "/plots/visualization/kernel_pca/" + str(config.architecture)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.style.use('seaborn-dark')

    clean_embeddings, noisy_embeddings, adv_embeddings, adv_embeddings_all, thresholds, all_noises, all_epsilons = get_all_embeddings(config)
    embedding_init = clean_embeddings + noisy_embeddings + adv_embeddings_all
    transform_input, kpca = process(config, embedding_init)
    filename = directory + f"/{config.attack_type}.png"

    labels = len(clean_embeddings)*list(["Clean"]) #+ len(noisy_embeddings)*list(["Noisy"])
    for noise in all_noises:
        labels += (len(noisy_embeddings)//len(all_noises))*list([f"Noisy {noise}"])
    if config.attack_type != "All":
        for epsilon in all_epsilons:
             labels += len(adv_embeddings[epsilon])*list([f"Adv {config.attack_type} {epsilon}"])
    else:
        for att in ["FGSM", "BIM", "DeepFool", "CW"]:
            if att in ["FGSM", "BIM"]:
                for epsilon in all_epsilons:
                    labels += len(adv_embeddings[att][epsilon])*list([f"Adv {att} {epsilon}"])
            else:
                labels += len(adv_embeddings[att][1])*list([f"Adv {att} {1}"])

    le = len(all_epsilons)
    if config.attack_type == "FGSM":
        pal = sns.color_palette(["#0F0F0F", "#C9C9C9", "#A0A0A0", "#757575"] + sns.color_palette("Blues", le))
    elif config.attack_type == "BIM":
        pal = sns.color_palette(["#0F0F0F", "#C9C9C9", "#A0A0A0", "#757575"] + sns.color_palette("Greens", le))
    elif config.attack_type == "DeepFool":
        pal = sns.color_palette(["#0F0F0F", "#C9C9C9", "#A0A0A0", "#757575", "#FEB307"])
    elif config.attack_type == "CW":
        pal = sns.color_palette(["#0F0F0F", "#C9C9C9", "#A0A0A0", "#757575", "#C50000"])
    else:
        pal = sns.color_palette(["#0F0F0F", "#C9C9C9", "#A0A0A0", "#757575"] + sns.color_palette("Blues", le) + sns.color_palette("Greens", le) + ["#FEB307"] + ["#C50000"])

    logger.info(f"Plotting figure...")
    p = sns.scatterplot(x=[1000*item[0] for item in transform_input], y=[1000*item[1] for item in transform_input],
        hue=labels, palette=pal, alpha=0.8)
    plt.savefig(filename, dpi=600)
    plt.clf()
    logger.info(f"Closing figure")

    labs = len(clean_embeddings)*list(["Clean"]) #+ len(noisy_embeddings)*list(["Noisy"])
    for noise in all_noises:
        labs += (len(noisy_embeddings)//len(all_noises))*list([f"Noisy {noise}"])
    logger.info(f"len labs = {len(labs)}")

    if config.attack_type == "All":
        for att in ["FGSM", "BIM", "DeepFool", "CW"]:
            clean_noisy_end = len(clean_embeddings) + len(noisy_embeddings)
            filename2 = directory + f"/{config.attack_type}_viz_{att}.png"
            if att == "FGSM":
                fgsm_end = clean_noisy_end + len(adv_embeddings[att])*config.dataset_size
                other_transform_input = transform_input[0:fgsm_end]
                pal2 = sns.color_palette(["#0F0F0F", "#C9C9C9", "#A0A0A0", "#757575"] + sns.color_palette("Blues", le))
                labels2 = copy.deepcopy(labs)
                for epsilon in all_epsilons:
                    labels2 += len(adv_embeddings[att][epsilon])*list([f"Adv {att} {epsilon}"])
            elif att == "BIM":
                bim_end = fgsm_end + len(adv_embeddings[att])*config.dataset_size
                other_transform_input = list(transform_input[:clean_noisy_end]) + list(transform_input[fgsm_end:bim_end])
                pal2 = sns.color_palette(["#0F0F0F", "#C9C9C9", "#A0A0A0", "#757575"] + sns.color_palette("Greens", le))
                labels2 = copy.deepcopy(labs)
                for epsilon in all_epsilons:
                    labels2 += len(adv_embeddings[att][epsilon])*list([f"Adv {att} {epsilon}"])
            elif att == "DeepFool":
                deepfool_end = bim_end + len(adv_embeddings[att])*config.dataset_size
                other_transform_input = list(transform_input[:clean_noisy_end]) + list(transform_input[bim_end:deepfool_end])
                pal2 = sns.color_palette(["#0F0F0F", "#C9C9C9", "#A0A0A0", "#757575", "#FEB307"])
                labels2 = copy.deepcopy(labs) + len(adv_embeddings[att][1])*list([f"Adv {att} {1}"])
            elif att == "CW":
                other_transform_input = list(transform_input[:clean_noisy_end]) + list(transform_input[deepfool_end:])
                pal2 = sns.color_palette(["#0F0F0F", "#C9C9C9", "#A0A0A0", "#757575", "#C50000"])
                labels2 = copy.deepcopy(labs) + len(adv_embeddings[att][1])*list([f"Adv {att} {1}"])
            logger.info(f"Plotting figure for all attacks...")
            p2 = sns.scatterplot(x=[1000*item[0] for item in other_transform_input], y=[1000*item[1] for item in other_transform_input],
                hue=labels2, palette=pal2, alpha=0.8)
            plt.savefig(filename2, dpi=600)
            plt.clf()
            logger.info(f"Closing figure")

if __name__ == "__main__":
    config = get_config()
    plot_kernel_pca(config)

