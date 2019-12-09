#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import time
import typing
from multiprocessing import Pool

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
from tda.graph_dataset import get_dataset
from tda.models.architectures import mnist_mlp, get_architecture
from tda.rootpath import db_path
from tda.thresholds import process_thresholds

start_time = time.time()

my_db = ExperimentDB(db_path=db_path)

################
# Parsing args #
################

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"[{args.experiment_id}_{args.run_id}]")

binary_path = os.path.dirname(os.path.realpath(__file__))
binary_path_split = pathlib.Path(binary_path)
directory = str(pathlib.Path(*binary_path_split.parts[:-3])) + "/plots/visualization/kernel_pca/" + str(args.architecture)
if not os.path.exists(directory):
    os.makedirs(directory)

# save np.load and modify the default parameters of np.load
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#####################
# Fetching datasets #
#####################

architecture = get_architecture(args.architecture)

if args.embedding_type == EmbeddingType.OriginalDataPoint:
    retain_data_point = True
else:
    retain_data_point = False

thresholds = process_thresholds(
    raw_thresholds=args.thresholds,
    dataset=args.dataset,
    architecture=args.architecture,
    epochs=args.epochs
)


# f"stats/{dataset}_{architecture}_{str(epochs)}_epochs.npy"

def get_embeddings(epsilon: float, noise: float, start: int = 0, attack=args.attack_type) -> typing.List:
    """
    Helper function to get list of embeddings
    """
    my_embeddings = list()
    for line in get_dataset(
            num_epochs=args.epochs,
            epsilon=epsilon,
            noise=noise,
            adv=epsilon > 0.0,
            retain_data_point=retain_data_point,
            architecture=architecture,
            source_dataset_name=args.dataset,
            dataset_size=args.dataset_size,
            thresholds=thresholds,
            only_successful_adversaries=args.successful_adv > 0,
            attack_type=attack,
            num_iter=args.num_iter,
            start=start,
            train_noise=args.train_noise,
    ):
        my_embeddings.append(get_embedding(
            embedding_type=args.embedding_type,
            graph=line.graph,
            params={
                "hash_size": int(args.hash_size),
                "height": int(args.height),
                "node_labels": args.node_labels,
                "steps": args.steps
            }
        ))
    logger.info(
        f"Computed embeddings for (attack = {args.attack_type}, eps={epsilon}, noise={noise}), number of sample = {len(my_embeddings)}")
    return my_embeddings


# Clean embeddings
train_eps = 0.005

start = 0

# Clean dataset
logger.info(f"Clean train dataset !!")
clean_embeddings = get_embeddings(epsilon=0.0, noise=0.0, start=0)
if args.identical_train_samples < 0.5:
    start += args.dataset_size

# Noisy dataset
if args.noise > 0.0:
    logger.info(f"Noisy train dataset !!")
    noisy_embeddings = get_embeddings(epsilon=0.0, noise=args.noise, start=start)
else:
    noisy_embeddings = list()
if args.identical_train_samples < 0.5:
    start += args.dataset_size

if args.kernel_type == KernelType.RBF:
    param_space = [
        {'gamma': gamma}
        for gamma in np.logspace(-6, -3, 10)
    ]
#logger.info(f"kernel type = {args.kernel_type} and {args.kernel_type == KernelType.SlicedWasserstein}")
#if args.kernel_type == KernelType.SlicedWasserstein:
else:
    param_space = [
        {'M': 10, 'sigma': 5 * 10 ** (-5)}
    ]

if args.attack_type in ["FGSM", "BIM", "All"]:
    all_epsilons = list([0.005, 0.01, 0.02])
elif args.attack_type in ["DeepFool", "CW"]:
    all_epsilons = [1]

if args.attack_type != "All":
    logger.info(f"Not all")
    adv_embeddings = dict()
    adv_embeddings_all = list()
    for epsilon in all_epsilons[:]:
        logger.info(f"Adversarial dataset for espilon = {epsilon} !!")
        adv_embeddings[epsilon] = get_embeddings(epsilon=epsilon, noise=0.0, start=start)
        adv_embeddings_all += adv_embeddings[epsilon]
else:
    logger.info(f"All")
    adv_embeddings = dict()
    adv_embeddings_all = list()
    for att in ["FGSM", "BIM", "DeepFool", "CW"]:
        adv_embeddings[att] = dict()
        if att in ["FGSM", "BIM"]:
            for epsilon in all_epsilons:
                logger.info(f"        Attack type = {att} and epsilon = {epsilon}")
                adv_embeddings[att][epsilon] = get_embeddings(epsilon=epsilon, noise=0.0, start=start, attack=att)
                adv_embeddings_all += adv_embeddings[att][epsilon]
        else:
            adv_embeddings[att][1] = get_embeddings(epsilon=1, noise=0.0, start=start, attack=att)
            adv_embeddings_all += adv_embeddings[att][1]

def process(embedding, fit=True, kpca0=None, embedding_init=None):
    """
    Compute the AUC for a given epsilon and returns also the scores
    of the best OneClass SVM
    """
    best_auc = 0.0

    for i, param in enumerate(param_space):
        if fit == True:
            gram_matrix = get_gram_matrix(
            kernel_type=args.kernel_type,
            embeddings_in=embedding,
            embeddings_out=None,
            params=param
            )
            kpca = KernelPCA(
                2,
                kernel="precomputed")
            transform_input = kpca.fit_transform(gram_matrix)
        else:
            other_gram_matrix = get_gram_matrix(
            kernel_type=args.kernel_type,
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

plt.style.use('seaborn-dark')

labels = len(clean_embeddings)*list(["Clean"]) + len(noisy_embeddings)*list(["Noisy"])
if args.attack_type != "All":
    for epsilon in all_epsilons:
         labels += len(adv_embeddings[epsilon])*list([f"Adv {args.attack_type} {epsilon}"])
else:
    for att in ["FGSM", "BIM", "DeepFool", "CW"]:
        if att in ["FGSM", "BIM"]:
            for epsilon in all_epsilons:
                labels += len(adv_embeddings[att][epsilon])*list([f"Adv {att} {epsilon}"])
        else:
            labels += len(adv_embeddings[att][1])*list([f"Adv {att} {1}"])
logger.info(f"Labels = {labels}")

embedding_init = clean_embeddings + noisy_embeddings + adv_embeddings_all
transform_input, kpca = process(embedding_init)
filename = directory + f"/{args.attack_type}.png"

le = len(all_epsilons)
if args.attack_type == "FGSM":
    pal = sns.color_palette(["#0F0F0F", "#A0A0A0"] + sns.color_palette("Blues", le))
elif args.attack_type == "BIM":
    pal = sns.color_palette(["#0F0F0F", "#A0A0A0"] + sns.color_palette("Greens", le))
elif args.attack_type == "DeepFool":
    pal = sns.color_palette(["#0F0F0F", "#A0A0A0", "#FEB307"])
elif args.attack_type == "CW":
    pal = sns.color_palette(["#0F0F0F", "#A0A0A0", "#C50000"])
else:
    pal = sns.color_palette(["#0F0F0F", "#A0A0A0"] + sns.color_palette("Blues", le) + sns.color_palette("Greens", le) + ["#FEB307"] + ["#C50000"])

p = sns.scatterplot(x=[item[0] for item in transform_input], y=[item[1] for item in transform_input],
    hue=labels, palette=pal, alpha=0.8)
plt.savefig(filename, dpi=600)
plt.clf()
logger.info(f"Closing figure")

if args.attack_type == "All":
    for att in ["FGSM", "BIM", "DeepFool", "CW"]:
        clean_noisy_end = 2*args.dataset_size
        labels2 = len(clean_embeddings)*list(["Clean"]) + len(noisy_embeddings)*list(["Noisy"])
        filename2 = directory + f"/{args.attack_type}_viz_{att}.png"
        if att == "FGSM":
            fgsm_end = clean_noisy_end + len(adv_embeddings[att])*args.dataset_size
            other_transform_input = transform_input[0:fgsm_end]
            pal2 = sns.color_palette(["#0F0F0F", "#A0A0A0"] + sns.color_palette("Blues", le))
            for epsilon in all_epsilons:
                labels2 += len(adv_embeddings[att][epsilon])*list([f"Adv {att} {epsilon}"])
        elif att == "BIM":
            bim_end = fgsm_end + len(adv_embeddings[att])*args.dataset_size
            other_transform_input = list(transform_input[:clean_noisy_end]) + list(transform_input[fgsm_end:bim_end])
            pal2 = sns.color_palette(["#0F0F0F", "#A0A0A0"] + sns.color_palette("Greens", le))
            for epsilon in all_epsilons:
                labels2 += len(adv_embeddings[att][epsilon])*list([f"Adv {att} {epsilon}"])
        elif att == "DeepFool":
            deepfool_end = bim_end + len(adv_embeddings[att])*args.dataset_size
            other_transform_input = list(transform_input[:clean_noisy_end]) + list(transform_input[bim_end:deepfool_end])
            pal2 = sns.color_palette(["#0F0F0F", "#A0A0A0", "#FEB307"])
            labels2 += len(adv_embeddings[att][1])*list([f"Adv {att} {1}"])
        elif att == "CW":
            other_transform_input = list(transform_input[:clean_noisy_end]) + list(transform_input[deepfool_end:])
            pal2 = sns.color_palette(["#0F0F0F", "#A0A0A0", "#C50000"])
            labels2 += len(adv_embeddings[att][1])*list([f"Adv {att} {1}"])
        p2 = sns.scatterplot(x=[item[0] for item in other_transform_input], y=[item[1] for item in other_transform_input],
            hue=labels2, palette=pal2, alpha=0.8)
        plt.savefig(filename2, dpi=600)
        plt.clf()
        logger.info(f"Closing figure")

end_time = time.time()
logger.info(f"Finished in {end_time - start_time} seconds.")

