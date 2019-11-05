import argparse
import logging
import time
import typing
from multiprocessing import Pool
from random import shuffle
from dionysus import wasserstein_distance
import os
import copy

import numpy as np
from r3d3.experiment_db import ExperimentDB
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
import seaborn as sns
import matplotlib.pyplot as plt

from tda.embeddings import get_embedding, EmbeddingType, \
    get_gram_matrix, KernelType
from tda.embeddings.weisfeiler_lehman import NodeLabels
from tda.graph_dataset import get_dataset
from tda.models.architectures import mnist_mlp, get_architecture
from tda.rootpath import db_path

start_time = time.time()

directory = "plots"
if not os.path.exists(directory):
    os.mkdir(directory)

################
# Parsing args #
################

parser = argparse.ArgumentParser(
    description='Transform a dataset in pail files to tf records.')
parser.add_argument('--experiment_id', type=int, default=-1)
parser.add_argument('--run_id', type=int, default=-1)
parser.add_argument('--embedding_type', type=str, default=EmbeddingType.PersistentDiagram)
parser.add_argument('--kernel_type', type=str, default=KernelType.RBF)
parser.add_argument('--thresholds', type=str, default='15000_15000_15000')
parser.add_argument('--height', type=int, default=1)
parser.add_argument('--hash_size', type=int, default=100)
parser.add_argument('--node_labels', type=str, default=NodeLabels.NONE)
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--architecture', type=str, default=mnist_mlp.name)
parser.add_argument('--dataset_size', type=int, default=100)
parser.add_argument('--successful_adv', type=int, default=1)
parser.add_argument('--attack_type', type=str, default="FGSM")
parser.add_argument('--num_iter', type=int, default=10)
parser.add_argument('--check_nb_sample', type=int, default=20)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"[{args.experiment_id}_{args.run_id}]")

#####################
# Fetching datasets #
#####################

architecture = get_architecture(args.architecture)

if args.embedding_type == EmbeddingType.OriginalDataPoint:
    retain_data_point = True
else:
    retain_data_point = False

thresholds = [int(x) for x in args.thresholds.split("_")]
print(thresholds)
stats = {}
last_start_i = 0

#############################
# Embeddings for each class #
#############################

def get_embeddings_per_class(epsilon: float, noise: float, start: int = 0) -> typing.List:
    """
    Helper function to get list of embeddings
    """
    global last_start_i
    my_embeddings = dict()
    for k in range(10): my_embeddings[k] = list()
    inds_class = [0]*len(range(10))
    start_i = start
    while True:
        for line in get_dataset(
            num_epochs=args.epochs,
            epsilon=epsilon,
            noise=noise,
            adv=epsilon > 0.0,
            retain_data_point=retain_data_point,
            architecture=architecture,
            source_dataset_name=args.dataset,
            dataset_size=args.check_nb_sample,
            thresholds=thresholds,
            only_successful_adversaries=args.successful_adv > 0,
            attack_type=args.attack_type,
            num_iter=args.num_iter,
            start=start_i
        ):
            logger.info(f"Line = {line[:3]} and diff = {line[4]}")
            if inds_class[line[2]] < args.dataset_size:
                inds_class[line[2]] += 1
                stats[epsilon].append(line[4])
                my_embeddings[line[2]].append(get_embedding(
                embedding_type=args.embedding_type,
                graph=line[0],
                params={
                    "hash_size": int(args.hash_size),
                    "height": int(args.height),
                    "node_labels": args.node_labels,
                    "steps": args.steps
                }))
        start_i += args.check_nb_sample
        logger.info(f"inds class = {inds_class} and start i = {start_i}")
        if all([inds_class[k] >= args.dataset_size for k in range(len(inds_class))]):
            break

    last_start_i = copy.deepcopy(start_i)
    for k in range(10): logger.info(f"Size embedding class {k} = {len(my_embeddings[k])}")
    return my_embeddings


########################
# Computing embeddings #
########################

stats[0.0] = list()
clean_embeddings = get_embeddings_per_class(epsilon=0.0, noise=0.0)
logger.info(f"Last start i = {last_start_i}")
noisy_embeddings = get_embeddings_per_class(epsilon=0.0, noise=args.noise, start=last_start_i)

if args.attack_type in ["FGSM", "BIM"]:
    all_epsilons = list(sorted(np.linspace(0.1, 0.1, num=1)))
else:
    all_epsilons = [1]

adv_embeddings = dict()
for epsilon in all_epsilons:
    stats[epsilon] = list()
    adv_embeddings[epsilon] = get_embeddings_per_class(epsilon=epsilon, noise=0.0, start=last_start_i)
    logger.info(f"Stats for diff btw clean and adv: {np.quantile(stats[epsilon], 0.1), np.quantile(stats[epsilon], 0.25), np.median(stats[epsilon]), np.quantile(stats[epsilon], 0.75), np.quantile(stats[epsilon], 0.9)}")

##########################
# Wasserstein distances  #
##########################

def persistent_dgm_dist(dgms1, dgms2=None):
    dist_dict = {}

    if dgms2 != None:
        for ind in range(10):
            dist = []
            for i, dgm1 in enumerate(dgms1[ind]):
                for j, dgm2 in enumerate(dgms2[ind]):
                    dist.append(wasserstein_distance(dgm1, dgm2, q=2))
            dist_dict[ind] = dist
    else:
        for ind in range(10):
            dist = []
            for i, dgm1 in enumerate(dgms1[ind]):
                for j, dgm2 in enumerate(dgms1[ind][i+1:]):
                    dist.append(wasserstein_distance(dgm1, dgm2, q=2))
            dist_dict[ind] = dist
    return dist_dict

######################
# Per class analysis #
######################

dist_clean = persistent_dgm_dist(clean_embeddings)
t1 = time.time()
logger.info(f"Clean distances computed in {t1 - start_time} seconds !")
dist_clean_noisy = persistent_dgm_dist(clean_embeddings, noisy_embeddings)
t2 = time.time()
logger.info(f"Clean vs Noisy distances computed in {t2 - t1} seconds !")
dist_clean_adv = {}
for epsilon in all_epsilons:
    dist_clean_adv[epsilon] = persistent_dgm_dist(clean_embeddings, adv_embeddings[epsilon])
t3 = time.time()
logger.info(f"Clean vs Adv distances computed in {t3 - t2} seconds !")


for ind in range(10):
    logger.info(f"For class {ind}, size clean = {len(clean_embeddings[ind])}, noisy = {len(noisy_embeddings[ind])} and adv = {len(adv_embeddings[0.1][ind])}")
    sns.distplot(dist_clean[ind], hist=False, label="Clean")
    sns.distplot(dist_clean_noisy[ind], hist=False, label="Noisy")
    for epsilon in all_epsilons:
        sns.distplot(dist_clean_adv[epsilon][ind], hist=False, label="Adv epsilon = " + str(epsilon))
    plt.savefig(directory + "/dist_plot_" + args.attack_type + "_class_" + str(ind) + ".png", dpi=800)
    plt.close()

########################
# All classes analysis #
########################

logger.info(f"computing distances for every class")
clean_all = [elem for v in clean_embeddings.values() for elem in v]
noisy_all = [elem for v in noisy_embeddings.values() for elem in v]
adv_all = dict()
for epsilon in all_epsilons:
    adv_all[epsilon] = [elem for v in adv_embeddings[epsilon].values() for elem in v]
dist_clean_tot = []
for i, dgm1 in enumerate(clean_all):
    for j, dgm2 in enumerate(clean_all[i+1:]):
        dist_clean_tot.append(wasserstein_distance(dgm1, dgm2, q=2))
logger.info(f"Clean done ! Size = {len(dist_clean_tot)}")
dist_noisy_tot = []
for i, dgm1 in enumerate(clean_all):
    for j, dgm2 in enumerate(noisy_all):
        dist_noisy_tot.append(wasserstein_distance(dgm1, dgm2, q=2))
logger.info(f"Noisy done ! Size = {len(dist_noisy_tot)}")
dist_adv_tot = dict()
for epsilon in all_epsilons:
    dist_adv_tot[epsilon] = list()
    for i, dgm1 in enumerate(clean_all):
        for j, dgm2 in enumerate(adv_all[epsilon]):
            dist_adv_tot[epsilon].append(wasserstein_distance(dgm1, dgm2, q=2))
    logger.info(f"Adv done for eps = {epsilon} ! Size = {len(dist_adv_tot[epsilon])}")
sns.distplot(dist_clean_tot, hist=False, label="Clean")
sns.distplot(dist_noisy_tot, hist=False, label="Noisy")
for epsilon in all_epsilons:
    sns.distplot(dist_adv_tot[epsilon], hist=False, label="Adv epsilon = " + str(epsilon))
plt.savefig(directory + "/dist_plot_" + args.attack_type + "_tot" + ".png", dpi=800)
plt.close()

end_time = time.time()

logger.info(f"Successfully ended in {end_time - start_time} seconds !")
