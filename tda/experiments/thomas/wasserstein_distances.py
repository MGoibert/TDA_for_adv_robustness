import argparse
import logging
import os
import time
import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dionysus import wasserstein_distance

from tda.embeddings import get_embedding, EmbeddingType
from tda.embeddings.persistent_diagrams import sliced_wasserstein_kernel
from tda.graph_dataset import get_dataset
from tda.models.architectures import mnist_mlp, svhn_lenet, get_architecture

start_time = time.time()

directory = "plots/wasserstein_distance"
if not os.path.exists(directory):
    os.mkdir(directory)

################
# Parsing args #
################

parser = argparse.ArgumentParser()
parser.add_argument('--thresholds', type=str, default='15000_15000_15000')
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--architecture', type=str, default=mnist_mlp.name)
parser.add_argument('--dataset_size', type=int, default=3)
parser.add_argument('--successful_adv', type=int, default=1)
parser.add_argument('--attack_type', type=str, default="FGSM")
parser.add_argument('--num_iter', type=int, default=10)
parser.add_argument('--check_nb_sample', type=int, default=20)
parser.add_argument('--approx_wasserstein', type=int, default=0)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[WassersteinDistances]")
logger.info(f"architecture = {svhn_lenet} ")

#####################
# Fetching datasets #
#####################

architecture = get_architecture(args.architecture)

thresholds = [int(x) for x in args.thresholds.split("_")]
print(thresholds)
stats = {}

all_classes = list(range(10))
# all_classes = [0, 1, 2, 3, 4, 5, 6, 8, 9]

#############################
# Embeddings for each class #
#############################


def get_embeddings_per_class(
        epsilon: float,
        noise: float,
        start_index: int = 0) -> (typing.List, int):
    """
    Helper function to get list of embeddings
    """
    my_embeddings = {
        k: list() for k in all_classes
    }
    last_consumed_index = start_index
    inds_class = [0 for _ in all_classes]
    while True:
        for line in get_dataset(
                num_epochs=args.epochs,
                epsilon=epsilon,
                noise=noise,
                adv=epsilon > 0.0,
                retain_data_point=False,
                architecture=architecture,
                source_dataset_name=args.dataset,
                dataset_size=args.check_nb_sample,
                thresholds=thresholds,
                only_successful_adversaries=args.successful_adv > 0,
                attack_type=args.attack_type,
                num_iter=args.num_iter,
                start=last_consumed_index
        ):

            predicted_label = line[2]
            perturbation_size = line[4]

            logger.info(f"Line = {line[:3]} and diff = {perturbation_size}")
            # If we still need samples with this predicted class
            if inds_class[predicted_label] < args.dataset_size:
                inds_class[predicted_label] += 1
                stats[epsilon].append(perturbation_size)
                my_embeddings[predicted_label].append(get_embedding(
                    embedding_type=EmbeddingType.PersistentDiagram,
                    graph=line[0]
                ))
        last_consumed_index += args.check_nb_sample
        logger.info(f"inds class = {inds_class} and last consumed index = {last_consumed_index}")
        classes_are_full = [inds_class[k] >= args.dataset_size
                            for k in all_classes
                            ]
        logger.info(f"Classes are full {classes_are_full}")
        if all(classes_are_full):
            break

    for k in all_classes:
        logger.info(f"Size embedding class {k} = {len(my_embeddings[k])}")
    return my_embeddings, last_consumed_index


########################
# Computing embeddings #
########################

stats[0.0] = list()
clean_embeddings, end_index = get_embeddings_per_class(
    epsilon=0.0,
    noise=0.0
)
logger.info(f"Consumed dataset up to index {end_index}")
noisy_embeddings, end_index = get_embeddings_per_class(
    epsilon=0.0,
    noise=args.noise,
    start_index=end_index
)
logger.info(f"Consumed dataset up to index {end_index}")

if args.attack_type in ["FGSM", "BIM"]:
    all_epsilons = list(sorted(np.linspace(0.01, 0.01, num=1)))
else:
    all_epsilons = [1]

adv_embeddings = dict()
q_values = [0.1, 0.25, 0.75, 0.9]
for epsilon in all_epsilons:
    stats[epsilon] = list()
    adv_embeddings[epsilon] = get_embeddings_per_class(
        epsilon=epsilon,
        noise=0.0,
        start_index=end_index
    )[0]
    quantiles = [np.quantile(stats[epsilon], q) for q in q_values]
    logger.info(
        f"Stats for diff btw clean and adv: {quantiles}")


##########################
# Wasserstein distances  #
##########################

def persistent_dgm_dist(dgms1, dgms2=None):

    distance_function = wasserstein_distance
    if args.approx_wasserstein > 0:
        distance_function = sliced_wasserstein_kernel

    if dgms2 is not None:
        dist = list()
        for dgm1 in dgms1:
            for dgm2 in dgms2:
                dist.append(distance_function(dgm1, dgm2))
        return dist
    else:
        dist = list()
        for i, dgm1 in enumerate(dgms1):
            for dgm2 in dgms1[i + 1:]:
                dist.append(distance_function(dgm1, dgm2))
        return dist


######################
# Per class analysis #
######################

dist_clean = {
    sample_class: persistent_dgm_dist(clean_embeddings[sample_class])
    for sample_class in range(10)
}
t1 = time.time()
logger.info(f"Clean distances computed in {t1 - start_time} seconds !")
dist_clean_noisy = {
    sample_class: persistent_dgm_dist(
        clean_embeddings[sample_class],
        noisy_embeddings[sample_class])
    for sample_class in range(10)
}
t2 = time.time()
logger.info(f"Clean vs Noisy distances computed in {t2 - t1} seconds !")
dist_clean_adv = {}
for epsilon in all_epsilons:
    dist_clean_adv[epsilon] = {
        sample_class: persistent_dgm_dist(
            clean_embeddings[sample_class],
            adv_embeddings[epsilon][sample_class])
        for sample_class in range(10)
    }
t3 = time.time()
logger.info(f"Clean vs Adv distances computed in {t3 - t2} seconds !")

for ind in all_classes:
    if len(dist_clean[ind]) <= 1 or len(dist_clean_noisy[ind]) <= 1:
        continue
    logger.info(f"For class {ind}, size clean = {len(clean_embeddings[ind])}, noisy = {len(noisy_embeddings[ind])} and adv = {[len(adv_embeddings[epsilon][ind]) for epsilon in all_epsilons]}")
    sns.distplot(dist_clean[ind], hist=False, label="Clean")
    sns.distplot(dist_clean_noisy[ind], hist=False, label="Noisy")
    for epsilon in all_epsilons:
        sns.distplot(dist_clean_adv[epsilon][ind], hist=False, label="Adv epsilon = " + str(epsilon))
    plt.savefig(directory + "/dist_plot_" + args.attack_type + "_" + args.dataset + "_class_" + str(ind) + ".png", dpi=800)
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
dist_clean_tot = persistent_dgm_dist(clean_all)
logger.info(f"Clean done ! Size = {len(dist_clean_tot)}")
dist_noisy_tot = persistent_dgm_dist(clean_all, noisy_all)
logger.info(f"Noisy done ! Size = {len(dist_noisy_tot)}")
dist_adv_tot = dict()
for epsilon in all_epsilons:
    dist_adv_tot[epsilon] = persistent_dgm_dist(clean_all, adv_all[epsilon])
    logger.info(f"Adv done for eps = {epsilon} ! Size = {len(dist_adv_tot[epsilon])}")

sns.distplot(dist_clean_tot, hist=False, label="Clean")
sns.distplot(dist_noisy_tot, hist=False, label="Noisy")
for epsilon in all_epsilons:
    sns.distplot(dist_adv_tot[epsilon], hist=False, label="Adv epsilon = " + str(epsilon))
plt.savefig(directory + "/dist_plot_" + args.dataset + "_" + args.attack_type + "_tot" + ".png", dpi=800)
plt.close()

end_time = time.time()

logger.info(f"Successfully ended in {end_time - start_time} seconds !")
