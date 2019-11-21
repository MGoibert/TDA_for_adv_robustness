#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import time
import typing
import numpy as np
import matplotlib.pyplot as plt

from tda.graph import Graph
from tda.graph_dataset import get_dataset, compute_adv_accuracy
from tda.models.architectures import mnist_mlp, get_architecture, svhn_lenet, mnist_lenet

import numpy as np
from r3d3 import ExperimentDB

from tda.graph_dataset import compute_adv_accuracy
from tda.models.architectures import get_architecture, svhn_lenet
from tda.rootpath import db_path

start_time = time.time()
directory = "plots/attack_perf/"

my_db = ExperimentDB(db_path=db_path)

################
# Parsing args #
################

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', type=int, default=-1)
parser.add_argument('--run_id', type=int, default=-1)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--dataset', type=str, default="SVHN")
parser.add_argument('--architecture', type=str, default=svhn_lenet.name)
parser.add_argument('--train_noise', type=float, default=0.0)
parser.add_argument('--dataset_size', type=int, default=50)
parser.add_argument('--attack_type', type=str, default="FGSM")
parser.add_argument('--num_iter', type=int, default=10)

args, _ = parser.parse_known_args()

logger = logging.getLogger("GraphStats")

if args.attack_type in ["FGSM", "BIM"]:
    all_epsilons = list(sorted(np.linspace(0.0, 0.4, num=21)))
else:
    all_epsilons = [0.0, 1]

architecture = get_architecture(args.architecture)
accuracies = dict()

for epsilon in all_epsilons:

    adversarial_acc = compute_adv_accuracy(
        epsilon=epsilon,
        noise=args.noise,
        num_epochs=args.epochs,
        source_dataset_name=args.dataset,
        architecture=architecture,
        dataset_size=args.dataset_size,
        attack_type=args.attack_type,
        num_iter=args.num_iter,
        train_noise=args.train_noise

    )

    logging.info(f"Epsilon={epsilon}: acc={adversarial_acc}")
    accuracies[epsilon] = adversarial_acc

logging.info(accuracies)
file_name = directory + str(args.dataset) + "_" + str(args.architecture) + str(args.epochs) + "_" + str(args.attack_type) + ".png"
logger.info(f"file name = {file_name}")
plt.style.use('ggplot')

plt.plot(all_epsilons,list(accuracies.values()), "o-", linewidth=1.5)
plt.title("Standard and adversarial accuracies")
plt.xlabel("Perturbation value")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.savefig(file_name, dpi=800)
plt.close()

my_db.update_experiment(
    experiment_id=args.experiment_id,
    run_id=args.run_id,
    metrics={
        "accuracies": accuracies
    }
)

end_time = time.time()

logging.info(f"Success in {end_time - start_time} seconds")
