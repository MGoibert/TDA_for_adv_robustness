#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import time

from r3d3.experiment_db import ExperimentDB

from tda.graph_dataset import process_sample
from tda.models import mnist_mlp, Dataset, get_deep_model, mnist_lenet
from tda.models.architectures import get_architecture, Architecture
from tda.rootpath import db_path

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from tda.models.architectures import SoftMaxLayer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

start_time = time.time()

plot_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

my_db = ExperimentDB(db_path=db_path)

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', type=int, default=-1)
parser.add_argument('--run_id', type=int, default=-1)

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--architecture', type=str, default=mnist_lenet.name)
parser.add_argument('--dataset_size', type=int, default=100)
parser.add_argument('--attack_type', type=str, default="FGSM")
parser.add_argument('--epsilon', type=float, default=0.02)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--number_of_nn', type=int, default=20)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"[{args.experiment_id}_{args.run_id}]")

for arg in vars(args):
    logger.info(f"{arg} => {getattr(args, arg)}")

dataset = Dataset(name=args.dataset)

logger.info(f"Getting deep model...")
model, loss_func = get_deep_model(
    num_epochs=args.epochs,
    dataset=dataset.Dataset_,
    architecture=get_architecture(args.architecture),
    train_noise=0.0
)

archi: Architecture = model

logger.info(f"Got deep model...")

all_lids_norm = list()
all_lids_adv = list()
all_lids_noisy = list()

for batch_idx in range(10):
    raw_batch = dataset.Dataset_.test_and_val_dataset[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
    b_norm = [s[0] for s in raw_batch]
    true_labels = [s[1] for s in raw_batch]

    #########################
    # Creating noisy batch  #
    #########################
    b_noisy = list()

    for x_norm in b_norm:
        x_noisy, _ = process_sample(
            sample=(x_norm, 0),
            adversarial=False,
            noise=args.noise,
            epsilon=None,
            model=None,
            num_classes=None,
            attack_type=None
        )
        b_noisy.append(x_noisy)

    #######################
    # Creating adv batch  #
    #######################
    b_adv = list()

    for i, x_norm in enumerate(b_norm):
        x_adv, _ = process_sample(
            sample=(x_norm, true_labels[i]),
            adversarial=True,
            noise=0.0,
            epsilon=args.epsilon,
            model=archi,
            num_classes=10,
            attack_type=args.attack_type
        )
        b_adv.append(x_adv)

    b_norm = torch.cat([x.unsqueeze(0) for x in b_norm])
    b_adv = torch.cat([x.unsqueeze(0) for x in b_adv])
    b_noisy = torch.cat([x.unsqueeze(0) for x in b_noisy])

    #########################
    # Computing activations #
    #########################

    a_norm = archi.forward(b_norm, output="all_inner")
    a_adv = archi.forward(b_adv, output="all_inner")
    a_noisy = archi.forward(b_noisy, output="all_inner")

    lids_norm = np.zeros((args.batch_size, len(archi.layers) - 1))
    lids_adv = np.zeros((args.batch_size, len(archi.layers) - 1))
    lids_noisy = np.zeros((args.batch_size, len(archi.layers) - 1))

    for layer_idx in archi.layer_visit_order:
        if layer_idx == -1:
            # Skipping input
            continue
        if isinstance(archi.layers[layer_idx], SoftMaxLayer):
            # Skipping softmax
            continue
        a_norm_layer = a_norm[layer_idx].reshape(args.batch_size, -1).cpu().detach().numpy()
        a_adv_layer = a_adv[layer_idx].reshape(args.batch_size, -1).cpu().detach().numpy()
        a_noisy_layer = a_noisy[layer_idx].reshape(args.batch_size, -1).cpu().detach().numpy()

        d_norm = euclidean_distances(a_norm_layer, a_norm_layer)
        d_adv = euclidean_distances(a_adv_layer, a_adv_layer)
        d_noisy = euclidean_distances(a_noisy_layer, a_noisy_layer)

        for sample_idx in range(args.batch_size):
            z_norm = d_norm[sample_idx]
            z_norm = z_norm[z_norm.argsort()[1:args.number_of_nn + 1]]

            lid_norm = -1 / (sum([np.log(x / z_norm[-1]) for x in z_norm]) / args.number_of_nn)

            z_adv = d_adv[sample_idx]
            z_adv = z_adv[z_adv.argsort()[1:args.number_of_nn + 1]]

            lid_adv = -1 / (sum([np.log(x / z_adv[-1]) for x in z_adv]) / args.number_of_nn)

            z_noisy = d_noisy[sample_idx]
            z_noisy = z_noisy[z_noisy.argsort()[1:args.number_of_nn + 1]]

            lid_noisy = -1 / (sum([np.log(x / z_noisy[-1]) for x in z_noisy]) / args.number_of_nn)

            lids_norm[sample_idx, layer_idx] = lid_norm
            lids_adv[sample_idx, layer_idx] = lid_adv
            lids_noisy[sample_idx, layer_idx] = lid_noisy

    all_lids_norm.append(lids_norm)
    all_lids_adv.append(lids_adv)
    all_lids_noisy.append(lids_noisy)

all_lids_norm = np.concatenate(all_lids_norm, axis=0)
all_lids_adv = np.concatenate(all_lids_adv, axis=0)
all_lids_noisy = np.concatenate(all_lids_noisy, axis=0)

N = len(all_lids_norm)
logger.info(f"Computed LIDs for {N} points")

train_ds = np.concatenate([
    all_lids_norm[:N // 2],
    all_lids_adv[:N // 2]
])

train_labels = np.concatenate([
    np.zeros((N // 2, 1)),
    np.ones((N // 2, 1))
])

logger.info(f"Shape of train ds for LR is {np.shape(train_ds)}")
logger.info(f"Shape of train labels for LR is {np.shape(train_labels)}")

test_ds = np.concatenate([
    all_lids_norm[N // 2:],
    all_lids_adv[N // 2:]
])

test_labels = np.concatenate([
    np.zeros((N // 2, 1)),
    np.ones((N // 2, 1))
])

logger.info(f"Shape of train ds for LR is {np.shape(test_ds)}")
logger.info(f"Shape of train labels for LR is {np.shape(test_labels)}")

detector = LogisticRegression(
    fit_intercept=True,
    verbose=1,
    tol=1e-5,
    max_iter=1000,
    solver='lbfgs'
)

detector.fit(X=train_ds, y=train_labels)
coefs = list(detector.coef_.flatten())
logger.info(f"Coefs of detector {coefs}")

test_predictions = detector.predict_proba(X=test_ds)[:, 1]
auc = roc_auc_score(y_true=test_labels, y_score=test_predictions)
logger.info(f"AUC is {auc}")


my_db.update_experiment(
    experiment_id=args.experiment_id,
    run_id=args.run_id,
    metrics={
        "time": time.time()-start_time,
        "classifier_coefs": coefs,
        "auc": auc
    }
)
