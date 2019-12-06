#!/usr/bin/env python
# coding: utf-8

import time
import argparse
import logging
import numpy as np
import typing
import os
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from r3d3.experiment_db import ExperimentDB

from tda.graph_dataset import process_sample
from tda.models import mnist_mlp, Dataset, get_deep_model
from tda.models.architectures import get_architecture, Architecture
from tda.rootpath import db_path

import matplotlib.pyplot as plt

start_time = time.time()

plot_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

my_db = ExperimentDB(db_path=db_path)

parser = argparse.ArgumentParser(
    description='Transform a dataset in pail files to tf records.')
parser.add_argument('--experiment_id', type=int, default=-1)
parser.add_argument('--run_id', type=int, default=-1)

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--architecture', type=str, default=mnist_mlp.name)
parser.add_argument('--dataset_size', type=int, default=100)
parser.add_argument('--attack_type', type=str, default="FGSM")
parser.add_argument('--epsilon', type=float, default=0.02)
parser.add_argument('--preproc_epsilon', type=float, default=0.0)
parser.add_argument('--noise', type=float, default=0.0)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"[{args.experiment_id}_{args.run_id}]")

for arg in vars(args):
    logger.info(f"{arg} => {getattr(args, arg)}")

dataset = Dataset(name=args.dataset)

logger.info(f"Getting deep model...")
model, loss_func = get_deep_model(
    num_epochs=args.epochs,
    dataset=dataset,
    architecture=get_architecture(args.architecture),
    train_noise=0.0
)

archi: Architecture = model

logger.info(f"Got deep model...")

##########################################
# Step 1: Compute all mu_clazz and sigma #
#         for each layer                 #
##########################################

max_sample_for_classes = 100
nb_sample_for_classes = 0

logger.info(f"I am going to go through a dataset of {max_sample_for_classes} points...")

i = 0
corr = 0

class_counters = dict()
class_means = dict()

features_per_class = dict()
mean_per_class = dict()
sigma_per_class = dict()


def sum_list_of_lists(
        list_a: typing.List[typing.List],
        list_b: typing.List[typing.List]):
    if len(list_a) == 0:
        return list_b
    else:
        assert len(list_a) == len(list_b)
        return [list_a[i] + list_b[i]
                for i in range(len(list_a))]


# As proposed in the paper, we use the train samples here
for x, y in dataset.train_dataset:

    if nb_sample_for_classes >= max_sample_for_classes:
        break

    m_features = archi.get_all_inner_activations(x)

    for i, feat in enumerate(m_features):
        if i not in features_per_class:
            features_per_class[i] = dict()
        features_per_class[i][y] = features_per_class[i].get(y, list()) + [feat]

    nb_sample_for_classes += 1

all_feature_indices = sorted(list(features_per_class.keys()))
logger.info(f"All indices for features are {all_feature_indices}")
all_classes = sorted(list(features_per_class[all_feature_indices[0]].keys()))
logger.info(f"All classes are {all_classes}")


for layer_idx in all_feature_indices:
    mean_per_class[layer_idx] = dict()
    sigma_per_class[layer_idx] = None
    counters_per_class = 0

    for clazz in all_classes:

        # Shortcut for the name
        arr = features_per_class[layer_idx][clazz]

        # Computing mu_clazz
        mu_clazz = sum(arr) / len(arr)
        mean_per_class[layer_idx][clazz] = mu_clazz

        # Computing sigma_clazz
        sigma_clazz = sum([np.transpose(v-mu_clazz)@(v-mu_clazz) for v in arr])
        if sigma_per_class[layer_idx] is None:
            sigma_per_class[layer_idx] = sigma_clazz
        else:
            sigma_per_class[layer_idx] += sigma_clazz
        counters_per_class += len(arr)

    sigma_per_class[layer_idx] = sigma_per_class[layer_idx]/counters_per_class

    plt.imshow(sigma_per_class[layer_idx])
    plt.savefig(f"{plot_path}/{layer_idx}_sigma")


################################################################
# Step 2: (OPT) Evaluate classifier based on confidence scores #
################################################################

i = 0
corr = 0
while i < args.dataset_size:
    x, y = dataset.test_and_val_dataset[i]
    m_features = archi.get_all_inner_activations(x)[-1]

    best_score = np.inf
    best_class = -1

    last_layer_idx = all_feature_indices[-1]

    for clazz in all_classes:
        mu_clazz = mean_per_class[last_layer_idx][clazz]
        sigma_clazz = sigma_per_class[last_layer_idx]

        score_clazz = (m_features-mu_clazz)@np.linalg.pinv(sigma_clazz)@np.transpose(m_features-mu_clazz)

        if score_clazz < best_score:
            best_score = score_clazz
            best_class = clazz

    if best_class == y:
        corr += 1

    i += 1

logger.info(f"Accuracy on test set = {corr/args.dataset_size}")


#############################################
# Step 3: Evaluate performance of detector  #
#############################################

def create_dataset(start: int) -> pd.DataFrame:
    i = start
    ret = list()

    while i < start + args.dataset_size:
        sample = dataset.test_and_val_dataset[i]

        if i % 2 == 0:
            adv = True
            epsilon = args.epsilon
            noise = 0.0
        else:
            adv = False
            epsilon = 0
            noise = args.noise

        x, y = process_sample(
            sample=sample,
            adversarial=adv,
            noise=noise,
            epsilon=epsilon,
            model=model,
            num_classes=10,
            attack_type=args.attack_type
        )

        m_features = archi.get_all_inner_activations(x)

        scores = list()

        for layer_idx in all_feature_indices:

            best_score = np.inf
            sigma_l = sigma_per_class[layer_idx]
            mu_l = None

            for clazz in all_classes:
                mu_clazz = mean_per_class[layer_idx][clazz]
                gap = m_features[layer_idx]-mu_clazz
                score_clazz = gap@np.linalg.pinv(sigma_l)@np.transpose(gap)

                if score_clazz < best_score:
                    best_score = score_clazz[0, 0]
                    mu_l = mu_clazz

            # OPT: Add perturbation on x to increase its score
            # (mainly useful for
            if args.preproc_epsilon > 0:
                # Let's forget about the past (attack, clamping)
                # and make x a leaf with require grad
                x = x.detach()
                x.requires_grad = True

                inv_sigma_tensor = torch.from_numpy(np.linalg.pinv(sigma_l))
                mu_tensor = torch.from_numpy(mu_l)

                # Adding perturbation on x
                f = archi.get_activations_for_layer(x, layer_idx=layer_idx)
                live_score = (f - mu_tensor) @ inv_sigma_tensor @ (f - mu_tensor).T

                assert np.isclose(live_score.detach().numpy(), best_score)

                archi.zero_grad()
                live_score.backward()
                assert x.grad is not None
                xhat = x + args.preproc_epsilon * x.grad.data.sign()
                xhat = torch.clamp(xhat, -0.5, 0.5)

                # Computing new score
                fhat = archi.get_activations_for_layer(xhat, layer_idx=layer_idx)
                new_score = (fhat - mu_tensor) @ inv_sigma_tensor @ (fhat - mu_tensor).T

                logger.info(f"Added perturbation to x {live_score.detach().numpy()[0,0]} "
                            f"-> {new_score.detach().numpy()[0,0]}")

                best_score = new_score.detach().numpy()[0, 0]

            scores.append(best_score)

        ret.append(scores + [int(adv)])
        i += 1

    return pd.DataFrame(ret, columns=[f"layer_{idx}" for idx in all_feature_indices]+["label"])


detector_train_dataset = create_dataset(start=0)
logger.info("Generated train dataset for detector !")

detector_test_dataset = detector_train_dataset # create_dataset(start=args.dataset_size)
logger.info("Generated test dataset for detector !")

detector = LogisticRegression(
    fit_intercept=True,
    verbose=1,
    tol=1e-5,
    max_iter=1000,
    solver='lbfgs'
)

detector.fit(X=detector_train_dataset.iloc[:, :-1], y=detector_train_dataset.iloc[:, -1])
coefs = list(detector.coef_.flatten())
logger.info(f"Coefs of detector {coefs}")

test_predictions = detector.predict_proba(X=detector_test_dataset.iloc[:, :-1])[:, 1]
auc = roc_auc_score(y_true=detector_test_dataset.iloc[:, -1], y_score=test_predictions)
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
