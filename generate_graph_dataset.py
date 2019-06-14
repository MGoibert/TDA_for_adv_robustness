#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:30:59 2019

"""

import logging
import os
import pickle

import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
torch.set_default_tensor_type(torch.DoubleTensor)
sns.set()

# Files from the project to import
from NN_architecture import MNISTMLP
from datasets import (train_loader_MNIST, test_loader_MNIST, val_loader_MNIST,
                      test_set)
from functions import (train_NN, compute_val_acc, compute_test_acc,
                       compute_all_edge_values, process_sample)
from utils import parse_cmdline_args

try:
    os.mkdir("trained_models")
except FileExistsError:
    pass

try:
    os.mkdir("graph_datasets")
except FileExistsError:
    pass

# ------------------------
# ----- Command line arguments
# ------------------------

args = parse_cmdline_args()
num_epochs = args.num_epochs
epsilon = args.epsilon
noise = args.noise
threshold = args.threshold
save = args.save


# --------------------
# ----- Neural Network
# --------------------


def get_model(
        num_epochs: int
) -> (nn.Module, nn.Module):

    model_filename = f"trained_models/mnist_{num_epochs}_epochs.model"
    loss_func = nn.CrossEntropyLoss()

    try:
        net = torch.load(model_filename)
        print(f"Loaded successfully model from {model_filename}")
    except FileNotFoundError:
        print(f"Unable to find model in {model_filename}... Retrying it...")

        # Use the MLP model
        model = MNISTMLP()

        # MNIST dataset
        train_loader = train_loader_MNIST
        test_loader = test_loader_MNIST
        val_loader = val_loader_MNIST

        # Train the NN
        net = train_NN(model, train_loader, val_loader, loss_func, num_epochs)[0]

        # Compute accuracies
        compute_val_acc(model, val_loader)
        compute_test_acc(model, test_loader)

        # Saving model
        torch.save(net, model_filename)

    return net, loss_func


# ----------------
# ----------------
# Run experiment !
# ----------------
# ----------------

if __name__ == "__main__":
    model, loss_func = get_model(num_epochs=num_epochs)

    print(model)
    print(loss_func)

    ###########################
    # Non-adversarial samples #
    ###########################

    graph_test_set = list()

    for i in tqdm(range(int(len(test_set)*0.1))):
        sample = test_set[i]

        x, y = process_sample(
            sample=sample,
            adversarial=False,
            noise=noise,
            epsilon=epsilon,
            model=model,
            loss_func=loss_func,
            num_classes=10
        )

        y_adv = 0  # not an adversarial sample
        x_graph = compute_all_edge_values(model, x.double())
        graph_test_set.append((x_graph, y, y_adv))

    with open(f"graph_datasets/mnist_{num_epochs}", "wb") as f:
        pickle.dump(graph_test_set, f)


    #######################
    # Adversarial samples #
    #######################

    graph_test_set = list()

    for i in tqdm(range(int(len(test_set)*0.1))):
        sample = test_set[i]

        x, y = process_sample(
            sample=sample,
            adversarial=True,
            noise=noise,
            epsilon=epsilon,
            model=model,
            loss_func=loss_func,
            num_classes=10
        )

        y_adv = 1  # not an adversarial sample
        x_graph = compute_all_edge_values(model, x.double())
        graph_test_set.append((x_graph, y, y_adv))

    with open(f"graph_datasets/mnist_{num_epochs}_adv", "wb") as f:
        pickle.dump(graph_test_set, f)
