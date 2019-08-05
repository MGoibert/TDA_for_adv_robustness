#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:30:59 2019

"""

import logging
import os
import pickle

import torch
import torch.nn as nn
from tqdm import tqdm

torch.set_default_tensor_type(torch.DoubleTensor)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# Files from the project to import
from NN_architecture import MNISTMLP
from datasets import (train_loader_MNIST, test_loader_MNIST, val_loader_MNIST,
                      test_set)
from functions import (train_NN, compute_val_acc, compute_test_acc,
                       compute_all_edge_values, process_sample)

try:
    os.mkdir("trained_models")
except FileExistsError:
    pass

try:
    os.mkdir("graph_datasets")
except FileExistsError:
    pass


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


def _get_dataset_path(
        num_epochs: int,
        epsilon: float,
        noise: float,
        adv: bool
):
    suffix = "_adv" if adv else ""
    return f"graph_datasets/mnist_{num_epochs}_" \
           f"{str(epsilon).replace('.', '_')}_" \
           f"{str(noise).replace('.', '_')}{suffix}"


def get_dataset(
        num_epochs: int,
        epsilon: float,
        noise: float,
        adv: bool
):
    dataset_path = _get_dataset_path(
        num_epochs=num_epochs,
        epsilon=epsilon,
        noise=noise,
        adv=adv
    )

    if os.path.exists(dataset_path):
        with open(dataset_path, "rb") as f:
            return pickle.load(f)

    # Else we have to compute the dataset first
    model, loss_func = get_model(num_epochs=num_epochs)
    dataset = list()
    N = int(len(test_set) * 0.1)
    correct = 0

    for i in tqdm(range(N)):
        sample = test_set[i]

        x, y = process_sample(
            sample=sample,
            adversarial=adv,
            noise=noise,
            epsilon=epsilon,
            model=model,
            num_classes=10
        )

        y_pred = model(x).argmax(dim=-1).item()
        y_adv = 0 if not adv else 1  # is it adversarial
        correct += 1 if y_pred == y else 0
        x_graph = compute_all_edge_values(model,
                                          x.view(-1, 28 * 28).double())
        dataset.append((x_graph, y, y_pred, y_adv))

    logger.info(f"Successfully generated dataset of {N} points"
                f" (model accuracy {100 * float(correct) / N}%)")

    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)

    return dataset


# ----------------
# ----------------
# Run experiment !
# ----------------
# ----------------


if __name__ == "__main__":

    for adv in [True, False]:
        dataset = get_dataset(
            num_epochs=20,
            epsilon=0.02,
            noise=0.0,
            adv=adv
        )
