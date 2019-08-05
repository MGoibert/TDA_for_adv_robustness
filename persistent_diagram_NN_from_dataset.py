#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:30:59 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

import logging
from tqdm import tqdm
import seaborn as sns
import torch

logger = logging.getLogger()
logger.setLevel(logging.INFO)
torch.set_default_tensor_type(torch.DoubleTensor)
sns.set()

# Files from the project to import
from functions import compute_dgm_from_edges
from generate_graph_dataset import get_dataset, get_model


# ----------------
# ----------------
# Run experiment !
# ----------------
# ----------------


if __name__ == "__main__":
    n = 50
    threshold = 20000
    epsilon = 0.08
    noise = 0.0
    start_n = 0
    num_epochs = 20
    num_classes = 10

    ##############################
    # Loading datasets and model #
    ##############################

    model, loss_func = get_model(num_epochs=num_epochs)

    dataset_non_adv = get_dataset(
        num_epochs=num_epochs,
        epsilon=epsilon,
        noise=noise,
        adv=False
    )

    dataset_adv = get_dataset(
        num_epochs=num_epochs,
        epsilon=epsilon,
        noise=noise,
        adv=True
    )

    dataset_full = dataset_non_adv + dataset_adv

    logger.info("Loaded datasets successfully !")

    for sample in tqdm(dataset_full):
        edges = sample[0]
        dgm = compute_dgm_from_edges(edges, threshold=15000)
        orig_class = sample[1]
        is_adv = sample[3]



    # Experiment with all (clean, adv, noisy) inputs
    # result = run_dist_detection(n, start_n=start_n)
