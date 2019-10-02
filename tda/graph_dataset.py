#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import pathlib
import os
import pickle
import torch
import numpy as np
import typing

from tda.models import get_deep_model
from tda.graph import Graph
from tda.models.datasets import Dataset
from tda.models.architectures import Architecture, mnist_mlp

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

pathlib.Path("/tmp/tda/graph_datasets").mkdir(parents=True, exist_ok=True)


# One-hot vector based on scalar
def one_hot(y, num_classes=None):
    if num_classes is None:
        classes, _ = y.max(0)
        num_classes = (classes.max() + 1).item()
    if y.dim() > 0:
        y_ = torch.zeros(len(y), num_classes, device=y.device)
    else:
        y_ = torch.zeros(1, num_classes)
    y_.scatter_(1, y.unsqueeze(-1), 1)
    return y_


def ce_loss(outputs, labels, num_classes=None):
    """
    Cross_entropy loss
    (output = post-softmax output of the model,
     and label =  one-hot)
    """
    labels = one_hot(labels, num_classes=num_classes)
    size = len(outputs)
    if outputs[0].dim() == 0:
        for i in range(size):
            outputs[i] = outputs[i].unsqueeze(-1)
    if labels[0].dim() == 0:
        for i in range(size):
            labels[i] = labels[i].unsqueeze(-1)

    res = 1. / size * sum([torch.dot(torch.log(outputs[i]), labels[i])
                           for i in range(size)])
    return -res


def adversarial_generation(model, x, y,
                           epsilon=0.25,
                           loss_func=ce_loss,
                           num_classes=10):
    """
    Create an adversarial example (FGMS only for now)
    """
    x_clean = x.double()
    x_clean.requires_grad = True
    y_clean = torch.from_numpy(np.asarray(y)).unsqueeze(0)
    output = model(x_clean)
    loss = loss_func(output, y_clean, num_classes)
    model.zero_grad()
    loss.backward()
    x_adv = torch.clamp(x_clean + epsilon * x_clean.grad.data.sign(), -0.5, 0.5).double()

    return x_adv


def process_sample(
        sample: typing.Tuple,
        adversarial: bool = False,
        noise: float = 0,
        epsilon: float = 0,
        model: typing.Optional[torch.nn.Module] = None,
        num_classes: int = 10
):
    # Casting to double
    x, y = sample
    x = x.double()

    # If we use adversarial or noisy example!
    if adversarial:
        x = adversarial_generation(model, x, y, epsilon, num_classes=num_classes)
    if noise > 0:
        x = torch.clamp(x + noise * torch.randn(x.size()), -0.5, 0.5).double()

    return x, y


def get_dataset(
        num_epochs: int,
        epsilon: float,
        noise: float,
        adv: bool,
        source_dataset_name: str = "MNIST",
        architecture: Architecture = mnist_mlp,
        retain_data_point: bool = False
) -> typing.List:

    # Else we have to compute the dataset first
    source_dataset = Dataset(name=source_dataset_name)

    model, loss_func = get_deep_model(
        num_epochs=num_epochs,
        dataset=source_dataset,
        architecture=architecture
    )
    dataset = list()
    N = int(len(source_dataset.test_and_val_dataset) * 0.1)
    correct = 0

    for i in range(N):
        sample = source_dataset.test_and_val_dataset[i]

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
        x_graph = Graph.from_architecture_and_data_point(
            model=model,
            x=x.view(-1, 28 * 28).double(),
            retain_data_point=retain_data_point
        )
        dataset.append((x_graph, y, y_pred, y_adv))

    logger.info(f"Successfully generated dataset of {N} points"
                f" (model accuracy {100 * float(correct) / N}%)")

    return dataset


if __name__ == "__main__":

    for adv in [True, False]:
        dataset = get_dataset(
            num_epochs=20,
            epsilon=0.02,
            noise=0.0,
            adv=adv
        )
