#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import pathlib
import os
import pickle
import time
import torch
import numpy as np
import typing

from tda.models import get_deep_model
from tda.graph import Graph
from tda.models.datasets import Dataset
from tda.models.architectures import Architecture, mnist_mlp
from tda.models.attacks import FGSM, BIM, DeepFool, CW

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


#def adversarial_generation(model, x, y,
#                           epsilon=0.25,
#                           loss_func=ce_loss,
#                           num_classes=10):
#    """
#    Create an adversarial example (FGMS only for now)
#    """
#    x_clean = x.double()
#    x_clean.requires_grad = True
#    y_clean = torch.from_numpy(np.asarray(y)).unsqueeze(0)
#    output = model(x_clean)
#    loss = loss_func(output, y_clean, num_classes)
#    model.zero_grad()
#    loss.backward()
#    x_adv = torch.clamp(x_clean + epsilon * x_clean.grad.data.sign(), -0.5, 0.5).double()
#
#    return x_adv

def adversarial_generation(model, x, y,
                           epsilon=0.25,
                           loss_func=ce_loss,
                           num_classes=10,
                           attack_type='FGSM',
                           num_iter=10,
                           lims = (-0.5, 0.5)):
    """
    Create an adversarial example (FGMS only for now)
    """
    y = torch.tensor([y])
    x.requires_grad = True
    if attack_type == "FGSM":
        attacker = FGSM(model, loss_func)
    elif attack_type == "BIM":
        attacker = BIM(model, loss_func, lims=lims, num_iter=num_iter)
    elif attack_type == "DeepFool":
        attacker = DeepFool(model, num_classes=num_classes, num_iter=num_iter)
    elif attack_type == "CW":
        attacker = CW(model, lims=lims, num_iter=num_iter)
    else:
        raise NotImplementedError(attack_type)

    if attack_type in ["FGSM", "BIM"]:
        x_adv = attacker(x, y, epsilon)
    elif attack_type == "CW":
        x_adv = attacker(x, y)
    elif attack_type == "DeepFool":
        x_adv = attacker(x, y)

    return x_adv


def process_sample(
        sample: typing.Tuple,
        adversarial: bool,
        noise: float = 0,
        epsilon: float = 0,
        model: typing.Optional[torch.nn.Module] = None,
        num_classes: int = 10,
        attack_type: str = "FGSM",
        num_iter: int = 10
):
    # Casting to double
    x, y = sample
    x = x.double()

    # If we use adversarial or noisy example!

    if adversarial:
        x = adversarial_generation(model, x, y, epsilon, num_classes=num_classes, attack_type=attack_type, num_iter=num_iter)
    if noise > 0:
        x = torch.clamp(x + noise * torch.randn(x.size()), -0.5, 0.5).double()

    return x, y


def compute_adv_accuracy(
        num_epochs: int,
        epsilon: float,
        noise: float,
        source_dataset_name: str = "MNIST",
        architecture: Architecture = mnist_mlp,
        dataset_size: int = 100,
        attack_type: str = "FGSM"
) -> float:
    # Else we have to compute the dataset first
    logger.info(f"Getting source dataset {source_dataset_name}")
    source_dataset = Dataset(name=source_dataset_name)
    logger.info(f"Got source dataset {source_dataset_name} !!")

    logger.info(f"Getting deep model...")
    model, loss_func = get_deep_model(
        num_epochs=num_epochs,
        dataset=source_dataset,
        architecture=architecture
    )
    logger.info(f"Got deep model...")

    logger.info(f"I am going to generate a dataset of {dataset_size} points...")

    nb_samples = 0
    i = 0
    corr = 0

    while nb_samples < dataset_size:
        sample = source_dataset.test_and_val_dataset[nb_samples]

        x, y = process_sample(
            sample=sample,
            adversarial=epsilon > 0,
            noise=noise,
            epsilon=epsilon,
            model=model,
            num_classes=10,
            attack_type=attack_type
        )

        y_pred = model(x).argmax(dim=-1).item()

        if y == y_pred:
            corr += 1

        nb_samples += 1

    return corr / dataset_size


def get_dataset(
        num_epochs: int,
        epsilon: float,
        noise: float,
        adv: bool,
        source_dataset_name: str = "MNIST",
        architecture: Architecture = mnist_mlp,
        retain_data_point: bool = False,
        dataset_size: int = 100,
        thresholds: typing.Optional[typing.List[int]] = None,
        only_successful_adversaries: bool = True,
        attack_type: str = "FGSM",
        num_iter: int = 10
) -> typing.Generator:
    # Else we have to compute the dataset first
    logger.info(f"Getting source dataset {source_dataset_name}")
    source_dataset = Dataset(name=source_dataset_name)
    logger.info(f"Got source dataset {source_dataset_name} !!")

    logger.info(f"Getting deep model...")
    model, loss_func = get_deep_model(
        num_epochs=num_epochs,
        dataset=source_dataset,
        architecture=architecture
    )
    logger.info(f"Got deep model...")

    logger.info(f"I am going to generate a dataset of {dataset_size} points...")
    logger.info(f"Only successful adversaries ? {'yes' if only_successful_adversaries else 'no'}")
    logger.info(f"Which attack ? {attack_type}")

    nb_samples = 0
    i = 0

    while nb_samples < dataset_size:
        sample = source_dataset.test_and_val_dataset[i]

        x, y = process_sample(
            sample=sample,
            adversarial=adv,
            noise=noise,
            epsilon=epsilon,
            model=model,
            num_classes=10,
            attack_type=attack_type,
            num_iter=num_iter
        )
        stat = np.linalg.norm(torch.abs((sample[0].double() - x.double()).flatten()).detach().numpy(), np.inf)
        #logger.info(f"x from process sample = {x}")
        y_pred = model(x).argmax(dim=-1).item()
        y_adv = 0 if not adv else 1  # is it adversarial

        if adv and only_successful_adversaries and y_pred == y:
            logger.info(f"Rejecting point (epsilon={epsilon}, y={y}, y_pred={y_pred}, y_adv={y_adv}) and diff = {stat}")
            i += 1
            continue
        #if (not adv) and y_pred != y:
        #    logger.info(f"Rejecting point (epsilon={epsilon}, y={y}, y_pred={y_pred}, y_adv={y_adv})")
        #    i += 1
        #    continue
        else:

            # st = time.time()
            x_graph = Graph.from_architecture_and_data_point(
                model=model,
                x=x.double(),
                retain_data_point=retain_data_point,
                thresholds=thresholds
            )
            # logger.info(f"Computed graph in {time.time()-st} secs")
            nb_samples += 1
            i += 1
            yield (x_graph, y, y_pred, y_adv, stat)


if __name__ == "__main__":

    for adv in [True, False]:
        dataset = get_dataset(
            num_epochs=20,
            epsilon=0.02,
            noise=0.0,
            adv=adv
        )
