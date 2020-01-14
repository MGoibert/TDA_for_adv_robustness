#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import typing

import numpy as np
import torch

from tda.graph import Graph
from tda.models.architectures import Architecture, mnist_mlp
from tda.models.attacks import FGSM, BIM, DeepFool, CW
from tda.models.datasets import Dataset
from tda.devices import device
from tda.logging import get_logger

logger = get_logger("GraphDataset")


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
    if device.type == "cuda":
        y_ = y_.to(device)
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
                           num_classes=10,
                           attack_type='FGSM',
                           num_iter=10,
                           lims=(-0.5, 0.5)):
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
        x = adversarial_generation(model, x, y, epsilon, num_classes=num_classes, attack_type=attack_type,
                                   num_iter=num_iter)
    if noise > 0:
        x = torch.clamp(x + noise * torch.randn(x.size()), -0.5, 0.5).double()

    return x, y


class DatasetLine(typing.NamedTuple):
    graph: Graph
    y: int
    y_pred: int
    y_adv: int
    l2_norm: float
    linf_norm: float
    sample_id: int
    x: torch.tensor


def get_graph_dataset(
        epsilon: float,
        noise: float,
        adv: bool,
        dataset: Dataset,
        architecture: Architecture = mnist_mlp,
        dataset_size: int = 100,
        thresholds: typing.Optional[typing.List[float]] = None,
        only_successful_adversaries: bool = True,
        attack_type: str = "FGSM",
        num_iter: int = 10,
        start: int = 0,
        per_class: bool = False,
) -> typing.Generator[DatasetLine, None, None]:
    # Else we have to compute the dataset first
    logger.info(f"Using source dataset {dataset.name}")

    logger.info(f"Checking that the received architecture has been trained")
    assert architecture.is_trained
    logger.info(f"OK ! Architecture is ready")

    logger.info(f"I am going to generate a dataset of {dataset_size} points...")
    logger.info(f"Only successful adversaries ? {'yes' if only_successful_adversaries else 'no'}")
    logger.info(f"Which attack ? {attack_type}")

    nb_samples = 0
    i = start
    per_class_nb_samples = np.repeat(0, 10)

    while nb_samples < dataset_size:
        sample = dataset.test_and_val_dataset[i]
        logger.debug(f"per class : {per_class_nb_samples} and nb samples = {nb_samples}")

        x, y = process_sample(
            sample=sample,
            adversarial=adv,
            noise=noise,
            epsilon=epsilon,
            model=architecture,
            num_classes=10,
            attack_type=attack_type,
            num_iter=num_iter
        )
        if per_class and per_class_nb_samples[y] >= dataset_size:
            i += 1
            continue
        l2_norm = np.linalg.norm(torch.abs((sample[0].double() - x.double()).flatten()).detach().numpy(), 2)
        linf_norm = np.linalg.norm(torch.abs((sample[0].double() - x.double()).flatten()).detach().numpy(), np.inf)
        y_pred = architecture(x).argmax(dim=-1).item()
        y_adv = 0 if not adv else 1  # is it adversarial

        if adv and only_successful_adversaries and y_pred == y:
            logger.debug(
                f"Rejecting point (epsilon={epsilon}, y={y}, y_pred={y_pred}, y_adv={y_adv}) and diff = {l2_norm}")
            i += 1
            continue
        else:
            x_graph = Graph.from_architecture_and_data_point(
                architecture=architecture,
                x=x.double(),
                thresholds=thresholds
            )
            nb_samples += 1
            i += 1
            per_class_nb_samples[y] += 1
            if per_class and any(np.asarray(per_class_nb_samples) < dataset_size):
                nb_samples = 0
            elif per_class and all(np.asarray(per_class_nb_samples) >= dataset_size):
                nb_samples = dataset_size + 1
            yield DatasetLine(
                graph=x_graph,
                y=y,
                y_pred=y_pred,
                y_adv=y_adv,
                l2_norm=l2_norm,
                linf_norm=linf_norm,
                sample_id=i - 1,
                x=x
            )
