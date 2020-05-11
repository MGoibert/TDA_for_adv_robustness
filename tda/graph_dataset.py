#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import typing
import glob
import pathlib

import numpy as np
import torch

from tda.cache import cached
from tda.devices import device
from tda.graph import Graph
from tda.tda_logging import get_logger
from tda.models.architectures import Architecture, mnist_mlp
from tda.models.attacks import FGSM, BIM, DeepFool, CW
from tda.models.datasets import Dataset
from tda.rootpath import rootpath

from art.attacks import (
    FastGradientMethod,
    ProjectedGradientDescent,
    DeepFool as DeepFoolArt,
    CarliniL2Method,
)

logger = get_logger("GraphDataset")


def saved_adv_path():
    directory = f"{rootpath}/saved_adversaries/"
    pathlib.Path(directory).mkdir(exist_ok=True, parents=True)
    return str(directory)


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

    res = (
        1.0
        / size
        * sum([torch.dot(torch.log(outputs[i]), labels[i]) for i in range(size)])
    )
    return -res


def adversarial_generation(
    model: Architecture,
    x,
    y,
    epsilon=0.25,
    loss_func=ce_loss,
    num_classes=10,
    attack_type="FGSM",
    num_iter=10,
    lims=(0, 1),
):
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
    elif attack_type == "FGSM_art":
        attacker = FastGradientMethod(
            classifier=model.get_art_classifier(), eps=epsilon
        )
    elif attack_type == "BIM_art":
        attacker = ProjectedGradientDescent(
            classifier=model.get_art_classifier(),
            max_iter=num_iter,
            eps=epsilon,
            eps_step=2 * epsilon / num_iter,
        )
    elif attack_type == "DeepFool_art":
        attacker = DeepFoolArt(classifier=model.get_art_classifier(), max_iter=num_iter)
    elif attack_type == "CW_art":
        attacker = CarliniL2Method(
            classifier=model.get_art_classifier(),
            max_iter=num_iter,
            binary_search_steps=15,
        )
    else:
        raise NotImplementedError(attack_type)

    x_adv = None

    if attack_type in ["FGSM", "BIM"]:
        x_adv = attacker(x, y, epsilon)
    elif attack_type == "CW":
        x_adv = attacker(x, y)
    elif attack_type == "DeepFool":
        x_adv = attacker(x, y)
    elif attack_type in ["FGSM_art", "BIM_art", "DeepFool_art", "CW_art"]:
        x_adv = torch.Tensor(attacker.generate(x.detach()))
        x_adv.to(device)

    return x_adv


def process_sample(
    sample: typing.Tuple,
    adversarial: bool,
    noise: float = 0,
    epsilon: float = 0,
    model: typing.Optional[torch.nn.Module] = None,
    num_classes: int = 10,
    attack_type: str = "FGSM",
    num_iter: int = 10,
):
    # Casting to double
    x, y = sample
    x = x.double()

    # If we use adversarial or noisy example!

    if adversarial:
        x = adversarial_generation(
            model,
            x,
            y,
            epsilon,
            num_classes=num_classes,
            attack_type=attack_type,
            num_iter=num_iter,
        )
    if noise > 0:
        x = torch.clamp(x + noise * torch.randn(x.size(), device=device), 0, 1).double()

    return x, y


class DatasetLine(typing.NamedTuple):
    graph: typing.Optional[Graph]
    y: int
    y_pred: int
    y_adv: int
    l2_norm: float
    linf_norm: float
    sample_id: int
    x: torch.tensor


@cached
def get_sample_dataset(
    epsilon: float,
    noise: float,
    adv: bool,
    dataset: Dataset,
    train: bool,
    succ_adv: bool,
    archi: Architecture = mnist_mlp,
    dataset_size: int = 100,
    attack_type: str = "FGSM",
    num_iter: int = 10,
    offset: int = 0,
    per_class: bool = False,
    compute_graph: bool = False,
    transfered_attacks: bool = False,
) -> typing.List[DatasetLine]:
    logger.info(f"Using source dataset {dataset.name}")

    logger.info(f"Checking that the received architecture has been trained")
    assert archi.is_trained
    logger.info(f"OK ! Architecture is ready")

    logger.info(f"I am going to generate a dataset of {dataset_size} points...")
    logger.info(f"Only successful adversaries ? {'yes' if succ_adv else 'no'}")
    logger.info(f"Which attack ? {attack_type}")

    nb_samples = 0
    per_class_nb_samples = np.repeat(0, 10)

    current_sample_id = offset

    if transfered_attacks:
        pathname = saved_adv_path() + f"{dataset.name}/{archi.name}/*{attack_type}*"
        path = glob.glob(pathname)
        logger.info(f"Using transfered attacks file {path}")
        source_dataset = torch.load(path[0], map_location=device)[f"{attack_type}"]
        lsd = (
            len(source_dataset["y"])
            if attack_type not in ["FGSM", "BIM"]
            else len(source_dataset[epsilon]["y"])
        )
    else:
        source_dataset = (
            dataset.train_dataset if train else dataset.test_and_val_dataset
        )
        lsd = len(source_dataset)

    ret = list()

    while nb_samples < dataset_size and current_sample_id < lsd:

        sample = None
        processed_sample = None
        y_pred = None

        while processed_sample is None:
            if transfered_attacks:
                if attack_type in ["FGSM", "BIM"]:
                    sample = [
                        source_dataset[epsilon]["x"][current_sample_id],
                        source_dataset[epsilon]["y"][current_sample_id],
                    ]
                    if adv:
                        processed_sample = [
                            source_dataset[epsilon]["x_adv"][current_sample_id],
                            source_dataset[epsilon]["y"][current_sample_id],
                        ]
                    else:
                        processed_sample = sample
                else:
                    sample = [
                        source_dataset["x"][current_sample_id],
                        source_dataset["y"][current_sample_id],
                    ]
                    if adv:
                        processed_sample = [
                            source_dataset["x_adv"][current_sample_id],
                            source_dataset["y"][current_sample_id],
                        ]
                    else:
                        processed_sample = sample
            else:
                sample = source_dataset[current_sample_id]
                processed_sample = process_sample(
                    sample=sample,
                    adversarial=adv,
                    noise=noise,
                    epsilon=epsilon,
                    model=archi,
                    num_classes=10,
                    attack_type=attack_type,
                    num_iter=num_iter,
                )

            assert sample[1] == processed_sample[1]

            y_pred = archi(processed_sample[0]).argmax(dim=-1).item()
            if adv and succ_adv and y_pred == sample[1]:
                logger.debug(
                    f"Rejecting point (epsilon={epsilon}, y={sample[1]}, y_pred={y_pred}, adv={adv})"
                )
                processed_sample = None
                current_sample_id += 1
                if current_sample_id >= lsd:
                    break

        if processed_sample is None:
            break

        # Ok we have found a point
        l2_norm = np.linalg.norm(
            torch.abs((processed_sample[0].double() - sample[0].double()).flatten())
            .cpu()
            .detach()
            .numpy(),
            2,
        )
        linf_norm = np.linalg.norm(
            torch.abs((processed_sample[0].double() - sample[0].double()).flatten())
            .cpu()
            .detach()
            .numpy(),
            np.inf,
        )

        nb_samples += 1
        if nb_samples % 10 == 0:
            logger.info(f"computing sample number = {nb_samples}/{dataset_size}")
        per_class_nb_samples[processed_sample[1]] += 1
        if per_class and any(np.asarray(per_class_nb_samples) < dataset_size):
            nb_samples = 0
        elif per_class and all(np.asarray(per_class_nb_samples) >= dataset_size):
            nb_samples = dataset_size + 1

        if compute_graph:
            graph = Graph.from_architecture_and_data_point(
                architecture=archi, x=processed_sample[0].double()
            )
        else:
            graph = None

        ret.append(
            DatasetLine(
                graph=graph,
                x=processed_sample[0],
                y=processed_sample[1],
                y_pred=y_pred,
                y_adv=adv,
                l2_norm=l2_norm,
                linf_norm=linf_norm,
                sample_id=current_sample_id,
            )
        )

        current_sample_id += 1

    if nb_samples < dataset_size:
        logger.warn(
            f"I was only able to generate {nb_samples} points even if {dataset_size} was requested. "
            f"This is probably a lack of adversarial points."
        )

    return ret
