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
from tda.models.datasets import Dataset
from tda.rootpath import rootpath

from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    DeepFool as DeepFoolArt,
    CarliniL2Method,
    HopSkipJump,
)

from tda.models.attacks import FGSM, BIM, DeepFool, CW

import foolbox as fb

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


class AttackBackend(object):
    CUSTOM = "CUSTOM"
    FOOLBOX = "FOOLBOX"
    ART = "ART"


def adversarial_generation(
    model: Architecture,
    x,
    y,
    epsilon=0.25,
    attack_type="FGSM",
    num_iter=10,
    attack_backend: str = AttackBackend.ART,
):
    """
    Create an adversarial example (FGMS only for now)
    """
    x.requires_grad = True

    if attack_backend == AttackBackend.ART:
        if attack_type == "FGSM":
            attacker = FastGradientMethod(estimator=model.art_classifier, eps=epsilon)
        elif attack_type == "PGD":
            attacker = ProjectedGradientDescent(
                estimator=model.art_classifier,
                max_iter=num_iter,
                eps=epsilon,
                eps_step=2 * epsilon / num_iter,
            )
        elif attack_type == "DeepFool":
            attacker = DeepFoolArt(classifier=model.art_classifier, max_iter=num_iter)
        elif attack_type == "CW":
            attacker = CarliniL2Method(
                classifier=model.art_classifier,
                max_iter=num_iter,
                binary_search_steps=15,
            )
        elif attack_type == "SQUARE":
            # attacker = SquareAttack(estimator=model.get_art_classifier())
            raise NotImplementedError("Work in progress")
        elif attack_type == "HOPSKIPJUMP":
            attacker = HopSkipJump(
                classifier=model.art_classifier,
                targeted=False,
                max_eval=100,
                max_iter=10,
                init_eval=10,
            )
        else:
            raise NotImplementedError(f"{attack_type} is not available in ART")

        attacked = attacker.generate(x=x.detach().cpu())
        attacked = torch.from_numpy(attacked).to(device)

    elif attack_backend == AttackBackend.FOOLBOX:
        if attack_type == "FGSM":
            attacker = fb.attacks.LinfFastGradientAttack()

            attacked, _, _ = attacker(
                model.foolbox_classifier,
                x.detach(),
                torch.from_numpy(y).to(device),
                epsilons=epsilon,
            )
        elif attack_type == "PGD":
            attacker = fb.attacks.LinfProjectedGradientDescentAttack()
            attacked, _, _ = attacker(
                model.foolbox_classifier,
                x.detach(),
                torch.from_numpy(y).to(device),
                epsilons=epsilon,
            )
        else:
            raise NotImplementedError(f"{attack_type} is not available in Foolbox")
    elif attack_backend == AttackBackend.CUSTOM:
        if attack_type == "FGSM":
            attacker = FGSM(model, ce_loss)
            attacked = attacker.run(
                data=x.detach(), target=torch.from_numpy(y).to(device), epsilon=epsilon
            )
        elif attack_type == "PGD":
            attacker = BIM(model, ce_loss, lims=(0, 1), num_iter=num_iter)
            attacked = attacker.run(
                data=x.detach(), target=torch.from_numpy(y).to(device), epsilon=epsilon
            )
        elif attack_type == "DeepFool":
            attacker = DeepFool(model, num_classes=10, num_iter=num_iter)
            attacked = attacker(x, y)
        elif attack_type == "CW":
            attacker = CW(model, lims=(0, 1), num_iter=num_iter)
            attacked = attacker(x, y)
        else:
            raise NotImplementedError(
                f"{attack_type} is not available as custom implementation"
            )
    else:
        raise NotImplementedError(f"Unknown backend {attack_backend}")

    def _to_tensor(x):
        return x if torch.is_tensor(x) else torch.from_numpy(x)

    #x_adv = torch.cat([_to_tensor(x) for x in attacked], 0).to(
    #    device
    #)

    return attacked.detach()


def process_sample(
    sample: typing.Tuple,
    adversarial: bool,
    noise: float = 0,
    epsilon: float = 0,
    model: typing.Optional[Architecture] = None,
    attack_type: str = "FGSM",
    attack_backend: str = AttackBackend.ART,
    num_iter: int = 10,
):
    # Casting to double
    x, y = sample
    x = x.double()

    # If we use adversarial or noisy example!

    if adversarial:
        x = adversarial_generation(
            model=model,
            x=x,
            y=y,
            epsilon=epsilon,
            attack_type=attack_type,
            num_iter=num_iter,
            attack_backend=attack_backend,
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
    attack_backend: str = AttackBackend.ART,
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
    logger.info(f"Which backend ? {attack_backend}")

    if transfered_attacks:
        pathname = saved_adv_path() + f"{dataset.name}/{archi.name}/*{attack_type}*"
        path = glob.glob(pathname)
        source_dataset = torch.load(path[0], map_location=device)[f"{attack_type}"]
        if attack_type in ["FGSM", "PGD"]:
            source_dataset = source_dataset[epsilon]
        source_dataset_size = len(source_dataset["y"])
    else:
        source_dataset = (
            dataset.train_dataset if train else dataset.test_and_val_dataset
        )
        source_dataset_size = len(source_dataset)

    final_dataset = list()

    per_class_nb_samples = np.repeat(0, 10)

    current_sample_id = offset

    dataset_done = False
    batch_size = 128

    while not dataset_done and current_sample_id < source_dataset_size:

        samples = None
        processed_samples = None
        y_pred = None

        while processed_samples is None and current_sample_id < source_dataset_size:

            if transfered_attacks:
                samples = (
                    source_dataset["x"][
                        current_sample_id : current_sample_id + batch_size
                    ],
                    source_dataset["y"][
                        current_sample_id : current_sample_id + batch_size
                    ],
                )
                if adv:
                    processed_samples = (
                        source_dataset["x_adv"][
                            current_sample_id : current_sample_id + batch_size
                        ],
                        source_dataset["y"][
                            current_sample_id : current_sample_id + batch_size
                        ],
                    )
                else:
                    processed_samples = samples
            else:
                # Fetching a batch of samples and concatenating them
                batch = source_dataset[
                    current_sample_id : current_sample_id + batch_size
                ]

                x = torch.cat([torch.unsqueeze(s[0], 0) for s in batch], 0).to(device)
                y = np.array([s[1] for s in batch])
                samples = (x, y)

                # Calling process_sample on the batch
                # TODO: Ensure process_sample handles batched examples
                processed_samples = process_sample(
                    sample=samples,
                    adversarial=adv,
                    noise=noise,
                    epsilon=epsilon,
                    model=archi,
                    attack_type=attack_type,
                    num_iter=num_iter,
                    attack_backend=attack_backend,
                )

            # Increasing current_sample_id
            current_sample_id += batch_size

            assert (samples[1] == processed_samples[1]).all()

            y_pred = archi(processed_samples[0]).argmax(dim=-1).cpu().numpy()

            if adv and succ_adv:
                # Check where the attack was successful
                valid_attacks = np.where(samples[1] != y_pred)[0]
                logger.debug(
                    f"Attack succeeded on {len(valid_attacks)} points over {len(samples[1])}"
                )

                if len(valid_attacks) == 0:
                    processed_samples = None
                else:
                    processed_samples = (
                        processed_samples[0][valid_attacks],
                        processed_samples[1][valid_attacks],
                    )

                    samples = (
                        samples[0][valid_attacks],
                        samples[1][valid_attacks],
                    )

        # If the while loop did not return any samples, let's stop here
        if processed_samples is None:
            break

        # Compute the norms on the batch
        l2_norms = (
            torch.norm(
                (processed_samples[0].double() - samples[0].double()).flatten(1),
                p=2,
                dim=1,
            )
            .cpu()
            .numpy()
        )

        linf_norms = (
            torch.norm(
                (processed_samples[0].double() - samples[0].double()).flatten(1),
                p=float("inf"),
                dim=1,
            )
            .cpu()
            .numpy()
        )

        # Update the counter per class
        for clazz in processed_samples[1]:
            per_class_nb_samples[clazz] += 1

        # Unbatching and return DatasetLine
        # TODO: see if we can avoid unbatching
        for i in range(len(processed_samples[1])):

            x = torch.unsqueeze(processed_samples[0][i], 0).double()

            # (OPT) Compute the graph
            graph = (
                Graph.from_architecture_and_data_point(architecture=archi, x=x)
                if compute_graph
                else None
            )

            # Add the line to the dataset
            final_dataset.append(
                DatasetLine(
                    graph=graph,
                    x=x,
                    y=processed_samples[1][i],
                    y_pred=y_pred[i],
                    y_adv=adv,
                    l2_norm=l2_norms[i],
                    linf_norm=linf_norms[i],
                    sample_id=current_sample_id,
                )
            )

            # Are we done yet ?
            if not per_class:
                # We are done if we have enough points in the dataset
                dataset_done = len(final_dataset) >= dataset_size
            else:
                # We are done if all classes have enough points
                dataset_done = all(np.asarray(per_class_nb_samples) >= dataset_size)

            if dataset_done:
                break

        logger.info(f"Compputed {len(final_dataset)}/{dataset_size} samples.")

    if len(final_dataset) < dataset_size:
        logger.warn(
            f"I was only able to generate {len(final_dataset)} points even if {dataset_size} was requested. "
            f"This is probably a lack of adversarial points."
        )

    return final_dataset
