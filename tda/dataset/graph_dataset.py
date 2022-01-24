#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import typing
import glob
import pathlib
import socket
import inspect
import os

import numpy as np
import torch

from tda.cache import cached
from tda.dataset.adversarial_generation import AttackBackend, adversarial_generation
from tda.devices import device
from tda.graph import Graph
from tda.tda_logging import get_logger
from tda.models.architectures import Architecture, mnist_mlp
from tda.dataset.datasets import Dataset
from tda.rootpath import rootpath
from tda.models import get_deep_model


logger = get_logger("GraphDataset")


def saved_adv_path():
    directory = f"{rootpath}/saved_adversaries/"
    pathlib.Path(directory).mkdir(exist_ok=True, parents=True)
    return str(directory)


def process_sample(
    sample: typing.Tuple,
    adversarial: bool,
    noise: float = 0,
    epsilon: float = 0,
    model: typing.Optional[Architecture] = None,
    attack_type: str = "FGSM",
    attack_backend: str = AttackBackend.FOOLBOX,
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
    x0: torch.tensor

def get_my_path(list_locals):
    if os.path.exists("/var/opt/data/user_data"):
        # We are on gpu
        cache_root = f"/var/opt/data/user_data/tda/"
    elif "mesos" in socket.gethostname():
        # We are in mozart
        cache_root = f"{os.environ['HOME']}/tda_cache/"
    else:
        # Other cases (local)
        cache_root = f"{rootpath}/cache/"
    base_path = f"{cache_root}get_sample_dataset/"
    pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
    if list_locals['adv']==True:
        remove_keys = ['list_arg', 'architecture', 'per_class']
    else:
        remove_keys = ['list_arg', 'architecture', 'num_iter', 'per_class']
    if 'transfered_attacks' in list_locals.keys():
        list_locals['transfered_attacks'] = "False"
    cache_path = (
        base_path
        + "_".join(sorted([f"{key}={str(list_locals[key])}" for key in list_locals.keys() if key not in remove_keys]))
        + ".cached"
        )
    logger.info(f"cache_path = {cache_path}")
    return cache_path

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
    attack_backend: str = AttackBackend.FOOLBOX,
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

    if adv:
        logger.info(f"Only successful adversaries ? {'yes' if succ_adv else 'no'}")
        logger.info(f"Which attack ? {attack_type}")
        logger.info(f"Which backend ? {attack_backend}")
    else:
        logger.info("This dataset will be non-adversarial !")

    if transfered_attacks:
        logger.info(f"Loading the architecture to generate adversaries with transferred attacks with {archi.epochs} epochs")
        archi = architecture = get_deep_model(
            num_epochs=archi.epochs,
            dataset=dataset,
            architecture=archi,
            train_noise=archi.train_noise,
        )
        if archi.epochs % 10 == 0:
            archi.epochs += 1
            list_locals = locals()
            logger.info(f"locals = {list_locals}")
            source_dataset_path = get_my_path(list_locals)
            if os.path.exists(source_dataset_path):
                source_dataset = torch.load(source_dataset_path)
            source_dataset_size = len(source_dataset)
            logger.info(f"Successfully loaded dataset of trsf attacks (len {source_dataset_size})")
            archi.epochs -= 1
            current_sample_id = 0
        else:
            source_dataset = (
            dataset.train_dataset if train else dataset.test_and_val_dataset
            )
            source_dataset_size = len(source_dataset)
            current_sample_id = offset
    else:
        source_dataset = (
            dataset.train_dataset if train else dataset.test_and_val_dataset
        )
        source_dataset_size = len(source_dataset)
        current_sample_id = offset

    final_dataset = list()

    if dataset.name in ["tinyimagenet"]:
        per_class_nb_samples = np.repeat(0, 200)
    elif dataset.name in ["cifar100"]:
        per_class_nb_samples = np.repeat(0,100)
    else:
        per_class_nb_samples = np.repeat(0, 10)

    #current_sample_id = offset

    dataset_done = False
    batch_size = 32

    while not dataset_done and current_sample_id < source_dataset_size:

        samples = None
        processed_samples = None
        y_pred = None

        while processed_samples is None and current_sample_id < source_dataset_size:

            if transfered_attacks:
                #logger.info(f"source dataset keys = {source_dataset[0]}")
                batch = source_dataset[
                    current_sample_id : current_sample_id + batch_size
                ]
                if isinstance(batch[0], DatasetLine):
                    #logger.info(f"line = {batch[0]}")
                    x = torch.cat([torch.unsqueeze(s.x, 0) for s in batch], 0).to(device)
                    y = np.array([s.y for s in batch])
                    logger.info(f"shape of x = {x.shape}")
                    if adv:
                        x_adv = torch.cat([torch.unsqueeze(s.x, 0) for s in batch], 0).to(device)
                        y_adv = np.array([s.y for s in batch])
                    else:
                        x_adv = x
                        y_adv = y
                #else:
                #    x = torch.cat([torch.unsqueeze(s[0], 0) for s in batch], 0).to(device)
                #    y = np.array([s[1] for s in batch])
                #    if adv:
                #        x_adv = 
                #        y_adv = 
                samples = (x, y)
                processed_samples = (x_adv, y_adv)

                #if adv:
                #    processed_samples = (
                #        source_dataset[current_sample_id].x_adv,#["x_adv"],
                #        source_dataset[current_sample_id].y#["y"],
                #    )
                #else:
                #    processed_samples = samples
            else:
                # Fetching a batch of samples and concatenating them
                batch = source_dataset[
                    current_sample_id : current_sample_id + batch_size
                ]
                if isinstance(batch[0], DatasetLine):
                    x = torch.cat([torch.unsqueeze(s.x, 0) for s in batch], 0).to(device)
                    y = np.array([s.y for s in batch])
                    logger.info(f"shape of x = {x.shape}")
                else:
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
            .cpu().detach()
            .numpy()
        )

        linf_norms = (
            torch.norm(
                (processed_samples[0].double() - samples[0].double()).flatten(1),
                p=float("inf"),
                dim=1,
            )
            .cpu().detach()
            .numpy()
        )

        # Update the counter per class
        for clazz in processed_samples[1]:
            per_class_nb_samples[clazz] += 1

        # Unbatching and return DatasetLine
        # TODO: see if we can avoid unbatching
        for i in range(len(processed_samples[1])):

            x = torch.unsqueeze(processed_samples[0][i], 0).double()
            x_origin = torch.unsqueeze(samples[0][i], 0).double()

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
                    x0=x_origin,
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
