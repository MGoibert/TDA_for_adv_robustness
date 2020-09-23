import random

import numpy as np
import torch
import pytest
from functools import reduce

from tda.models import get_deep_model, cifar_lenet
from tda.models.architectures import (
    mnist_mlp,
    Architecture,
    mnist_lenet,
    svhn_lenet,
    cifar_toy_resnet,
    cifar_resnet_1,
)
from tda.dataset.datasets import Dataset
from tda.tda_logging import get_logger

logger = get_logger("test_models")


def test_get_mnist_model():
    torch.manual_seed(37)
    random.seed(38)
    np.random.seed(39)

    source_dataset = Dataset("MNIST")
    _, val_acc, test_acc = get_deep_model(
        dataset=source_dataset,
        num_epochs=1,
        architecture=mnist_lenet,
        with_details=True,
        force_retrain=True,
    )
    print(val_acc)
    print(test_acc)


def test_get_svhn_model():
    source_dataset = Dataset("SVHN")
    get_deep_model(dataset=source_dataset, num_epochs=1, architecture=svhn_lenet)


def test_train_eval():
    archi: Architecture = mnist_mlp
    archi.set_train_mode()
    training_modes = [layer.func.training for layer in archi.layers]
    print(training_modes)
    assert all(training_modes)
    archi.set_eval_mode()
    eval_modes = [not layer.func.training for layer in archi.layers]
    print(eval_modes)
    assert all(eval_modes)


@pytest.mark.parametrize(
    "dataset, architecture",
    [
        ("CIFAR10", cifar_lenet),
        ("CIFAR10", cifar_toy_resnet),
        ("CIFAR10", cifar_resnet_1),
    ],
)
def test_shapes(dataset, architecture):
    source_dataset = Dataset(dataset)

    archi = get_deep_model(
        dataset=source_dataset,
        num_epochs=1,
        architecture=architecture,
        with_details=False,
        force_retrain=False,
    )

    x, y = source_dataset.train_dataset[0]

    inner_values = archi.forward(x, output="all_inner")
    shape = None
    mat_shapes_for_underopt = dict()

    for key in inner_values:
        print(f"Layer {key}: {archi.layers[key]}")
        new_shape = inner_values[key].shape

        if shape is not None and isinstance(archi.layers[key].func, torch.nn.Conv2d):
            outs = reduce(lambda x, y: x * y, list(new_shape), 1)
            ins = reduce(lambda x, y: x * y, list(shape), 1)
            print(f"Matrix shape is {ins} x {outs}")
            mat_shapes_for_underopt[key] = (outs, ins)

        shape = new_shape
        print(f"New shape is {new_shape}")
        print("--------")

    mat_shapes_for_underopt = [mat_shapes_for_underopt[k] for k in sorted(mat_shapes_for_underopt.keys())]
    print(mat_shapes_for_underopt)
