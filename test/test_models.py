import torch
import numpy as np
import random

from tda.models import get_deep_model
from tda.models.datasets import Dataset
from tda.models.architectures import mnist_mlp, svhn_cnn_simple, Architecture


def test_get_mnist_model():
    torch.manual_seed(37)
    random.seed(38)
    np.random.seed(39)

    source_dataset = Dataset("MNIST")
    _, val_acc, test_acc = get_deep_model(
        dataset=source_dataset.Dataset_,
        num_epochs=1,
        architecture=mnist_mlp,
        with_details=True,
        force_retrain=True
    )
    print(val_acc)
    print(test_acc)


def test_get_svhn_model():
    source_dataset = Dataset("SVHN")
    get_deep_model(
        dataset=source_dataset.Dataset_,
        num_epochs=2,
        architecture=svhn_cnn_simple
    )


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

