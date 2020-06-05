import torch
import torch.nn as nn
import numpy as np
import random
from collections import OrderedDict

from tda.models import get_deep_model
from tda.models.datasets import Dataset
from tda.models.architectures import *  # mnist_mlp, svhn_cnn_simple, Architecture, mnist_lenet
from tda.threshold_underoptimized_edges import process_thresholds_underopt
from tda.graph_dataset import get_sample_dataset
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
    get_deep_model(dataset=source_dataset, num_epochs=2, architecture=svhn_cnn_simple)


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


def test_conv():
    cnn = Architecture(
        name="cnn",
        layers=[
            ConvLayer(1, 2, 2, bias=False, name="conv1"),  # output 6 * 28 * 28
            ConvLayer(2, 3, 2, bias=False, name="conv2"),  # output 6 * 28 * 28
            LinearLayer(3, 2, name="fc1"),
            SoftMaxLayer(),
        ],
    )
    model = cnn
    for p in model.parameters():
        logger.info(f"p = {p}")
    x = torch.round(10 * torch.randn([1, 1, 3, 3]))
    logger.info(f"x = {x}")
    out = cnn(x)
    logger.info(f"out = {out}")
    logger.info(f"{np.round(cnn.get_graph_values(x)[(0,1)].todense(),2)}")


def test_new_threshold():
    architecture = get_architecture(mnist_lenet.name)
    dataset = Dataset.get_or_create(name="MNIST")

    architecture = get_deep_model(
        num_epochs=53, dataset=dataset, architecture=architecture, train_noise=0.0
    )

    thresholds = process_thresholds_underopt(
        raw_thresholds="0.1_0.1_0.1_0.1_0.1", architecture=architecture,
    )


def test_cw_l2norm():
    dataset = Dataset.get_or_create(name="MNIST")
    architecture = get_architecture(mnist_lenet.name)
    architecture = get_deep_model(
        num_epochs=50, dataset=dataset, architecture=architecture, train_noise=False,
    )
    data = get_sample_dataset(
        adv=True,
        noise=0.0,
        dataset=dataset,
        train=False,
        succ_adv=True,
        archi=architecture,
        attack_type="CW",
        epsilon=1,
        num_iter=1000,
        dataset_size=50,
        offset=0,
        compute_graph=False,
    )
    list_norm = list()
    for d in data:
        list_norm.append(d.l2_norm)

    def summary(my_list):
        s = dict()
        s["min"] = np.min(my_list)
        for q in [0.05, 0.1, 0.25, 0.4, 0.5]:
            s[str(q)] = np.quantile(my_list, q)
        s["mean"] = np.mean(my_list)
        for q in [0.6, 0.75, 0.9, 0.95]:
            s[str(q)] = np.quantile(my_list, q)
        s["max"] = np.max(my_list)
        return s

    print(f"Summary l2 norm = {summary(list_norm)}")


if __name__ == "__main__":
    test_cw_l2norm()
