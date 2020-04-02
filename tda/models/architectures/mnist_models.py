import torch.nn.functional as F

from tda.models.layers import (
    ConvLayer,
    MaxPool2dLayer,
    DropOut,
    LinearLayer,
    SoftMaxLayer,
)
from .architecture import Architecture


def mnist_preprocess(x):
    return x.view(-1, 28 * 28)


def mnist_preprocess2(x):
    return x.view(-1, 1, 28, 28)


def mnist_preprocess_cnn_05(x):
    return x.view(-1, 1, 28, 28) - 0.5


mnist_mlp = Architecture(
    name="simple_fcn_mnist",
    preprocess=mnist_preprocess,
    layers=[
        LinearLayer(28 * 28, 500),
        LinearLayer(500, 256),
        LinearLayer(256, 10),
        SoftMaxLayer(),
    ],
)

mnist_mlp_relu = Architecture(
    name="simple_fcn_mnist_relu",
    preprocess=mnist_preprocess,
    layers=[
        LinearLayer(28 * 28, 500, activ="relu"),
        LinearLayer(500, 256, activ="relu"),
        LinearLayer(256, 10),
        SoftMaxLayer(),
    ],
)

mnist_small_mlp = Architecture(
    name="small_mlp",
    preprocess=mnist_preprocess,
    layers=[
        LinearLayer(28 * 28, 200),
        LinearLayer(200, 50),
        LinearLayer(50, 10),
        SoftMaxLayer(),
    ],
)

mnist_lenet = Architecture(
    name="mnist_lenet",
    preprocess=mnist_preprocess2,
    layers=[
        ConvLayer(
            1, 10, 5, activ=F.relu, bias=True, name="conv1"
        ),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        ConvLayer(
            10, 20, 5, activ=F.relu, bias=True, name="conv2"
        ),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        LinearLayer(320, 50, activ=F.relu, name="fc1"),
        DropOut(),
        LinearLayer(50, 10, name="fc2"),
        SoftMaxLayer(),
    ],
)
