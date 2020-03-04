import torch.nn.functional as F

from tda.models.layers import (
    ConvLayer,
    MaxPool2dLayer,
    LinearLayer,
    SoftMaxLayer,
)
from .architecture import Architecture
from .mnist_models import (
    mnist_preprocess_cnn,
    mnist_preprocess_flatten,
    mnist_preprocess_cnn_0_1,
)

fashion_mnist_lenet = Architecture(
    name="fashion_mnist_lenet",
    preprocess=mnist_preprocess_cnn,
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
        # DropOut(),
        LinearLayer(50, 10, name="fc2"),
        SoftMaxLayer(),
    ],
)

# Same model as above but with the pixels between 0 and 1
# (instead of -0.5 / 0.5)
fashion_mnist_lenet = Architecture(
    name="fashion_mnist_lenet_01",
    preprocess=mnist_preprocess_cnn_0_1,
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
        # DropOut(),
        LinearLayer(50, 10, name="fc2"),
        SoftMaxLayer(),
    ],
)

fashion_mnist_mlp = Architecture(
    name="fashion_mnist_mlp",
    preprocess=mnist_preprocess_flatten,
    layers=[
        LinearLayer(28 * 28, 512, activ=F.relu, name="fc0"),
        LinearLayer(512, 256, activ=F.relu, name="fc1"),
        LinearLayer(256, 128, activ=F.relu, name="fc2"),
        LinearLayer(128, 64, activ=F.relu, name="fc3"),
        LinearLayer(64, 10, activ=F.relu, name="fc4"),
        SoftMaxLayer(),
    ],
)
