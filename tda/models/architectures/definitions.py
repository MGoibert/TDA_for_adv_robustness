from .architecture import Architecture
import torch.nn.functional as F
from typing import List
from tda.models.layers import (
    ConvLayer,
    MaxPool2dLayer,
    DropOut,
    LinearLayer,
    SoftMaxLayer,
    BatchNorm2d,
    ReluLayer,
    AvgPool2dLayer,
)

#######################
# MNIST Architectures #
#######################


def mnist_preprocess(x):
    return x.view(-1, 28 * 28)


def mnist_preprocess2(x):
    return x.view(-1, 1, 28, 28)


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


######################
# SVHN Architectures #
######################


def svhn_preprocess(x):
    return x.reshape(-1, 3, 32, 32)


svhn_cnn_simple = Architecture(
    name="simple_cnn_svhn",
    preprocess=svhn_preprocess,
    layers=[
        ConvLayer(3, 8, 5),  # output 8 * 28 * 28
        ConvLayer(8, 3, 5),  # output 3 * 24 * 24
        LinearLayer(3 * 24 * 24, 500),
        LinearLayer(500, 256),
        LinearLayer(256, 10),
        SoftMaxLayer(),
    ],
)

svhn_lenet = Architecture(
    name="svhn_lenet",
    preprocess=svhn_preprocess,
    layers=[
        ConvLayer(3, 6, 5, activ=F.relu),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        ConvLayer(6, 16, 5, activ=F.relu),
        MaxPool2dLayer(2),  # output 16 * 5 * 5
        LinearLayer(16 * 5 * 5, 120, activ=F.relu),
        LinearLayer(120, 84, activ=F.relu),
        LinearLayer(84, 10),
        SoftMaxLayer(),
    ],
)

svhn_resnet = Architecture(
    name="svhn_resnet",
    preprocess=svhn_preprocess,
    layers=[
        # 1st layer / no stack or block
        ConvLayer(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        #  Stack 1
        # Block a
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=64),
        ReluLayer(),
        # Block b
        ConvLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=64),
        ReluLayer(),
        # Stack 2
        # Block a
        ConvLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=128, activ=F.relu),
        ConvLayer(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=128),
        ReluLayer(),
        # Block b
        ConvLayer(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=128, activ=F.relu),
        ConvLayer(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=128),
        ReluLayer(),
        # Stack 3
        # Block a
        ConvLayer(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=256, activ=F.relu),
        ConvLayer(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=256),
        ReluLayer(),
        # Block b
        ConvLayer(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=256, activ=F.relu),
        ConvLayer(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=256),
        ReluLayer(),
        # Stack 4
        # Block a
        ConvLayer(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=512, activ=F.relu),
        ConvLayer(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=512),
        ReluLayer(),
        # Block b
        ConvLayer(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=512, activ=F.relu),
        ConvLayer(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=512),
        ReluLayer(),
        # End part
        AvgPool2dLayer(kernel_size=4),
        LinearLayer(512, 10),
        SoftMaxLayer(),
        # Layer to reduce dimension in residual blocks
        ConvLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=False,
        ),
        ConvLayer(
            in_channels=128,
            out_channels=256,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=False,
        ),
        ConvLayer(
            in_channels=256,
            out_channels=512,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=False,
        ),
    ],
    layer_links=[(i - 1, i) for i in range(45)]
    + [
        (1, 6),
        (6, 11),
        (16, 21),
        (26, 31),
        (36, 41),
        (11, 45),
        (45, 16),
        (21, 46),
        (46, 26),
        (31, 47),
        (47, 36),
    ],
)

svhn_resnet_test = Architecture(
    name="svhn_resnet_test",
    preprocess=svhn_preprocess,
    layers=[
        # 1st layer / no stack or block
        ConvLayer(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        #  Stack 1
        # Block a
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(channels=64),
        ReluLayer(),
        # End part
        AvgPool2dLayer(kernel_size=32),
        LinearLayer(64, 10),
        SoftMaxLayer(),
    ],
    layer_links=[(i - 1, i) for i in range(10)] + [(1, 6)],
)

#########################
# CIFAR10 Architectures #
#########################

cifar_lenet = Architecture(
    name="cifar_lenet",
    preprocess=svhn_preprocess,
    layers=[
        ConvLayer(3, 6, 5, activ=F.relu),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        ConvLayer(6, 16, 5, activ=F.relu),
        MaxPool2dLayer(2),  # output 16 * 5 * 5
        LinearLayer(16 * 5 * 5, 120, activ=F.relu),
        LinearLayer(120, 84, activ=F.relu),
        LinearLayer(84, 10),
        SoftMaxLayer(),
    ],
)

###############################
# Fashion MNIST Architectures #
###############################

fashion_mnist_lenet = Architecture(
    name="fashion_mnist_lenet",
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
        # DropOut(),
        LinearLayer(50, 10, name="fc2"),
        SoftMaxLayer(),
    ],
)

fashion_mnist_mlp = Architecture(
    name="fashion_mnist_mlp",
    preprocess=mnist_preprocess,
    layers=[
        LinearLayer(28 * 28, 512, activ=F.relu, name="fc0"),
        LinearLayer(512, 256, activ=F.relu, name="fc1"),
        LinearLayer(256, 128, activ=F.relu, name="fc2"),
        LinearLayer(128, 64, activ=F.relu, name="fc3"),
        LinearLayer(64, 10, activ=F.relu, name="fc4"),
        SoftMaxLayer()
    ])
