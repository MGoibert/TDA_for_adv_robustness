import torch.nn.functional as F

from tda.models.layers import (
    ConvLayer,
    MaxPool2dLayer,
    LinearLayer,
    SoftMaxLayer,
    BatchNorm2d,
    ReluLayer,
    AvgPool2dLayer,
)
from .architecture import Architecture


######################
# SVHN Architectures #
######################


def svhn_preprocess(x):
    return x.reshape(-1, 3, 32, 32)

def svhn_preprocess_bandw(x):
    return x.view(-1, 1, 32, 32)


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

svhn_lenet_bandw = Architecture(
    name="svhn_lenet_bandw",
    preprocess=svhn_preprocess_bandw,
    layers=[
        ConvLayer(1, 6, 5, activ=F.relu),  # output 6 * 28 * 28
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
