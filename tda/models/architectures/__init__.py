from typing import List
from .architecture import Architecture
from .definitions import (
    mnist_lenet,
    mnist_mlp,
    svhn_lenet,
    svhn_cnn_simple,
    svhn_resnet,
    cifar_lenet,
    fashion_mnist_lenet,
    svhn_preprocess,
    mnist_preprocess,
    mnist_preprocess2,
    fashion_mnist_mlp,
    mnist_small_mlp,
    svhn_resnet_test,
)

known_architectures: List[Architecture] = [
    mnist_mlp,
    svhn_cnn_simple,
    svhn_lenet,
    svhn_resnet,
    mnist_lenet,
    mnist_small_mlp,
    svhn_resnet_test,
    cifar_lenet,
    fashion_mnist_lenet,
    fashion_mnist_mlp,
]


def get_architecture(architecture_name: str) -> Architecture:
    for archi in known_architectures:
        if architecture_name == archi.name:
            return archi


# Hack to deserialize models
# (old serialized models are importing their Layers from the
#  wrong place)
from tda.models.layers import (
    Layer,
    ConvLayer,
    LinearLayer,
    SoftMaxLayer,
    ReluLayer,
    DropOut,
    BatchNorm2d,
    MaxPool2dLayer,
    AvgPool2dLayer,
)
