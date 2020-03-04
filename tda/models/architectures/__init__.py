from typing import List
from .architecture import Architecture

from .mnist_models import (
    mnist_lenet,
    mnist_mlp,
    mnist_preprocess_flatten,
    mnist_preprocess_cnn,
    mnist_preprocess_cnn_0_1,
    mnist_small_mlp,
)

from .svhn_models import (
    svhn_lenet,
    svhn_resnet,
    svhn_resnet_test,
    svhn_cnn_simple,
    svhn_preprocess,
)

from .fashion_mnist_models import fashion_mnist_mlp, fashion_mnist_lenet

from .cifar10_models import cifar_lenet

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
