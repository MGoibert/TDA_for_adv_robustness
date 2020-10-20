from typing import List
from tda.devices import device
from .architecture import Architecture

from .mnist_models import (
    mnist_lenet,
    mnist_mlp,
    mnist_preprocess,
    mnist_preprocess2,
    mnist_preprocess_cnn_05,
    mnist_small_mlp,
    mnist_mlp_relu,
)

from .svhn_models import (
    svhn_lenet,
    svhn_lenet_bandw,
    svhn_lenet_bandw2,
    svhn_resnet,
    svhn_resnet_test,
    svhn_cnn_simple,
    svhn_preprocess,
)

from .fashion_mnist_models import (
    fashion_mnist_mlp,
    fashion_mnist_lenet,
    fashion_mnist_lenet_05,
)

from .cifar10_models import cifar_lenet, cifar_toy_resnet
from .cifar10_resnet_model_1 import cifar_resnet_1
from .svhn_resnet_model_1 import svhn_resnet_1

known_architectures: List[Architecture] = [
    mnist_mlp,
    svhn_cnn_simple,
    svhn_lenet,
    svhn_lenet_bandw,
    svhn_lenet_bandw2,
    svhn_resnet,
    mnist_lenet,
    mnist_small_mlp,
    mnist_mlp_relu,
    svhn_resnet_test,
    cifar_lenet,
    fashion_mnist_lenet,
    fashion_mnist_mlp,
    fashion_mnist_lenet_05,
    cifar_toy_resnet,
    cifar_resnet_1,
    svhn_resnet_1
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
