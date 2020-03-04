from .architecture import Architecture
from .definitions import (
    get_architecture,
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
    fashion_mnist_mlp
)

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
