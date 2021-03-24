import torch.nn.functional as F

from tda.models.layers import (
    ConvLayer,
    MaxPool2dLayer,
    DropOut,
    LinearLayer,
    SoftMaxLayer,
)
from .architecture import Architecture


toy_mlp = Architecture(
    name="toy_mlp",
    #preprocess=mnist_preprocess,
    layers=[
        LinearLayer(2, 4, activ=F.relu, name="fc1"),
        LinearLayer(4, 2, name="fc2"),
        SoftMaxLayer(),
    ],
)

toy_mlp2 = Architecture(
    name="toy_mlp2",
    #preprocess=mnist_preprocess,
    layers=[
        LinearLayer(2, 8, activ=F.relu, name="fc1"),
        LinearLayer(8, 2, name="fc2"),
        SoftMaxLayer(),
    ],
)

toy_mlp3 = Architecture(
    name="toy_mlp3",
    #preprocess=mnist_preprocess,
    layers=[
        LinearLayer(2, 20, activ=F.relu, name="fc1"),
        LinearLayer(20, 2, name="fc2"),
        SoftMaxLayer(),
    ],
)

toy_mlp4 = Architecture(
    name="toy_mlp4",
    #preprocess=mnist_preprocess,
    layers=[
        LinearLayer(2, 100, activ=F.relu, name="fc1"),
        LinearLayer(100, 2, name="fc2"),
        SoftMaxLayer(),
    ],
)