import torch.nn.functional as F

from tda.models.layers import (
    ConvLayer,
    MaxPool2dLayer,
    DropOut,
    LinearLayer,
    SoftMaxLayer,
)
from .architecture import Architecture

def viz_preprocess(x):
    return x.view(-1, 1, 3, 3)
def viz_preprocess2(x):
    return x.view(-1, 9)

toy_viz = Architecture(
    name="toy_viz",
    preprocess=viz_preprocess,
    layers=[
        ConvLayer(1, 2, 2, stride=1, padding=0, activ=F.relu, bias=True, name="conv1"),
        LinearLayer(8, 4, activ=F.relu, bias=True, name="fc1"),
        LinearLayer(4, 3, activ=F.relu, bias=True, name="fc2"),
        LinearLayer(3, 2, name="fc3"),
        SoftMaxLayer(),
    ],
)

toy_viz2 = Architecture(
    name="toy_viz2",
    preprocess=viz_preprocess,
    layers=[
        ConvLayer(1, 1, 2, stride=1, padding=0, activ=F.relu, bias=True, name="conv1"),
        LinearLayer(4, 3, activ=F.relu, name="fc1"),
        LinearLayer(3, 2, name="fc2"),
        SoftMaxLayer(),
    ],
)

toy_viz3 = Architecture(
    name="toy_viz3",
    preprocess=viz_preprocess,
    layers=[
        ConvLayer(1, 1, 2, stride=1, padding=0, activ=F.relu, bias=False, name="conv1"),
        LinearLayer(4, 3, activ=F.relu, name="fc1"),
        LinearLayer(3, 2, name="fc2"),
        SoftMaxLayer(),
    ],
)


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
