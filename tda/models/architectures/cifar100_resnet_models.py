from tda.models import Architecture
from tda.models.layers import ConvLayer, LinearLayer, Layer
import torch.nn.functional as F
import tempfile
import requests
import torch
from typing import NamedTuple, List, Tuple, Dict
from tda.models.layers import BatchNorm2d, ReluLayer, AdaptativeAvgPool2dLayer


def conv3x3_layer(in_planes, out_planes, stride=1, name=None):
    """3x3 convolution with padding"""
    return ConvLayer(
        in_channels=in_planes,
        out_channels=out_planes,
        stride=stride,
        kernel_size=3,
        bias=False,
        padding=1,
        name=name,
    )


def conv1x1_layer(in_planes, out_planes, stride=1, name=None):
    """1x1 convolution"""
    return ConvLayer(
        in_channels=in_planes,
        out_channels=out_planes,
        stride=stride,
        kernel_size=1,
        bias=False,
        name=name,
    )


def _basic_block(inplanes, planes, stride=1, downsample=None, name=None):

    conv1 = conv3x3_layer(inplanes, planes, stride, name=f"{name}-C1")
    bn1 = BatchNorm2d(planes, activ=F.relu)

    conv2 = conv3x3_layer(planes, planes, name=f"{name}-C2")
    bn2 = BatchNorm2d(planes, activ=None)
    relu = ReluLayer()

    layers = [conv1, bn1, conv2, bn2]

    edges = [
        (-1, 0),  # x -> conv1
        (0, 1),  # conv1 -> bn1
        (1, 2),  # bn1 -> conv2
        (2, 3),  # conv2 -> bn2
    ]

    if downsample is None:
        edges.append((3, 4))
        edges.append((-1, 4))  # skip connexion x -> relu
    else:
        input_idx = -1
        for layer in downsample:
            layers.append(layer)
            layer_idx = len(layers) - 1
            edges.append((input_idx, layer_idx))
            input_idx = layer_idx
        edges.append((layer_idx, layer_idx + 1))
        edges.append((3, layer_idx + 1))

    layers.append(relu)

    return layers, edges


def _basic_layer(inplanes, planes, blocks, stride=1, name=None):
    downsample = None
    expansion = 1
    if stride != 1 or inplanes != planes * expansion:
        downsample = [
            conv1x1_layer(inplanes, planes * expansion, stride, name=f"{name}-DS"),
            BatchNorm2d(planes * expansion),
        ]

    layers = list()
    layers.append(_basic_block(inplanes, planes, stride, downsample, name=f"{name}-B0"))
    inplanes = planes * expansion
    for i in range(1, blocks):
        layers.append(_basic_block(inplanes, planes, name=f"{name}-B{i}"))

    return layers, inplanes


def _cifar_resnet(layer_sizes):

    conv1 = conv3x3_layer(3, 16, name="IntroConv")
    bn1 = BatchNorm2d(16, activ=F.relu)

    layers = [conv1, bn1]

    edges = [(-1, 0), (0, 1)]

    inplanes = 16
    layer1, inplanes = _basic_layer(inplanes, 16, layer_sizes[0], name="L1")
    layer2, inplanes = _basic_layer(inplanes, 32, layer_sizes[1], stride=2, name="L2")
    layer3, inplanes = _basic_layer(inplanes, 64, layer_sizes[2], stride=2, name="L3")

    for layer_block in [layer1, layer2, layer3]:
        for block_layers, block_edges in layer_block:
            start_idx = max([v for u, v in edges])
            mapped_edges = [
                (u + start_idx + 1, v + start_idx + 1) for u, v in block_edges
            ]
            layers.extend(block_layers)
            edges.extend(mapped_edges)

    avgpool = AdaptativeAvgPool2dLayer(output_size=(1, 1))
    fc = LinearLayer(64, 100)

    start_idx = max([v for u, v in edges])
    layers.extend([avgpool, fc])

    last_edges = [(start_idx, start_idx + 1), (start_idx + 1, start_idx + 2)]

    edges.extend(last_edges)

    return layers, edges


def cifar100_preprocess(x):
    return x.reshape(-1, 3, 32, 32)


def _architecture(model_name: str) -> Architecture:

    layer_sizes = dict(
        zip(
            ["resnet20", "resnet32", "resnet44", "resnet56"],
            [[3] * 3, [5] * 3, [7] * 3, [9] * 3],
        )
    )[model_name]

    layers, edges = _cifar_resnet(layer_sizes=layer_sizes)
    archi = Architecture(
        layers=layers,
        layer_links=edges,
        name="resnet32",
        preprocess=cifar100_preprocess,
    )
    archi.epochs = 42
    return archi


def _state_dict(model_name: str) -> Dict:

    urls = {
        "resnet20": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt",
        "resnet32": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt",
        "resnet44": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt",
        "resnet56": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt",
    }

    with tempfile.NamedTemporaryFile() as f:
        r = requests.get(urls.get(model_name))
        f.write(r.content)
        sd = torch.load(f.name)
    return sd


def _update_archi_with_weights(archi: Architecture, model_name: str):
    state_dict = _state_dict(model_name)
    foreign_keys = [
        k for k in state_dict.keys() if "running" not in k and "batches" not in k
    ]
    keymap = dict(zip(foreign_keys, archi.state_dict().keys()))
    mapped_sd = {keymap[k]: v for k, v in state_dict.items() if k in foreign_keys}
    archi.load_state_dict(mapped_sd)


def cifar100_architecture_with_weights(model_name: str) -> Architecture:
    archi = _architecture(model_name)
    _update_archi_with_weights(archi, model_name)
    return archi


if __name__ == "__main__":
    for model_name in ["resnet20", "resnet32", "resnet44", "resnet56"]:
        print(f"Trying {model_name}...")
        model = cifar100_architecture_with_weights(model_name)
