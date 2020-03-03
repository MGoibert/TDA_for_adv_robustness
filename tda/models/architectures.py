#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Callable, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tda.devices import device
from tda.tda_logging import get_logger
from tda.models.layers import Layer, ConvLayer, MaxPool2dLayer, DropOut, LinearLayer, SoftMaxLayer, BatchNorm2d, \
    ReluLayer, AvgPool2dLayer

torch.set_default_tensor_type(torch.DoubleTensor)
logger = get_logger("Architecture")

#################
# Architectures #
#################
logger = get_logger("Architecture")


class Architecture(nn.Module):

    def __init__(self,
                 layers: List[Layer],
                 preprocess: Callable=None,
                 layer_links: List[Tuple[int, int]] = None,
                 name: str = ""):
        """
        Instantiating architecture with a list of layers and edges.
        The graph on the layers should be a DAG (we won't check for cycles)
        """
        super().__init__()
        self.name = name
        self.layers = layers
        self.layer_links = layer_links or [(-1, 0)] + [(i, i + 1) for i in range(len(layers) - 1)]

        self.preprocess = preprocess

        self.layer_visit_order = Architecture.walk_through_dag(self.layer_links)
        self.parent_dict = Architecture.get_parent_dict(self.layer_links)

        for i, layer in enumerate(layers):
            layer_params = dict(layer.func.named_parameters())
            layer_name = layer.name or f"layer{i}"
            for name in layer_params:
                self.register_parameter(f"{layer_name}_{name}", layer_params[name])

        self.is_trained = False

        self.epochs = 0

    def __repr__(self):
        return f"{self.name}_{self.epochs}"

    def __str__(self):
        return f"{self.name}_{self.epochs}"

    def set_train_mode(self):
        for layer in self.layers:
            layer.func.train()

    def set_eval_mode(self):
        for layer in self.layers:
            layer.func.eval()

    def get_pre_softmax_idx(self):
        softmax_layer_idx = None
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, SoftMaxLayer):
                softmax_layer_idx = layer_idx
                break
        if softmax_layer_idx is None:
            raise RuntimeError(f"Didn't find any SoftMax in {self.layers}")

        return [link for link in self.layer_links if link[1] == softmax_layer_idx][0][0]

    @staticmethod
    def walk_through_dag(
            edges: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Helper function to build an ordered walkthrough in the DAG
        """

        all_nodes = set([edge[0] for edge in edges]).union(set([edge[1] for edge in edges]))

        # Step 1: find the roots and add them to the stack
        stack = [node for node in all_nodes if not any([node == edge[1] for edge in edges])]

        order = list()

        while len(stack) > 0:
            current_node = stack.pop()
            order.append(current_node)

            for child in [edge[1] for edge in edges if edge[0] == current_node]:
                all_parents = [edge[0] for edge in edges if edge[1] == child]
                if all([parent in order for parent in all_parents]):
                    stack.append(child)

        return order

    @staticmethod
    def get_parent_dict(
            edges: List[Tuple[int, int]]
    ):
        ret = dict()

        for node in [edge[1] for edge in edges]:
            parents = [edge[0] for edge in edges if edge[1] == node]
            ret[node] = sorted(parents)

        return ret

    def forward(self, x, store_for_graph=False, output="final"):
        # List to store intermediate results if needed
        if device.type == "cuda":
            x = x.to(device)
        if self.preprocess is not None:
            x = self.preprocess(x)

        outputs = {-1: x.double()}

        # Going through all layers
        for layer_idx in self.layer_visit_order:
            #logger.info(f"Layer nb {layer_idx}")
            if layer_idx != -1:
                layer = self.layers[layer_idx]
                input = {
                    parent_idx: outputs[parent_idx].double()
                    for parent_idx in self.parent_dict[layer_idx]
                }
                outputs[layer_idx] = layer.process(input, store_for_graph=store_for_graph)

        # Returning final result
        if output == "presoft":
            return outputs[self.layer_visit_order[-2]]
        elif output == "all_inner":
            return outputs
        elif output == "final":
            return outputs[self.layer_visit_order[-1]]
        else:
            raise RuntimeError(f"Unknown output type {output}")

    def get_graph_values(self, x) -> Dict:
        # Processing sample
        # logging.info(f"Shape of x is {x.shape}")
        self.forward(x, store_for_graph=True)
        # Getting matrix for each layer
        ret = dict()
        for layer_idx, layer in enumerate(self.layers):
            # (f"Processing layer {layer_idx}")
            if layer.graph_layer:
                m = layer.get_matrix()
                for parentidx in m:
                    ret[(parentidx, layer_idx)] = m[parentidx]
        return ret

    def get_nb_graph_layers(self) -> int:
        return sum([
            1 for layer in self.layers if layer.graph_layer
        ])


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
        SoftMaxLayer()
    ])

mnist_small_mlp = Architecture(
    name="small_mlp",
    preprocess=mnist_preprocess,
    layers=[
        LinearLayer(28 * 28, 200),
        LinearLayer(200, 50),
        LinearLayer(50, 10),
        SoftMaxLayer()
    ])

mnist_lenet = Architecture(
    name="mnist_lenet",
    preprocess=mnist_preprocess2,
    layers=[
        ConvLayer(1, 10, 5, activ=F.relu, bias=True, name="conv1"),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        ConvLayer(10, 20, 5, activ=F.relu, bias=True, name="conv2"),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        LinearLayer(320, 50, activ=F.relu, name="fc1"),
        DropOut(),
        LinearLayer(50, 10, name="fc2"),
        SoftMaxLayer()
    ])


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
        SoftMaxLayer()
    ])

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
        SoftMaxLayer()
    ])

svhn_resnet = Architecture(
    name="svhn_resnet",
    preprocess=svhn_preprocess,
    layers=[
        # 1st layer / no stack or block
        ConvLayer(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),

        #  Stack 1
        # Block a
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64),
        ReluLayer(),
        # Block b
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64),
        ReluLayer(),

        # Stack 2
        # Block a
        ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
        BatchNorm2d(channels=128, activ=F.relu),
        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=128),
        ReluLayer(),
        # Block b
        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=128, activ=F.relu),
        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=128),
        ReluLayer(),

        # Stack 3
        # Block a
        ConvLayer(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
        BatchNorm2d(channels=256, activ=F.relu),
        ConvLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=256),
        ReluLayer(),
        # Block b
        ConvLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=256, activ=F.relu),
        ConvLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=256),
        ReluLayer(),

        # Stack 4
        # Block a
        ConvLayer(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
        BatchNorm2d(channels=512, activ=F.relu),
        ConvLayer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=512),
        ReluLayer(),
        # Block b
        ConvLayer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=512, activ=F.relu),
        ConvLayer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=512),
        ReluLayer(),

        # End part
        AvgPool2dLayer(kernel_size=4),
        LinearLayer(512, 10),
        SoftMaxLayer(),

        # Layer to reduce dimension in residual blocks
        ConvLayer(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False),
        ConvLayer(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False),
        ConvLayer(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0, bias=False)
    ],
    layer_links=[(i-1,i) for i in range(45)]+[
        (1,6), (6,11), (16,21), (26,31), (36,41),
        (11,45), (45,16), (21,46), (46,26), (31,47), (47,36)
    ])

svhn_resnet_test = Architecture(
    name="svhn_resnet_test",
    preprocess=svhn_preprocess,
    layers=[
        # 1st layer / no stack or block
        ConvLayer(in_channels=3, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False),

        #  Stack 1
            # Block a
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(in_channels=64, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(in_channels=64, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64),
        ReluLayer(),

        # End part
        AvgPool2dLayer(kernel_size=32),
        LinearLayer(64,10),
        SoftMaxLayer(),

        ],
    layer_links=[(i-1,i) for i in range(10)]+[
        (1,6)
    ])

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
        SoftMaxLayer()
    ])

###############################
# Fashion MNIST Architectures #
###############################

fashion_mnist_lenet = Architecture(
    name="fashion_mnist_lenet",
    preprocess=mnist_preprocess2,
    layers=[
        ConvLayer(1, 10, 5, activ=F.relu, bias=True, name="conv1"),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        ConvLayer(10, 20, 5, activ=F.relu, bias=True, name="conv2"),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        LinearLayer(320, 50, activ=F.relu, name="fc1"),
        #DropOut(),
        LinearLayer(50, 10, name="fc2"),
        SoftMaxLayer()
    ])

known_architectures: List[Architecture] = [
    mnist_mlp,
    svhn_cnn_simple,
    svhn_lenet,
    svhn_resnet,
    mnist_lenet,
    mnist_small_mlp,
    svhn_resnet_test,
    cifar_lenet,
    fashion_mnist_lenet
]


def get_architecture(architecture_name: str) -> Architecture:
    for archi in known_architectures:
        if architecture_name == archi.name:
            return archi
