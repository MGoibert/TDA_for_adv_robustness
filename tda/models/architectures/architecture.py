#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Callable, Tuple, Dict

import torch
import torch.nn as nn
from art.classifiers import PyTorchClassifier

from tda.devices import device
from tda.models.layers import (
    Layer,
    SoftMaxLayer,
)
from tda.rootpath import rootpath
from tda.tda_logging import get_logger

torch.set_default_tensor_type(torch.DoubleTensor)
logger = get_logger("Architecture")

#################
# Architectures #
#################
logger = get_logger("Architecture")


class Architecture(nn.Module):
    def __init__(
        self,
        layers: List[Layer],
        preprocess: Callable = None,
        layer_links: List[Tuple[int, int]] = None,
        name: str = "",
    ):
        """
        Instantiating architecture with a list of layers and edges.
        The graph on the layers should be a DAG (we won't check for cycles)
        """
        super().__init__()
        self.name = name
        self.layers = layers
        self.layer_links = layer_links or [(-1, 0)] + [
            (i, i + 1) for i in range(len(layers) - 1)
        ]

        self.preprocess = preprocess

        self.layer_visit_order = Architecture.walk_through_dag(self.layer_links)
        self.parent_dict = Architecture.get_parent_dict(self.layer_links)

        for i, layer in enumerate(layers):
            layer_params = dict(layer.func.named_parameters())
            layer_name = layer.name or f"layer{i}"
            for name in layer_params:
                self.register_parameter(f"{layer_name}_{name}", layer_params[name])

        self.is_trained = False

        self.art_classifier = None

        self.epochs = 0
        self.train_noise = 0.0
        self.tot_prune_percentile = 0.0

        self.to(device)

    def build_matrices(self):
        for layer in self.layers:
            if layer.graph_layer:
                layer.build_matrix()

    def get_layer_matrices(self):
        return {
            i: layer.matrix for i, layer in enumerate(self.layers) if layer.graph_layer
        }

    def get_model_savepath(self, initial: bool = False):
        return f"{rootpath}/trained_models/{self.get_full_name(initial=initial)}.model"

    def get_initial_model(self) -> "Architecture":
        """
        Return the initial version of the model if available
        """
        return torch.load(self.get_model_savepath(initial=True), map_location=device)

    def __repr__(self):
        return self.get_full_name()

    def __str__(self):
        return self.get_full_name()

    def get_full_name(self, initial: bool = False):
        postfix = "_init" if initial else ""

        nprefix = ""
        if hasattr(self, "train_noise") and self.train_noise > 0.0:
            nprefix += f"_tn_{self.train_noise}"

        if hasattr(self, "tot_prune_percentile") and self.tot_prune_percentile > 0.0:
            nprefix += f"_p_{1.0 - self.tot_prune_percentile}"

        return f"{self.name}_e_{self.epochs}{nprefix}{postfix}"

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

    def get_art_classifier(self):
        if not hasattr(self, "art_classifier") or self.art_classifier is None:
            if "bandw" in self.name:
                input_shape = (1, 32, 32)
            elif "svhn" in self.name or "cifar" in self.name:
                input_shape = (3, 32, 32)
            elif "mnist" in self.name:
                input_shape = (1, 28, 28)

            self.art_classifier = PyTorchClassifier(
                model=self,
                clip_values=(0, 1),
                loss=torch.nn.CrossEntropyLoss(),
                optimizer=None,
                input_shape=input_shape,
                nb_classes=10,
            )
        return self.art_classifier

    @staticmethod
    def walk_through_dag(edges: List[Tuple[int, int]]) -> List[int]:
        """
        Helper function to build an ordered walkthrough in the DAG
        """

        all_nodes = set([edge[0] for edge in edges]).union(
            set([edge[1] for edge in edges])
        )

        # Step 1: find the roots and add them to the stack
        stack = [
            node for node in all_nodes if not any([node == edge[1] for edge in edges])
        ]

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
    def get_parent_dict(edges: List[Tuple[int, int]]):
        ret = dict()

        for node in [edge[1] for edge in edges]:
            parents = [edge[0] for edge in edges if edge[1] == node]
            ret[node] = sorted(parents)

        return ret

    def forward(self, x, store_for_graph=False, output="final"):
        # List to store intermediate results if needed

        if not torch.is_tensor(x):
            x = torch.Tensor(x)

        x = x.to(device)
        if self.preprocess is not None:
            x = self.preprocess(x)

        outputs = {-1: x.double()}

        # Going through all layers
        for layer_idx in self.layer_visit_order:
            # logger.info(f"Layer nb {layer_idx}")
            if layer_idx != -1:
                layer = self.layers[layer_idx]
                input = {
                    parent_idx: outputs[parent_idx].double()
                    for parent_idx in self.parent_dict[layer_idx]
                }
                outputs[layer_idx] = layer.process(
                    input, store_for_graph=store_for_graph
                )

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
        return sum([1 for layer in self.layers if layer.graph_layer])
