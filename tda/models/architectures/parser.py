"""
Synopsis: Utilities to utomatically infer architecture of a pytorch model
Author: Elvis Dohmatob <e.dohmatob@criteo.com>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import tda.models.layers as tda_layers
from tda.models import Architecture
from tda.tda_logging import get_logger

logger = get_logger("Arch-parser")


def _unroll_children(model):
    for child in model.children():
        if len(list(child.children())):
            for grandchild in _unroll_children(child):
                yield grandchild
        else:
            yield child


def _get_layers(model):
    layers_ = []
    layers = []
    for layer in _unroll_children(model):
        layer_name = layer.__class__.__name__
        layers_.append((layer, layer_name))

    for l, (layer, layer_name) in enumerate(layers_):
        kwargs = {}
        if layer_name in ["MaxPool2d", "BatchNorm2d", "Conv2d", "Linear"]:
            # infer activation function by checking next layer
            if l < len(layers_) - 1:
                next_layer_name = layers_[l + 1][1]
                if next_layer_name == "ReLU":
                    kwargs["activ"] = F.relu
                elif next_layer_name == "Sigmoid":
                    kwargs["activ"] = F.sigmoid
                else:
                    pass
                    # XXX implement support for other activaiton functions
        if layer_name == "MaxPool2d":
            tda_layer = tda_layers.MaxPool2dLayer(
                kernel_size=layer.kernel_size, **kwargs)
        elif layer_name == "Conv2d":
            tda_layer = tda_layers.ConvLayer(in_channels=layer.in_channels,
                                             out_channels=layer.out_channels,
                                             kernel_size=layer.kernel_size,
                                             stride=layer.stride[0],
                                             padding=layer.padding[0],
                                             **kwargs)
        elif layer_name == "BatchNorm2d":
            tda_layer = tda_layers.BatchNorm2d(layer.num_features,
                                               **kwargs)
        elif layer_name == "Linear":
            tda_layer = tda_layers.LinearLayer(in_width=layer.in_features,
                                               out_width=layer.out_features,
                                               **kwargs)
        elif layer_name == "ReLU":
            continue
        else:
            raise NotImplementedError(layer_name)
        layers.append(tda_layer)
    return layers


def model_to_architecture(model: (nn.Module, Architecture),
                          name: str=None, device=None,
                          x: torch.Tensor=None,
                          **kwargs) -> Architecture:
    """
    Tries to parse an arbitrary pytorch model into a format
    compatible with the tda pipelines.

    Notes
    -----
    For the heuristic to work, input model should have been constructed
    using torch.nn modules and not functions, whenever possible.
    For example, torch.nn.ReLU should've been used rather than F.relu; etc.
    """
    if isinstance(model, Architecture):
        model_arch = model
        for key, val in kwargs.items():
            setattr(model_arch, key, val)
        try:
            model_arch = torch.load(model_arch.get_model_savepath(),
                                    map_location=device)
            logger.info(f"Loaded successfully model from %s" % (
                model_arch.get_model_savepath()))
        except FileNotFoundError:
            logger.info(f"Unable to find model in %s" % (
                model_arch.get_model_savepath()))
            raise
    else:
        layers = _get_layers(model)
        model_arch = Architecture(name=name, layers=layers,
                                  preprocess=lambda x: x.unsqueeze(0))
    model_arch.set_eval_mode()
    model_arch.is_trained = True

    if x is not None:
        model_arch.forward(x);
        model_arch.build_matrices();
    return model_arch
