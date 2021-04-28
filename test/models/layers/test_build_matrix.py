import torch
import pytest
import numpy as np
from tda.models.layers import (
    LinearLayer,
    ConvLayer,
    AvgPool2dLayer,
    AdaptativeAvgPool2dLayer, MaxPool2dLayer,
)


@pytest.mark.parametrize(
    "layer,input_shape",
    [
        (LinearLayer(3, 2, activ=None, bias=False), (3, 1)),
        (
            ConvLayer(in_channels=2, out_channels=3, kernel_size=2, input_shape=(7, 7)),
            (1, 2, 7, 7),
        ),
        (
            ConvLayer(
                in_channels=1,
                out_channels=1,
                kernel_size=2,
                stride=2,
                input_shape=(4, 4),
            ),
            (1, 1, 4, 4),
        ),
        (
            ConvLayer(
                in_channels=2,
                out_channels=2,
                kernel_size=2,
                grouped_channels=True,
                input_shape=(7, 7),
            ),
            (1, 2, 7, 7),
        ),
        (
            ConvLayer(
                in_channels=1,
                out_channels=1,
                kernel_size=(2, 1),
                stride=2,
                padding=0,
                input_shape=(5, 5),
            ),
            (1, 1, 5, 5),
        ),
        (
            ConvLayer(
                in_channels=1,
                out_channels=1,
                kernel_size=(2, 1),
                stride=2,
                padding=1,
                input_shape=(5, 5),
            ),
            (1, 1, 5, 5),
        ),
        (
            ConvLayer(
                in_channels=2,
                out_channels=3,
                kernel_size=(3, 2),
                stride=2,
                padding=2,
                input_shape=(11, 13),
            ),
            (1, 2, 11, 13),
        ),
        (AvgPool2dLayer(kernel_size=3, ceil_mode=False), (1, 2, 7, 7)),
        (AvgPool2dLayer(kernel_size=3, ceil_mode=True), (1, 2, 7, 7)),
        (AdaptativeAvgPool2dLayer(output_size=(2, 2)), (1, 2, 7, 7)),
        (AdaptativeAvgPool2dLayer(output_size=(1, 1)), (1, 2, 7, 7)),
        (MaxPool2dLayer(kernel_size=2, stride=2), (1, 2, 4, 5))
    ],
)
def test_build_matrix(layer, input_shape):
    sample = torch.randn(input_shape)

    input = {0: sample}

    out = layer.process(input, store_for_graph=True)
    print(f"Out shape is {out.shape}")
    out = out.detach().cpu().numpy().reshape(-1)

    layer.build_matrix()
    out_mat = layer.get_matrix()[0].todense().sum(axis=1).reshape(-1)
    print(f"Out_mat_shape is {out_mat.shape}")

    difference = np.linalg.norm(out_mat - out)
    print(f"Difference is {difference}")

    assert np.isclose(difference, 0)
