import torch
import pytest
import numpy as np
from tda.models.layers import LinearLayer, ConvLayer, AvgPool2dLayer, AdaptativeAvgPool2dLayer


@pytest.mark.parametrize("layer,input_shape", [
    (LinearLayer(3, 2, activ=None, bias=False), (3, 1)),
    (ConvLayer(in_channels=2, out_channels=3, kernel_size=2, input_shape=(7, 7)), (1, 2, 7, 7)),
    (ConvLayer(in_channels=2, out_channels=3, kernel_size=2, stride=2, input_shape=(7, 7)), (1, 2, 7, 7)),
    (ConvLayer(in_channels=2, out_channels=2, kernel_size=2, grouped_channels=True, input_shape=(7, 7)), (1, 2, 7, 7)),
    (AvgPool2dLayer(kernel_size=3), (1, 2, 7, 7)),
    (AdaptativeAvgPool2dLayer(output_size=(2, 2)), (1, 2, 7, 7))
])
def test_build_matrix(layer, input_shape):

    input = {0: torch.randn(input_shape)}

    out = layer.process(input, store_for_graph=True)
    print(f"Out shape is {out.shape}")
    out = out.detach().cpu().numpy().reshape(-1)

    layer.build_matrix()
    out_mat = layer.get_matrix()[0].todense().sum(axis=1).reshape(-1)
    print(f"Out_mat_shape is {out_mat.shape}")

    difference = np.linalg.norm(out_mat - out)
    print(f"Difference is {difference}")

    assert np.isclose(difference, 0)
