from .avgpool_layer import AvgPool2dLayer


class AdaptativeAvgPool2dLayer(AvgPool2dLayer):

    def __init__(self, input_size, output_size):
        stride = (input_size // output_size)
        kernel_size = input_size - (output_size - 1) * stride

        super().__init__(
            kernel_size=kernel_size,
            stride=stride
        )

