from .avgpool_layer import AvgPool2dLayer


class AdaptativeAvgPool2dLayer(AvgPool2dLayer):

    def __init__(self, input_size, output_size):

        output_height, output_width = output_size
        input_height, input_width = input_size

        stride_width = input_width // output_width
        kernel_width = input_width - (output_width-1) * stride_width

        stride_height = input_height // output_height
        kernel_height = input_height - (output_height-1) * stride_height

        super().__init__(
            kernel_size=(kernel_height, kernel_width),
            stride=(stride_height, stride_width)
        )

