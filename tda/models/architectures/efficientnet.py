from .architecture import Architecture
from .svhn_models import svhn_preprocess, svhn_preprocess_resize
from tda.models.layers import (
    ConvLayer,
    BatchNorm2d,
    ReluLayer,
    AvgPool2dLayer,
    LinearLayer,
    SoftMaxLayer,
)
import torch.nn.functional as F

from ..layers.adaptativeavgpool_layer import AdaptativeAvgPool2dLayer

efficientnet = Architecture(
    name="efficientnet",
    preprocess=svhn_preprocess_resize,
    layers=[
        # Beginning
        # 0
        ConvLayer(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        # 1
        BatchNorm2d(channels=32, activ=F.relu),

        # Block 1
        # Sub-block 1
        # 2
        ConvLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            grouped_channels=True,
            bias=False,
        ),
        # 3
        BatchNorm2d(channels=32, activ=F.relu),
        # 4
        ConvLayer(
            in_channels=32,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 5
        BatchNorm2d(channels=16),

        # Sub-block 2
        # 6
        ConvLayer(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            grouped_channels=True,
            bias=False,
        ),
        # 7
        BatchNorm2d(channels=16, activ=F.relu),
        # 8
        ConvLayer(
            in_channels=16,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 9
        BatchNorm2d(channels=16),

        # Block 2
        # Sub-block 1
        # 10
        ConvLayer(
            in_channels=16,
            out_channels=96,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 11
        BatchNorm2d(channels=96, activ=F.relu),
        # 12
        ConvLayer(
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            stride=2,
            padding=1,
            grouped_channels=True,
            bias=False,
        ),
        # 13
        BatchNorm2d(channels=96, activ=F.relu),
        # 14
        ConvLayer(
            in_channels=96,
            out_channels=24,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 15
        BatchNorm2d(channels=24),

        # Sub-block 2
        # 16
        ConvLayer(
            in_channels=24,
            out_channels=144,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 17
        BatchNorm2d(channels=144, activ=F.relu),
        # 18
        ConvLayer(
            in_channels=144,
            out_channels=144,
            kernel_size=3,
            stride=1,
            padding=1,
            grouped_channels=True,
            bias=False,
        ),
        # 19
        BatchNorm2d(channels=144, activ=F.relu),
        # 20
        ConvLayer(
            in_channels=144,
            out_channels=24,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 21
        BatchNorm2d(channels=24),

        # Sub-block 3
        # 22
        ConvLayer(
            in_channels=24,
            out_channels=144,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 23
        BatchNorm2d(channels=144, activ=F.relu),
        # 24
        ConvLayer(
            in_channels=144,
            out_channels=144,
            kernel_size=3,
            stride=1,
            padding=1,
            grouped_channels=True,
            bias=False,
        ),
        # 25
        BatchNorm2d(channels=144, activ=F.relu),
        # 26
        ConvLayer(
            in_channels=144,
            out_channels=24,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 27
        BatchNorm2d(channels=24),

        # Block 3
        # Sub-block 1
        # 28
        ConvLayer(
            in_channels=24,
            out_channels=144,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 29
        BatchNorm2d(channels=144, activ=F.relu),
        # 30
        ConvLayer(
            in_channels=144,
            out_channels=144,
            kernel_size=5,
            stride=2,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 31
        BatchNorm2d(channels=144, activ=F.relu),
        # 32
        ConvLayer(
            in_channels=144,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 33
        BatchNorm2d(channels=40),

        # Sub-block 2
        # 34
        ConvLayer(
            in_channels=40,
            out_channels=240,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 35
        BatchNorm2d(channels=240, activ=F.relu),
        # 36
        ConvLayer(
            in_channels=240,
            out_channels=240,
            kernel_size=5,
            stride=1,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 37
        BatchNorm2d(channels=240, activ=F.relu),
        # 38
        ConvLayer(
            in_channels=240,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 39
        BatchNorm2d(channels=40),

        # Sub-block 3
        # 40
        ConvLayer(
            in_channels=40,
            out_channels=240,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 41
        BatchNorm2d(channels=240, activ=F.relu),
        # 42
        ConvLayer(
            in_channels=240,
            out_channels=240,
            kernel_size=5,
            stride=1,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 43
        BatchNorm2d(channels=240, activ=F.relu),
        # 44
        ConvLayer(
            in_channels=240,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 45
        BatchNorm2d(channels=40),


        # Block 4
        # Sub-block 1
        # 46
        ConvLayer(
            in_channels=40,
            out_channels=240,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 47
        BatchNorm2d(channels=240, activ=F.relu),
        # 48
        ConvLayer(
            in_channels=240,
            out_channels=240,
            kernel_size=3,
            stride=2,
            padding=1,
            grouped_channels=True,
            bias=False,
        ),
        # 49
        BatchNorm2d(channels=240, activ=F.relu),
        # 50
        ConvLayer(
            in_channels=240,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 51
        BatchNorm2d(channels=80),

        # Sub-block 2
        # 52
        ConvLayer(
            in_channels=80,
            out_channels=480,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 53
        BatchNorm2d(channels=480, activ=F.relu),
        # 54
        ConvLayer(
            in_channels=480,
            out_channels=480,
            kernel_size=3,
            stride=1,
            padding=1,
            grouped_channels=True,
            bias=False,
        ),
        # 55
        BatchNorm2d(channels=480, activ=F.relu),
        # 56
        ConvLayer(
            in_channels=480,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 57
        BatchNorm2d(channels=80),

        # Sub-block 3
        # 58
        ConvLayer(
            in_channels=80,
            out_channels=480,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 59
        BatchNorm2d(channels=480, activ=F.relu),
        # 60
        ConvLayer(
            in_channels=480,
            out_channels=480,
            kernel_size=3,
            stride=1,
            padding=1,
            grouped_channels=True,
            bias=False,
        ),
        # 61
        BatchNorm2d(channels=480, activ=F.relu),
        # 62
        ConvLayer(
            in_channels=480,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 63
        BatchNorm2d(channels=80),

        # Sub-block 4
        # 64
        ConvLayer(
            in_channels=80,
            out_channels=480,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 65
        BatchNorm2d(channels=480, activ=F.relu),
        # 66
        ConvLayer(
            in_channels=480,
            out_channels=480,
            kernel_size=3,
            stride=1,
            padding=1,
            grouped_channels=True,
            bias=False,
        ),
        # 67
        BatchNorm2d(channels=480, activ=F.relu),
        # 68
        ConvLayer(
            in_channels=480,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 69
        BatchNorm2d(channels=80),


        # Block 5
        # Sub-block 1
        # 70
        ConvLayer(
            in_channels=80,
            out_channels=480,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 71
        BatchNorm2d(channels=480, activ=F.relu),
        # 72
        ConvLayer(
            in_channels=480,
            out_channels=480,
            kernel_size=5,
            stride=1,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 73
        BatchNorm2d(channels=480, activ=F.relu),
        # 74
        ConvLayer(
            in_channels=480,
            out_channels=112,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 75
        BatchNorm2d(channels=112),

        # Sub-block 2
        # 76
        ConvLayer(
            in_channels=112,
            out_channels=672,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 77
        BatchNorm2d(channels=672, activ=F.relu),
        # 78
        ConvLayer(
            in_channels=672,
            out_channels=672,
            kernel_size=5,
            stride=1,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 79
        BatchNorm2d(channels=672, activ=F.relu),
        # 80
        ConvLayer(
            in_channels=672,
            out_channels=112,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 81
        BatchNorm2d(channels=112),

        # Sub-block 3
        # 82
        ConvLayer(
            in_channels=112,
            out_channels=672,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 83
        BatchNorm2d(channels=672, activ=F.relu),
        # 84
        ConvLayer(
            in_channels=672,
            out_channels=672,
            kernel_size=5,
            stride=1,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 85
        BatchNorm2d(channels=672, activ=F.relu),
        # 86
        ConvLayer(
            in_channels=672,
            out_channels=112,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 87
        BatchNorm2d(channels=112),

        # Sub-block 4
        # 88
        ConvLayer(
            in_channels=112,
            out_channels=672,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 89
        BatchNorm2d(channels=672, activ=F.relu),
        # 90
        ConvLayer(
            in_channels=672,
            out_channels=672,
            kernel_size=5,
            stride=1,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 91
        BatchNorm2d(channels=672, activ=F.relu),
        # 92
        ConvLayer(
            in_channels=672,
            out_channels=112,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 93
        BatchNorm2d(channels=112),



        # Block 6
        # Sub-block 1
        # 94
        ConvLayer(
            in_channels=112,
            out_channels=672,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 95
        BatchNorm2d(channels=672, activ=F.relu),
        # 96
        ConvLayer(
            in_channels=672,
            out_channels=672,
            kernel_size=5,
            stride=2,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 97
        BatchNorm2d(channels=672, activ=F.relu),
        # 98
        ConvLayer(
            in_channels=672,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 99
        BatchNorm2d(channels=192),

        # Sub-block 2
        # 100
        ConvLayer(
            in_channels=192,
            out_channels=1152,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 101
        BatchNorm2d(channels=1152, activ=F.relu),
        # 102
        ConvLayer(
            in_channels=1152,
            out_channels=1152,
            kernel_size=5,
            stride=1,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 103
        BatchNorm2d(channels=1152, activ=F.relu),
        # 104
        ConvLayer(
            in_channels=1152,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 105
        BatchNorm2d(channels=192),

        # Sub-block 3
        # 106
        ConvLayer(
            in_channels=192,
            out_channels=1152,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 107
        BatchNorm2d(channels=1152, activ=F.relu),
        # 108
        ConvLayer(
            in_channels=1152,
            out_channels=1152,
            kernel_size=5,
            stride=1,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 109
        BatchNorm2d(channels=1152, activ=F.relu),
        # 110
        ConvLayer(
            in_channels=1152,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 111
        BatchNorm2d(channels=192),

        # Sub-block 4
        # 112
        ConvLayer(
            in_channels=192,
            out_channels=1152,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 113
        BatchNorm2d(channels=1152, activ=F.relu),
        # 114
        ConvLayer(
            in_channels=1152,
            out_channels=1152,
            kernel_size=5,
            stride=1,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 115
        BatchNorm2d(channels=1152, activ=F.relu),
        # 116
        ConvLayer(
            in_channels=1152,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 117
        BatchNorm2d(channels=192),

        # Sub-block 5
        # 118
        ConvLayer(
            in_channels=192,
            out_channels=1152,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 119
        BatchNorm2d(channels=1152, activ=F.relu),
        # 120
        ConvLayer(
            in_channels=1152,
            out_channels=1152,
            kernel_size=5,
            stride=1,
            padding=2,
            grouped_channels=True,
            bias=False,
        ),
        # 121
        BatchNorm2d(channels=1152, activ=F.relu),
        # 122
        ConvLayer(
            in_channels=1152,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 123
        BatchNorm2d(channels=192),



        # Block 7
        # Sub-block 1
        # 124
        ConvLayer(
            in_channels=192,
            out_channels=1152,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 125
        BatchNorm2d(channels=1152, activ=F.relu),
        # 126
        ConvLayer(
            in_channels=1152,
            out_channels=1152,
            kernel_size=3,
            stride=1,
            padding=1,
            grouped_channels=True,
            bias=False,
        ),
        # 127
        BatchNorm2d(channels=1152, activ=F.relu),
        # 128
        ConvLayer(
            in_channels=1152,
            out_channels=320,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 129
        BatchNorm2d(channels=320),

        # Sub-block 2
        # 130
        ConvLayer(
            in_channels=320,
            out_channels=1920,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 131
        BatchNorm2d(channels=1920, activ=F.relu),
        # 132
        ConvLayer(
            in_channels=1920,
            out_channels=1920,
            kernel_size=3,
            stride=1,
            padding=1,
            grouped_channels=True,
            bias=False,
        ),
        # 133
        BatchNorm2d(channels=1920, activ=F.relu),
        # 134
        ConvLayer(
            in_channels=1920,
            out_channels=320,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 135
        BatchNorm2d(channels=320),


        # End
        # 136
        ConvLayer(
            in_channels=320,
            out_channels=1280,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 137
        AdaptativeAvgPool2dLayer(output_size=1),
        # 138
        LinearLayer(1280, 10),
        # 139
        SoftMaxLayer(),


        #############
        #############
        # Skip layers

        # For block 2
        # 140
        AvgPool2dLayer(kernel_size=2, stride=2),
        # 141
        ConvLayer(
            in_channels=16,
            out_channels=24,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 142
        BatchNorm2d(channels=24),

        # For block 3
        # 143
        AvgPool2dLayer(kernel_size=2, stride=2),
        # 144
        ConvLayer(
            in_channels=24,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 145
        BatchNorm2d(channels=40),

        # For block 4
        # 146
        AvgPool2dLayer(kernel_size=2, stride=2),
        # 147
        ConvLayer(
            in_channels=40,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 148
        BatchNorm2d(channels=80),

        # For block 5
        # 149
        ConvLayer(
            in_channels=80,
            out_channels=112,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 150
        BatchNorm2d(channels=112),

        # For block 6
        # 151
        AvgPool2dLayer(kernel_size=2, stride=2),
        # 152
        ConvLayer(
            in_channels=112,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 153
        BatchNorm2d(channels=192),

        # For block 6
        # 154
        ConvLayer(
            in_channels=192,
            out_channels=320,
            kernel_size=1,
            stride=1,
            padding=0,
            grouped_channels=False,
            bias=False,
        ),
        # 155
        BatchNorm2d(channels=320),
],
    layer_links=[(i - 1, i) for i in range(139)]
    + [(140,141), (141,142)]
    + [(143,144), (144,145)]
    + [(146,147), (147,148)]
    + [(149,150)]
    + [(151,152), (152,153)]
    + [(154,155)]
    + [
        (9, 140),
        (142, 16),

        (27, 143),
        (145, 34),

        (45, 146),
        (148, 52),

        (69, 149),
        (150, 76),

        (93, 151),
        (153, 100),

        (123, 154),
        (155, 130),
    ],
)
