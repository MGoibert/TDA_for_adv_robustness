from .architecture import Architecture
from .svhn_models import svhn_preprocess
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

cifar_resnet_1 = Architecture(
    name="efficientnet",
    preprocess=svhn_preprocess,
    layers=[
        # Beginning
        # -1
        ConvLayer(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        # 0
        BatchNorm2d(channels=32, activ=F.relu),

        # Block 1
        # Sub-block 1
        # 1
        ConvLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=32,
            bias=False,
        ),
        # 2
        BatchNorm2d(channels=32, activ=F.relu),
        # 3
        ConvLayer(
            in_channels=32,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 4
        BatchNorm2d(channels=16),

        # Sub-block 2
        # 5
        ConvLayer(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=16,
            bias=False,
        ),
        # 6
        BatchNorm2d(channels=16, activ=F.relu),
        # 7
        ConvLayer(
            in_channels=16,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 8
        BatchNorm2d(channels=16),

        # Block 2
        # Sub-block 1
        # 9
        ConvLayer(
            in_channels=16,
            out_channels=96,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 10
        BatchNorm2d(channels=96, activ=F.relu),
        # 11
        ConvLayer(
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=96,
            bias=False,
        ),
        # 12
        BatchNorm2d(channels=96, activ=F.relu),
        # 13
        ConvLayer(
            in_channels=96,
            out_channels=24,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 14
        BatchNorm2d(channels=24),

        # Sub-block 2
        # 15
        ConvLayer(
            in_channels=24,
            out_channels=144,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 16
        BatchNorm2d(channels=144, activ=F.relu),
        # 17
        ConvLayer(
            in_channels=144,
            out_channels=144,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=144,
            bias=False,
        ),
        # 18
        BatchNorm2d(channels=144, activ=F.relu),
        # 19
        ConvLayer(
            in_channels=144,
            out_channels=24,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 20
        BatchNorm2d(channels=24),

        # Sub-block 3
        # 21
        ConvLayer(
            in_channels=24,
            out_channels=144,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 22
        BatchNorm2d(channels=144, activ=F.relu),
        # 23
        ConvLayer(
            in_channels=144,
            out_channels=144,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=144,
            bias=False,
        ),
        # 24
        BatchNorm2d(channels=144, activ=F.relu),
        # 25
        ConvLayer(
            in_channels=144,
            out_channels=24,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 26
        BatchNorm2d(channels=24),

        # Block 3
        # Sub-block 1
        # 27
        ConvLayer(
            in_channels=24,
            out_channels=144,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 28
        BatchNorm2d(channels=144, activ=F.relu),
        # 29
        ConvLayer(
            in_channels=144,
            out_channels=144,
            kernel_size=5,
            stride=2,
            padding=1,
            groups=144,
            bias=False,
        ),
        # 30
        BatchNorm2d(channels=144, activ=F.relu),
        # 31
        ConvLayer(
            in_channels=144,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 32
        BatchNorm2d(channels=40),

        # Sub-block 2
        # 33
        ConvLayer(
            in_channels=40,
            out_channels=240,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 34
        BatchNorm2d(channels=240, activ=F.relu),
        # 35
        ConvLayer(
            in_channels=240,
            out_channels=240,
            kernel_size=5,
            stride=1,
            padding=1,
            groups=240,
            bias=False,
        ),
        # 36
        BatchNorm2d(channels=240, activ=F.relu),
        # 37
        ConvLayer(
            in_channels=240,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 38
        BatchNorm2d(channels=40),

        # Sub-block 3
        # 39
        ConvLayer(
            in_channels=40,
            out_channels=240,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 40
        BatchNorm2d(channels=240, activ=F.relu),
        # 41
        ConvLayer(
            in_channels=240,
            out_channels=240,
            kernel_size=5,
            stride=1,
            padding=1,
            groups=240,
            bias=False,
        ),
        # 42
        BatchNorm2d(channels=240, activ=F.relu),
        # 43
        ConvLayer(
            in_channels=240,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 44
        BatchNorm2d(channels=40),


        # Block 4
        # Sub-block 1
        # 45
        ConvLayer(
            in_channels=40,
            out_channels=240,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 46
        BatchNorm2d(channels=240, activ=F.relu),
        # 47
        ConvLayer(
            in_channels=240,
            out_channels=240,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=240,
            bias=False,
        ),
        # 48
        BatchNorm2d(channels=240, activ=F.relu),
        # 49
        ConvLayer(
            in_channels=240,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 50
        BatchNorm2d(channels=80),

        # Sub-block 2
        # 51
        ConvLayer(
            in_channels=80,
            out_channels=480,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 52
        BatchNorm2d(channels=480, activ=F.relu),
        # 53
        ConvLayer(
            in_channels=480,
            out_channels=480,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=480,
            bias=False,
        ),
        # 54
        BatchNorm2d(channels=480, activ=F.relu),
        # 55
        ConvLayer(
            in_channels=480,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 56
        BatchNorm2d(channels=80),

        # Sub-block 3
        # 57
        ConvLayer(
            in_channels=80,
            out_channels=480,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 58
        BatchNorm2d(channels=480, activ=F.relu),
        # 59
        ConvLayer(
            in_channels=480,
            out_channels=480,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=480,
            bias=False,
        ),
        # 60
        BatchNorm2d(channels=480, activ=F.relu),
        # 61
        ConvLayer(
            in_channels=480,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 62
        BatchNorm2d(channels=80),

        # Sub-block 4
        # 63
        ConvLayer(
            in_channels=80,
            out_channels=480,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 64
        BatchNorm2d(channels=480, activ=F.relu),
        # 65
        ConvLayer(
            in_channels=480,
            out_channels=480,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=480,
            bias=False,
        ),
        # 66
        BatchNorm2d(channels=480, activ=F.relu),
        # 67
        ConvLayer(
            in_channels=480,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 68
        BatchNorm2d(channels=80),


        # Block 5
        # Sub-block 1
        # 69
        ConvLayer(
            in_channels=80,
            out_channels=480,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 70
        BatchNorm2d(channels=480, activ=F.relu),
        # 71
        ConvLayer(
            in_channels=480,
            out_channels=480,
            kernel_size=5,
            stride=1,
            padding=1,
            groups=480,
            bias=False,
        ),
        # 72
        BatchNorm2d(channels=480, activ=F.relu),
        # 73
        ConvLayer(
            in_channels=480,
            out_channels=112,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 74
        BatchNorm2d(channels=112),

        # Sub-block 2
        # 75
        ConvLayer(
            in_channels=112,
            out_channels=672,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 76
        BatchNorm2d(channels=672, activ=F.relu),
        # 77
        ConvLayer(
            in_channels=672,
            out_channels=672,
            kernel_size=5,
            stride=1,
            padding=1,
            groups=672,
            bias=False,
        ),
        # 78
        BatchNorm2d(channels=672, activ=F.relu),
        # 79
        ConvLayer(
            in_channels=672,
            out_channels=112,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 80
        BatchNorm2d(channels=112),

        # Sub-block 3
        # 81
        ConvLayer(
            in_channels=112,
            out_channels=672,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 82
        BatchNorm2d(channels=672, activ=F.relu),
        # 83
        ConvLayer(
            in_channels=672,
            out_channels=672,
            kernel_size=5,
            stride=1,
            padding=1,
            groups=672,
            bias=False,
        ),
        # 84
        BatchNorm2d(channels=672, activ=F.relu),
        # 85
        ConvLayer(
            in_channels=672,
            out_channels=112,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 86
        BatchNorm2d(channels=112),

        # Sub-block 4
        # 87
        ConvLayer(
            in_channels=112,
            out_channels=672,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 88
        BatchNorm2d(channels=672, activ=F.relu),
        # 89
        ConvLayer(
            in_channels=672,
            out_channels=672,
            kernel_size=5,
            stride=1,
            padding=1,
            groups=672,
            bias=False,
        ),
        # 90
        BatchNorm2d(channels=672, activ=F.relu),
        # 91
        ConvLayer(
            in_channels=672,
            out_channels=112,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 92
        BatchNorm2d(channels=112),



        # Block 6
        # Sub-block 1
        # 93
        ConvLayer(
            in_channels=112,
            out_channels=672,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 94
        BatchNorm2d(channels=672, activ=F.relu),
        # 95
        ConvLayer(
            in_channels=672,
            out_channels=672,
            kernel_size=5,
            stride=2,
            padding=1,
            groups=672,
            bias=False,
        ),
        # 96
        BatchNorm2d(channels=672, activ=F.relu),
        # 97
        ConvLayer(
            in_channels=672,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 98
        BatchNorm2d(channels=192),

        # Sub-block 2
        # 99
        ConvLayer(
            in_channels=192,
            out_channels=1152,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 100
        BatchNorm2d(channels=1152, activ=F.relu),
        # 101
        ConvLayer(
            in_channels=1152,
            out_channels=1152,
            kernel_size=5,
            stride=1,
            padding=1,
            groups=1152,
            bias=False,
        ),
        # 102
        BatchNorm2d(channels=1152, activ=F.relu),
        # 103
        ConvLayer(
            in_channels=1152,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 104
        BatchNorm2d(channels=192),

        # Sub-block 3
        # 105
        ConvLayer(
            in_channels=192,
            out_channels=1152,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 106
        BatchNorm2d(channels=1152, activ=F.relu),
        # 107
        ConvLayer(
            in_channels=1152,
            out_channels=1152,
            kernel_size=5,
            stride=1,
            padding=1,
            groups=1152,
            bias=False,
        ),
        # 108
        BatchNorm2d(channels=1152, activ=F.relu),
        # 109
        ConvLayer(
            in_channels=1152,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 110
        BatchNorm2d(channels=192),

        # Sub-block 4
        # 111
        ConvLayer(
            in_channels=192,
            out_channels=1152,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 112
        BatchNorm2d(channels=1152, activ=F.relu),
        # 113
        ConvLayer(
            in_channels=1152,
            out_channels=1152,
            kernel_size=5,
            stride=1,
            padding=1,
            groups=1152,
            bias=False,
        ),
        # 114
        BatchNorm2d(channels=1152, activ=F.relu),
        # 115
        ConvLayer(
            in_channels=1152,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 116
        BatchNorm2d(channels=192),

        # Sub-block 5
        # 117
        ConvLayer(
            in_channels=192,
            out_channels=1152,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 118
        BatchNorm2d(channels=1152, activ=F.relu),
        # 119
        ConvLayer(
            in_channels=1152,
            out_channels=1152,
            kernel_size=5,
            stride=1,
            padding=1,
            groups=1152,
            bias=False,
        ),
        # 120
        BatchNorm2d(channels=1152, activ=F.relu),
        # 121
        ConvLayer(
            in_channels=1152,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 122
        BatchNorm2d(channels=192),



        # Block 7
        # Sub-block 1
        # 123
        ConvLayer(
            in_channels=192,
            out_channels=1152,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 124
        BatchNorm2d(channels=1152, activ=F.relu),
        # 125
        ConvLayer(
            in_channels=1152,
            out_channels=1152,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1152,
            bias=False,
        ),
        # 126
        BatchNorm2d(channels=1152, activ=F.relu),
        # 127
        ConvLayer(
            in_channels=1152,
            out_channels=320,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 128
        BatchNorm2d(channels=320),

        # Sub-block 2
        # 129
        ConvLayer(
            in_channels=320,
            out_channels=1920,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 130
        BatchNorm2d(channels=1920, activ=F.relu),
        # 131
        ConvLayer(
            in_channels=1920,
            out_channels=1920,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1920,
            bias=False,
        ),
        # 132
        BatchNorm2d(channels=1920, activ=F.relu),
        # 133
        ConvLayer(
            in_channels=1920,
            out_channels=320,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 134
        BatchNorm2d(channels=320),


        # End
        # 135
        ConvLayer(
            in_channels=320,
            out_channels=1280,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 136
        AdaptativeAvgPool2dLayer(input_size=..., output_size=1),
        # 137
        LinearLayer(1280, 10),
        # 138
        SoftMaxLayer(),



        # Skip layers

        # For block 2
        # 139
        AvgPool2dLayer(kernel_size=2, stride=2),
        # 140
        ConvLayer(
            in_channels=16,
            out_channels=24,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 141
        BatchNorm2d(channels=24),

        # For block 3
        # 142
        AvgPool2dLayer(kernel_size=2, stride=2),
        # 143
        ConvLayer(
            in_channels=24,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 144
        BatchNorm2d(channels=40),

        # For block 4
        # 145
        AvgPool2dLayer(kernel_size=2, stride=2),
        # 146
        ConvLayer(
            in_channels=40,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 147
        BatchNorm2d(channels=48),

        # For block 5
        # 148
        ConvLayer(
            in_channels=80,
            out_channels=112,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 149
        BatchNorm2d(channels=112),

        # For block 6
        # 150
        AvgPool2dLayer(kernel_size=2, stride=2),
        # 151
        ConvLayer(
            in_channels=112,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 152
        BatchNorm2d(channels=192),

        # For block 6
        # 153
        ConvLayer(
            in_channels=192,
            out_channels=320,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        # 154
        BatchNorm2d(channels=320),
],
    layer_links=[(i - 1, i) for i in range(138)]
    + [(139,140), (140,141)]
    + [(142,143), (143,144)]
    + [(145,146), (146,147)]
    + [(148,149)]
    + [(150,151), (151,152)]
    + [(153,154)]
    + [
        (8, 139),
        (141, 15),

        (26, 142),
        (144, 33),

        (44, 145),
        (147, 51),

        (68, 148),
        (149, 75),

        (92, 150),
        (152, 99),

        (122, 153),
        (154, 129),
    ],
)
