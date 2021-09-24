from typing import List, Tuple
import torch
import numpy as np
from torch import nn, Tensor
from math import gcd


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes, out_planes, kernel_size=1, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=False
    )


def resize_conv3x3(in_planes, out_planes, scale=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact"""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=scale), conv3x3(in_planes, out_planes)
        )


def resize_conv1x1(in_planes, out_planes, scale=1):
    """upsample + 1x1 convolution with padding to avoid checkerboard artifact"""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=scale), conv1x1(in_planes, out_planes)
        )


class ResnetBlock:
    expansion: int


class EncoderBlock(nn.Module, ResnetBlock):
    """
    ResNet block, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EncoderBottleneck(nn.Module, ResnetBlock):
    """
    ResNet bottleneck, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L75
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride=stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module, ResnetBlock):
    """
    ResNet block, but convs replaced with resize convs, and channel increase is in
    second conv, not first
    """

    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderBottleneck(nn.Module, ResnetBlock):
    """
    ResNet bottleneck, but convs replaced with resize convs
    """

    expansion = 4

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = resize_conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = resize_conv3x3(width, width, scale)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.scale = scale

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        block: ResnetBlock,
        layers: List[int],
        latent_related_dim: int,
        latent_unrelated_dim: int,
        img_dim: Tuple[int, int, int],
        hidden_dim: int = 64,
        first_conv: bool = False,
        maxpool1: bool = False,
    ):
        super().__init__()

        self.inplanes = 64
        self.latent_related_dim = latent_related_dim
        self.latent_unrelated_dim = latent_unrelated_dim
        self.hidden_dim = hidden_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(
                img_dim[0],
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        else:
            self.conv1 = nn.Conv2d(
                img_dim[0],
                self.inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, hidden_dim * 1, layers[0])
        self.layer2 = self._make_layer(block, hidden_dim * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_dim * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_dim * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(
            hidden_dim * 8, latent_related_dim + (latent_unrelated_dim * 2)
        )

    def _make_layer(self, block: ResnetBlock, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        z_related = x[..., : self.latent_related_dim]
        x_unrelated = x[..., self.latent_related_dim :]
        mu = x_unrelated[..., : self.latent_unrelated_dim]
        logvar = x_unrelated[..., self.latent_unrelated_dim :]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_unrelated = mu + eps * std
        return z_related, z_unrelated

    def encode(self, x):
        z_related, _ = self.forward(x)
        return z_related


class ResNetDecoder(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(
        self,
        block: ResnetBlock,
        layers,
        latent_related_dim: int,
        latent_unrelated_dim: int,
        img_dim: Tuple[int, int, int],
        hidden_dim: int = 64,
        first_conv=False,
        maxpool1=False,
    ):
        super().__init__()
        img_dim = np.asarray(img_dim)
        self.expansion = block.expansion
        self.inplanes = hidden_dim * 8 * block.expansion
        self.latent_related_dim = latent_related_dim
        self.latent_unrelated_dim = latent_unrelated_dim
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.upscale_factor = 8

        self.init_dim = (img_dim[1:] / gcd(*img_dim[1:]) * 4).astype(np.int)

        self.linear = nn.Linear(
            latent_related_dim + latent_unrelated_dim,
            self.inplanes * np.prod(self.init_dim),
        )

        self.layer1 = self._make_layer(block, hidden_dim * 4, layers[0], scale=2)
        self.layer2 = self._make_layer(block, hidden_dim * 2, layers[1], scale=2)
        self.layer3 = self._make_layer(block, hidden_dim, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, hidden_dim, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, hidden_dim, layers[3])

        if self.first_conv:
            self.upscale = nn.Upsample(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = nn.Upsample(scale_factor=1)

        self.upscale1 = nn.Upsample(size=tuple(img_dim[1:] // self.upscale_factor))

        self.conv1 = nn.Conv2d(
            hidden_dim * block.expansion,
            img_dim[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def _make_layer(self, block: ResnetBlock, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, z_related, z_unrelated):
        z = torch.cat([z_related, z_unrelated], dim=-1)
        x: Tensor = self.linear(z)

        x = x.view(x.size(0), self.hidden_dim * 8 * self.expansion, *self.init_dim)
        x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x


__valid_setups = {
    "resnet9": {
        "enc_block": EncoderBlock,
        "dec_block": DecoderBlock,
        "layers": [1, 1, 1, 1],
        "hidden_dim": 16,
    },
    "resnet9_8": {
        "enc_block": EncoderBlock,
        "dec_block": DecoderBlock,
        "layers": [1, 1, 1, 1],
        "hidden_dim": 8,
    },
    "resnet9_32": {
        "enc_block": EncoderBlock,
        "dec_block": DecoderBlock,
        "layers": [1, 1, 1, 1],
        "hidden_dim": 32,
    },
    "resnet18": {
        "enc_block": EncoderBlock,
        "dec_block": DecoderBlock,
        "layers": [2, 2, 2, 2],
        "hidden_dim": 64,
    },
    "resnet50": {
        "enc_block": EncoderBottleneck,
        "dec_block": DecoderBottleneck,
        "layers": [3, 4, 6, 3],
        "hidden_dim": 64,
    },
}


def create_encoder(
    enc_type: str,
    latent_related_dim,
    latent_unrelated_dim,
    img_dim,
    first_conv,
    maxpool1,
):
    if enc_type not in __valid_setups:
        enc_type = "resnet18"
    setups = __valid_setups[enc_type]
    return ResNetEncoder(
        setups["enc_block"],
        setups["layers"],
        latent_related_dim,
        latent_unrelated_dim,
        img_dim,
        setups["hidden_dim"],
        first_conv,
        maxpool1,
    )


def create_decoder(
    enc_type: str,
    latent_related_dim,
    latent_unrelated_dim,
    img_dim,
    first_conv,
    maxpool1,
):
    if enc_type not in __valid_setups:
        enc_type = "resnet18"
    setups = __valid_setups[enc_type]
    return ResNetDecoder(
        setups["dec_block"],
        setups["layers"],
        latent_related_dim,
        latent_unrelated_dim,
        img_dim,
        setups["hidden_dim"],
        first_conv,
        maxpool1,
    )
