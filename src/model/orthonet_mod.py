from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn
from torch import Tensor
import math


def gram_schmidt(input):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x / x.norm(p=2)
        output.append(x)
    return torch.stack(output)


def initialize_orthogonal_filters(c, h, w):

    if h * w < c:
        n = c // (h * w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))


class GramSchmidtTransform(torch.nn.Module):
    instance: Dict[int, Optional[GramSchmidtTransform]] = {}
    constant_filter: Tensor

    @staticmethod
    def build(c: int, h: int):
        if c not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(  # type:ignore
                c, h
            )
        return GramSchmidtTransform.instance[(c, h)]  # type:ignore

    def __init__(self, c: int, h: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
        self.register_buffer(
            "constant_filter", rand_ortho_filters.to(self.device).detach()
        )

    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, FWT: GramSchmidtTransform, input: Tensor):
        # happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input.to(self.device))
        b = input.size(0)
        return input.view(b, -1)

    __all__ = [
        "OrthoNet",
        "orthonet_mod_18",
        "orthonet_mod_34",
        "orthonet_mod_50",
        "orthonet_mod_101",
        "orthonet_mod_152",
    ]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, height, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self._process: nn.Module = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.downsample = downsample
        self.stride = stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.planes = planes
        self._excitation = nn.Sequential(
            nn.Linear(
                in_features=planes,
                out_features=round(planes / 16),
                device=self.device,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=round(planes / 16),
                out_features=planes,
                device=self.device,
                bias=False,
            ),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention()
        self.F_C_A = GramSchmidtTransform.build(planes, height)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self._process(x)
        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0), out.size(1)
        excitation = self._excitation(compressed).view(b, c, 1, 1)
        attention = excitation * out
        attention += residual
        activated = torch.relu(attention)
        return activated


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, height, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self._preprocess: nn.Module = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(planes),
        )
        self._scale: nn.Module = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.downsample = downsample
        self.stride = stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.planes = planes

        self._excitation = nn.Sequential(
            nn.Linear(
                in_features=planes,
                out_features=round(planes / 16),
                device=self.device,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=round(planes / 16),
                out_features=planes,
                device=self.device,
                bias=False,
            ),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention()
        self.F_C_A = GramSchmidtTransform.build(planes, height)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        preprocess_out = self._preprocess(x)
        compressed = self.OrthoAttention(self.F_C_A, preprocess_out)
        b, c = preprocess_out.size(0), preprocess_out.size(1)
        attention = self._excitation(compressed).view(b, c, 1, 1)
        attentended = attention * preprocess_out
        scale_out = self._scale(attentended)
        scale_out += residual
        activated = torch.relu(scale_out)

        return activated


class OrthoNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inplanes = 64
        super(OrthoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 16, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.to(self._device)

    def _make_layer(self, block, planes, height, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, height, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, height))

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def orthonet_mod_18(n_classes, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        n_classes (int) : number of dataset classes
    """
    model = OrthoNet(BasicBlock, [2, 2, 2, 2], num_classes=n_classes, **kwargs)
    return model


def orthonet_mod_34(n_classes, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        n_classes (int) : number of dataset classes
    """
    model = OrthoNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes, **kwargs)
    return model


def orthonet_mod_50(n_classes, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        n_classes (int) : number of dataset classes
    """
    model = OrthoNet(Bottleneck, [3, 4, 6, 3], num_classes=n_classes, **kwargs)
    return model


def orthonet_mod_101(n_classes, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        n_classes (int) : number of dataset classes
    """
    model = OrthoNet(Bottleneck, [3, 4, 23, 3], num_classes=n_classes, **kwargs)
    return model


def orthonet_mod_152(n_classes, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        n_classes (int) : number of dataset classes
    """
    model = OrthoNet(Bottleneck, [3, 8, 36, 3], num_classes=n_classes, **kwargs)
    return model
