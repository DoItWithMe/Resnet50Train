import torch
import torch.nn as nn
import torch.nn.functional as nnF
import timm
from timm.models.layers import SqueezeExcite

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze step
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )  # Excitation step

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  # Scale the input feature map


def add_se_layer_to_block(block, reduction=16):
    """Add a SqueezeExcite to each block in the given module."""
    for name, child in block.named_children():
        if isinstance(child, nn.Conv2d) and "conv" in name:
            setattr(
                block,
                name,
                nn.Sequential(child, SqueezeExcite(child.out_channels, reduction)),
            )
        elif len(list(child.children())) > 0:
            add_se_layer_to_block(child, reduction)


def modify_resnet_with_se(original_model, layer_name, reduction=16):
    """Modify the specified layer of the ResNet model to include SELayers."""
    for name, child in original_model.named_children():
        if name == layer_name:
            add_se_layer_to_block(child, reduction)
        elif len(list(child.children())) > 0:
            modify_resnet_with_se(child, layer_name, reduction)


