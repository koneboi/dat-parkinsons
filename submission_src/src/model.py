"""Model definitions for the submission.

Mirrors src/train_resnet.py (backbones + heads) but builds with weights=None so
it loads across torchvision versions in the official runtime; the trained
state_dict is then loaded. Supports resnet18/34/50, densenet121,
efficientnet_b0/b1. Also carries the 3D ResNet (DATScan3DCNN) used by vol3d
members.
"""

import torch
import torch.nn as nn


def _swap_conv1(model, in_channels):
    if in_channels == 3:
        return model
    old = None
    if hasattr(model, "conv1"):
        old = model.conv1
    elif hasattr(model, "features"):
        cand = getattr(model.features, "conv0", None)
        if not (isinstance(cand, nn.Conv2d) and cand.in_channels == 3):
            cand = model.features[0][0] if len(model.features[0]) else None
        if isinstance(cand, nn.Conv2d) and cand.in_channels == 3:
            old = cand
    if not isinstance(old, nn.Conv2d):
        return model
    nw = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride,
                   old.padding, old.dilation, old.groups, old.bias is not None,
                   old.padding_mode)
    if old.bias is not None:
        nw.bias.copy_(old.bias)
    if hasattr(model, "conv1"):
        model.conv1 = nw
    elif hasattr(model.features, "conv0") and model.features.conv0 is old:
        model.features.conv0 = nw
    else:
        model.features[0][0] = nw
    return model


def conv3d_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3d_block(in_ch, out_ch, stride)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        return self.relu(out)


class DATScan3DCNN(nn.Module):
    """Small 3D ResNet: stem + 3 residual stages + global pool + head."""

    def __init__(self, in_channels=1, base_channels=32, num_classes=1, dropout=0.3):
        super().__init__()
        self.stem = conv3d_block(in_channels, base_channels)
        self.stage1 = ResidualBlock(base_channels, base_channels, stride=1)
        self.stage2 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.stage3 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


def build_model(backbone="resnet18", dropout=0.5, in_channels=3):
    from torchvision import models as tv

    builders = {
        "resnet18": tv.resnet18,
        "resnet34": tv.resnet34,
        "resnet50": tv.resnet50,
        "densenet121": tv.densenet121,
        "efficientnet_b0": tv.efficientnet_b0,
        "efficientnet_b1": tv.efficientnet_b1,
        "convnext_t": tv.convnext_tiny,
        "swin_t": tv.swin_t,
        "vit_b16": tv.vit_b_16,
    }
    model = builders[backbone](weights=None)
    if in_channels != 3:
        model = _swap_conv1(model, in_channels)
    if backbone.startswith("resnet"):
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 1))
    elif backbone.startswith("efficientnet"):
        f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout, inplace=True),
                                         nn.Linear(f, 1))
    elif backbone.startswith("densenet"):
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1024, 1))
    elif backbone == "convnext_t":
        model.classifier = nn.Sequential(nn.Flatten(start_dim=1),
                                         nn.Dropout(dropout),
                                         nn.Linear(768, 1))
    elif backbone == "swin_t":
        model.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.head.in_features, 1))
    elif backbone == "vit_b16":
        model.heads.head = nn.Linear(model.heads.head.in_features, 1)
    return model
