"""Lightweight 3D ResNet for DaT scan classification.

Kept deliberately small so training fits on a low-VRAM GPU while staying
expressive enough for 64^3 single-channel volumes.
"""

import torch
import torch.nn as nn


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
        # spatial dims: 64 -> 32 -> 16 -> 8
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
