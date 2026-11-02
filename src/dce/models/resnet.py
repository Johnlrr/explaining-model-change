"""ResNet-8 for the second-architecture check (D-052 X4): stem conv, three basic residual blocks, GAP, linear.

Layers with weights: 1 stem conv + 3 blocks × 2 convs + 1 linear = 8. Same input (3 × 32 × 32 rendered images) and
output (2 logits) as SmallCNN, so every margin, solver and baseline runs unchanged. BatchNorm is used in eval mode
whenever the model is searched (model.eval()).
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.short = (nn.Sequential() if stride == 1 and c_in == c_out else
                      nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(out)) + self.short(x))


class ResNet8(nn.Module):
    """Stem Conv(3→32)–BN–ReLU → blocks 32 → 64 (stride 2) → 64 (stride 2) → GAP → Linear(64→2)."""

    def __init__(self, in_channels: int = 3, n_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.blocks = nn.Sequential(BasicBlock(32, 32, 1), BasicBlock(32, 64, 2), BasicBlock(64, 64, 2))
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)).mean(dim=(2, 3)))
