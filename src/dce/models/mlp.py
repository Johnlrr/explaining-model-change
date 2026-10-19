"""Stage B model (v1.9 §4, D-040): MLP 6→64→64→2 with ReLU, and its margin for the shared solver.

`TabMargin` mirrors `dce.eval.tier0_margins.Tier0Margin` with D = identity on [0, 1]^6: a float32 model for phase 1
and feasibility checks, a float64 copy of the same weights only when the solver evaluates float64 inputs.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from dce.models.cnn import margin


class MLP(nn.Module):
    def __init__(self, n_in: int = 6, width: int = 64, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def as_batch(x: np.ndarray, gen=None) -> torch.Tensor:
    """`make_batch` for dce.models.cnn.train: tabular inputs need no rendering and get no noise."""
    return torch.as_tensor(x, dtype=torch.float32)


class TabMargin:
    """Callable (z, idx) → (n,) margins w.r.t. the per-point labels y[idx]."""

    def __init__(self, model: MLP, y: np.ndarray):
        self.m32 = model.eval()
        self.m64 = copy.deepcopy(model).double().eval()
        for net in (self.m32, self.m64):
            for prm in net.parameters():
                prm.requires_grad_(False)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return (self.m64 if z.dtype == torch.float64 else self.m32)(z)

    def __call__(self, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return margin(self.logits(z), self.y[idx])
