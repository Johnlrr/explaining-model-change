"""Classifier shared by all tiers (spec §5.1) and its training loops."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn


class SmallCNN(nn.Module):
    """Conv(3→32)–ReLU–MaxPool → Conv(32→64)–ReLU–MaxPool → Conv(64→64)–ReLU–GAP → Linear(64→2)."""

    def __init__(self, in_channels: int = 3, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def margin(logits: torch.Tensor, y: torch.Tensor | int) -> torch.Tensor:
    """m = f_y − f_{1−y} (spec §3.1)."""
    if isinstance(y, int):
        y = torch.full((logits.shape[0],), y, dtype=torch.long, device=logits.device)
    sign = 1 - 2 * y.to(logits.dtype)  # y=0 → +1, y=1 → −1
    return sign * (logits[:, 0] - logits[:, 1])


def train(
    model: nn.Module,
    inputs: np.ndarray,
    labels: np.ndarray,
    make_batch: Callable[[np.ndarray, torch.Generator], torch.Tensor],
    *,
    epochs: int,
    lr: float,
    batch_size: int = 256,
    order_seed: int,
    noise_seed: int,
) -> list[float]:
    """Adam training; `make_batch` turns raw inputs (e.g. factors) into a noisy image batch.

    The data order and the pixel noise have their own seeds, so f_old and the N0 fine-tunes differ
    only in the data order and noise (spec §4.4).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order_rng = np.random.default_rng(order_seed)
    noise_gen = torch.Generator().manual_seed(noise_seed)
    y_all = torch.as_tensor(labels, dtype=torch.long)
    losses = []
    model.train()
    for _ in range(epochs):
        perm = order_rng.permutation(len(labels))
        total, count = 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            x = make_batch(inputs[idx], noise_gen)
            loss = nn.functional.cross_entropy(model(x), y_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        losses.append(total / count)
    model.eval()
    return losses


@torch.no_grad()
def predict_logits(
    model: nn.Module, inputs: np.ndarray, make_batch: Callable[[np.ndarray], torch.Tensor],
    batch_size: int = 1000,
) -> torch.Tensor:
    model.eval()
    out = [model(make_batch(inputs[i : i + batch_size])) for i in range(0, len(inputs), batch_size)]
    return torch.cat(out)
