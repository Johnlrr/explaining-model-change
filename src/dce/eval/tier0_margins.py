"""Tier 0 margins for the solvers (spec §3.1; D-024): m(z) = margin_y(CNN(render(z))).

The float32 model is the evaluation model (phase 1 and every feasibility check). A float64 copy of
the same weights is used only when the solver evaluates at float64 inputs (SLSQP, phase 2).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from dce.data.renderer import render
from dce.models.cnn import SmallCNN, margin


class Tier0Margin:
    """Callable (z, idx) → (n,) margins of one classifier, w.r.t. the per-point labels y[idx]."""

    def __init__(self, model: SmallCNN, y: np.ndarray):
        self.m32 = model.eval()
        self.m64 = copy.deepcopy(model).double().eval()
        for net in (self.m32, self.m64):
            for prm in net.parameters():
                prm.requires_grad_(False)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        net = self.m64 if z.dtype == torch.float64 else self.m32
        return net(render(z))

    def __call__(self, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return margin(self.logits(z), self.y[idx])


def float64_gap(model: SmallCNN, phi: np.ndarray, batch: int = 1000) -> float:
    """D-024: max |logit64 − logit32| over phi (same weights, renderer at each precision)."""
    m = Tier0Margin(model, np.zeros(len(phi), dtype=int))
    gap = 0.0
    with torch.no_grad():
        for i in range(0, len(phi), batch):
            z = torch.as_tensor(phi[i : i + batch])
            l32 = m.logits(z.float()).double()
            l64 = m.logits(z.double())
            gap = max(gap, float((l64 - l32).abs().max()))
    return gap


def valid_points(phi: np.ndarray, y_clean: np.ndarray, m_old: Tier0Margin, m_new: Tier0Margin, n: int,
                 seed: int, kappa: float) -> np.ndarray:
    """Indices of the first n valid points under default_rng(seed).permutation (spec §3.2, D-024)."""
    order = np.random.default_rng(seed).permutation(len(phi))
    z = torch.as_tensor(phi[order], dtype=torch.float32)
    y = torch.as_tensor(y_clean[order], dtype=torch.long)
    with torch.no_grad():
        mo = margin(m_old.logits(z), y)
        mn = margin(m_new.logits(z), y)
    ok = ((mo >= kappa) & (mn >= kappa)).numpy()
    keep = order[ok][:n]
    if len(keep) < n:
        raise RuntimeError(f"only {len(keep)} valid points, need {n}")
    return keep
