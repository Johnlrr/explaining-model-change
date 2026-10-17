"""Projected Δφ̃ (D-033 (b), D-034 C3): remove the component along ∇_z m_o(z0) of f_old.

P = I − g gᵀ / ‖g‖², Δφ̃' = P Δφ̃. Recovery then uses the same top-factor tie rule (dce.eval.metrics).
"""

from __future__ import annotations

import numpy as np
import torch

from dce.eval.metrics import MethodOutput
from dce.eval.tier0_margins import Tier0Margin


def old_margin_gradients(f_old, z0: np.ndarray, y: np.ndarray) -> np.ndarray:
    """g_i = ∇_z m_o(z0_i) for the f_old margin w.r.t. the clean label y_i (float32 model, Tier 0 factor space)."""
    m = Tier0Margin(f_old, y)
    z = torch.as_tensor(z0, dtype=torch.float32).clone().requires_grad_(True)
    idx = torch.arange(len(z0))
    return torch.autograd.grad(m(z, idx).sum(), z)[0].numpy().astype(float)


def project(out: MethodOutput, g: np.ndarray) -> MethodOutput:
    """Δφ̃' = Δφ̃ − (Δφ̃·g / ‖g‖²) g per point; not-found rows stay zero; a zero gradient leaves Δφ̃ unchanged."""
    gg = (g * g).sum(1, keepdims=True)
    coef = np.where(gg > 0, (out.dphi * g).sum(1, keepdims=True) / np.where(gg > 0, gg, 1.0), 0.0)
    d = out.dphi - coef * g
    d[~out.found] = 0.0
    return MethodOutput(out.found, d)
