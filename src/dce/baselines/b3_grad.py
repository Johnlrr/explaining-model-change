"""B3 — gradient difference (Artelt et al. Eq. 14; spec §6.4).

g = ∇m_n/‖∇m_n‖ − ∇m_o/‖∇m_o‖ at z0. Tier 0: Δφ̃ = −g (already in factor space).
Valid unless ‖g‖ < 1e-8.
"""

from __future__ import annotations

import torch

from dce.solvers.dce import Margin


def b3_grad(z0: torch.Tensor, m_old: Margin, m_new: Margin) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(z0.shape[0])
    z = z0.detach().clone().requires_grad_(True)
    g_o = torch.autograd.grad(m_old(z, idx).sum(), z)[0]
    g_n = torch.autograd.grad(m_new(z, idx).sum(), z)[0]
    g = g_n / g_n.norm(dim=1, keepdim=True) - g_o / g_o.norm(dim=1, keepdim=True)
    return g.norm(dim=1) >= 1e-8, -g
