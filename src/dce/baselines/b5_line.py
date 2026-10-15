"""B5 — DIG-style line search (after Renard et al.; EVALUATION §7, amendment A4).

From z0, straight segments to the K = 20 nearest training points (Euclidean, same space Z), 50 equally
spaced steps each (t = 1/50 … 1). Returns the point of F^N_κ with the smallest d(z0, ·) if ≤ r,
otherwise NOT_FOUND. Δφ̃ = z − z0 on Tier 0.
"""

from __future__ import annotations

import torch

from dce.solvers.dce import Margin


def b5_line(z0: torch.Tensor, train_z: torch.Tensor, m_old: Margin, m_new: Margin, kappa: float, r: float,
            K: int = 20, steps: int = 50) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, p = z0.shape
    best_z = z0.clone()
    best_d = torch.full((B,), float("inf"), dtype=z0.dtype)
    t = torch.arange(1, steps + 1, dtype=z0.dtype) / steps
    with torch.no_grad():
        nn_idx = torch.cdist(z0, train_z).topk(K, largest=False).indices  # (B, K)
        for i in range(B):
            ends = train_z[nn_idx[i]]                                     # (K, p)
            pts = (z0[i] + t[None, :, None] * (ends - z0[i])[:, None, :]).reshape(-1, p)
            idx = torch.full((len(pts),), i, dtype=torch.long)
            ok = (m_new(pts, idx) <= -kappa) & (m_old(pts, idx) >= kappa)
            if ok.any():
                d = (pts[ok] - z0[i]).norm(dim=1)
                j = d.argmin()
                best_d[i], best_z[i] = d[j], pts[ok][j]
    found = torch.isfinite(best_d) & (best_d <= r)
    return found, best_z - z0, best_d
