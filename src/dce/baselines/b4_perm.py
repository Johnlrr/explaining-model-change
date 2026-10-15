"""B4 — difference of global permutation importance (Muschalik-style; Tier 0 only; spec §6.4).

PI_k(f) = acc(f) − acc(f with factor k permuted), accuracy w.r.t. y_clean, mean over 5 permutations
(seeds 0–4). One vector Δ = PI(f_new) − PI(f_old), shared by every point; always valid (D-024).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def permutation_importance(predict: Callable[[np.ndarray], np.ndarray], phi: np.ndarray, y: np.ndarray,
                           n_perm: int = 5) -> np.ndarray:
    base = (predict(phi) == y).mean()
    out = np.zeros(phi.shape[1])
    for k in range(phi.shape[1]):
        drops = []
        for s in range(n_perm):
            moved = phi.copy()
            moved[:, k] = np.random.default_rng(s).permutation(moved[:, k])
            drops.append(base - (predict(moved) == y).mean())
        out[k] = float(np.mean(drops))
    return out


def b4_perm(predict_old, predict_new, phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return permutation_importance(predict_new, phi, y) - permutation_importance(predict_old, phi, y)
