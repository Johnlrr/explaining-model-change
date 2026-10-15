"""Tier L: random linear model pairs and the closed forms of Propositions 3–4 (spec §4.1, §3.3).

m_o(z) = w_o·z + b_o, m_n(z) = w_n·z + b_n; w_o ~ N(0, I), Δw ~ N(0, 0.3² I), b ~ N(0, 0.1²).
Reading (spec silent): z0 ~ N(0, I) is redrawn until m_o(z0) ≥ κ and m_n(z0) ≥ κ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearPair:
    w_o: np.ndarray
    w_n: np.ndarray
    b_o: float
    b_n: float
    z0: np.ndarray

    @property
    def dw(self) -> np.ndarray:
        return self.w_n - self.w_o


def sample_pair(seed: int, p: int, kappa: float = 0.5, max_tries: int = 100_000) -> LinearPair:
    rng = np.random.default_rng(seed)
    w_o = rng.normal(0.0, 1.0, p)
    w_n = w_o + rng.normal(0.0, 0.3, p)
    b_o, b_n = rng.normal(0.0, 0.1, 2)
    for _ in range(max_tries):
        z0 = rng.normal(0.0, 1.0, p)
        if w_o @ z0 + b_o >= kappa and w_n @ z0 + b_n >= kappa:
            return LinearPair(w_o, w_n, float(b_o), float(b_n), z0)
    raise RuntimeError(f"no valid z0 for seed {seed}")


def closed_form(pair: LinearPair, kappa: float = 0.5) -> dict:
    """Proposition 3: minimum-norm u with w_n·u ≤ −a and w_o·u ≥ −c."""
    w_o, w_n, z0 = pair.w_o, pair.w_n, pair.z0
    a = w_n @ z0 + pair.b_n + kappa
    c = w_o @ z0 + pair.b_o - kappa
    u_a = -a / (w_n @ w_n) * w_n
    if w_o @ u_a >= -c:  # equivalent to a (w_o·w_n)/‖w_n‖² ≤ c
        return {"case": "A", "u": u_a, "delta": float(a / np.linalg.norm(w_n))}
    W = np.stack([w_n, w_o])
    rhs = np.array([-a, -c])
    G_inv_r = np.linalg.solve(W @ W.T, rhs)
    return {"case": "B", "u": W.T @ G_inv_r, "delta": float(np.sqrt(rhs @ G_inv_r))}


def project_out(v: np.ndarray, w_o: np.ndarray) -> np.ndarray:
    """P v with P = I − w_o w_oᵀ/‖w_o‖² (Proposition 4)."""
    return v - (v @ w_o) / (w_o @ w_o) * w_o


def abs_cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def ce_closed_form(w: np.ndarray, b: float, z0: np.ndarray, kappa: float = 0.5) -> dict:
    """Ordinary CE of a linear margin (amendment v1.4 E3): u = −(m(z0) + κ)/‖w‖² · w."""
    a = w @ z0 + b + kappa
    return {"u": -a / (w @ w) * w, "delta": float(a / np.linalg.norm(w))}
