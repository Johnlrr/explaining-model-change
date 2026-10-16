"""Paired bootstrap and Holm correction (EVALUATION §10–§11).

Points are the resampling unit; resampling is stratified by pair. The statistic is the mean over pairs of
(per-pair mean of method A − per-pair mean of method B). The same resampled indices are used for A and B,
so the test is paired. Percentile 95% CI; two-sided bootstrap p = 2·min(P*(≤0), P*(≥0)), floored at 1/B.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    diff: float
    ci_lo: float
    ci_hi: float
    p: float


def paired_bootstrap(a_pairs: Sequence[np.ndarray], b_pairs: Sequence[np.ndarray], rng: np.random.Generator,
                     B: int = 10_000, level: float = 0.95) -> BootstrapResult:
    """a_pairs[j], b_pairs[j]: per-point values (N_j,) of the two methods on pair j, the same points."""
    if len(a_pairs) != len(b_pairs) or not a_pairs:
        raise ValueError("need the same non-zero number of pairs for both methods")
    J = len(a_pairs)
    point = 0.0
    stat = np.zeros(B)
    for a, b in zip(a_pairs, b_pairs):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape != b.shape or a.ndim != 1:
            raise ValueError("per-pair arrays must be 1-D and aligned (same points)")
        point += (a.mean() - b.mean()) / J
        idx = rng.integers(0, a.size, size=(B, a.size))
        stat += (a[idx].mean(axis=1) - b[idx].mean(axis=1)) / J
    alpha = 1 - level
    lo, hi = np.quantile(stat, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((stat <= 0).mean(), (stat >= 0).mean()))
    return BootstrapResult(float(point), float(lo), float(hi), float(max(p, 1.0 / B)))


def holm(pvals: Mapping[str, float], alpha: float = 0.05) -> dict[str, tuple[float, bool]]:
    """Holm step-down over a named family. Returns name → (Holm-adjusted p, reject at alpha)."""
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    out: dict[str, tuple[float, bool]] = {}
    running = 0.0
    for j, name in enumerate(names):
        running = max(running, min(1.0, (m - j) * pvals[name]))
        out[name] = (running, running < alpha)
    return out
