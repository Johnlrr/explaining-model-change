"""Evaluation metrics (EVALUATION §2–§3, D-021 D2). Pure numpy: no model, no split access.

Every function takes one method's output on the N points of one pair (MethodOutput) and the injected
factor index k*. Recovery counts a point as 0 when it is not found within r (EVALUATION §2).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class MethodOutput:
    """found: (N,) bool, witness or CE found within r. dphi: (N, K) factor change; its rows on not-found points are ignored."""

    found: np.ndarray
    dphi: np.ndarray

    def __post_init__(self):
        found = np.asarray(self.found, dtype=bool)
        dphi = np.asarray(self.dphi, dtype=float)
        if dphi.ndim != 2 or found.shape != (dphi.shape[0],):
            raise ValueError("found must be (N,) and dphi (N, K)")
        if not np.isfinite(dphi[found]).all():
            raise ValueError("non-finite Δφ̃ on found points")
        object.__setattr__(self, "found", found)
        object.__setattr__(self, "dphi", dphi)


TIE_TOL = 1e-12


def factor_index(factors: Sequence[str], name: str) -> int:
    """Column of factor `name`. The order comes from the tier config (`factors`), never from a hard-coded list."""
    return list(factors).index(name)


def top_factor(dphi: np.ndarray) -> np.ndarray:
    """argmax_k |Δφ̃_ik| per point. −1 (undefined, the point scores 0) where the vector is zero or the maximum
    is attained by more than one factor (within TIE_TOL; D-028)."""
    mag = np.abs(dphi)
    mx = mag.max(axis=1)
    n_at_max = (mag >= (mx - TIE_TOL)[:, None]).sum(axis=1)
    return np.where((mx > 0) & (n_at_max == 1), mag.argmax(axis=1), -1)


def recovery_indicator(out: MethodOutput, k_star: int) -> np.ndarray:
    """Per-point term of Recovery@r: v_i · 1[top_i = k*], as floats (N,)."""
    return (out.found & (top_factor(out.dphi) == k_star)).astype(float)


def recovery_at_r(out: MethodOutput, k_star: int) -> float:
    return float(recovery_indicator(out, k_star).mean())


def found_rate(out: MethodOutput) -> float:
    return float(out.found.mean())


def share_matrix(out: MethodOutput) -> np.ndarray:
    """share_ik = |Δφ̃_ik| / Σ_k |Δφ̃_ik| on found points; NaN on not-found or zero-sum points."""
    mag = np.abs(out.dphi)
    tot = mag.sum(axis=1)
    ok = out.found & (tot > 0)
    share = np.full(mag.shape, np.nan)
    share[ok] = mag[ok] / tot[ok, None]
    return share


def share_of_kstar(out: MethodOutput, k_star: int) -> float:
    """Share of k*, mean over found points with a non-zero change (zero-sum points are excluded)."""
    s = share_matrix(out)[:, k_star]
    return _nanmean(s)


def median_norm(out: MethodOutput) -> float:
    """Median ‖Δφ̃_i‖₂ over found points (pair score for B1, B1-det, B3; EVALUATION §3 S)."""
    norms = np.linalg.norm(out.dphi[out.found], axis=1)
    return float(np.median(norms)) if norms.size else float("nan")


def pooled_found_rate(outs: Sequence[MethodOutput]) -> float:
    """Δ-CE pair score S: found-rate over the pooled N and O directions (EVALUATION §3)."""
    return float(np.concatenate([o.found for o in outs]).mean())


def recovery_given_reach(reach: np.ndarray, outs: Mapping[str, MethodOutput], k_star: int) -> dict[str, float]:
    """Recovery|reach (D-021 D2): the same subset `reach` (Δ-CE found) for every method. Descriptive only."""
    reach = np.asarray(reach, dtype=bool)
    if not reach.any():
        return {m: float("nan") for m in outs}
    return {m: float(recovery_indicator(o, k_star)[reach].mean()) for m, o in outs.items()}


def sar_pair(out: MethodOutput) -> bool:
    """Systematic attribution on a null pair (EVALUATION §3): max_k mean share ≥ 0.5 and found-rate ≥ 0.2."""
    if found_rate(out) < 0.2:
        return False
    mean_share = _nanmean_axis0(share_matrix(out))
    if np.isnan(mean_share).all():
        return False
    return bool(np.nanmax(mean_share) >= 0.5)


def auroc(drift_scores: Sequence[float], null_scores: Sequence[float]) -> float:
    """P(score_drift > score_null) with ties counted half (Mann–Whitney). Descriptive H3 only."""
    d = np.asarray(drift_scores, dtype=float)
    n = np.asarray(null_scores, dtype=float)
    if d.size == 0 or n.size == 0:
        raise ValueError("both groups must be non-empty")
    ranks = rankdata(np.concatenate([d, n]))
    u = ranks[: d.size].sum() - d.size * (d.size + 1) / 2
    return float(u / (d.size * n.size))


def _nanmean(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else float("nan")


def _nanmean_axis0(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=0)
