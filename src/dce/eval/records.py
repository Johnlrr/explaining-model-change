"""Wiring for the final run (EVALUATION §8 steps 5–8): solver and baseline outputs → MethodOutput, B0,
runtime rows, and schema checks. No metric value is computed or printed here (D-028).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch

from .metrics import MethodOutput, recovery_indicator
from .rules import PairRecord

EXPECTED = {0: {"pairs": 17, "points": 300}, 1: {"pairs": 11, "points": 200}}  # EVALUATION §8 step 7

Phi = Callable[[torch.Tensor], torch.Tensor]  # φ̃(D(·)); identity on Tier 0 (σ_k = 1, z is the factor vector)


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def from_points(found, z_new: torch.Tensor, z_ref: torch.Tensor, phi: Phi | None = None) -> MethodOutput:
    """Δφ̃ = φ̃(z_new) − φ̃(z_ref): Δ-CE and B5 (z_new = witness, z_ref = z0), B1 (z_new = CE of f_new,
    z_ref = CE of f_old). Values on not-found points are zeroed so they cannot leak into any statistic."""
    d = (phi(z_new) - phi(z_ref)) if phi else (z_new - z_ref)
    found = _np(found).astype(bool)
    dphi = _np(d).astype(float)
    dphi[~found] = 0.0
    return MethodOutput(found, dphi)


def from_direction(valid, dphi: torch.Tensor) -> MethodOutput:
    """B3: a direction in factor space (already Δφ̃) valid unless ‖g‖ < 1e-8."""
    valid = _np(valid).astype(bool)
    d = _np(dphi).astype(float)
    d[~valid] = 0.0
    return MethodOutput(valid, d)


def b0_disagreement(pred_old: np.ndarray, pred_new: np.ndarray) -> float:
    """B0 (EVALUATION §3): share of ALL samples of the split (unfiltered) with argmax f_old ≠ argmax f_new."""
    pred_old, pred_new = np.asarray(pred_old), np.asarray(pred_new)
    if pred_old.shape != pred_new.shape or pred_old.ndim != 1 or pred_old.size == 0:
        raise ValueError("predictions must be aligned non-empty 1-D arrays")
    return float((pred_old != pred_new).mean())


def runtime_row(res, wall_s: float, n: int) -> dict:
    """Runtime per point (v1.6 G3): wall and CPU, phase-2 CPU split by points with / without a witness."""
    row = {"wall_s_per_point": wall_s / n, "cpu_s_per_point": res.cpu_seconds / n, "device": "cpu"}
    c, f = res.phase2_cpu_per_point, res.found
    if c is not None:
        row["phase1_wall_s"] = float(res.phase1_wall)
        row["phase2_cpu_s_per_point_with_witness"] = float(c[f].mean()) if f.any() else None
        row["phase2_cpu_s_per_point_without_witness"] = float(c[~f].mean()) if (~f).any() else None
    return row


def check_outputs(outputs: Mapping[str, MethodOutput], n: int, n_factors: int) -> None:
    """Schema, count and NaN checks for one pair (EVALUATION §16). Raises ValueError."""
    for name, o in outputs.items():
        if o.found.shape != (n,) or o.dphi.shape != (n, n_factors):
            raise ValueError(f"{name}: expected {n} points × {n_factors} factors, got {o.dphi.shape}")
        if not np.isfinite(o.dphi).all():
            raise ValueError(f"{name}: non-finite Δφ̃")


def check_run_counts(tier: int, points_per_pair: Mapping[int, int]) -> None:
    """pair_id → number of points. 17 × 300 (Tier 0) or 11 × 200 (Tier 1)."""
    exp = EXPECTED[tier]
    if len(points_per_pair) != exp["pairs"] or any(v != exp["points"] for v in points_per_pair.values()):
        raise ValueError(f"tier {tier}: expected {exp['pairs']} pairs × {exp['points']} points")


def pair_record(pair_id: int, seed: int, outputs: Mapping[str, MethodOutput], k_star: int) -> PairRecord:
    """Per-point Recovery indicators of every method on a drift pair (null pairs have no k*)."""
    return PairRecord(pair_id, seed, {m: recovery_indicator(o, k_star) for m, o in outputs.items()})
