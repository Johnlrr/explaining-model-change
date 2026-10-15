"""Tier 0 data: clean splits and injected-drift adaptation sets (spec §4.2, §4.4).

Reading of the spec where it is silent (research/DECISIONS.md D-017):
- adaptation sets carry no label noise (the 5% noise of §4.2 is for the clean training split only);
- in T1 the colour–label correlation applies to every adaptation sample, including the 30% drawn
  outside the ambiguous band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .renderer import N_FACTORS, sample_factors, task_label

CONDITIONS = ("T1", "T3", "T4", "N0", "N1")
# Injected factor k* per condition (index into FACTOR_NAMES); None for null conditions.
K_STAR = {"T1": 2, "T3": 1, "T4": 0, "N0": None, "N1": None}


@dataclass(frozen=True)
class Split:
    phi: np.ndarray  # [n, 5]
    y: np.ndarray  # labels used for training/evaluation (train split: with label noise)
    y_clean: np.ndarray  # noise-free labels y = 1[φ1 > 0.5]


@dataclass(frozen=True)
class DriftParams:
    """Defaults are spec §4.4; the G0 knobs may change them (at most two adjustments)."""

    t1_ambiguous_frac: float = 0.7
    t1_ambiguous_low: float = 0.4
    t1_ambiguous_high: float = 0.6
    t1_correlation: float = 0.95
    t3_size_threshold: float = 0.75
    t4_threshold: float = 0.6
    n1_size_low: float = 0.5


def clean_splits(
    seed: int = 1234, n_train: int = 20_000, n_val: int = 5_000, n_test: int = 5_000,
    label_noise: float = 0.05,
) -> dict[str, Split]:
    """Fixed train/val/test splits; label noise only on train (spec §4.2)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        phi = sample_factors(n, rng)
        y_clean = task_label(phi)
        y = y_clean.copy()
        if name == "train" and label_noise > 0:
            flip = rng.random(n) < label_noise
            y[flip] = 1 - y[flip]
        out[name] = Split(phi=phi, y=y, y_clean=y_clean)
    return out


def drift_label(condition: str, phi: np.ndarray, params: DriftParams) -> np.ndarray:
    """Labelling rule after the drift (used for the adaptation set and for G0 checks)."""
    y = task_label(phi)
    if condition == "T3":
        sub = phi[:, 1] > params.t3_size_threshold
        y = np.where(sub, 1 - y, y)
    elif condition == "T4":
        y = task_label(phi, threshold=params.t4_threshold)
    return y


def adaptation_set(
    condition: str, seed: int, n: int = 10_000, params: DriftParams | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptation data (φ, y) for fine-tuning f_new (spec §4.4)."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    params = params or DriftParams()
    rng = np.random.default_rng(seed)
    phi = sample_factors(n, rng)

    if condition == "T1":
        amb = rng.random(n) < params.t1_ambiguous_frac
        phi[amb, 0] = rng.uniform(params.t1_ambiguous_low, params.t1_ambiguous_high, amb.sum())
        y = task_label(phi)
        corr = rng.random(n) < params.t1_correlation
        high = rng.uniform(0.5, 1.0, n)
        low = rng.uniform(0.0, 0.5, n)
        phi[:, 2] = np.where(corr, np.where(y == 1, high, low), phi[:, 2])
        return phi, y

    if condition == "N1":
        phi[:, 1] = rng.uniform(params.n1_size_low, 1.0, n)

    return phi, drift_label(condition, phi, params)


def tier0_pairs(n_seeds_drift: int = 3, n_seeds_n0: int = 5, n_seeds_n1: int = 3) -> list[dict]:
    """Model pairs of Tier 0 with stable pair_id (spec §4.4: 17 pairs)."""
    counts = {"T1": n_seeds_drift, "T3": n_seeds_drift, "T4": n_seeds_drift, "N0": n_seeds_n0,
              "N1": n_seeds_n1}
    pairs, pid = [], 0
    for cond in CONDITIONS:
        for s in range(counts[cond]):
            pairs.append({"pair_id": pid, "condition": cond, "seed": s, "k_star": K_STAR[cond]})
            pid += 1
    return pairs


assert N_FACTORS == 5
