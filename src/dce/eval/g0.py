"""G0 phenomenon check for Tier 0 (spec §9, on validation per D-011).

Flip_k(f) = share of predictions that change when factor k is re-sampled uniformly, all other
factors fixed. The same re-sampled values are used for f_old and f_new, so ΔFlip_k is paired.
"""

from __future__ import annotations

import numpy as np

FLIP_SEED_BASE = 7000


def flip_rates(predict, phi: np.ndarray, n_factors: int = 5) -> np.ndarray:
    base = predict(phi)
    rates = np.zeros(n_factors)
    for k in range(n_factors):
        rng = np.random.default_rng(FLIP_SEED_BASE + k)
        moved = phi.copy()
        moved[:, k] = rng.uniform(0.0, 1.0, len(phi))
        rates[k] = float((predict(moved) != base).mean())
    return rates


def g0_criteria(condition, phi, pred_old, pred_new, y_new, flip_old, flip_new, params) -> dict:
    """Pass/fail of the spec §9 G0 table for one pair (thresholds fixed in the spec)."""
    agree = pred_old == pred_new
    out = {"agreement_all": float(agree.mean()), "delta_flip": (flip_new - flip_old).tolist()}
    if condition == "T1":
        out["delta_flip_color"] = float(flip_new[2] - flip_old[2])
        out["pass"] = out["delta_flip_color"] >= 0.15
    elif condition == "T3":
        sub = phi[:, 1] > params.t3_size_threshold
        out["n_subgroup"] = int(sub.sum())
        out["acc_new_labels_subgroup"] = float((pred_new[sub] == y_new[sub]).mean())
        out["agreement_outside_subgroup"] = float(agree[~sub].mean())
        out["pass"] = out["acc_new_labels_subgroup"] >= 0.85 and out["agreement_outside_subgroup"] >= 0.95
    elif condition == "T4":
        band = (phi[:, 0] > 0.5) & (phi[:, 0] < params.t4_threshold)
        out["n_band"] = int(band.sum())
        out["acc_new_labels_band"] = float((pred_new[band] == y_new[band]).mean())
        out["pass"] = out["acc_new_labels_band"] >= 0.80
    else:  # N0, N1
        out["max_abs_delta_flip"] = float(np.abs(flip_new - flip_old).max())
        out["pass"] = out["agreement_all"] >= 0.98 and out["max_abs_delta_flip"] <= 0.05
    out["pass"] = bool(out["pass"])
    return out
