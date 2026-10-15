"""G0 phenomenon check for Tier 0 (spec §9 as amended by v1.2 B3; validation per D-011).

Flip_k(f) = share of predictions that change when factor k is re-sampled uniformly, all other
factors fixed. The same re-sampled values are used for f_old and f_new, so ΔFlip_k is paired.
"""

from __future__ import annotations

import numpy as np

FLIP_SEED_BASE = 7000


def decisive_disagreement(logits_old, logits_new, kappa: float) -> np.ndarray:
    """Per point: argmax differs AND |m_o| ≥ κ AND |m_n| ≥ κ (amendment v1.2 B3).

    This is the observable part of F_κ (type N ∪ O): threshold jitter is excluded because near the
    decision threshold at least one model has margin < κ.
    """
    lo, ln = np.asarray(logits_old), np.asarray(logits_new)
    m_o = np.abs(lo[:, 0] - lo[:, 1])
    m_n = np.abs(ln[:, 0] - ln[:, 1])
    return (lo.argmax(1) != ln.argmax(1)) & (m_o >= kappa) & (m_n >= kappa)


def flip_rates(predict, phi: np.ndarray, n_factors: int = 5) -> np.ndarray:
    base = predict(phi)
    rates = np.zeros(n_factors)
    for k in range(n_factors):
        rng = np.random.default_rng(FLIP_SEED_BASE + k)
        moved = phi.copy()
        moved[:, k] = rng.uniform(0.0, 1.0, len(phi))
        rates[k] = float((predict(moved) != base).mean())
    return rates


def g0_criteria(condition, phi, logits_old, logits_new, y_new, flip_old, flip_new, params,
                kappa: float) -> dict:
    """Pass/fail of the G0 table for one pair (spec §9 with amendment v1.2 B3)."""
    pred_old, pred_new = np.asarray(logits_old).argmax(1), np.asarray(logits_new).argmax(1)
    agree = pred_old == pred_new
    dd = decisive_disagreement(logits_old, logits_new, kappa)
    out = {"agreement_all": float(agree.mean()), "decisive_disagreement_all": float(dd.mean()),
           "delta_flip": (flip_new - flip_old).tolist()}
    if condition == "T1":
        out["delta_flip_color"] = float(flip_new[2] - flip_old[2])
        out["pass"] = out["delta_flip_color"] >= 0.15
    elif condition == "T3":
        sub = phi[:, 1] > params.t3_size_threshold
        out["n_subgroup"] = int(sub.sum())
        out["acc_new_labels_subgroup"] = float((pred_new[sub] == y_new[sub]).mean())
        out["agreement_outside_subgroup"] = float(agree[~sub].mean())  # descriptive
        out["dd_outside_subgroup"] = float(dd[~sub].mean())
        out["pass"] = out["acc_new_labels_subgroup"] >= 0.85 and out["dd_outside_subgroup"] <= 0.05
    elif condition == "T4":
        band = (phi[:, 0] > 0.5) & (phi[:, 0] < params.t4_threshold)
        out["n_band"] = int(band.sum())
        out["acc_new_labels_band"] = float((pred_new[band] == y_new[band]).mean())
        out["pass"] = out["acc_new_labels_band"] >= 0.80
    else:  # N0, N1
        out["max_abs_delta_flip"] = float(np.abs(flip_new - flip_old).max())
        out["pass"] = out["decisive_disagreement_all"] <= 0.02 and out["max_abs_delta_flip"] <= 0.05
    out["pass"] = bool(out["pass"])
    return out
