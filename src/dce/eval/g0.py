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


T3_CRITERIA = ("v1.2", "v1.3")


def g0_criteria(condition, phi, logits_old, logits_new, y_new, flip_old, flip_new, params,
                kappa: float, t3_criteria: str = "v1.2") -> dict:
    """Pass/fail of the G0 table for one pair (spec §9 with amendments v1.2 B3 and v1.3 C1).

    t3_criteria="v1.2": subgroup accuracy w.r.t. the new labels ≥ 0.85 and DD outside ≤ 0.05.
    t3_criteria="v1.3" (C1): ΔFlip_{k*} ≥ 0.15, max_{k≠k*} |ΔFlip_k| ≤ 0.05, DD outside ≤ 0.05.
    The v1.2 accuracies are always computed and reported (C2).
    """
    if t3_criteria not in T3_CRITERIA:
        raise ValueError(f"unknown t3_criteria {t3_criteria!r}")
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
        decisive = sub & (np.abs(np.asarray(logits_new)[:, 0] - np.asarray(logits_new)[:, 1]) >= kappa)
        out["n_subgroup_decisive"] = int(decisive.sum())
        out["acc_new_labels_subgroup_decisive"] = float((pred_new[decisive] == y_new[decisive]).mean())
        k_star = 1  # size
        delta = flip_new - flip_old
        out["delta_flip_k_star"] = float(delta[k_star])
        out["max_abs_delta_flip_other"] = float(np.abs(np.delete(delta, k_star)).max())
        out["t3_criteria"] = t3_criteria
        out["pass_v1_2"] = out["acc_new_labels_subgroup"] >= 0.85 and out["dd_outside_subgroup"] <= 0.05
        out["pass_v1_3"] = (out["delta_flip_k_star"] >= 0.15 and out["max_abs_delta_flip_other"] <= 0.05
                            and out["dd_outside_subgroup"] <= 0.05)
        out["pass"] = out["pass_v1_3"] if t3_criteria == "v1.3" else out["pass_v1_2"]
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
