"""G0 phenomenon check for Tier 1 (spec §9 with v1.2 B3, v1.3 C1/C3/C4, D-021 D1). Validation only (A1).

Flip_k is defined for the re-sampleable factors {color, bg, thick} (shape has none). The same re-drawn
images are used for f_old and f_new, so ΔFlip_k is paired.
"""

from __future__ import annotations

import numpy as np

from dce.data.tier1 import RESAMPLE, resample_factor
from dce.eval.g0 import decisive_disagreement

FLIP_FACTORS = tuple(RESAMPLE)


def flip_rates(predict, split) -> dict[str, float]:
    base = predict(split.images())
    return {f: float((predict(resample_factor(split, f)) != base).mean()) for f in FLIP_FACTORS}


def g0_criteria(condition: str, logits_old, logits_new, y_new, sub: np.ndarray | None, flip_old: dict,
                flip_new: dict, kappa: float, k_star: str | None = None) -> dict:
    """Pass/fail for one Tier 1 pair. T1: ΔFlip_color ≥ 0.15 (C4). T3 (C1): ΔFlip_k* ≥ 0.15, max over the other
    re-sampleable factors ≤ 0.05, DD outside the subgroup ≤ 0.05. N0: DD ≤ 0.02 and max |ΔFlip_k| ≤ 0.05."""
    lo, ln = np.asarray(logits_old), np.asarray(logits_new)
    pred_old, pred_new = lo.argmax(1), ln.argmax(1)
    dd = decisive_disagreement(lo, ln, kappa)
    delta = {f: flip_new[f] - flip_old[f] for f in FLIP_FACTORS}
    out = {"agreement_all": float((pred_old == pred_new).mean()), "decisive_disagreement_all": float(dd.mean()),
           "delta_flip": delta}
    if condition == "T1":
        out["pass"] = delta["color"] >= 0.15
    elif condition == "T3":
        out["n_subgroup"] = int(sub.sum())
        out["acc_new_labels_subgroup"] = float((pred_new[sub] == y_new[sub]).mean())  # C2: reported
        out["dd_outside_subgroup"] = float(dd[~sub].mean())
        out["k_star"] = k_star
        out["delta_flip_k_star"] = delta[k_star]
        out["max_abs_delta_flip_other"] = max(abs(v) for f, v in delta.items() if f != k_star)
        out["pass"] = (delta[k_star] >= 0.15 and out["max_abs_delta_flip_other"] <= 0.05
                       and out["dd_outside_subgroup"] <= 0.05)
    elif condition == "N0":
        out["max_abs_delta_flip"] = max(abs(v) for v in delta.values())
        out["pass"] = out["decisive_disagreement_all"] <= 0.02 and out["max_abs_delta_flip"] <= 0.05
    else:
        raise ValueError(f"unknown Tier 1 condition {condition!r}")
    out["pass"] = bool(out["pass"])
    return out
