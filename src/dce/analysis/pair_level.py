"""D-056 S1/S2 (saved outputs only, descriptive, not registered): the pair is the unit of inference.

S1 (a) primary: per-pair estimate for every row of Table 3 (paper/main.tex, tab:tests) whose per-point quantity is in the
saved outputs; exact one-sided sign test over pairs (claimed side as the row states); for differences an exact sign-flip
permutation test on the pair means (all 2^n sign patterns); 95% CI from the t distribution on the pair means for n >= 4
(none for n < 4). Splits A and B use the same model pairs on different data, so "pooling" A and B means the per-pair mean
of the two splits' estimates (the number of pairs does not grow).
S1 (b) sensitivity, rows with >= 8 pairs: statsmodels MixedLM, point-level outcome, random intercept per pair.
No two-level bootstrap, no binomial GLMM (D-056).
S2: time per query and method from the saved `runtime` fields (no forward/backward counts are saved).
"""

from __future__ import annotations

import itertools
import json
import warnings
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

BAND = 0.10          # equivalence band of the "approximately equal" and "tie" rows
MIN_P_FLAG = 0.05


# ----------------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------------
def sign_test_p(k: int, n: int) -> float:
    """Exact one-sided sign test: P(X >= k), X ~ Binomial(n, 1/2)."""
    if n == 0:
        return float("nan")
    return sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n


def sign_test(values, c: float, side: str) -> dict:
    """k = number of pairs strictly on the claimed side of c ('below': v < c, 'above': v > c); pairs equal to c are dropped."""
    v = np.asarray(values, dtype=float)
    d = v - c
    keep = np.abs(d) > 1e-12
    k = int(((d < 0) if side == "below" else (d > 0))[keep].sum())
    n = int(keep.sum())
    return {"c": c, "side": side, "k": k, "n": n, "n_ties_dropped": int(len(v) - n), "p_one_sided": sign_test_p(k, n),
            "min_attainable_p": 0.5 ** n if n else None}


def signflip_p(d, side: str = "greater") -> dict:
    """Exact sign-flip permutation test on the pair means d_i: all 2^n sign patterns, statistic mean(s * d).
    side: 'greater' (mean(s d) >= observed), 'less', or 'two-sided' (|mean| >= |observed|)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n > 22:
        raise ValueError("exact enumeration limited to n <= 22")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    m = (signs * d).mean(axis=1)
    obs = float(d.mean())
    tol = 1e-12
    if side == "greater":
        p = float((m >= obs - tol).mean())
    elif side == "less":
        p = float((m <= obs + tol).mean())
    else:
        p = float((np.abs(m) >= abs(obs) - tol).mean())
    return {"side": side, "mean": obs, "n": n, "n_patterns": int(2 ** n), "p": p,
            "min_attainable_p": float(1 / 2 ** n) if side != "two-sided" else float(2 / 2 ** n)}


def t_ci(values, level: float = 0.95) -> dict | None:
    """t interval on the mean of the pair means; None for n < 4 (D-056)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 4:
        return None
    se = v.std(ddof=1) / np.sqrt(n)
    h = float(stats.t.ppf(0.5 + level / 2, n - 1) * se)
    return {"level": level, "df": n - 1, "lo": float(v.mean() - h), "hi": float(v.mean() + h), "se": float(se)}


def summarize(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"n": int(len(v)), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "t_ci95": t_ci(v)}


def mixed_lm(y, pair, x=None, xname: str = "x", claim: tuple[float, str] | None = None) -> dict:
    """statsmodels MixedLM, outcome y (point level), random intercept per pair; fixed part: intercept, or intercept + x.
    claim = (c, side) adds a Wald z test of the intercept against c (side 'below'/'above'), next to the default p (H0: 0)."""
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "pair": np.asarray(pair)})
    if x is not None:
        df[xname] = np.asarray(x, dtype=float)
    out = {"n_points": int(len(df)), "n_pairs": int(df["pair"].nunique()), "formula": f"y ~ {xname if x is not None else '1'} + (1 | pair)",
           "reml": True}
    if df["y"].nunique() < 2:
        out.update(status="not fitted: constant outcome", constant_value=float(df["y"].iloc[0]))
        return out
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = smf.mixedlm(f"y ~ {xname}" if x is not None else "y ~ 1", df, groups=df["pair"]).fit(reml=True)
        except Exception as e:   # reported, not hidden
            out.update(status=f"fit failed: {type(e).__name__}: {e}")
            return out
    var_u = float(res.cov_re.iloc[0, 0])
    out.update(status="fitted", converged=bool(res.converged), random_intercept_var=var_u, residual_var=float(res.scale),
               singular_random_effect=bool(var_u < 1e-8), warnings=sorted({str(x.message)[:160] for x in w}),
               coef={k: {"coef": float(res.params[k]), "se": float(res.bse[k]), "z": float(res.tvalues[k]), "p_two_sided_h0_zero": float(res.pvalues[k])}
                     for k in res.params.index if k != "Group Var"})
    if claim is not None and x is None:
        c, side = claim
        b, se = out["coef"]["Intercept"]["coef"], out["coef"]["Intercept"]["se"]
        z = (b - c) / se
        out["wald_vs_claim"] = {"c": c, "side": side, "z": float(z),
                                "p_one_sided": float(stats.norm.cdf(z) if side == "below" else stats.norm.sf(z))}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# row assembly
# ----------------------------------------------------------------------------------------------------------------------
def make_row(rid: str, table_row: str, quantity: str, pairs: list[dict], claim: dict, *, crosscheck=None, notes=None, mixed=None,
             extra=None) -> dict:
    """pairs: [{pair_id, value, ...}] one per pair (the unit). claim: kind in threshold | diff_pos | band | none."""
    vals = np.array([p["value"] for p in pairs], dtype=float)
    r = {"id": rid, "table3_row": table_row, "quantity": quantity, "pairs": pairs, "summary": summarize(vals), "claim": claim}
    k = claim["kind"]
    if k == "threshold":
        r["sign_test"] = sign_test(vals, claim["c"], claim["side"])
    elif k == "diff_pos":
        r["sign_test"] = sign_test(vals, 0.0, "above")
        r["signflip"] = signflip_p(vals, "greater")
    elif k == "band":
        inside = int((np.abs(vals) <= BAND + 1e-12).sum())
        r["within_band"] = {"band": BAND, "k": inside, "n": len(vals)}
        r["signflip"] = signflip_p(vals, "two-sided") if len(vals) <= 22 else None
    n = len(vals)
    r["pair_level_flag"] = (f"min attainable one-sided p = {0.5 ** n:.4g} with {n} pairs" +
                            ("; not confirmable at pair level (p < 0.05 unreachable)" if 0.5 ** n >= MIN_P_FLAG else ""))
    if crosscheck:
        r["crosscheck"] = crosscheck
    if notes:
        r["notes"] = notes
    if mixed is not None:
        r["mixedlm"] = mixed
    if extra:
        r.update(extra)
    return r


def xcheck(name: str, saved, recomputed, tol: float = 1e-9) -> dict:
    s, c = np.asarray(saved, dtype=float), np.asarray(recomputed, dtype=float)
    return {"name": name, "saved": s.tolist() if s.ndim else float(s), "recomputed": c.tolist() if c.ndim else float(c),
            "max_abs_diff": float(np.abs(s - c).max()), "match": bool(np.abs(s - c).max() <= tol)}


# ----------------------------------------------------------------------------------------------------------------------
# loaders (saved outputs only)
# ----------------------------------------------------------------------------------------------------------------------
def load_pairs(pairs_dir: Path) -> dict[int, tuple[dict, dict]]:
    """pair_id -> (json row, arrays) for every pair_*.npz that has a pair_*.json next to it."""
    out = {}
    for npz in sorted(Path(pairs_dir).glob("pair_*.npz")):
        js = npz.with_suffix(".json")
        if not js.exists():
            continue
        meta = json.loads(js.read_text())
        out[int(meta["pair_id"])] = (meta, dict(np.load(npz)))
    return out


def tree_sha(paths: list[Path]) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()
