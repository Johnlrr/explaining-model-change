"""D-036 / D-038 / D-039: analysis of the v1.9 Stage A test run (H5a–H5d, §5.3 consistency, descriptive (i)–(iii)).

Resampling follows dce.eval.stats: points are the unit, stratified by pair. A statistic is a mean over pairs of
per-pair means. Within one cell set and two methods on the same points (H5d) the indices are shared (paired); across
different cells (H5b, H5c) the pairs have different test points, so each pair is resampled independently.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dce.analysis.projection import project
from dce.eval.metrics import MethodOutput, recovery_indicator
from dce.eval.stats import holm

K_INDEX = {"aspect": 0, "color": 2}
TOST_MARGIN = 0.10
H5A_MAX = 0.20
SEEDS = {"H5a": 6500, "H5b": 6501, "H5c": 6502, "H5d": 6503}   # 6500 + index (D-039, as D-028)
CONSISTENCY = {"b3_recovery_max": 0.20, "dce_found_rate": (0.30, 0.65)}


def boot_mean(groups: Sequence[np.ndarray], rng: np.random.Generator, B: int) -> tuple[float, np.ndarray]:
    """Mean over pairs of the per-pair mean, and its B bootstrap replicates (each pair resampled on its own)."""
    point, stat = 0.0, np.zeros(B)
    for g in groups:
        g = np.asarray(g, dtype=float)
        point += g.mean() / len(groups)
        stat += g[rng.integers(0, g.size, size=(B, g.size))].mean(axis=1) / len(groups)
    return float(point), stat


def boot_paired(a: Sequence[np.ndarray], b: Sequence[np.ndarray], rng, B) -> tuple[float, np.ndarray]:
    """Same scheme as dce.eval.stats.paired_bootstrap (shared indices per pair)."""
    point, stat = 0.0, np.zeros(B)
    for x, y in zip(a, b):
        point += (x.mean() - y.mean()) / len(a)
        idx = rng.integers(0, x.size, size=(B, x.size))
        stat += (x[idx].mean(axis=1) - y[idx].mean(axis=1)) / len(a)
    return float(point), stat


def two_sided_p(stat: np.ndarray) -> float:
    return float(max(min(1.0, 2 * min((stat <= 0).mean(), (stat >= 0).mean())), 1.0 / stat.size))


def tost_p(stat: np.ndarray, margin: float = TOST_MARGIN) -> float:
    return float(max((stat <= -margin).mean(), (stat >= margin).mean(), 1.0 / stat.size))


def outputs(arr: dict, method: str) -> MethodOutput:
    return MethodOutput(arr[f"{method}__found"], arr[f"{method}__dphi"])


def rec(arr: dict, method: str, k: str, projected: bool = False) -> np.ndarray:
    o = outputs(arr, method)
    if projected:
        o = project(o, arr["grad_old"])
    return recovery_indicator(o, K_INDEX[k])


def cell_pairs(pairs: dict, cells, exclude=()) -> list[tuple[dict, dict]]:
    """(row, arrays) of every pair in the given (α, t) cells, minus excluded pair_ids, ordered by pair_id."""
    return [pairs[p] for p in sorted(pairs) if (pairs[p][0]["alpha"], pairs[p][0]["t"]) in cells and p not in exclude]


def exclusions_d038(grid: dict, pair_cells: dict) -> dict:
    """D-038 reading of v1.9.1: (2b) drops no pair; every test drops pairs failing (1)/(3)/(2a); H5c also drops the
    seeds failing (2c). `grid` is `grid_checks_v191` output, `pair_cells` maps pair_id → (α, t, seed)."""
    from dce.analysis.qualification import TESTS

    per = {int(k): v for k, v in grid["per_pair"].items()}
    c2c = {int(k): v for k, v in grid["c2c"].items()}
    out = {}
    for name, cells in TESTS.items():
        rows = []
        for pid, (a, t, s) in sorted(pair_cells.items()):
            if (a, t) not in cells:
                continue
            why = [] if per[pid]["base_qualified"] else ["not qualified (1)/(3)/(2a)"]
            if name == "H5c" and not c2c.get(s, {}).get("ok", False):
                why.append("(2c)")
            if why:
                rows.append({"pair_id": pid, "alpha": a, "t": t, "seed": s, "reasons": why})
        out[name] = rows
    return out


def consistency(pairs: dict) -> dict:
    """v1.9 §5.3 on cell (0, 0.60): pooled Recovery(B3) ≤ 0.20 and Δ-CE found-rate in [0.30, 0.65]."""
    sel = cell_pairs(pairs, {(0.0, 0.60)})
    b3 = [float(rec(a, "b3", "aspect").mean()) for _, a in sel]
    fr = [float(a["dce_N__found"].mean()) for _, a in sel]
    lo, hi = CONSISTENCY["dce_found_rate"]
    pooled_b3, pooled_fr = float(np.mean(b3)), float(np.mean(fr))
    return {"pairs": [r["pair_id"] for r, _ in sel], "b3_recovery_per_pair": b3, "dce_found_rate_per_pair": fr,
            "b3_recovery": pooled_b3, "dce_found_rate": pooled_fr, "locked_t4_reference": {"b3_recovery": 0.07, "dce_found_rate": 0.47},
            "pass": bool(pooled_b3 <= CONSISTENCY["b3_recovery_max"] and lo <= pooled_fr <= hi)}


def h5(pairs: dict, exclusions: dict, B: int = 10_000) -> dict:
    """H5a–H5d (v1.9 §3, D-038 exclusions, D-039 test statistics); Holm over the four p-values (m = 4)."""
    res, pv = {}, {}

    def groups(name, cells, fn):
        sel = cell_pairs(pairs, cells, {e["pair_id"] for e in exclusions.get(name, [])})
        return sel, [fn(r, a) for r, a in sel]

    # H5a: pooled Recovery(B3) on α = 0, t > 0.5, k* = aspect; one-sided p for mean ≥ 0.20
    sel, g = groups("H5a", {(0.0, 0.53), (0.0, 0.56), (0.0, 0.60)}, lambda r, a: rec(a, "b3", "aspect"))
    if sel:
        point, stat = boot_mean(g, np.random.default_rng(SEEDS["H5a"]), B)
        lo, hi = np.quantile(stat, [0.025, 0.975])
        pv["H5a"] = float(max((stat >= H5A_MAX).mean(), 1.0 / B))
        res["H5a"] = {"pairs": [r["pair_id"] for r, _ in sel], "estimate": point, "ci95": [float(lo), float(hi)],
                      "p_one_sided": pv["H5a"], "criterion": "ci95 upper <= 0.20 and p_holm < 0.05"}
    # H5b: Rec(B3; 0.5, 0.5) − Rec(B3; 0.1, 0.5), k* = colour; independent cells
    sa, ga = groups("H5b", {(0.5, 0.50)}, lambda r, a: rec(a, "b3", "color"))
    sb, gb = groups("H5b", {(0.1, 0.50)}, lambda r, a: rec(a, "b3", "color"))
    if sa and sb:
        rng = np.random.default_rng(SEEDS["H5b"])
        (pa, st_a), (pb, st_b) = boot_mean(ga, rng, B), boot_mean(gb, rng, B)
        stat = st_a - st_b
        lo, hi = np.quantile(stat, [0.025, 0.975])
        pv["H5b"] = two_sided_p(stat)
        res["H5b"] = {"pairs_a": [r["pair_id"] for r, _ in sa], "pairs_b": [r["pair_id"] for r, _ in sb],
                      "rec_a": pa, "rec_b": pb, "estimate": pa - pb, "ci95": [float(lo), float(hi)], "p_two_sided": pv["H5b"],
                      "criterion": "ci95 lower > 0 and p_holm < 0.05"}
    # H5c: Rec(B3; 0.25, 0.60) − Rec(B3; 0.25, 0.50), k* = colour; TOST ±0.10 with CI90
    sa, ga = groups("H5c", {(0.25, 0.60)}, lambda r, a: rec(a, "b3", "color"))
    sb, gb = groups("H5c", {(0.25, 0.50)}, lambda r, a: rec(a, "b3", "color"))
    if sa and sb:
        rng = np.random.default_rng(SEEDS["H5c"])
        (pa, st_a), (pb, st_b) = boot_mean(ga, rng, B), boot_mean(gb, rng, B)
        stat = st_a - st_b
        lo, hi = np.quantile(stat, [0.05, 0.95])
        pv["H5c"] = tost_p(stat)
        res["H5c"] = {"pairs_a": [r["pair_id"] for r, _ in sa], "pairs_b": [r["pair_id"] for r, _ in sb],
                      "rec_a": pa, "rec_b": pb, "estimate": pa - pb, "ci90": [float(lo), float(hi)], "p_tost": pv["H5c"],
                      "criterion": "ci90 within [-0.10, 0.10] and p_holm < 0.05"}
    # H5d: Rec_proj(Δ-CE) − Rec_proj(B3), α ∈ {0.25, 0.5}, t = 0.5, k* = colour; paired; TOST ±0.10 with CI90
    sel, ga = groups("H5d", {(0.25, 0.50), (0.5, 0.50)}, lambda r, a: rec(a, "dce_N", "color", projected=True))
    _, gb = groups("H5d", {(0.25, 0.50), (0.5, 0.50)}, lambda r, a: rec(a, "b3", "color", projected=True))
    if sel:
        point, stat = boot_paired(ga, gb, np.random.default_rng(SEEDS["H5d"]), B)
        lo, hi = np.quantile(stat, [0.05, 0.95])
        pv["H5d"] = tost_p(stat)
        unproj = float(np.mean([rec(a, "dce_N", "color").mean() - rec(a, "b3", "color").mean() for _, a in sel]))
        res["H5d"] = {"pairs": [r["pair_id"] for r, _ in sel], "estimate": point, "ci90": [float(lo), float(hi)],
                      "p_tost": pv["H5d"], "diff_unprojected": unproj,
                      "criterion": "ci90 within [-0.10, 0.10] and p_holm < 0.05"}
    adj = holm(pv)
    for k, v in res.items():
        v["p_holm"] = adj[k][0]
        if k == "H5a":
            v["pass"] = bool(v["ci95"][1] <= H5A_MAX and v["p_holm"] < 0.05)
        elif k == "H5b":
            v["pass"] = bool(v["ci95"][0] > 0 and v["p_holm"] < 0.05)
        else:
            v["pass"] = bool(-TOST_MARGIN <= v["ci90"][0] and v["ci90"][1] <= TOST_MARGIN and v["p_holm"] < 0.05)
    for k in ("H5a", "H5b", "H5c", "H5d"):
        res.setdefault(k, {"status": "not testable (no pairs left)"})
    res["holm_m"] = len(pv)
    return res


def descriptive(pairs: dict) -> dict:
    """Per-pair table and the descriptive predictions (i)–(iii) of v1.9 §3 (not tested)."""
    rows = []
    for pid in sorted(pairs):
        r, a = pairs[pid]
        row = {k: r[k] for k in ("pair_id", "alpha", "t", "seed", "alpha_hat", "t_hat", "k_star")}
        for m in ("dce_N", "b1_det", "b3", "a2"):
            row[f"found_{m}"] = float(a[f"{m}__found"].mean())
            for k in ("aspect", "color"):
                row[f"rec_{m}_{k}"] = float(rec(a, m, k).mean())
            if r["k_star"]:
                row[f"rec_proj_{m}"] = float(rec(a, m, r["k_star"], projected=True).mean())
        for kap in ("0", "0.25", "1.0"):
            if f"dce_N_k{kap}__found" in a:
                row[f"found_dce_N_k{kap}"] = float(a[f"dce_N_k{kap}__found"].mean())
        y1 = a["y"] == 1
        row["dce_found_y1"] = float(a["dce_N__found"][y1].mean()) if y1.any() else None
        row["dce_found_y0"] = float(a["dce_N__found"][~y1].mean()) if (~y1).any() else None
        rows.append(row)
    row0 = [r for r in rows if r["alpha"] == 0.0]
    return {"pairs": rows,
            "i_found_rate_alpha0_by_t": {t: float(np.mean([r["found_dce_N"] for r in row0 if r["t"] == t]))
                                         for t in sorted({r["t"] for r in row0})},
            "ii_alpha0_witness_only_y1": {r["pair_id"]: r["dce_found_y0"] == 0.0 for r in row0},
            "iii_a2_recovery_alpha0": {r["pair_id"]: r["rec_a2_aspect"] for r in row0}}
