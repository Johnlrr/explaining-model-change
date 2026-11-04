"""D-056 S1/S2 (saved outputs only): pair-level inference for the rows of Table 3 and runtime per query.

    uv run python -m dce.cli.pair_level run --out runs/experiments/exp_0058

No training, no solver run. Reads runs/experiments/exp_0017 (split A), exp_0019 (B), exp_0026 (grid), exp_0028 (Adult),
exp_0032, exp_0033, exp_0047 (cross-check), exp_0048, exp_0049 (D), exp_0050 and exp_0054 (ResNet-8). Writes results_s.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from dce.analysis import pair_level as pl
from dce.analysis.projection import project
from dce.cli.g0_tier0 import git_state
from dce.cli.review_0211 import k_index, perm_spearman, rank_metrics
from dce.eval.metrics import MethodOutput, recovery_indicator

R = Path("runs/experiments")
SPLIT_A, SPLIT_B, GRID, ADULT = R / "exp_0017", R / "exp_0019", R / "exp_0026", R / "exp_0028"
H7, H8, D, X1 = R / "exp_0032", R / "exp_0033", R / "exp_0049", R / "exp_0048"
RES_D, RES_E, E47 = R / "exp_0050", R / "exp_0054" / "test", R / "exp_0047"
PERM_N, PERM_SEED = 10_000, 5201          # exp_0047 config (same permutation test as E1a)
CODE_FILES = ("src/dce/analysis/pair_level.py", "src/dce/cli/pair_level.py", "tests/unit/test_pair_level.py", "pyproject.toml")


def jload(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def rec(a: dict, method: str, k, proj: np.ndarray | None = None) -> np.ndarray:
    o = MethodOutput(a[f"{method}__found"], a[f"{method}__dphi"])
    if proj is not None:
        o = project(o, proj)
    return recovery_indicator(o, k_index(k))


def union_found(a: dict) -> np.ndarray:
    return a["dce_N__found"] | a["dce_O__found"]


def prow(pid, value, **kw) -> dict:
    return {"pair_id": int(pid), "value": float(value), **kw}


# ----------------------------------------------------------------------------------------------------------------------
# abstention
# ----------------------------------------------------------------------------------------------------------------------
def abstention_rows() -> list[dict]:
    rows = []
    claim = {"kind": "threshold", "c": 0.05, "side": "below", "text": "witness share of the null queries <= 0.05 (the registered criterion), per pair"}
    # Null, B (exp_0019, pairs 9-16): per pair the N-or-O found rate saved in results.json (C2); arrays hold type N only
    b = pl.load_pairs(SPLIT_B / "pairs")
    c2 = jload(SPLIT_B / "results.json")["C2"]["pairs"]
    ids = sorted(int(k) for k in c2)
    arr_n = [float(b[i][1]["dce_N__found"].mean()) for i in ids]
    y = np.concatenate([b[i][1]["dce_N__found"] for i in ids]).astype(float)
    g = np.concatenate([[i] * len(b[i][1]["dce_N__found"]) for i in ids])
    rows.append(pl.make_row(
        "abst_null_B", "No witness on nulls | Null, B", "witness share (type N or O) per pair, exp_0019 results.json C2 found_rate_union",
        [prow(i, c2[str(i)]["found_rate_union"], n_points=int(b[i][0]["n_points"]), condition=b[i][0]["condition"]) for i in ids], claim,
        crosscheck=[pl.xcheck("type-N found rate from arrays (N only; type O is not in the arrays) vs union", [c2[str(i)]["found_rate_union"] for i in ids], arr_n)],
        mixed=pl.mixed_lm(y, g, claim=(0.05, "below")),
        notes="MixedLM outcome = type-N found indicator (type O arrays are not saved for split B); union rate in results.json is 0 in every pair."))
    # Adult (exp_0028, 5 pairs): h6.json union found rate; arrays hold N and O
    h6 = jload(ADULT / "h6.json")
    vals = []
    for p in h6["pairs"]:
        a = dict(np.load(ADULT / "pairs" / f"pair_{p['pair']}.npz"))
        vals.append((p["pair"], p["dce_union_found_rate"], float(union_found(a).mean())))
    rows.append(pl.make_row(
        "abst_adult", "No witness on nulls | Adult", "witness share (type N or O) per pair, exp_0028 h6.json",
        [prow(i, v, pair=n) for i, (n, v, _) in enumerate(vals)], claim,
        crosscheck=[pl.xcheck("union found rate from arrays vs h6.json", [v for _, v, _ in vals], [u for _, _, u in vals])],
        notes="5 pairs (< 8): no MixedLM. pair_id here is the index of B0..B4."))
    # Null, D (exp_0049, pairs 110-117)
    d = pl.load_pairs(D / "pairs")
    x3 = {p["pair_id"]: p for p in jload(D / "results_d.json")["x3"]["pairs"]}
    ids = sorted(x3)
    union = {i: union_found(d[i][1]) for i in ids}
    rows.append(pl.make_row(
        "abst_null_D", "No witness on unfiltered nulls | Null, D", "witness share (type N or O) per pair, exp_0049 results_d.json x3",
        [prow(i, x3[i]["witness_share_N_or_O"], n_points=int(d[i][0]["n_points"]), condition=d[i][0]["condition"]) for i in ids], claim,
        crosscheck=[pl.xcheck("union found rate from arrays vs results_d.json", [x3[i]["witness_share_N_or_O"] for i in ids],
                              [float(union[i].mean()) for i in ids])],
        mixed=pl.mixed_lm(np.concatenate([union[i] for i in ids]).astype(float), np.concatenate([[i] * len(union[i]) for i in ids]), claim=(0.05, "below"))))
    # ResNet-8: split D (exp_0050, pairs 128, 129; 100 queries each) and split E at kappa* = 2.5 (exp_0054 test)
    for rid, run, lab, kap in (("abst_resnet_D", RES_D, "ResNet-8, D | kappa = 0.5", None), ("abst_resnet_E", RES_E, "ResNet-8, E | kappa* = 2.5", 2.5)):
        pp = pl.load_pairs(run / "pairs")
        saved = {p["pair_id"]: p["witness_share"] for p in jload(run / ("results_x4.json" if run == RES_D else "results_x4_kstar.json"))["nulls"]}
        ids = sorted(saved)
        rows.append(pl.make_row(
            rid, f"No witness on nulls | {lab}", "witness share (type N or O) per pair (100 queries per pair)",
            [prow(i, union_found(pp[i][1]).mean(), n_points=int(pp[i][0]["n_points"])) for i in ids], claim,
            crosscheck=[pl.xcheck("union found rate from arrays vs results json", [saved[i] for i in ids], [float(union_found(pp[i][1]).mean()) for i in ids])],
            notes="2 pairs (< 8): no MixedLM."))
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# detection of a logit offset by CF-diff (exp_0032)
# ----------------------------------------------------------------------------------------------------------------------
def cfdiff_row() -> dict:
    res = jload(H7 / "results_h7.json")
    pairs = pl.load_pairs(H7 / "pairs")
    sel = sorted(i for i, (m, _) in pairs.items() if abs(m["b"]) < 1.0)    # |b| < 2 kappa = 1
    claim = {"kind": "threshold", "c": 0.90, "side": "above", "text": "CF-diff found rate > 0.90 per pair (H7c: every pair >= 0.90)"}
    vals = [float(pairs[i][1]["b1_det__found"].mean()) for i in sel]
    y = np.concatenate([pairs[i][1]["b1_det__found"] for i in sel]).astype(float)
    g = np.concatenate([[i] * 300 for i in sel])
    return pl.make_row(
        "cfdiff_flags", "CF-diff (no cut) flags |b| < 2 kappa | Logit offsets", "CF-diff found rate per pair (12 pairs: f_old seed 0-2 x b in {-0.9, -0.5, 0.5, 0.9})",
        [prow(i, v, b=pairs[i][0]["b"], seed=pairs[i][0]["seed"]) for i, v in zip(sel, vals)], claim,
        crosscheck=[pl.xcheck("per-pair found rate vs results_h7.json H7c.per_pair_found_b1_det (order of saved list)", res["H7c"]["per_pair_found_b1_det"], vals),
                    pl.xcheck("pair mean vs H7c estimate", res["H7c"]["estimate"], np.mean(vals))],
        mixed=pl.mixed_lm(y, g, claim=(0.90, "above")))


# ----------------------------------------------------------------------------------------------------------------------
# gradient differences and offsets
# ----------------------------------------------------------------------------------------------------------------------
def graddiff_offset_rows() -> list[dict]:
    rows = []
    g = pl.load_pairs(GRID / "pairs")
    res = jload(GRID / "results.json")
    h5 = res["H5"]
    # grid alpha = 0, pairs 24-27 (H5a)
    ids = h5["H5a"]["pairs"]
    vals = [float(rec(g[i][1], "b3", g[i][0]["k_star"]).mean()) for i in ids]
    e47 = jload(E47 / "results_e.json")["e1b_per_pair"]["graddiff_lt_0.2_grid_alpha0"]["per_pair"]
    rows.append(pl.make_row(
        "gd_lt02_grid", "Grad-diff < 0.2 | Grid, alpha = 0", "Recovery of Grad-diff per pair (k* = aspect), pairs 24-27",
        [prow(i, v, t=g[i][0]["t"], seed=g[i][0]["seed"]) for i, v in zip(ids, vals)],
        {"kind": "threshold", "c": 0.2, "side": "below", "text": "Grad-diff Recovery < 0.2 per pair"},
        crosscheck=[pl.xcheck("pair mean vs results.json H5a.estimate", h5["H5a"]["estimate"], np.mean(vals)),
                    pl.xcheck("per-pair vs exp_0047 e1b graddiff_lt_0.2_grid_alpha0", e47, vals)]))
    # split D offsets, 8 qualified pairs (exp_0049)
    d = pl.load_pairs(D / "pairs")
    x2 = jload(D / "results_d.json")["x2"]
    ids = sorted(p["pair_id"] for p in x2["pairs"] if p["qualified"])
    saved = {p["pair_id"]: p["graddiff_recovery"] for p in x2["pairs"]}
    vals = [float(rec(d[i][1], "b3", d[i][0]["k_star"]).mean()) for i in ids]
    y = np.concatenate([rec(d[i][1], "b3", d[i][0]["k_star"]) for i in ids])
    gg = np.concatenate([[i] * 300 for i in ids])
    rows.append(pl.make_row(
        "gd_lt02_D", "Grad-diff < 0.2 | Offsets, D", "Recovery of Grad-diff per pair (k* = aspect), 8 qualified offset pairs (pair 103 not qualified)",
        [prow(i, v, t=d[i][0]["t"], seed=d[i][0]["seed"]) for i, v in zip(ids, vals)],
        {"kind": "threshold", "c": 0.2, "side": "below", "text": "Grad-diff Recovery < 0.2 per pair"},
        crosscheck=[pl.xcheck("per-pair vs results_d.json x2.graddiff_recovery", [saved[i] for i in ids], vals),
                    pl.xcheck("sign-test p vs the saved 0.0039 (exact 2^-8)", 0.5 ** 8, pl.sign_test(vals, 0.2, "below")["p_one_sided"])],
        mixed=pl.mixed_lm(y, gg, claim=(0.2, "below"))))
    # H5c: Rec(0.25, 0.60) - Rec(0.25, 0.50), colour; pairs a = 42, 43 and b = 36, 37, matched by f_old seed
    a_ids, b_ids = h5["H5c"]["pairs_a"], h5["H5c"]["pairs_b"]
    ra = {g[i][0]["seed"]: float(rec(g[i][1], "b3", "color").mean()) for i in a_ids}
    rb = {g[i][0]["seed"]: float(rec(g[i][1], "b3", "color").mean()) for i in b_ids}
    ds = sorted(set(ra) & set(rb))
    byseed_a = {g[i][0]["seed"]: i for i in a_ids}
    byseed_b = {g[i][0]["seed"]: i for i in b_ids}
    rows.append(pl.make_row(
        "gd_offset_free", "Grad-diff offset-free | Grid", "Rec(Grad-diff; alpha 0.25, t 0.60) - Rec(Grad-diff; 0.25, 0.50), k* = colour, per f_old seed (2 matched pairs)",
        [prow(byseed_a[s], ra[s] - rb[s], seed=s, pair_t060=byseed_a[s], pair_t050=byseed_b[s], rec_t060=ra[s], rec_t050=rb[s]) for s in ds],
        {"kind": "band", "text": "|difference| <= 0.10 (equivalence); cells matched by f_old seed (a design choice, D-056 does not fix the matching)"},
        crosscheck=[pl.xcheck("mean of the 2 matched differences vs results.json H5c.estimate", h5["H5c"]["estimate"], np.mean([ra[s] - rb[s] for s in ds]))],
        notes="2 pairs per cell. No CI (n < 4)."))
    # H5b: Rec(alpha 0.5) - Rec(alpha 0.1) at t = 0.5, colour, matched by seed; and the 24-pair rotation analysis
    a_ids, b_ids = h5["H5b"]["pairs_a"], h5["H5b"]["pairs_b"]
    ra = {g[i][0]["seed"]: (i, float(rec(g[i][1], "b3", "color").mean())) for i in a_ids}
    rb = {g[i][0]["seed"]: (i, float(rec(g[i][1], "b3", "color").mean())) for i in b_ids}
    ds = sorted(set(ra) & set(rb))
    rows.append(pl.make_row(
        "gd_rises_contrast", "Grad-diff rises with alpha | Grid (registered contrast H5b)", "Rec(Grad-diff; alpha 0.5, t 0.5) - Rec(Grad-diff; alpha 0.1, t 0.5), k* = colour, per f_old seed (2 matched pairs)",
        [prow(ra[s][0], ra[s][1] - rb[s][1], seed=s, pair_alpha05=ra[s][0], pair_alpha01=rb[s][0], rec_alpha05=ra[s][1], rec_alpha01=rb[s][1]) for s in ds],
        {"kind": "diff_pos", "text": "difference > 0"},
        crosscheck=[pl.xcheck("mean of the 2 matched differences vs results.json H5b.estimate", h5["H5b"]["estimate"], np.mean([ra[s][1] - rb[s][1] for s in ds]))],
        notes="2 pairs per cell. No CI (n < 4)."))
    rows.append(rotation_row(g))
    return rows


def rotation_row(g) -> dict:
    """24 alpha > 0 grid pairs: per-pair Grad-diff Recovery vs measured alpha_hat (Spearman, permutation within f_old seed as in exp_0047)
    and the MixedLM Recovery ~ alpha_hat + (1 | pair) on the points."""
    rot = [(m, a) for _, (m, a) in sorted(g.items()) if m["alpha"] > 0]
    x = np.array([m["alpha_hat"] for m, _ in rot])
    pts = [rec(a, "b3", m["k_star"]) for m, a in rot]
    y = np.array([float(p.mean()) for p in pts])
    y_rank = np.array([rank_metrics(a["b3__found"], a["b3__dphi"], k_index(m["k_star"]))["recovery"] for m, a in rot])
    seeds = np.array([m["seed"] for m, _ in rot])
    rho, p = perm_spearman(x, y_rank, seeds, PERM_N, PERM_SEED)
    e = jload(E47 / "results_e.json")["e1a_rotation"]
    mixed = pl.mixed_lm(np.concatenate(pts), np.concatenate([[m["pair_id"]] * len(p_) for (m, _), p_ in zip(rot, pts)]),
                        x=np.concatenate([[m["alpha_hat"]] * len(p_) for (m, _), p_ in zip(rot, pts)]), xname="alpha_hat")
    by_alpha = {}
    for (m, _), v in zip(rot, y):
        by_alpha.setdefault(m["alpha"], []).append(float(v))
    return {
        "id": "gd_rises_rotation", "table3_row": "Grad-diff rises with alpha | Grid (rho = 0.89)",
        "quantity": "Recovery of Grad-diff per pair (k* = colour) against measured alpha_hat, 24 pairs with alpha > 0",
        "pairs": [prow(m["pair_id"], v, alpha=m["alpha"], t=m["t"], seed=m["seed"], alpha_hat=float(xi)) for (m, _), v, xi in zip(rot, y, x)],
        "summary": pl.summarize(y),
        "claim": {"kind": "monotone", "text": "Recovery increases with alpha_hat (Spearman rho > 0)"},
        "spearman": {"rho": rho, "perm_p_one_sided": p, "n_perm": PERM_N, "perm_seed": PERM_SEED, "strata": "f_old seed",
                     "note": "Monte Carlo permutation (the strata have 12! orderings each; exact enumeration is infeasible); same call as exp_0047 e1a"},
        "by_alpha": {str(a): {"n": len(v), "min": min(v), "max": max(v), "mean": float(np.mean(v))} for a, v in sorted(by_alpha.items())},
        "pair_level_flag": "24 pairs; the Monte Carlo p has a floor of 1/(n_perm + 1)",
        "crosscheck": [pl.xcheck("Spearman rho vs exp_0047 e1a_rotation", e["spearman_rho"], rho), pl.xcheck("permutation p vs exp_0047", e["perm_p_one_sided"], p),
                       pl.xcheck("per-pair Recovery (rank-based) vs exp_0047 graddiff_recovery", [q["graddiff_recovery"] for q in e["pairs"]], y_rank),
                       pl.xcheck("per-pair Recovery (recovery_indicator) vs rank-based", y_rank, y)],
        "mixedlm": mixed}


# ----------------------------------------------------------------------------------------------------------------------
# Delta-CE versus gradient differences
# ----------------------------------------------------------------------------------------------------------------------
def dce_minus_gd(a: dict, k, proj: np.ndarray | None = None) -> float:
    return float(rec(a, "dce_N", k, proj).mean() - rec(a, "b3", k, proj).mean())


def diff_rows() -> list[dict]:
    A, B = pl.load_pairs(SPLIT_A / "pairs"), pl.load_pairs(SPLIT_B / "pairs")
    e47 = jload(E47 / "results_e.json")["e1b_per_pair"]
    cond_ids = lambda P, c: sorted(i for i, (m, _) in P.items() if m["condition"] == c)
    dA = {c: {i: dce_minus_gd(A[i][1], A[i][0]["k_star"]) for i in cond_ids(A, c)} for c in ("T1", "T3", "T4")}
    dB = {c: {i: dce_minus_gd(B[i][1], B[i][0]["k_star"]) for i in cond_ids(B, c)} for c in ("T1", "T4")}
    c1, c19 = jload(SPLIT_A / "results.json"), jload(SPLIT_B / "results.json")
    rows = []
    name = {"T1": "shortcut", "T3": "subgroup", "T4": "threshold"}
    pos = {"kind": "diff_pos", "text": "Delta-CE Recovery - Grad-diff Recovery > 0"}
    band = {"kind": "band", "text": "|Delta-CE - Grad-diff| <= 0.10 (tie)"}

    def prs(d, **kw):
        return [prow(i, v, **kw) for i, v in sorted(d.items())]

    pool = lambda c: {i: (dA[c][i] + dB[c][i]) / 2 for i in dA[c]}
    # threshold: B (the row), A, pooled A+B (per-pair mean over the two splits; same model pairs)
    rows.append(pl.make_row("dce_gt_gd_threshold_B", "Delta-CE > Grad-diff | Threshold, B", "Recovery(Delta-CE) - Recovery(Grad-diff) per pair, split B, pairs 6-8 (k* = aspect)",
                            prs(dB["T4"]), pos, crosscheck=[pl.xcheck("per-pair vs exp_0047 e1b",e47["dce_minus_graddiff_threshold_splitB"]["per_pair"], list(dB["T4"].values()))]))
    rows.append(pl.make_row("dce_gt_gd_threshold_A", "Delta-CE > Grad-diff | Threshold, A (same claim, locked split)", "same quantity, split A",
                            prs(dA["T4"]), pos, crosscheck=[pl.xcheck("per-pair vs exp_0047 e1b", e47["dce_minus_graddiff_threshold_splitA"]["per_pair"], list(dA["T4"].values()))]))
    rows.append(pl.make_row("dce_gt_gd_threshold_AB", "Delta-CE > Grad-diff | Threshold, A+B pooled", "per-pair mean of the split-A and split-B differences (same models, two data sets)",
                            prs(pool("T4")), pos, notes="Pooling does not add pairs: A and B use the same 3 model pairs (same checkpoints, different test data)."))
    # subgroup: split A only (T3 is not in split B)
    rows.append(pl.make_row("dce_gt_gd_subgroup", "Delta-CE > Grad-diff | Subgroup", "Recovery(Delta-CE) - Recovery(Grad-diff) per pair, split A, pairs 3-5 (k* = size)",
                            prs(dA["T3"]), pos, crosscheck=[pl.xcheck("per-pair vs exp_0047 e1b", e47["dce_minus_graddiff_subgroup_splitA"]["per_pair"], list(dA["T3"].values()))],
                            notes="Claimed side: positive; the saved row reads 0/3."))
    # shortcut (not in Table 3 as a separate difference row; used by the tie row)
    for c in ("T1", "T4"):
        ids_pool = pool(c)
        rows.append(pl.make_row(f"tie_{name[c]}_A", f"Tie (|Delta| <= 0.1) | {name[c].capitalize()}, A", f"Recovery(Delta-CE) - Recovery(Grad-diff) per pair, split A", prs(dA[c]), band,
                                crosscheck=[pl.xcheck("per-pair vs exp_0047 e1b", e47[f"dce_minus_graddiff_{name[c]}_splitA"]["per_pair"], list(dA[c].values()))]))
        rows.append(pl.make_row(f"tie_{name[c]}_B", f"Tie (|Delta| <= 0.1) | {name[c].capitalize()}, B", "same, split B", prs(dB[c]), band,
                                crosscheck=[pl.xcheck("per-pair vs exp_0047 e1b", e47[f"dce_minus_graddiff_{name[c]}_splitB"]["per_pair"], list(dB[c].values()))]))
        rows.append(pl.make_row(f"tie_{name[c]}_AB", f"Tie (|Delta| <= 0.1) | {name[c].capitalize()}, A+B pooled", "per-pair mean of A and B", prs(ids_pool), band))
    # projected rows
    cs = jload(SPLIT_B / "results.json")["C3"]
    ids = cond_ids(B, "T1")
    vals = [dce_minus_gd(B[i][1], B[i][0]["k_star"], B[i][1]["grad_old"]) for i in ids]
    rows.append(pl.make_row("proj_shortcut_B", "Proj. Delta-CE ~ Grad-diff | Shortcut, B", "projected Recovery(Delta-CE) - projected Recovery(Grad-diff) per pair, split B, pairs 0-2 (k* = colour)",
                            [prow(i, v, seed=B[i][0]["seed"]) for i, v in zip(ids, vals)],
                            {"kind": "band", "text": "|difference| <= 0.10 (equivalence)"},
                            crosscheck=[pl.xcheck("per-pair vs exp_0019 results.json C3.seed_diffs_projected", [cs["seed_diffs_projected"][str(B[i][0]["seed"])] for i in ids], vals)]))
    G = pl.load_pairs(GRID / "pairs")
    h5d = jload(GRID / "results.json")["H5"]["H5d"]
    vals = [dce_minus_gd(G[i][1], G[i][0]["k_star"], G[i][1]["grad_old"]) for i in h5d["pairs"]]
    rows.append(pl.make_row("proj_grid", "Proj. Delta-CE ~ Grad-diff | grid", "projected Recovery(Delta-CE) - projected Recovery(Grad-diff) per pair, grid pairs 36, 37, 44, 45 (k* = colour)",
                            [prow(i, v, alpha=G[i][0]["alpha"], seed=G[i][0]["seed"]) for i, v in zip(h5d["pairs"], vals)],
                            {"kind": "band", "text": "|difference| <= 0.10 (equivalence)"},
                            crosscheck=[pl.xcheck("pair mean vs results.json H5d.estimate", h5d["estimate"], np.mean(vals))]))
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# H8: normal at z*
# ----------------------------------------------------------------------------------------------------------------------
def h8_row() -> dict:
    from dce.analysis.v110 import CELLS, REFERENCE, REGISTERED, h8_indicators

    res = jload(H8 / "results_h8.json")["H8"]
    P = pl.load_pairs(H8 / "pairs")
    ids = res["pairs"]
    out, gap_ref, gap_reg = [], [], []
    for i in ids:
        a = P[i][1]
        dce = MethodOutput(a["dce__found"], a["dce__dphi"])
        b3 = {k: MethodOutput(a[f"b3_{k}__found"], a[f"b3_{k}__dphi"]) for k in ("z0", "zs")}
        ind = h8_indicators(dce, b3, {"z0": a["grad_z0"], "zs": a["grad_zs"]}, k_index("color"))
        gaps = {c: float(ind[c][0].mean() - ind[c][1].mean()) for c in CELLS}
        gap_ref.append(gaps[REFERENCE]), gap_reg.append(gaps[REGISTERED])
        out.append(prow(i, abs(gaps[REFERENCE]) - 2 * abs(gaps[REGISTERED]), support=int(dce.found.sum()), gap_z0_z0=gaps[REFERENCE],
                        gap_zs_z0=gaps[REGISTERED], alpha=P[i][0]["alpha"], seed=P[i][0]["seed"]))
    saved_ref, saved_reg = res["gap"]["z0,z0"], res["gap"]["zs,z0"]
    row = pl.make_row(
        "normal_zstar", "Normal at z* halves the gap or more | Grid, new", "T_i = |gap(z0,z0)| - 2 |gap(z*,z0)| per pair (T_i > 0: the gap narrows by at least a half), 4 pairs, support = queries with a witness",
        out, {"kind": "threshold", "c": 0.0, "side": "above", "text": "T_i > 0"},
        crosscheck=[pl.xcheck("mean over pairs of gap(z0,z0) vs results_h8.json H8.gap[z0,z0]", saved_ref, np.mean(gap_ref)),
                    pl.xcheck("mean over pairs of gap(z*,z0) vs results_h8.json H8.gap[z*,z0]", saved_reg, np.mean(gap_reg)),
                    pl.xcheck("registered T = |mean gap(z0,z0)| - 2 |mean gap(z*,z0)| from the pair means vs H8.T", res["T"], abs(np.mean(gap_ref)) - 2 * abs(np.mean(gap_reg)))],
        notes="T_i is the pair-level analogue of the registered T (which uses the pooled gaps); the mean of the T_i differs from H8.T because |.| is taken per pair.",
        extra={"gap_zs_z0_summary": pl.summarize(gap_reg), "gap_z0_z0_summary": pl.summarize(gap_ref)})
    return row


# ----------------------------------------------------------------------------------------------------------------------
# S2
# ----------------------------------------------------------------------------------------------------------------------
def s2() -> dict:
    out = {"runs": {}, "note": "time per query is wall_s_per_point (pair wall time / n points; the solver ran with parallel workers) and cpu_s_per_point (solver CPU seconds / n points). No forward/backward counts are saved in any pair json."}
    for label, run in (("A exp_0017", SPLIT_A), ("B exp_0019", SPLIT_B), ("grid exp_0026", GRID), ("H7 exp_0032", H7), ("H8 exp_0033", H8), ("D exp_0049", D)):
        P = pl.load_pairs(run / "pairs")
        acc = {}
        for i, (m, _) in sorted(P.items()):
            for meth, rt in (m.get("runtime") or {}).items():
                for k, v in rt.items():
                    if isinstance(v, (int, float)) and v is not None:
                        acc.setdefault(meth, {}).setdefault(k, []).append(float(v))
        out["runs"][label] = {"n_pairs": len(P), "n_points_per_pair": sorted({int(m["n_points"]) for m, _ in P.values()}),
                              "methods": {meth: {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n_pairs": len(v)}
                                                 for k, v in d.items()} for meth, d in sorted(acc.items())}}
    # X1 (exp_0048): total seconds per pair for IG, SHAP, Imp-diff(adapt), Seg-search x10
    P = pl.load_pairs(X1 / "pairs")
    acc = {}
    for i, (m, a) in sorted(P.items()):
        for meth, s in (m.get("runtime_s") or {}).items():
            acc.setdefault(meth, []).append(float(s) / int(m["n_points"]))
    out["runs"]["X1 exp_0048"] = {"n_pairs": len(P), "methods": {k: {"wall_s_per_point_from_pair_total": {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n_pairs": len(v)}} for k, v in acc.items()},
                                  "note": "runtime_s holds the total seconds per pair; divided by n_points here. Pairs ran on different machines (D-052)."}
    out["runs"]["ResNet exp_0050 / exp_0054"] = {"note": "no runtime fields saved (chunked local run); not available"}
    # shared parameters from the configs
    tier = yaml.safe_load(Path("configs/tiers/tier0.yaml").read_text())
    cfg_params = {}
    for exp in ("exp_0017", "exp_0019", "exp_0026", "exp_0032", "exp_0033", "exp_0048", "exp_0049", "exp_0050_run", "exp_0054"):
        c = yaml.safe_load(Path(f"configs/experiments/{exp}.yaml").read_text())
        cfg_params[exp] = {k: c[k] for k in ("n_points", "kappa", "r", "tier_config", "decision") if k in c}
    out["shared_parameters"] = {"tier0_yaml": {k: tier[k] for k in ("kappa", "r", "solver", "seeds") if k in tier}, "per_experiment_config_overrides": cfg_params,
                                "solver_note": "solver block: R restarts, T steps, lr, lam0, mu, sigma, lam_schedule/n_lam_steps, polish (SLSQP); kappa and r as in tier0.yaml unless listed per experiment"}
    return out


# ----------------------------------------------------------------------------------------------------------------------
def inputs_record() -> dict:
    rec_ = {}
    for run in (SPLIT_A, SPLIT_B, GRID, ADULT, H7, H8, D, X1, RES_D, RES_E, E47):
        files = [p for p in Path(run).glob("pairs/pair_*") if p.is_file()] + [p for p in Path(run).glob("*.json") if p.is_file() and p.name.startswith(("results", "h6"))]
        if run == RES_E:
            files += list(Path(run).glob("results_*.json"))
        rec_[str(run)] = {"n_files": len(files), "tree_sha256": pl.tree_sha(files)}
    return rec_


def git_info() -> dict:
    g = git_state()
    porcelain = subprocess.run(["git", "status", "--porcelain", "src", "configs", "tests", "pyproject.toml"], capture_output=True, text=True).stdout.splitlines()
    mine = subprocess.run(["git", "status", "--porcelain", "--", *CODE_FILES], capture_output=True, text=True).stdout.strip()
    return {**g, "dirty_files_src_configs_tests_pyproject": [l[3:] for l in porcelain], "analysis_code_files_clean": not mine}


def run(out: Path) -> int:
    path = out / "results_s.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    rows = abstention_rows() + [cfdiff_row()] + graddiff_offset_rows() + diff_rows() + [h8_row()]
    for r in rows:
        bad = [c for c in r.get("crosscheck", []) if not c["match"]]
        r["crosscheck_all_match"] = not bad if r.get("crosscheck") else None
    res = {"label": "D-056 S1/S2 (saved outputs only, descriptive, not registered)",
           "s1": {"unit": "pair", "rows": rows,
                  "rules": "exact one-sided sign test over pairs; exact sign-flip permutation (2^n patterns) on pair means for differences; t CI for n >= 4; "
                           "MixedLM (REML, random intercept per pair) for rows with >= 8 pairs; no bootstrap, no GLMM (D-056)"},
           "s2": s2(), "inputs": inputs_record(), "git": git_info()}
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2, default=float))
    for r in rows:
        s = r["summary"]
        print(f"{r['id']:26s} n={s['n']:2d} mean={s['mean']:+.4f} [{s['min']:+.3f},{s['max']:+.3f}] "
              f"sign={r.get('sign_test', {}).get('k')}/{r.get('sign_test', {}).get('n')} p={r.get('sign_test', {}).get('p_one_sided')} "
              f"xcheck={r['crosscheck_all_match']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("run",))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return run(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
