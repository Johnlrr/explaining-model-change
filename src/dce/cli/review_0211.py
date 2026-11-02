"""D-052 (advisor review 2026-11-02): additional analyses, rules fixed in research/DECISIONS.md before any number.

m-analyze    — exp_0046 (saved z0 plus hash-verified checkpoints, no solver run): margin control for detection.
               s_M(z0) = |h(z0)|, h = Δℓ_new − Δℓ_old, Δℓ = ℓ_1 − ℓ_0; s_M,r = max |h| over 256 uniform points in the
               r-ball ∩ [0, 1]^5 (plus z0). Thresholds on the exp_0017 nulls 9–16; readout on the exp_0032 logit
               offsets (per |b|) and the 7 fine-tuned offset pairs of exp_0044, next to Δ-CE type N and CF-diff.

e-analyze    — exp_0047 (saved outputs only): E1 pair-level inference, E2 baseline diagnosis (Imp-diff vectors,
               Seg-search found-rate), E3 rank metrics and set-recovery, E4 κ and r sensitivity; plus a check that
               Table 2 is reproduced from the same arrays.

Usage:
    python -m dce.cli.review_0211 m-analyze --config configs/experiments/exp_0046.yaml --out runs/experiments/exp_0046
    python -m dce.cli.review_0211 e-analyze --config configs/experiments/exp_0047.yaml --out runs/experiments/exp_0047
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dce.cli.g0_reeval import load_verified
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.cli.review_d048 import ASPECT, cfdiff_score, cmd_json
from dce.cli.v19_grid import grid_pairs
from dce.cli.v19_test import load_models
from dce.data.renderer import render
from dce.eval.pairs import load_pair
from dce.models.label_shift import LogitShift


@torch.no_grad()
def logit_gap(f, z: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Δℓ = ℓ_1 − ℓ_0 of f at the factor points z."""
    out = []
    for i in range(0, len(z), batch):
        lg = f(render(torch.as_tensor(z[i:i + batch], dtype=torch.float32)))
        out.append((lg[:, 1] - lg[:, 0]).numpy().astype(float))
    return np.concatenate(out)


def ball_samples(z0: np.ndarray, r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n points uniform in {‖z − z0‖ ≤ r} ∩ [0, 1]^d by rejection."""
    d, got = z0.shape[0], []
    while sum(len(g) for g in got) < n:
        u = rng.standard_normal((4 * n, d))
        u *= (r * rng.random(4 * n) ** (1 / d) / np.linalg.norm(u, axis=1))[:, None]
        z = z0 + u
        got.append(z[((z >= 0) & (z <= 1)).all(1)])
    return np.concatenate(got)[:n]


def margin_scores(f_old, f_new, z0: np.ndarray, y: np.ndarray | None, kappa: float, r: float, n_ball: int,
                  seed: int) -> dict[str, np.ndarray]:
    g_o, g_n = logit_gap(f_old, z0), logit_gap(f_new, z0)
    # identity check: every saved query must be valid for both models (margin ≥ κ in its class)
    yy = (g_o > 0).astype(int) if y is None else y
    sgn = np.where(yy == 1, 1.0, -1.0)
    if (sgn * g_o < kappa - 1e-4).any() or (sgn * g_n < kappa - 1e-4).any():
        raise SystemExit("saved z0 are not valid queries for these checkpoints (wrong models or points)")
    rng = np.random.default_rng(seed)
    balls = np.stack([ball_samples(z, r, n_ball, rng) for z in z0])
    flat = balls.reshape(-1, z0.shape[1])
    h_ball = (logit_gap(f_new, flat) - logit_gap(f_old, flat)).reshape(len(z0), n_ball)
    h0 = g_n - g_o
    return {"s_M": np.abs(h0), "s_Mr": np.maximum(np.abs(h0), np.abs(h_ball).max(1)), "y": yy}


def thresholds(null: np.ndarray) -> dict[str, float]:
    return {"fpr0": float(null.max()), "fpr0.05": float(np.quantile(null, 0.95))}


def m_analyze(cfg) -> dict:
    kappa, r, nb, base = cfg["kappa"], cfg["r"], cfg["n_ball"], cfg["ball_seed_base"]
    cache, grid = {}, {p["pair_id"]: p for p in grid_pairs()}

    def scores(f_old, f_new, a, pid, y):
        return margin_scores(f_old, f_new, a["z0"], y, kappa, r, nb, base + pid)

    null = {"s_M": [], "s_Mr": [], "cfdiff": []}
    for pid in cfg["null_pairs"]:
        a = dict(np.load(Path(cfg["null_run"]) / "pairs" / f"pair_{pid:02d}.npz"))
        _, f_old, f_new, _ = load_pair(pid, cache=cache)
        s = scores(f_old, f_new, a, pid, None)
        null["s_M"].append(s["s_M"]), null["s_Mr"].append(s["s_Mr"]), null["cfdiff"].append(cfdiff_score(a))
    null = {k: np.concatenate(v) for k, v in null.items()}
    taus = {k: thresholds(v) for k, v in null.items()}
    res = {"label": "D-052 M (descriptive)", "n_null_queries": int(null["s_M"].size), "tau": taus,
           "null_witness_share_dce": 0.0, "by_b": {}, "pairs": []}

    def readout(sc: dict, low: np.ndarray, dce: np.ndarray) -> dict:
        row = {"n_lowered": int(low.sum()), "dce_lowered": float(dce[low].mean()), "dce_all": float(dce.mean())}
        for name, s in sc.items():
            for k, t in taus[name].items():
                row[f"{name}_lowered_{k}"] = float((s[low] > t).mean())
                row[f"{name}_all_{k}"] = float((s > t).mean())
        return row

    rows_b = {}
    for npz in sorted((Path(cfg["h7_run"]) / "pairs").glob("pair_*.npz")):
        meta, a = json.loads(npz.with_suffix(".json").read_text()), dict(np.load(npz))
        f_old, _ = load_verified(Path(cfg["f_old_source"]), f"f_old_s{meta['seed']}", cache)
        s = scores(f_old, LogitShift(f_old, meta["b"]).eval(), a, meta["pair_id"], a["y"])
        if not np.allclose(s["s_M"], abs(meta["b"]), atol=1e-3):
            raise SystemExit(f"pair {meta['pair_id']}: |h| ≠ |b| for a logit shift")
        rows_b.setdefault(abs(meta["b"]), []).append(({"s_M": s["s_M"], "s_Mr": s["s_Mr"], "cfdiff": cfdiff_score(a)},
                                                     a["y"] == meta["lowered_class"], a["dce_N__found"]))
    for ab, rs in sorted(rows_b.items()):
        cat = {k: np.concatenate([x[0][k] for x in rs]) for k in ("s_M", "s_Mr", "cfdiff")}
        res["by_b"][str(ab)] = {"n_pairs": len(rs), **readout(cat, np.concatenate([x[1] for x in rs]),
                                                               np.concatenate([x[2] for x in rs]))}

    for run, pids in cfg["offset_runs"].items():
        for pid in pids:
            a = dict(np.load(Path(run) / "pairs" / f"pair_{pid}.npz"))
            if not np.array_equal(a["y"], (a["z0"][:, ASPECT] > 0.5).astype(a["y"].dtype)):
                raise SystemExit(f"{run} pair {pid}: saved y is not 1[aspect > 0.5]")
            if pid in grid and "exp_0026" in run:
                f_old, f_new, _ = load_models(cfg, grid[pid], cache)
            else:
                _, f_old, f_new, _ = load_pair(pid, cache=cache)
            s = scores(f_old, f_new, a, pid, a["y"])
            row = {"run": run, "pair_id": pid,
                   **readout({"s_M": s["s_M"], "s_Mr": s["s_Mr"], "cfdiff": cfdiff_score(a)}, a["y"] == 1,
                             a["dce_N__found"])}
            row["sM_minus_dce_fpr0"] = row["s_M_lowered_fpr0"] - row["dce_lowered"]
            res["pairs"].append(row)

    ps = res["pairs"]
    n_cheap = sum(p["s_M_lowered_fpr0"] >= p["dce_lowered"] - 0.05 for p in ps)
    witness_better = all(p["dce_lowered"] >= p["s_M_lowered_fpr0"] + 0.20 for p in ps)
    res["n_pairs_sM_ge_dce_minus_0.05"] = int(n_cheap)
    res["branch"] = ("whether is cheap" if n_cheap >= 5 else
                     "witness more sensitive" if witness_better else "report only (framing as whether is cheap)")
    return res


FACTORS = ("aspect", "size", "color", "bg", "posx")
SPLIT_A_METHODS = ("dce_N", "b1_det", "b3", "b4", "b5", "a1", "a2")
GRID_METHODS = ("dce_N", "b1_det", "b3", "a2")


def k_index(k) -> int:
    return FACTORS.index(k) if isinstance(k, str) else int(k)


def kstar_rank(found: np.ndarray, dphi: np.ndarray, k: int) -> np.ndarray:
    """Rank of k* in |Δφ| (1 = largest; ties count as the worst rank); 0 where not found."""
    mag = np.abs(dphi)
    rank = (mag >= mag[:, [k]] - 1e-12).sum(1)
    return np.where(found & (mag.max(1) > 0), rank, 0)


def rank_metrics(found, dphi, k) -> dict:
    rk = kstar_rank(found, dphi, k)
    return {"mrr": float(np.where(rk > 0, 1.0 / np.maximum(rk, 1), 0.0).mean()), "top2": float(((rk > 0) & (rk <= 2)).mean()),
            "recovery": float(((rk == 1) & found).mean())}


def load_run(run: Path) -> list[tuple[dict, dict]]:
    out = []
    for npz in sorted((run / "pairs").glob("pair_*.npz")):
        meta = json.loads(npz.with_suffix(".json").read_text())
        out.append((meta, dict(np.load(npz))))
    return out


def perm_spearman(x: np.ndarray, y: np.ndarray, strata: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    """Spearman ρ(x, y) and the one-sided permutation p of ρ > 0, x permuted within strata."""
    from scipy.stats import spearmanr
    rho = float(spearmanr(x, y).statistic)
    rng, ge = np.random.default_rng(seed), 0
    for _ in range(n):
        xp = x.copy()
        for s in np.unique(strata):
            m = strata == s
            xp[m] = rng.permutation(xp[m])
        ge += spearmanr(xp, y).statistic >= rho - 1e-12
    return rho, (ge + 1) / (n + 1)


def e_analyze(cfg) -> dict:
    A, G, B = load_run(Path(cfg["split_a_run"])), load_run(Path(cfg["grid_run"])), load_run(Path(cfg["split_b_run"]))
    res = {"label": "D-052 E1–E4 (descriptive)"}

    # consistency with Table 2 (split A, Recovery and found-rate per condition)
    t2 = {}
    for meta, a in A:
        if meta["k_star"] is None:
            continue
        for m in SPLIT_A_METHODS:
            r = rank_metrics(a[f"{m}__found"], a[f"{m}__dphi"], k_index(meta["k_star"]))
            t2.setdefault(meta["condition"], {}).setdefault(m, []).append((r["recovery"], float(a[f"{m}__found"].mean())))
    res["table2_check"] = {c: {m: [float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))]
                               for m, v in d.items()} for c, d in t2.items()}

    # E1a: rotation, Grad-diff Recovery per pair vs measured α̂ (24 α > 0 pairs)
    rot = [(meta, a) for meta, a in G if meta["alpha"] > 0]
    x = np.array([m["alpha_hat"] for m, _ in rot])
    y = np.array([rank_metrics(a["b3__found"], a["b3__dphi"], k_index(m["k_star"]))["recovery"] for m, a in rot])
    rho, p = perm_spearman(x, y, np.array([m["seed"] for m, _ in rot]), cfg["n_perm"], cfg["perm_seed"])
    res["e1a_rotation"] = {"n_pairs": len(rot), "spearman_rho": rho, "perm_p_one_sided": p,
                           "pairs": [{"pair_id": m["pair_id"], "alpha": m["alpha"], "t": m["t"], "seed": m["seed"],
                                      "alpha_hat": float(xi), "graddiff_recovery": float(yi)} for (m, _), xi, yi in zip(rot, x, y)]}

    # E1b: per-pair estimates of the Table 3 rows computable from saved outputs
    def per_pair(rows, fn):
        v = [float(fn(m, a)) for m, a in rows]
        return {"per_pair": v, "min": min(v), "max": max(v), "n": len(v)}

    rec = lambda meth: (lambda m, a: rank_metrics(a[f"{meth}__found"], a[f"{meth}__dphi"], k_index(m["k_star"]))["recovery"])
    diff = lambda m, a: rec("dce_N")(m, a) - rec("b3")(m, a)
    off = [(m, a) for m, a in G if m["alpha"] == 0 and m["pair_id"] in cfg["offset_pairs"]]
    e1b = {"graddiff_lt_0.2_grid_alpha0": per_pair(off, rec("b3"))}
    e1b["graddiff_lt_0.2_grid_alpha0"]["k_below_0.2"] = int(sum(v < 0.2 for v in e1b["graddiff_lt_0.2_grid_alpha0"]["per_pair"]))
    for alpha in sorted({m["alpha"] for m, _ in rot}):
        e1b[f"graddiff_grid_alpha{alpha}"] = per_pair([(m, a) for m, a in rot if m["alpha"] == alpha], rec("b3"))
    for cond, name in (("T1", "shortcut"), ("T3", "subgroup"), ("T4", "threshold")):
        e1b[f"dce_minus_graddiff_{name}_splitA"] = per_pair([(m, a) for m, a in A if m["condition"] == cond], diff)
        if any(m["condition"] == cond for m, _ in B):
            e1b[f"dce_minus_graddiff_{name}_splitB"] = per_pair([(m, a) for m, a in B if m["condition"] == cond], diff)
    nulls = [(m, a) for m, a in A if m["k_star"] is None]
    e1b["null_witnesses"] = per_pair(nulls, lambda m, a: a["dce_N__found"].sum())
    res["e1b_per_pair"] = e1b

    # E2: Imp-diff vector per pair (shared by all points) and Seg-search found-rate / Recovery
    e2 = []
    for m, a in A:
        v = a["b4__dphi"]
        e2.append({"pair_id": m["pair_id"], "condition": m["condition"], "k_star": m["k_star"],
                   "shared_by_all_points": bool(np.allclose(v, v[0])), "delta_pi": [float(t) for t in v[0]],
                   "argmax": FACTORS[int(np.abs(v[0]).argmax())],
                   "segsearch_found": float(a["b5__found"].mean())})
    res["e2_baselines"] = e2

    # E3: rank metrics (split A per condition; grid by α) and set-recovery {aspect, colour} on α > 0
    e3 = {}
    for m, a in A:
        if m["k_star"] is None:
            continue
        for meth in SPLIT_A_METHODS:
            e3.setdefault(f"splitA_{m['condition']}", {}).setdefault(meth, []).append(
                rank_metrics(a[f"{meth}__found"], a[f"{meth}__dphi"], k_index(m["k_star"])))
    for m, a in G:
        if m["k_star"] is None:
            continue
        for meth in GRID_METHODS:
            e3.setdefault(f"grid_alpha{m['alpha']}", {}).setdefault(meth, []).append(
                rank_metrics(a[f"{meth}__found"], a[f"{meth}__dphi"], k_index(m["k_star"])))
    res["e3_rank"] = {g: {meth: {k: float(np.mean([r[k] for r in rs])) for k in ("recovery", "mrr", "top2")}
                          for meth, rs in d.items()} for g, d in e3.items()}
    order = {}
    for g, d in res["e3_rank"].items():
        by_rec = sorted(d, key=lambda k: -d[k]["recovery"])
        by_mrr = sorted(d, key=lambda k: -d[k]["mrr"])
        order[g] = {"by_recovery": by_rec, "by_mrr": by_mrr, "same": by_rec == by_mrr}
    res["e3_order"] = order
    setrec = {}
    for m, a in rot:
        for meth in GRID_METHODS:
            mag = np.abs(a[f"{meth}__dphi"])
            top2 = np.argsort(-mag, axis=1)[:, :2]
            hit = a[f"{meth}__found"] & (np.sort(top2, 1) == np.array([0, 2])).all(1)
            setrec.setdefault(meth, []).append(float(hit.mean()))
    res["e3_set_recovery_alpha_gt0"] = {k: float(np.mean(v)) for k, v in setrec.items()}

    # E4: κ and r sensitivity
    e4 = {"grid_alpha0_offsets": {}, "splitA": {}, "null": {}}
    kap_grid = {"0": "dce_N_k0", "0.25": "dce_N_k0.25", "0.5": "dce_N", "1.0": "dce_N_k1.0"}
    for kap, meth in kap_grid.items():
        rows = [rank_metrics(a[f"{meth}__found"], a[f"{meth}__dphi"], k_index(m["k_star"])) | {"found": float(a[f"{meth}__found"].mean())}
                for m, a in G if m["pair_id"] in cfg["offset_pairs"]]
        e4["grid_alpha0_offsets"][kap] = {k: float(np.mean([r[k] for r in rows])) for k in ("recovery", "found")}
        e4["null"][f"grid_kappa{kap}"] = float(np.mean([a[f"{meth}__found"].mean() for m, a in G if m["pair_id"] in cfg["grid_null_pairs"]]))
    kap_a = {"0": "a3_k0", "0.5": "dce_N", "1.0": "a3_k1"}
    for kap, meth in kap_a.items():
        for cond in ("T1", "T3", "T4"):
            rows = [rank_metrics(a[f"{meth}__found"], a[f"{meth}__dphi"], k_index(m["k_star"])) | {"found": float(a[f"{meth}__found"].mean())}
                    for m, a in A if m["condition"] == cond]
            e4["splitA"].setdefault(kap, {})[cond] = {k: float(np.mean([r[k] for r in rows])) for k in ("recovery", "found")}
        e4["null"][f"splitA_kappa{kap}"] = float(np.mean([a[f"{meth}__found"].mean() for m, a in nulls]))
    e4["r"] = {}
    for rp in cfg["r_levels"]:
        for cond in ("T1", "T3", "T4"):
            vals = []
            for m, a in A:
                if m["condition"] != cond:
                    continue
                f = a["dce_N__found"] & (a["dce_N_distance"] <= rp + 1e-9)
                vals.append(rank_metrics(f, a["dce_N__dphi"], k_index(m["k_star"]))["recovery"])
            e4["r"].setdefault(str(rp), {})[cond] = float(np.mean(vals))
    res["e4"] = e4
    return res


def integrated_gradients(m, z0: torch.Tensor, base: torch.Tensor, steps: int) -> np.ndarray:
    """IG of the margin m along the straight path base → z0 (midpoint rule, `steps` points)."""
    n, d = z0.shape
    t = (torch.arange(steps, dtype=z0.dtype) + 0.5) / steps
    pts = (base[:, None, :] + t[None, :, None] * (z0 - base)[:, None, :]).reshape(-1, d).requires_grad_(True)
    idx = torch.arange(n).repeat_interleave(steps)
    g = torch.autograd.grad(m(pts, idx).sum(), pts)[0].reshape(n, steps, d).mean(1)
    return ((z0 - base) * g).detach().numpy().astype(float)


@torch.no_grad()
def exact_shapley(m, z0: torch.Tensor, bg: torch.Tensor, batch: int = 32768) -> np.ndarray:
    """Exact interventional Shapley values of m over the d factors (2^d coalitions, background bg)."""
    from math import factorial
    n, d = z0.shape
    nb = len(bg)
    v = torch.empty(2 ** d, n, dtype=torch.float64)
    idx = torch.arange(n).repeat_interleave(nb)
    for S in range(2 ** d):
        cols = torch.tensor([(S >> k) & 1 for k in range(d)], dtype=torch.bool)
        pts = bg.repeat(n, 1)
        pts[:, cols] = z0.repeat_interleave(nb, 0)[:, cols]
        vals = torch.cat([m(pts[i:i + batch], idx[i:i + batch]) for i in range(0, len(pts), batch)])
        v[S] = vals.reshape(n, nb).double().mean(1)
    phi = torch.zeros(n, d, dtype=torch.float64)
    for S in range(2 ** d):
        s = S.bit_count()
        for k in range(d):
            if not (S >> k) & 1:
                w = factorial(s) * factorial(d - s - 1) / factorial(d)
                phi[:, k] += w * (v[S | (1 << k)] - v[S])
    return phi.numpy()


def x1_pair(cfg, pid, split, train, cache) -> tuple[dict, dict]:
    from dce.baselines.b4_perm import b4_perm
    from dce.baselines.b5_line import b5_line
    from dce.data.drift import DriftParams, adaptation_set
    from dce.eval.tier0_margins import Tier0Margin
    from dce.models.cnn import predict_logits
    pair, f_old, f_new, entries = load_pair(pid, cache=cache)
    saved = dict(np.load(Path(cfg["split_a_run"]) / "pairs" / f"pair_{pid:02d}.npz"))
    keep = saved["point_ids"]
    if not np.allclose(split.phi[keep], saved["z0"], atol=1e-6):
        raise SystemExit(f"pair {pid}: split-A points do not match the saved z0")
    z0 = torch.as_tensor(split.phi[keep], dtype=torch.float32)
    y = split.y_clean[keep]
    m_old, m_new = Tier0Margin(f_old, y), Tier0Margin(f_new, y)
    out, rt = {"point_ids": keep, "z0": z0.numpy()}, {}
    n = len(keep)

    import time
    t0 = time.time()
    base = torch.as_tensor(train.phi.mean(0), dtype=torch.float32).expand_as(z0)
    out["ig__dphi"] = (integrated_gradients(m_new, z0, base, cfg["ig_steps"])
                       - integrated_gradients(m_old, z0, base, cfg["ig_steps"]))
    rt["ig"] = time.time() - t0
    t0 = time.time()
    rng = np.random.default_rng(cfg["shap_bg_seed_base"] + pid)
    bg = torch.as_tensor(train.phi[rng.choice(len(train.phi), cfg["shap_bg"], replace=False)], dtype=torch.float32)
    out["shap__dphi"] = exact_shapley(m_new, z0, bg) - exact_shapley(m_old, z0, bg)
    rt["shap"] = time.time() - t0
    t0 = time.time()
    params = DriftParams(**cfg.get("drift_params", {}).get(pair["condition"], {}))
    phi_a, y_a = adaptation_set(pair["condition"], seed=cfg["adapt_seed_base"] + pid, n=cfg["b4_points"],
                                params=params, label_noise=0.0)
    pred = lambda f: (lambda phi: predict_logits(f, phi, lambda p: render(torch.as_tensor(p, dtype=torch.float32))).argmax(1).numpy())
    out["b4a__dphi"] = np.tile(b4_perm(pred(f_old), pred(f_new), phi_a, y_a), (n, 1))
    rt["b4a"] = time.time() - t0
    t0 = time.time()
    if pair["k_star"] is None:   # D-052 cost amendment: Seg-search×10 only on change pairs
        row = {**pair, "n_points": n, "runtime_s": rt, "checkpoints": entries}
        for k in ("ig", "shap", "b4a"):
            out[f"{k}__found"] = np.ones(n, bool)
        return out, row
    found, d, dist = b5_line(z0, torch.as_tensor(train.phi, dtype=torch.float32), m_old, m_new, cfg["kappa"],
                             float(cfg["r"]), K=cfg["b5x_K"])
    out["b5x__found"], out["b5x__dphi"], out["b5x_distance"] = found.numpy(), d.numpy().astype(float), dist.numpy()
    rt["b5x"] = time.time() - t0
    for k in ("ig", "shap", "b4a"):
        out[f"{k}__found"] = np.ones(n, bool)
    row = {**pair, "n_points": n, "runtime_s": rt, "checkpoints": entries}
    return out, row


def cmd_x1_run(cfg, out: Path, pairs: list[int] | None = None) -> int:
    from dce.data.drift import clean_splits
    splits = clean_splits(seed=cfg["data_seed"])
    split, train = splits["test"], splits["train"]
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata_x1.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"experiment": cfg["experiment_id"], "git_runs": []}
    meta["git_runs"].append(git_state())
    meta_path.write_text(json.dumps(meta, indent=2))
    cache = {}
    for pid in pairs or cfg["pair_ids"]:
        npz, js = out / "pairs" / f"pair_{pid:02d}.npz", out / "pairs" / f"pair_{pid:02d}.json"
        if npz.exists() and js.exists():
            print(f"pair {pid}: already done, skipped", flush=True)
            continue
        arrays, row = x1_pair(cfg, pid, split, train, cache)
        np.savez_compressed(npz, **arrays)
        js.write_text(json.dumps(row, indent=2, default=float))
        print(f"pair {pid} ({row['condition']}) done {row['runtime_s']}", flush=True)
    meta["status"] = "RUN_COMPLETE"
    meta_path.write_text(json.dumps(meta, indent=2))
    return 0


X1_METHODS = ("ig", "shap", "b4a", "b5x")


def x1_analyze(cfg) -> dict:
    rows = {}
    for meta, a in load_run(Path(cfg["out_run"])):
        for meth in X1_METHODS:
            if f"{meth}__found" not in a:
                continue
            f = a[f"{meth}__found"]
            rec = rank_metrics(f, a[f"{meth}__dphi"], k_index(meta["k_star"])) if meta["k_star"] is not None else {}
            rows.setdefault(meta["condition"], {}).setdefault(meth, []).append({**rec, "found": float(f.mean())})
        if meta["k_star"] is not None:
            rows[meta["condition"]].setdefault("b4a_vector", []).append([float(t) for t in a["b4a__dphi"][0]])
    res = {"label": "D-052 X1 (descriptive)"}
    for cond, d in rows.items():
        res[cond] = {meth: ({k: float(np.mean([r[k] for r in v])) for k in v[0]} if meth != "b4a_vector" else v)
                     for meth, v in d.items()}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("m-analyze", "e-analyze", "x1-run", "x1-analyze"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", default=None, help="comma-separated subset of pair_ids (x1-run; one Kaggle kernel each)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    torch.set_num_threads(int(cfg.get("threads", 8)))
    if args.stage == "x1-run":
        return cmd_x1_run(cfg, args.out, [int(x) for x in args.pairs.split(",")] if args.pairs else None)
    if args.stage == "x1-analyze":
        return cmd_json(x1_analyze, "results_x1.json")(cfg, args.out)
    if args.stage == "e-analyze":
        return cmd_json(e_analyze, "results_e.json")(cfg, args.out)
    return cmd_json(m_analyze, "results_m.json")(cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
