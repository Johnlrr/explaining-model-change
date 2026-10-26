"""Response to the external review (D-048): two descriptive analyses, rules declared in DECISIONS.md before any number.

e1-run       — exp_0041: Δ-CE type N, type O and N ∪ O on Tier 0 Threshold (T4) pairs 6–8, fresh data
               new_test_set(9100 + pair_id, 5 000), 300 valid points; B1-det, B3, CF-new on the same points.
e1-describe  — found-rate and Recovery (k* = aspect) per method and per query class, paired bootstrap stratified by pair.
e2-analyze   — exp_0042 (saved outputs only): CF-diff (B1-det) with a detection threshold at matched false-positive rate
               on the null pairs of exp_0017; detection per logit offset |b| on the pairs of exp_0032 next to Δ-CE.
e3-analyze   — exp_0043 (D-049; exp_0033 outputs plus model gradients at z0, no solver run): the linear recovery
               condition θ > π/4 or d_o(sin θ + cos θ) < d_n, evaluated at the query, against Δ-CE Recovery on the support.

Usage:
    python -m dce.cli.review_d048 e1-run --config configs/experiments/exp_0041.yaml --out runs/experiments/exp_0041
    python -m dce.cli.review_d048 e1-describe --config configs/experiments/exp_0041.yaml --out runs/experiments/exp_0041
    python -m dce.cli.review_d048 e2-analyze --config configs/experiments/exp_0042.yaml --out runs/experiments/exp_0042
    python -m dce.cli.review_d048 e3-analyze --config configs/experiments/exp_0043.yaml --out runs/experiments/exp_0043
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dce.analysis.v19 import boot_mean
from dce.baselines.b1_diff import b1_diff
from dce.baselines.b3_grad import b3_grad
from dce.cli.calibrate_r import solver_config
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.cli.v110_test import Timer, data_for, run_pairs
from dce.cli.v19_grid import grid_pairs
from dce.cli.v19_test import load_models
from dce.eval.metrics import top_factor
from dce.data.drift import tier0_pairs
from dce.eval.metrics import MethodOutput, auroc, recovery_indicator
from dce.eval.pairs import load_pair
from dce.eval.records import from_direction, from_points, runtime_row
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.solvers.dce import delta_ce
from dce.solvers.ordinary_ce import ordinary_ce

ASPECT, COLOUR = 0, 2
E1_METHODS = ("dce_N", "dce_O", "dce_NO", "b1_det", "b3", "a2")


def nearer_union(res_n, res_o, z0: torch.Tensor) -> MethodOutput:
    """N ∪ O: per point the nearer witness when both exist, else the one that exists."""
    take_n = res_n.found & (~res_o.found | (res_n.distance <= res_o.distance))
    z = torch.where(take_n[:, None], res_n.z, res_o.z)
    return from_points(res_n.found | res_o.found, z, z0)


def e1_pair(cfg, p, cache):
    pid, n = p["pair_id"], cfg["n_points"]
    base = solver_config(cfg, r=float(cfg["r"]))
    _, f_old, f_new, entries = load_pair(pid, cache=cache)
    phi, y, digest = data_for(cfg, pid)
    keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), n, cfg["seeds"]["points_base"] + pid, cfg["kappa"])
    z0 = torch.as_tensor(phi[keep], dtype=torch.float32)
    yk = y[keep]
    m_old, m_new = Tier0Margin(f_old, yk), Tier0Margin(f_new, yk)
    seed = cfg["seeds"]["restart_base"] + pid
    outs, rt = {}, {}
    with Timer() as t:
        res_n = delta_ce(z0, m_old, m_new, base, seed, kind="N")
    rt["dce_N"] = runtime_row(res_n, t.wall, n)
    with Timer() as t:
        res_o = delta_ce(z0, m_old, m_new, base, seed, kind="O")
    rt["dce_O"] = runtime_row(res_o, t.wall, n)
    outs["dce_N"], outs["dce_O"] = from_points(res_n.found, res_n.z, z0), from_points(res_o.found, res_o.z, z0)
    outs["dce_NO"] = nearer_union(res_n, res_o, z0)
    valid, _, ce_o, ce_n = b1_diff(z0, m_old, m_new, base, seed, deterministic=True)
    outs["b1_det"] = from_points(valid, ce_n.z, ce_o.z)
    valid, d = b3_grad(z0, m_old, m_new)
    outs["b3"] = from_direction(valid, d)
    res_a2 = ordinary_ce(z0, m_new, base, seed)
    outs["a2"] = from_points(res_a2.found, res_a2.z, z0)
    arrays = {"point_ids": keep, "y": yk, "z0": z0.numpy(), "dce_N_distance": res_n.distance.numpy(),
              "dce_O_distance": res_o.distance.numpy()}
    for k, v in outs.items():
        arrays[f"{k}__found"], arrays[f"{k}__dphi"] = v.found, v.dphi
    row = {**p, "n_points": n, "data_seed": cfg["data_seed_base"] + pid, "data_sha256": digest,
           "runtime": rt, "checkpoints": entries}
    return arrays, row


def e1_describe(pairs: dict, seed: int, B: int) -> dict:
    ids = sorted(pairs)
    res = {"label": "D-048 E1 (descriptive)", "bootstrap_seed": seed, "pair_ids": ids, "methods": {}}
    for m in E1_METHODS:
        outs = {pid: MethodOutput(pairs[pid]["%s__found" % m], pairs[pid]["%s__dphi" % m]) for pid in ids}
        rec = [recovery_indicator(outs[pid], ASPECT) for pid in ids]
        fnd = [outs[pid].found.astype(float) for pid in ids]
        entry = {}
        for name, groups in (("recovery", rec), ("found_rate", fnd)):
            point, stat = boot_mean(groups, np.random.default_rng(seed), B)
            entry[name] = {"mean": point, "ci95": [float(np.quantile(stat, 0.025)), float(np.quantile(stat, 0.975))]}
        entry["per_pair_recovery"] = [float(g.mean()) for g in rec]
        for c in (0, 1):
            sel = [pairs[pid]["y"] == c for pid in ids]
            entry[f"class{c}"] = {"n": int(sum(s.sum() for s in sel)),
                                  "found_rate": float(np.mean(np.concatenate([f[s] for f, s in zip(fnd, sel)]))),
                                  "recovery": float(np.mean(np.concatenate([r[s] for r, s in zip(rec, sel)])))}
        res["methods"][m] = entry
    return res


def cmd_e1_describe(cfg, out: Path) -> int:
    path = out / "results_e1.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    missing = [p for p in cfg["pair_ids"] if not (out / "pairs" / f"pair_{p}.json").exists()]
    if missing:
        raise SystemExit(f"run stage not complete: {missing}")
    pairs = {p: dict(np.load(out / "pairs" / f"pair_{p}.npz")) for p in cfg["pair_ids"]}
    res = e1_describe(pairs, cfg["bootstrap_seed"], cfg["bootstrap_resamples"])
    res["git"] = git_state()
    path.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({m: {k: v for k, v in e.items() if k != "per_pair_recovery"} for m, e in res["methods"].items()}, indent=1))
    return 0


def cfdiff_score(arrays: dict) -> np.ndarray:
    """Query score of CF-diff: ‖Δφ̃‖₂ of B1-det, 0 where no counterfactual pair exists."""
    return np.where(arrays["b1_det__found"], np.linalg.norm(arrays["b1_det__dphi"], axis=1), 0.0)


def e2_analyze(cfg) -> dict:
    null = np.concatenate([cfdiff_score(dict(np.load(Path(cfg["null_run"]) / "pairs" / f"pair_{p:02d}.npz"))) for p in cfg["null_pairs"]])
    taus = {"fpr0": float(null.max())}
    for q in cfg["fpr_levels"]:
        taus[f"fpr{q}"] = float(np.quantile(null, 1 - q))
    res = {"label": "D-048 E2 (descriptive)", "n_null_queries": int(null.size), "tau": taus,
           "null_fpr_at_tau": {k: float((null > t).mean()) for k, t in taus.items()},
           "null_run": cfg["null_run"], "h7_run": cfg["h7_run"], "by_b": {}}
    h7 = Path(cfg["h7_run"])
    rows = {}
    for npz in sorted((h7 / "pairs").glob("pair_*.npz")):
        meta = json.loads(npz.with_suffix(".json").read_text())
        a = dict(np.load(npz))
        rows.setdefault(abs(meta["b"]), []).append((meta, a))
    for ab in sorted(rows):
        s = np.concatenate([cfdiff_score(a) for _, a in rows[ab]])
        dce = np.concatenate([a["dce_N__found"] for _, a in rows[ab]])
        low = np.concatenate([a["y"] == meta["lowered_class"] for meta, a in rows[ab]])
        res["by_b"][str(ab)] = {
            "n_queries": int(s.size), "n_pairs": len(rows[ab]),
            "cfdiff_detect": {k: float((s > t).mean()) for k, t in taus.items()},
            "cfdiff_detect_lowered_class": {k: float((s[low] > t).mean()) for k, t in taus.items()},
            "auroc_cfdiff_vs_null": auroc(s, null),
            "dce_found_rate": float(dce.mean()), "dce_found_rate_lowered_class": float(dce[low].mean())}
    return res


def cmd_e2(cfg, out: Path) -> int:
    path = out / "results_e2.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    out.mkdir(parents=True, exist_ok=True)
    res = e2_analyze(cfg)
    res["git"] = git_state()
    path.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({"tau": res["tau"], "null_fpr_at_tau": res["null_fpr_at_tau"]}, indent=1))
    for ab, r in res["by_b"].items():
        print(ab, r["cfdiff_detect"], "dce", round(r["dce_found_rate"], 3), "auroc", r["auroc_cfdiff_vs_null"])
    return 0


def margin_and_grad(f, z0: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = Tier0Margin(f, y)
    z = torch.as_tensor(z0, dtype=torch.float32).clone().requires_grad_(True)
    val = m(z, torch.arange(len(z0)))
    return val.detach().numpy().astype(float), torch.autograd.grad(val.sum(), z)[0].numpy().astype(float)


def linear_condition(m_o, g_o, m_n, g_n, kappa: float) -> dict[str, np.ndarray]:
    """P = 1[θ > π/4 or d_o(sin θ + cos θ) < d_n] with the local linearisation at z0 (P5, Δ-CE recovery condition)."""
    no, nn = np.linalg.norm(g_o, axis=1), np.linalg.norm(g_n, axis=1)
    theta = np.arccos(np.clip((g_o * g_n).sum(1) / (no * nn), -1.0, 1.0))
    d_o, d_n = (m_o - kappa) / no, (m_n + kappa) / nn
    pred = (theta > np.pi / 4) | (d_o * (np.sin(theta) + np.cos(theta)) < d_n)
    v = g_n - ((g_n * g_o).sum(1) / no**2)[:, None] * g_o
    return {"P": pred, "theta": theta, "d_o": d_o, "d_n": d_n,
            "top_g_o_aspect": top_factor(g_o) == ASPECT, "top_v_colour": top_factor(v) == COLOUR}


def e3_analyze(cfg) -> dict:
    src = Path(cfg["source_run"])
    by_id = {p["pair_id"]: p for p in grid_pairs()}
    groups, cache, rows = [], {}, []
    for pid in cfg["pair_ids"]:
        a = dict(np.load(src / "pairs" / f"pair_{pid}.npz"))
        f_old, f_new, _ = load_models(cfg, by_id[pid], cache)
        m_o, g_o = margin_and_grad(f_old, a["z0"], a["y"])
        m_n, g_n = margin_and_grad(f_new, a["z0"], a["y"])
        if not np.allclose(g_o, a["grad_z0"], atol=1e-5):
            raise SystemExit(f"pair {pid}: ∇m_o differs from the saved grad_z0 (wrong model or points)")
        c = linear_condition(m_o, g_o, m_n, g_n, cfg["kappa"])
        s = a["dce__found"]
        rec = top_factor(a["dce__dphi"]) == COLOUR
        groups.append((c["P"][s], rec[s]))
        rows.append({"pair_id": pid, "alpha": by_id[pid]["alpha"], "t": by_id[pid]["t"], "support": int(s.sum()),
                     "share_P1": float(c["P"][s].mean()), "rec": float(rec[s].mean()),
                     "median_theta_deg": float(np.degrees(np.median(c["theta"][s]))),
                     "share_top_g_o_aspect": float(c["top_g_o_aspect"][s].mean()),
                     "share_top_v_colour": float(c["top_v_colour"][s].mean())})
    P = np.concatenate([g[0] for g in groups])
    R = np.concatenate([g[1] for g in groups])

    def gap(ps, rs):
        p, r = np.concatenate(ps), np.concatenate(rs)
        return r[p].mean() - r[~p].mean() if p.any() and (~p).any() else np.nan

    rng = np.random.default_rng(cfg["bootstrap_seed"])
    boot = np.empty(cfg["bootstrap_resamples"])
    for b in range(boot.size):
        idx = [rng.integers(0, len(g[0]), len(g[0])) for g in groups]
        boot[b] = gap([g[0][i] for g, i in zip(groups, idx)], [g[1][i] for g, i in zip(groups, idx)])
    share = float(P.mean())
    return {"label": "D-049 E3 (descriptive)", "source_run": str(src), "n_support": int(P.size), "share_P1": share,
            "rec_P1": float(R[P].mean()) if P.any() else None, "rec_P0": float(R[~P].mean()) if (~P).any() else None,
            "gap": float(gap([P], [R])), "gap_ci95": [float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))],
            "agreement": float((P == R).mean()), "rec_all": float(R.mean()),
            "testable": bool(min(share, 1 - share) >= 0.05),
            "share_top_g_o_aspect": float(np.mean([r["share_top_g_o_aspect"] for r in rows])),
            "share_top_v_colour": float(np.mean([r["share_top_v_colour"] for r in rows])), "pairs": rows}


def cmd_e3(cfg, out: Path) -> int:
    path = out / "results_e3.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    out.mkdir(parents=True, exist_ok=True)
    res = e3_analyze(cfg)
    res["git"] = git_state()
    path.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("pairs", "git")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("e1-run", "e1-describe", "e2-analyze", "e3-analyze"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    if args.stage == "e1-run":
        by_id = {p["pair_id"]: p for p in tier0_pairs()}
        shard = tuple(map(int, args.shard.split("/")))
        return run_pairs(cfg, args.out, shard, by_id, e1_pair, cfg.get("decision", "D-048"))
    if args.stage == "e1-describe":
        return cmd_e1_describe(cfg, args.out)
    if args.stage == "e3-analyze":
        return cmd_e3(cfg, args.out)
    return cmd_e2(cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
