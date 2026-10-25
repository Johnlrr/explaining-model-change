"""Response to the external review (D-048): two descriptive analyses, rules declared in DECISIONS.md before any number.

e1-run       — exp_0041: Δ-CE type N, type O and N ∪ O on Tier 0 Threshold (T4) pairs 6–8, fresh data
               new_test_set(9100 + pair_id, 5 000), 300 valid points; B1-det, B3, CF-new on the same points.
e1-describe  — found-rate and Recovery (k* = aspect) per method and per query class, paired bootstrap stratified by pair.
e2-analyze   — exp_0042 (saved outputs only): CF-diff (B1-det) with a detection threshold at matched false-positive rate
               on the null pairs of exp_0017; detection per logit offset |b| on the pairs of exp_0032 next to Δ-CE.

Usage:
    python -m dce.cli.review_d048 e1-run --config configs/experiments/exp_0041.yaml --out runs/experiments/exp_0041
    python -m dce.cli.review_d048 e1-describe --config configs/experiments/exp_0041.yaml --out runs/experiments/exp_0041
    python -m dce.cli.review_d048 e2-analyze --config configs/experiments/exp_0042.yaml --out runs/experiments/exp_0042
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
from dce.data.drift import tier0_pairs
from dce.eval.metrics import MethodOutput, auroc, recovery_indicator
from dce.eval.pairs import load_pair
from dce.eval.records import from_direction, from_points, runtime_row
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.solvers.dce import delta_ce
from dce.solvers.ordinary_ce import ordinary_ce

ASPECT = 0
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("e1-run", "e1-describe", "e2-analyze"))
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
    return cmd_e2(cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
