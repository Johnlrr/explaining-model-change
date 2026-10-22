"""Tangential error of the solver on the CNN benchmark (VALIDATION only; descriptive; author's request 2026-10-22).

For ~100 valid validation points per pair (Shortcut 0, Subgroup 3, Threshold 6, grid α > 0 pairs 36 and 44, grid
α = 0 pair 24): Δ-CE with the experiments' solver (float32 Adam phase, SLSQP polish as configured) against a tight
reference solve of the same problem (float64 throughout, SLSQP maxiter 1000, ftol 1e-14, tighten 1e-7), and DeepXplore
(Δ-CE without proximity, no polish) in float32 against the same search in float64 (no unique optimum exists without
the proximity term, so this measures the sensitivity of its output to rounding). Reports the relative vector error
‖u − u_ref‖ / ‖u_ref‖ on points found by both: median and p95 per pair and pooled.

Usage: python -m dce.cli.tangent_check --config configs/experiments/exp_0038.yaml --out runs/experiments/exp_0038
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dce.cli.calibrate_r import solver_config
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.cli.v19_grid import grid_pairs
from dce.cli.v19_test import load_models
from dce.cli.v110_test import data_for, run_pairs
from dce.eval.pairs import load_pair
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.solvers.dce import delta_ce


def rel(u, ref, both):
    e = np.linalg.norm(u - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-12)
    return e[both]


def models(cfg, pid, cache):
    if pid < 20:                                              # Tier 0 pair (exp_0002 / exp_0003 checkpoints)
        q, f_old, f_new, _ = load_pair(pid, cache=cache)
        return f_old, f_new, {"kind": "tier0", "condition": q["condition"], "seed": q["seed"]}
    q = next(g for g in grid_pairs() if g["pair_id"] == pid)
    f_old, f_new, _ = load_models(cfg, q, cache)
    return f_old, f_new, {"kind": "grid", "alpha": q["alpha"], "t": q["t"], "seed": q["seed"]}


def pair_fn(cfg, p, cache):
    pid = p["pair_id"]
    f_old, f_new, info = models(cfg, pid, cache)
    phi, y, digest = data_for(cfg, pid)
    keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), cfg["n_points"],
                        cfg["seeds"]["points_base"] + pid, cfg["kappa"])
    yk = y[keep]
    m_old, m_new = Tier0Margin(f_old, yk), Tier0Margin(f_new, yk)
    z32 = torch.as_tensor(phi[keep], dtype=torch.float32)
    z64 = z32.double()
    base = solver_config(cfg, r=float(cfg["r"]))
    tight = replace(base, slsqp_maxiter=1000, slsqp_ftol=1e-14, slsqp_tighten=1e-7)
    a1 = replace(base, lam0=0.0, polish=False)
    seed = cfg["seeds"]["restart_base"] + pid
    out, arrays = {}, {"z0": z32.numpy(), "y": yk}
    for name, cfg_s, z in (("dce_std", base, z32), ("dce_ref", tight, z64), ("a1_f32", a1, z32), ("a1_f64", a1, z64)):
        t = time.time()
        res = delta_ce(z, m_old, m_new, cfg_s, seed, kind="N")
        arrays[f"{name}_u"] = (res.z.double() - z64).numpy()
        arrays[f"{name}_found"] = res.found.numpy()
        out[f"{name}_seconds"] = round(time.time() - t, 1)
    for a, b in (("dce_std", "dce_ref"), ("a1_f32", "a1_f64")):
        both = arrays[f"{a}_found"] & arrays[f"{b}_found"]
        e = rel(arrays[f"{a}_u"], arrays[f"{b}_u"], both)
        out[f"{a}_vs_{b}"] = {"n_both": int(both.sum()), "found_disagree": int((arrays[f"{a}_found"] != arrays[f"{b}_found"]).sum()),
                              "median": float(np.median(e)) if len(e) else None, "p95": float(np.quantile(e, 0.95)) if len(e) else None,
                              "max": float(e.max()) if len(e) else None}
    row = {"pair_id": pid, **info, "data_sha256": digest,
           "n_points": len(yk), **out}
    return arrays, row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("run", "summary"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="dyn")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cfg["_out"] = str(args.out)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    if args.stage == "run":
        by_id = {pid: {"pair_id": pid} for pid in cfg["pair_ids"]}
        shard = None if args.shard == "dyn" else tuple(map(int, args.shard.split("/")))
        return run_pairs(cfg, args.out, shard, by_id, pair_fn, "tangent check (validation, descriptive)")
    rows = [json.loads((args.out / "pairs" / f"pair_{p}.json").read_text()) for p in cfg["pair_ids"]]
    pooled = {}
    for a, b in (("dce_std", "dce_ref"), ("a1_f32", "a1_f64")):
        e = []
        for p in cfg["pair_ids"]:
            z = np.load(args.out / "pairs" / f"pair_{p}.npz")
            both = z[f"{a}_found"] & z[f"{b}_found"]
            e.append(rel(z[f"{a}_u"], z[f"{b}_u"], both))
        e = np.concatenate(e)
        pooled[f"{a}_vs_{b}"] = {"n": len(e), "median": float(np.median(e)), "p95": float(np.quantile(e, 0.95)),
                                 "max": float(e.max())}
    res = {"label": "tangent check (validation, descriptive)", "pairs": rows, "pooled": pooled, "git": git_state()}
    (args.out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(pooled, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
