"""Run-to-run stability of Recovery for Δ-CE and DeepXplore (VALIDATION only; descriptive; author's request 2026-10-22).

Same ~100 valid validation points per pair (Shortcut 0, Subgroup 3, Threshold 6). Five runs per method: restart seeds
s0, s0+1, s0+2 in float32 and s0, s0+1 in float64 (s0 = restart_base + pair_id, the experiments' seed). Δ-CE uses the
experiments' solver; DeepXplore is Δ-CE without the proximity term and without polish. Per run: found-rate and
Recovery@r; per method: min–max and SD over runs; per run pairing: Recovery(Δ-CE) − Recovery(DeepXplore).

Usage: python -m dce.cli.stability_check run|summary --config configs/experiments/exp_0039.yaml --out runs/experiments/exp_0039
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
from dce.cli.tangent_check import models
from dce.cli.v110_test import data_for, run_pairs
from dce.eval.metrics import MethodOutput, recovery_indicator
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.solvers.dce import delta_ce

RUNS = (("f32_s0", torch.float32, 0), ("f32_s1", torch.float32, 1), ("f32_s2", torch.float32, 2),
        ("f64_s0", torch.float64, 0), ("f64_s1", torch.float64, 1))


def pair_fn(cfg, p, cache):
    pid = p["pair_id"]
    f_old, f_new, info = models(cfg, pid, cache)
    phi, y, digest = data_for(cfg, pid)
    keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), cfg["n_points"],
                        cfg["seeds"]["points_base"] + pid, cfg["kappa"])
    yk = y[keep]
    m_old, m_new = Tier0Margin(f_old, yk), Tier0Margin(f_new, yk)
    base = solver_config(cfg, r=float(cfg["r"]))
    methods = {"dce": base, "a1": replace(base, lam0=0.0, polish=False)}
    k = p["k_star"]
    arrays, row = {"y": yk}, {"pair_id": pid, **info, "k_star": k, "data_sha256": digest, "n_points": len(yk), "runs": {}}
    for m, scfg in methods.items():
        for name, dt, off in RUNS:
            t = time.time()
            z0 = torch.as_tensor(phi[keep], dtype=dt)
            res = delta_ce(z0, m_old, m_new, scfg, cfg["seeds"]["restart_base"] + pid + off, kind="N")
            d = (res.z.double() - z0.double()).numpy()
            out = MethodOutput(res.found.numpy(), d)
            arrays[f"{m}_{name}_found"], arrays[f"{m}_{name}_dphi"] = out.found, d
            row["runs"][f"{m}_{name}"] = {"found": float(out.found.mean()), "recovery": float(recovery_indicator(out, k).mean()),
                                          "seconds": round(time.time() - t, 1)}
    return arrays, row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("run", "summary"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cfg["_out"] = str(args.out)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    from dce.data.drift import tier0_pairs
    by_id = {q["pair_id"]: q for q in tier0_pairs() if q["pair_id"] in cfg["pair_ids"]}
    if args.stage == "run":
        return run_pairs(cfg, args.out, None, by_id, pair_fn, "stability check (validation, descriptive)")
    summ = {}
    for pid in cfg["pair_ids"]:
        r = json.loads((args.out / "pairs" / f"pair_{pid}.json").read_text())
        s = {}
        for m in ("dce", "a1"):
            rec = np.array([r["runs"][f"{m}_{n}"]["recovery"] for n, _, _ in RUNS])
            fr = np.array([r["runs"][f"{m}_{n}"]["found"] for n, _, _ in RUNS])
            s[m] = {"recovery": rec.round(3).tolist(), "rec_min_max": [float(rec.min()), float(rec.max())],
                    "rec_sd": float(rec.std(ddof=1)), "found": fr.round(3).tolist(), "found_sd": float(fr.std(ddof=1))}
        diff = np.array([r["runs"][f"dce_{n}"]["recovery"] - r["runs"][f"a1_{n}"]["recovery"] for n, _, _ in RUNS])
        s["dce_minus_a1"] = {"per_run": diff.round(3).tolist(), "min_max": [float(diff.min()), float(diff.max())]}
        summ[r["condition"]] = s
    res = {"label": "stability check (validation, descriptive)", "conditions": summ, "git": git_state()}
    (args.out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
