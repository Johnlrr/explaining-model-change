"""Run-to-run variability of the reported comparisons (VALIDATION only; descriptive; author's request 2026-10-22).

Decision rules fixed before any number was seen (author, 2026-10-22):
- Shortcut, Δ-CE vs CF-diff: |Δ| > 0.10 in every run → paper unchanged; otherwise the tie sentence is reworded.
- H8: recorded in proofs.md §6; the paper changes only if the gap at (z*, z0) has the opposite sign of the gap at
  (z0, z0) in some run.

Units = (pair, run). Runs: restart seeds s0, s0+1, s0+2 in float32 and s0, s0+1 in float64 (s0 = restart_base + pair).
- Shortcut (pair 0) and Threshold (pair 6), n_points_tier0 valid validation points: Recovery of Δ-CE and of CF-diff
  (deterministic counterfactual differencing; R = 1 without noise, so only the precision changes it).
- H5d pairs 36, 37, 44, 45, n_points_grid points: projected Recovery of Δ-CE and Grad-diff on the Δ-CE support, with
  ∇m_old at z0 (cell (z0, z0)) and at the witness z* (cell (z*, z0)); Grad-diff computed at z0.

Usage: python -m dce.cli.variability_check run|summary --config configs/experiments/exp_0040.yaml --out runs/experiments/exp_0040
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dce.analysis.projection import old_margin_gradients, project
from dce.baselines.b1_diff import b1_diff
from dce.baselines.b3_grad import b3_grad
from dce.cli.calibrate_r import solver_config
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.cli.tangent_check import models
from dce.cli.v110_test import data_for, run_pairs
from dce.eval.metrics import MethodOutput, recovery_indicator
from dce.eval.records import from_direction, from_points
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.solvers.dce import delta_ce

RUNS = (("f32_s0", torch.float32, 0), ("f32_s1", torch.float32, 1), ("f32_s2", torch.float32, 2),
        ("f64_s0", torch.float64, 0), ("f64_s1", torch.float64, 1))
TIER0 = {0: 2, 6: 0}                 # pair → k* (Shortcut: colour; Threshold: aspect)
H5D = (36, 37, 44, 45)               # k* = colour


def unit_fn(cfg, p, cache):
    pid, (name, dt, off) = p["pid"], RUNS[p["run"]]
    f_old, f_new, info = models(cfg, pid, cache)
    phi, y, _ = data_for(cfg, pid)
    n = cfg["n_points_tier0"] if pid in TIER0 else cfg["n_points_grid"]
    keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), n, cfg["seeds"]["points_base"] + pid, cfg["kappa"])
    yk = y[keep]
    m_old, m_new = Tier0Margin(f_old, yk), Tier0Margin(f_new, yk)
    z0 = torch.as_tensor(phi[keep], dtype=dt)
    base = solver_config(cfg, r=float(cfg["r"]))
    seed = cfg["seeds"]["restart_base"] + pid + off
    res = delta_ce(z0, m_old, m_new, base, seed, kind="N")
    dce = from_points(res.found, res.z.double(), z0.double())
    row = {"pair_id": pid, "run": name, **info, "n_points": len(yk)}
    if pid in TIER0:
        k = TIER0[pid]
        valid, d, _, _ = b1_diff(z0, m_old, m_new, base, seed, deterministic=True)
        cfd = MethodOutput(valid.numpy(), d.double().numpy())
        row.update(rec_dce=float(recovery_indicator(dce, k).mean()), rec_cf_diff=float(recovery_indicator(cfd, k).mean()),
                   found_dce=float(dce.found.mean()), found_cf_diff=float(cfd.found.mean()))
    else:
        k, s = 2, dce.found
        z32 = z0.float()
        b3 = from_direction(*b3_grad(z32, m_old, m_new))
        zs = torch.where(res.found[:, None], res.z.float(), z32).numpy()
        g = {"z0": old_margin_gradients(f_old, z32.numpy(), yk), "zs": old_margin_gradients(f_old, zs, yk)}
        for cell in ("z0", "zs"):
            rd = recovery_indicator(project(dce, g[cell]), k)[s]
            rb = recovery_indicator(project(b3, g[cell]), k)[s]
            row[f"gap_{cell}_z0"] = float(rd.mean() - rb.mean()) if s.any() else None
        row["support"] = int(s.sum())
    return {"found": dce.found}, row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("run", "summary"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cfg["_out"] = str(args.out)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    units = {100 * pid + r: {"pair_id": 100 * pid + r, "pid": pid, "run": r} for pid in cfg["pair_ids"] for r in range(len(RUNS))}
    cfg["pair_ids"] = list(units)
    if args.stage == "run":
        return run_pairs(cfg, args.out, None, units, unit_fn, "variability check (validation, descriptive)")
    rows = [json.loads((args.out / "pairs" / f"pair_{u}.json").read_text()) for u in units]
    out = {}
    for pid in TIER0:
        rs = [r for r in rows if r["pair_id"] == pid]
        diff = [r["rec_dce"] - r["rec_cf_diff"] for r in rs]
        out[f"pair_{pid}"] = {"runs": [r["run"] for r in rs], "rec_dce": [r["rec_dce"] for r in rs],
                              "rec_cf_diff": [r["rec_cf_diff"] for r in rs], "dce_minus_cf_diff": diff,
                              "min_abs_diff": min(abs(x) for x in diff)}
    h8 = []
    for name, *_ in RUNS:
        rs = [r for r in rows if r["run"] == name and r["pair_id"] in H5D]
        g00 = float(np.mean([r["gap_z0_z0"] for r in rs]))
        gs0 = float(np.mean([r["gap_zs_z0"] for r in rs]))
        h8.append({"run": name, "gap_z0z0": g00, "gap_zsz0": gs0, "T": abs(g00) - 2 * abs(gs0),
                   "opposite_sign_literal": bool(np.sign(g00) != np.sign(gs0)),
                   "zs_sign_differs_from_reported_plus": bool(gs0 < 0), "T_nonpositive": bool(abs(g00) - 2 * abs(gs0) <= 0)})
    out["H8"] = h8
    sc = out["pair_0"]["min_abs_diff"]
    out["decisions"] = {"shortcut_vs_cf_diff_all_runs_above_0.10": bool(sc > 0.10),
                        "h8_note": "the reported test values already have opposite signs (−0.190, +0.029); all three readings listed per run"}
    res = {"label": "variability check (validation, descriptive)", **out, "git": git_state()}
    (args.out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
