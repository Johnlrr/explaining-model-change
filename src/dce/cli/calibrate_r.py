"""Calibrate the Tier 0 distance budget r once (spec §6.3; D-024).

r = 2 × median d(z0, CE(z0; f_new)) over the found CEs, on the 300 valid validation points of pair 0
(T1 seed 0), with the shared two-phase solver and r = ∞ during calibration.

Usage:
    python -m dce.cli.calibrate_r --config configs/experiments/exp_0008.yaml --out runs/experiments/exp_0008
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from dce.cli.g0_tier0 import git_state, load_config
from dce.data.drift import clean_splits
from dce.eval.pairs import load_pair
from dce.eval.tier0_margins import Tier0Margin, float64_gap, valid_points
from dce.solvers.ordinary_ce import ordinary_ce
from dce.solvers.search import SolverConfig


def solver_config(cfg: dict, **overrides) -> SolverConfig:
    s = dict(cfg["solver"])
    s["clamp"] = tuple(s["clamp"]) if s.get("clamp") is not None else None
    return SolverConfig(**{**s, "kappa": cfg["kappa"], **overrides})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = args.out
    if (out / "metrics.json").exists():
        print(f"refusing to overwrite completed run {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.set_num_threads(cfg.get("threads", 8))
    meta = {"experiment": cfg["experiment_id"], "gate": "G1 (r calibration)", "tier": 0, "git": git_state(),
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": "cpu", "status": "RUNNING"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    val = clean_splits(seed=cfg["data_seed"])["val"]
    pair, f_old, f_new, entries = load_pair(cfg["pair_id"])
    meta["checkpoints"] = entries
    m_old, m_new = Tier0Margin(f_old, val.y_clean), Tier0Margin(f_new, val.y_clean)
    keep = valid_points(val.phi, val.y_clean, m_old, m_new, cfg["n_points"],
                        cfg["seeds"]["points_base"] + pair["pair_id"], cfg["kappa"])
    z0 = torch.as_tensor(val.phi[keep], dtype=torch.float32)
    m = Tier0Margin(f_new, val.y_clean[keep])
    start = time.time()
    res = ordinary_ce(z0, m, solver_config(cfg, r=float("inf")), cfg["seeds"]["restart_base"] + pair["pair_id"])
    d = res.distance[res.found].double().numpy()
    median = float(np.median(d))
    metrics = {
        "pair_id": pair["pair_id"], "condition": pair["condition"], "seed": pair["seed"], "split": "validation",
        "n_points": len(keep), "ce_found_share": float(res.found.float().mean()),
        "median_ce_distance": median, "r": 2 * median,
        "float64_gap_f_new": float64_gap(f_new, val.phi), "float64_gap_f_old": float64_gap(f_old, val.phi),
        "f64_ok_f32_rejected": res.n_f64_ok_f32_rejected, "polish_runs": res.n_polish_runs,
        "polish_feasible": res.n_polished, "wall_seconds": round(time.time() - start, 2),
        "cpu_seconds": round(res.cpu_seconds, 2),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    meta.update(status="SUCCESS", seconds=round(time.time() - start, 1))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
