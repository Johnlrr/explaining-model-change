"""G1 pilot on Tier 0 validation (spec §9 G1; EVALUATION §6 pilot discipline; D-024).

Seed-0 pair of every condition, N valid validation points each, every method. Persists ONLY found-rate,
the validation IDs of the points with a witness (v1.6 G4/G5), runtime (wall and CPU per point; phase-2 CPU
split by points with / without a witness, v1.6 G3) and solver counters. No Δφ̃, Recovery, share or per-point output is
computed into the record or written anywhere (pilot discipline).

Methods: dce_N (Δ-CE type N), dce_O (N0, N1 only), b1, b1_det, b3, b4, b5, a1 (λ0 = 0, phase 1 only),
a2 (no m_o ≥ κ constraint = ordinary CE of f_new), a3_k0 / a3_k1 (Δ-CE N with κ = 0 / 1.0).
F5 (v1.5): solver completeness on the first 50 points of pair 0 (diagnostic; distances only).

Usage:
    python -m dce.cli.g1_pilot --config configs/experiments/exp_0009.yaml --out runs/experiments/exp_0009
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

from dce.baselines.b1_diff import b1_diff
from dce.baselines.b3_grad import b3_grad
from dce.baselines.b4_perm import b4_perm
from dce.baselines.b5_line import b5_line
from dce.cli.calibrate_r import solver_config
from dce.cli.g0_tier0 import git_state, load_config
from dce.data.drift import clean_splits
from dce.data.renderer import render
from dce.eval.pairs import load_pair
from dce.eval.tier0_margins import Tier0Margin, float64_gap, valid_points
from dce.models.cnn import predict_logits
from dce.solvers.dce import delta_ce
from dce.solvers.ordinary_ce import ordinary_ce
from dce.solvers.search import run_phase2


class Timer:
    def __enter__(self):
        self.w, self.c = time.time(), time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall, self.cpu = time.time() - self.w, time.process_time() - self.c


def split_runtime(res) -> dict:
    """v1.6 G3: phase-2 CPU per point, split by points with / without a witness within r."""
    if res.phase2_cpu_per_point is None:
        return {}
    c, f = res.phase2_cpu_per_point, res.found
    mean = lambda x: float(x.mean()) if len(x) else None
    return {"phase1_wall_s": round(res.phase1_wall, 2),
            "phase2_cpu_s_per_point_with_witness": mean(c[f]),
            "phase2_cpu_s_per_point_without_witness": mean(c[~f])}


def solver_row(res, timer: Timer, n: int, ids=None) -> dict:
    return {**split_runtime(res), "found_ids": ids[res.found.numpy()].tolist() if ids is not None else None,
            "found_rate": float(res.found.float().mean()), "wall_s_per_point": timer.wall / n,
            "cpu_s_per_point": (res.cpu_seconds) / n, "wall_s_total": round(timer.wall, 2),
            "polish_runs": res.n_polish_runs, "polish_feasible": res.n_polished,
            "answers_from_phase2": res.n_from_phase2, "f64_ok_f32_rejected": res.n_f64_ok_f32_rejected}


def f5_completeness(z0, m_old, m_new, cfg, std_res, n_starts: int, seed: int) -> dict:
    """v1.5 F5: SLSQP from n_starts uniform starts in the box vs the standard Δ-CE (distances only)."""
    k = cfg.kappa

    def constraints(z, idx):
        return torch.stack([-k - m_new(z, idx), m_old(z, idx) - k], 1)

    gen = torch.Generator().manual_seed(seed)
    starts = torch.rand((n_starts, *z0.shape), generator=gen, dtype=z0.dtype)
    with Timer() as t:
        outs, _ = run_phase2(z0, constraints, cfg, starts)
    B = z0.shape[0]
    xs = torch.tensor(np.stack([x for o in outs for x, _ in o]), dtype=z0.dtype)
    idx = torch.arange(B).repeat_interleave(n_starts)
    with torch.no_grad():
        ok = (constraints(xs, idx) >= 0).all(1)
    d = torch.where(ok, (xs - z0[idx]).norm(dim=1), torch.tensor(float("inf"))).view(B, n_starts).min(1).values
    ref_found = torch.isfinite(d) & (d <= cfg.r)
    std_found, std_d = std_res.found, std_res.distance
    both = ref_found & std_found
    gap = ((std_d[both] - d[both]) / d[both]).double().numpy()
    return {"n_points": B, "n_starts": n_starts, "ref_found_rate": float(ref_found.float().mean()),
            "std_found_rate": float(std_found.float().mean()),
            "share_missed_by_std": float((ref_found & ~std_found).float().mean()),
            "share_found_by_std_not_ref": float((std_found & ~ref_found).float().mean()),
            "rel_gap_median": float(np.median(gap)) if len(gap) else None,
            "rel_gap_max": float(gap.max()) if len(gap) else None,
            "share_std_longer_by_1e-3": float((gap > 1e-3).mean()) if len(gap) else None,
            "wall_s": round(t.wall, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if cfg.get("r") is None:
        raise SystemExit("r is not calibrated (configs/tiers/tier0.yaml)")
    out = args.out
    if (out / "metrics.json").exists():
        print(f"refusing to overwrite completed run {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.set_num_threads(cfg.get("threads", 8))
    meta = {"experiment": cfg["experiment_id"], "gate": "G1 pilot", "tier": 0, "git": git_state(),
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": "cpu (16 logical cores)", "status": "RUNNING",
            "checkpoints": []}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    r, kappa, n = float(cfg["r"]), cfg["kappa"], cfg["n_points"]
    base = solver_config(cfg, r=r)
    splits = clean_splits(seed=cfg["data_seed"])
    val, train = splits["val"], splits["train"]
    g0_phi, g0_y = val.phi[:2000], val.y_clean[:2000]
    train_z = torch.as_tensor(train.phi, dtype=torch.float32)
    results = {"r": r, "n_points": n, "split": "validation", "pairs": {}}
    start = time.time()

    for pair_id in cfg["pair_ids"]:
        pair, f_old, f_new, entries = load_pair(pair_id)
        meta["checkpoints"] += entries
        cond = pair["condition"]
        mo_all, mn_all = Tier0Margin(f_old, val.y_clean), Tier0Margin(f_new, val.y_clean)
        keep = valid_points(val.phi, val.y_clean, mo_all, mn_all, n, cfg["seeds"]["points_base"] + pair_id, kappa)
        z0 = torch.as_tensor(val.phi[keep], dtype=torch.float32)
        m_old, m_new = Tier0Margin(f_old, val.y_clean[keep]), Tier0Margin(f_new, val.y_clean[keep])
        seed = cfg["seeds"]["restart_base"] + pair_id
        row = {"condition": cond, "seed": pair["seed"],
               "float64_gap": {"f_old": float64_gap(f_old, val.phi), "f_new": float64_gap(f_new, val.phi)},
               "methods": {}}
        M = row["methods"]

        with Timer() as t:
            res_n = delta_ce(z0, m_old, m_new, base, seed, kind="N")
        M["dce_N"] = solver_row(res_n, t, n, keep)
        if cond in ("N0", "N1"):
            with Timer() as t:
                res_o = delta_ce(z0, m_old, m_new, base, seed, kind="O")
            M["dce_O"] = solver_row(res_o, t, n, keep)
            M["dce_N_or_O_found_rate"] = float((res_n.found | res_o.found).float().mean())
        for name, det in (("b1", False), ("b1_det", True)):
            with Timer() as t:
                valid, _, ce_o, ce_n = b1_diff(z0, m_old, m_new, base, seed, deterministic=det)
            M[name] = {"found_rate": float(valid.float().mean()), "wall_s_per_point": t.wall / n,
                       "cpu_s_per_point": (ce_o.cpu_seconds + ce_n.cpu_seconds) / n,
                       "ce_old_found_rate": float(ce_o.found.float().mean()),
                       "ce_new_found_rate": float(ce_n.found.float().mean()),
                       "f64_ok_f32_rejected": ce_o.n_f64_ok_f32_rejected + ce_n.n_f64_ok_f32_rejected,
                       "found_ids": keep[valid.numpy()].tolist()}
            del _
        with Timer() as t:
            valid, _ = b3_grad(z0, m_old, m_new)
        M["b3"] = {"found_rate": float(valid.float().mean()), "wall_s_per_point": t.wall / n}
        del _
        with Timer() as t:
            pred = lambda f: (lambda phi: predict_logits(f, phi, lambda p: render(torch.as_tensor(p, dtype=torch.float32))).argmax(1).numpy())
            _ = b4_perm(pred(f_old), pred(f_new), g0_phi, g0_y)
        M["b4"] = {"found_rate": 1.0, "wall_s_total": round(t.wall, 2)}
        del _
        with Timer() as t:
            found, _, _d = b5_line(z0, train_z, m_old, m_new, kappa, r)
        M["b5"] = {"found_rate": float(found.float().mean()), "wall_s_per_point": t.wall / n,
                   "found_ids": keep[found.numpy()].tolist()}
        del _, _d
        with Timer() as t:
            res = delta_ce(z0, m_old, m_new, replace(base, lam0=0.0, polish=False), seed, kind="N")
        M["a1"] = solver_row(res, t, n, keep)
        with Timer() as t:
            res = ordinary_ce(z0, m_new, base, seed)
        M["a2"] = solver_row(res, t, n, keep)
        for name, k in (("a3_k0", 0.0), ("a3_k1", 1.0)):
            with Timer() as t:
                res = delta_ce(z0, m_old, m_new, replace(base, kappa=k), seed, kind="N")
            M[name] = solver_row(res, t, n, keep)

        dce_wall = M["dce_N"]["wall_s_total"] + (M["dce_O"]["wall_s_total"] if "dce_O" in M else 0.0)
        row["dce_wall_s_per_pair_at_N"] = round(dce_wall, 1)
        row["dce_wall_min_per_pair_extrapolated_N300"] = round(dce_wall * 300 / n / 60, 2)
        if pair_id == cfg["f5"]["pair_id"]:
            k5 = cfg["f5"]["n_points"]
            std = type(res_n)(z=res_n.z[:k5], distance=res_n.distance[:k5], found=res_n.found[:k5], seconds=0)
            sub_o, sub_n = Tier0Margin(f_old, val.y_clean[keep][:k5]), Tier0Margin(f_new, val.y_clean[keep][:k5])
            row["f5"] = f5_completeness(z0[:k5], sub_o, sub_n, base, std, cfg["f5"]["n_starts"], cfg["f5"]["seed"])
        results["pairs"][str(pair_id)] = row
        print(f"pair {pair_id} {cond}: " + json.dumps({k: v.get("found_rate") if isinstance(v, dict) else v
                                                       for k, v in M.items()}), f"({time.time() - start:.0f}s)", flush=True)

    t1 = results["pairs"].get("0")
    results["g1"] = {
        "dce_found_rate_T1": t1["methods"]["dce_N"]["found_rate"] if t1 else None,
        "found_rate_T1_ge_0.6": bool(t1 and t1["methods"]["dce_N"]["found_rate"] >= 0.6),
        "max_dce_min_per_pair_extrapolated_N300": max(p["dce_wall_min_per_pair_extrapolated_N300"]
                                                       for p in results["pairs"].values()),
    }
    results["g1"]["runtime_le_5min"] = results["g1"]["max_dce_min_per_pair_extrapolated_N300"] <= 5.0
    results["g1"]["pass"] = results["g1"]["found_rate_T1_ge_0.6"] and results["g1"]["runtime_le_5min"]
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    meta.update(status="SUCCESS", seconds=round(time.time() - start, 1))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(results["g1"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
