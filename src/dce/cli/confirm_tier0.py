"""D-034: confirmation on a NEW Tier 0 test set (C1, C2, C3). Post-hoc hypotheses confirmed on fresh data.

run      — generate the data once (default_rng(data_seed_new), sample_factors, clean labels; sha256 recorded), then per
           pair the declared methods on N valid points (seed 8000 + pair_id); resumable per pair; no metric printed.
analyze  — C1 (T4, Δ-CE − B3, §11 bootstrap seed 6200), C2 (null refusal, deterministic), C3 (T1, projected
           Δ-CE − B3, TOST ±0.10, 90% CI, seed 6201); Holm over C1 and C3 (m = 2).

Usage:
    python -m dce.cli.confirm_tier0 run --config configs/experiments/exp_0019.yaml --out runs/experiments/exp_0019
    python -m dce.cli.confirm_tier0 analyze --config configs/experiments/exp_0019.yaml --out runs/experiments/exp_0019
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dce.analysis.projection import old_margin_gradients, project
from dce.baselines.b1_diff import b1_diff
from dce.baselines.b3_grad import b3_grad
from dce.cli.calibrate_r import solver_config
from dce.cli.final_tier0 import Timer
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.data.renderer import sample_factors, task_label
from dce.eval.metrics import MethodOutput, recovery_indicator
from dce.eval.pairs import load_pair
from dce.eval.records import from_direction, from_points, runtime_row
from dce.eval.rules import SEEDS, PairRecord, compare
from dce.eval.stats import holm
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.solvers.dce import delta_ce

DRIFT_METHODS = ("dce_N", "b1_det", "b3")
NULL_METHODS = ("dce_N", "dce_O", "b1")
TOST_MARGIN = 0.10
C2_MAX = 0.05


def new_test_set(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, str]:
    phi = sample_factors(n, np.random.default_rng(seed))
    y = task_label(phi).astype(np.int64)
    digest = hashlib.sha256(np.ascontiguousarray(phi, dtype=np.float64).tobytes() + y.tobytes()).hexdigest()
    return phi, y, digest


def run_pair(cfg, pid, phi, y, n):
    r, kappa = float(cfg["r"]), cfg["kappa"]
    base = solver_config(cfg, r=r)
    pair, f_old, f_new, entries = load_pair(pid)
    keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), n, cfg["points_seed_base"] + pid, kappa)
    z0 = torch.as_tensor(phi[keep], dtype=torch.float32)
    m_old, m_new = Tier0Margin(f_old, y[keep]), Tier0Margin(f_new, y[keep])
    seed = cfg["seeds"]["restart_base"] + pid
    outs, rt, extra = {}, {}, {}
    null = pair["condition"] in ("N0", "N1")
    for kind in ("N", "O") if null else ("N",):
        with Timer() as t:
            res = delta_ce(z0, m_old, m_new, base, seed, kind=kind)
        outs[f"dce_{kind}"], rt[f"dce_{kind}"] = from_points(res.found, res.z, z0), runtime_row(res, t.wall, n)
        extra[f"dce_{kind}_distance"] = res.distance.numpy()
    det = not null  # drift pairs: B1-det; null pairs: B1 (D-034 scope)
    with Timer() as t:
        valid, _, ce_o, ce_n = b1_diff(z0, m_old, m_new, base, seed, deterministic=det)
    name = "b1_det" if det else "b1"
    outs[name], rt[name] = from_points(valid, ce_n.z, ce_o.z), {"wall_s_per_point": t.wall / n}
    if not null:
        with Timer() as t:
            v, d = b3_grad(z0, m_old, m_new)
        outs["b3"], rt["b3"] = from_direction(v, d), {"wall_s_per_point": t.wall / n}
        extra["grad_old"] = old_margin_gradients(f_old, z0.numpy(), y[keep])
    arrays = {"point_ids": keep, "z0": z0.numpy(), **extra}
    for k, o in outs.items():
        arrays[f"{k}__found"], arrays[f"{k}__dphi"] = o.found, o.dphi
    return arrays, {**pair, "n_points": n, "runtime": rt, "checkpoints": entries}


def cmd_run(cfg, out: Path) -> int:
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", 8))
    phi, y, digest = new_test_set(cfg["data_seed_new"], cfg["n_samples"])
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "experiment": cfg["experiment_id"], "decision": "D-034", "run_kind": cfg["run_kind"],
        "data_seed_new": cfg["data_seed_new"], "data_sha256": digest, "git_runs": [], "python": sys.version.split()[0],
        "torch": torch.__version__, "platform": platform.platform(), "device": "cpu (8 threads)"}
    if meta["data_sha256"] != digest:
        raise SystemExit("regenerated data differ from the recorded hash")
    meta["git_runs"].append({"git": git_state(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    meta["status"] = "RUNNING"
    meta_path.write_text(json.dumps(meta, indent=2))
    start = time.time()
    for pid in cfg["pair_ids"]:
        npz, js = out / "pairs" / f"pair_{pid:02d}.npz", out / "pairs" / f"pair_{pid:02d}.json"
        if npz.exists() and js.exists():
            print(f"pair {pid}: already done, skipped", flush=True)
            continue
        with Timer() as t:
            arrays, row = run_pair(cfg, pid, phi, y, cfg["n_points"])
        np.savez_compressed(npz, **arrays)
        row["wall_s_pair"] = round(t.wall, 1)
        js.write_text(json.dumps(row, indent=2))
        print(f"pair {pid} ({row['condition']} s{row['seed']}) done in {t.wall / 60:.1f} min "
              f"(total {(time.time() - start) / 3600:.2f} h)", flush=True)
    meta.update(status="RUN_COMPLETE")
    meta_path.write_text(json.dumps(meta, indent=2))
    return 0


def bootstrap_stats(a_pairs, b_pairs, rng, B):
    """Same resampling scheme as dce.eval.stats.paired_bootstrap (stratified by pair, paired), returning the
    point estimate and the B bootstrap statistics (needed for the TOST p-values)."""
    J = len(a_pairs)
    point, stat = 0.0, np.zeros(B)
    for a, b in zip(a_pairs, b_pairs):
        point += (a.mean() - b.mean()) / J
        idx = rng.integers(0, a.size, size=(B, a.size))
        stat += (a[idx].mean(axis=1) - b[idx].mean(axis=1)) / J
    return point, stat


def analyze(pairs: dict, cfg: dict) -> dict:
    B = cfg["bootstrap_resamples"]
    by = {}
    for pid, (row, arr) in pairs.items():
        by.setdefault(row["condition"], []).append((pid, row, arr))
    out = {"label": "D-034 confirmation (post-hoc hypotheses on fresh data)"}

    def outputs(arr, methods):
        return {m: MethodOutput(arr[f"{m}__found"], arr[f"{m}__dphi"]) for m in methods if f"{m}__found" in arr}

    # C1: T4, Δ-CE − B3, §11 bootstrap
    recs = []
    for pid, row, arr in by["T4"]:
        o = outputs(arr, DRIFT_METHODS)
        recs.append(PairRecord(pid, row["seed"], {m: recovery_indicator(o[m], row["k_star"]) for m in o}))
    c1 = compare("C1", recs, "dce_N", "b3", np.random.default_rng(cfg["seeds_confirm"]["C1"]), B=B)
    # C3: T1, projected Δ-CE − B3, TOST
    a_p, b_p, seed_d, unproj = [], [], {}, []
    for pid, row, arr in by["T1"]:
        o = outputs(arr, DRIFT_METHODS)
        g = arr["grad_old"]
        ia = recovery_indicator(project(o["dce_N"], g), row["k_star"])
        ib = recovery_indicator(project(o["b3"], g), row["k_star"])
        a_p.append(ia), b_p.append(ib)
        seed_d[row["seed"]] = float(ia.mean() - ib.mean())
        unproj.append(float(recovery_indicator(o["dce_N"], row["k_star"]).mean() - recovery_indicator(o["b3"], row["k_star"]).mean()))
    point, stat = bootstrap_stats(a_p, b_p, np.random.default_rng(cfg["seeds_confirm"]["C3"]), B)
    lo90, hi90 = np.quantile(stat, [0.05, 0.95])
    p_tost = max((stat <= -TOST_MARGIN).mean(), (stat >= TOST_MARGIN).mean(), 1.0 / B)
    adj = holm({"C1": c1.boot.p, "C3": float(p_tost)})
    out["C1"] = {"diff": c1.boot.diff, "ci95": [c1.boot.ci_lo, c1.boot.ci_hi], "p_two_sided": c1.boot.p,
                 "p_holm": adj["C1"][0], "seed_diffs": c1.seed_diffs,
                 "pass": bool(adj["C1"][0] < 0.05 and c1.boot.diff > 0 and all(c1.seed_diffs.get(s, -1) > 0 for s in SEEDS))}
    out["C3"] = {"diff_projected": float(point), "ci90": [float(lo90), float(hi90)], "p_tost": float(p_tost),
                 "p_holm": adj["C3"][0], "seed_diffs_projected": seed_d, "diff_unprojected_pair_mean": float(np.mean(unproj)),
                 "margin": TOST_MARGIN,
                 "pass": bool(adj["C3"][0] < 0.05 and lo90 >= -TOST_MARGIN and hi90 <= TOST_MARGIN)}
    # C2: null refusal, deterministic
    c2 = {}
    for c in ("N0", "N1"):
        for pid, row, arr in by.get(c, []):
            union = arr["dce_N__found"] | arr["dce_O__found"]
            c2[pid] = {"condition": c, "seed": row["seed"], "found_rate_union": float(union.mean()),
                       "b1_found_rate": float(arr["b1__found"].mean())}
    out["C2"] = {"pairs": c2, "max_found_rate_union": max(v["found_rate_union"] for v in c2.values()),
                 "pass": len(c2) == 8 and all(v["found_rate_union"] <= C2_MAX for v in c2.values())}
    out["holm_m"] = 2
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("run", "analyze"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    if args.stage == "run":
        return cmd_run(cfg, args.out)
    meta = json.loads((args.out / "metadata.json").read_text())
    if meta.get("status") != "RUN_COMPLETE":
        raise SystemExit("run stage not complete")
    if (args.out / "results.json").exists():
        raise SystemExit("refusing to overwrite results.json")
    pairs = {}
    for pid in cfg["pair_ids"]:
        row = json.loads((args.out / "pairs" / f"pair_{pid:02d}.json").read_text())
        if row["n_points"] != cfg["n_points"]:
            raise SystemExit(f"pair {pid} incomplete")
        pairs[pid] = (row, dict(np.load(args.out / "pairs" / f"pair_{pid:02d}.npz")))
    res = analyze(pairs, cfg)
    (args.out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    meta.update(status="ANALYZED", analysis_git=git_state())
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"results written to {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
