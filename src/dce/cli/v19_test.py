"""D-036 / D-038 / D-039: v1.9 Stage A test run on fresh renderer data, local CPU, locked solver.

run          — per pair (config order; resumable): test set new_test_set(7000 + pair_id, 5 000) with clean labels of the
               original rule (as T4 in exp_0019), 300 valid points (κ = 0.5, order seed points_base + pair_id), then
               Δ-CE N, B1-det, B3 (+ ∇m_o for the projection), A2 = ordinary CE of f_new (as exp_0017), restarts
               restart_base + pair_id; on the κ-curve pairs also Δ-CE N at κ ∈ {0, 0.25, 1.0} on the same points
               (as A3 in exp_0017). No metric is printed.
consistency  — v1.9 §5.3 on pairs 26–27 only; written before any H5 statistic is computed.
analyze      — H5a–H5d (refuses unless consistency passed and all 32 pairs are done) and the descriptive table.
figure       — v1.9 §5.2 map at the measured (α̂, t̂), from results.json only.
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

from dce.analysis.projection import old_margin_gradients
from dce.analysis.v19 import consistency, descriptive, h5
from dce.baselines.b1_diff import b1_diff
from dce.baselines.b3_grad import b3_grad
from dce.cli.calibrate_r import solver_config
from dce.cli.confirm_tier0 import new_test_set
from dce.cli.final_tier0 import Timer
from dce.cli.g0_reeval import load_verified
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.cli.v19_grid import grid_pairs, model_name
from dce.eval.records import check_outputs, from_direction, from_points, runtime_row
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.models.cnn import SmallCNN
from dce.solvers.dce import delta_ce
from dce.solvers.ordinary_ce import ordinary_ce


def load_models(cfg, p, cache):
    f_old, e_old = load_verified(Path(cfg["f_old_source"]), f"f_old_s{p['seed']}", cache)
    src = Path(cfg["models_run"])
    man = {a["name"]: a for a in json.loads((src / "artifact_manifest.json").read_text())["artifacts"]}
    entry = man[model_name(p)]
    path = src / entry["path"]
    if sha256(path) != entry["sha256"]:
        raise SystemExit(f"hash mismatch for {path}")
    f_new = SmallCNN()
    f_new.load_state_dict(torch.load(path))
    f_new.eval()
    return f_old, f_new, [e_old, {"name": entry["name"], "path": str(path), "sha256": entry["sha256"]}]


def run_pair(cfg, p, qual, cache):
    pid, n = p["pair_id"], cfg["n_points"]
    base = solver_config(cfg, r=float(cfg["r"]))
    f_old, f_new, entries = load_models(cfg, p, cache)
    phi, y, digest = new_test_set(cfg["data_seed_base"] + pid, cfg["n_samples"])
    keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), n, cfg["seeds"]["points_base"] + pid, cfg["kappa"])
    z0 = torch.as_tensor(phi[keep], dtype=torch.float32)
    m_old, m_new = Tier0Margin(f_old, y[keep]), Tier0Margin(f_new, y[keep])
    seed = cfg["seeds"]["restart_base"] + pid
    outs, rt, extra = {}, {}, {}

    def solver(name, fn):
        with Timer() as t:
            res = fn()
        outs[name], rt[name] = from_points(res.found, res.z, z0), runtime_row(res, t.wall, n)
        extra[f"{name}_distance"] = res.distance.numpy()

    solver("dce_N", lambda: delta_ce(z0, m_old, m_new, base, seed, kind="N"))
    with Timer() as t:
        valid, _, ce_o, ce_n = b1_diff(z0, m_old, m_new, base, seed, deterministic=True)
    outs["b1_det"], rt["b1_det"] = from_points(valid, ce_n.z, ce_o.z), {"wall_s_per_point": t.wall / n}
    with Timer() as t:
        v, d = b3_grad(z0, m_old, m_new)
    outs["b3"], rt["b3"] = from_direction(v, d), {"wall_s_per_point": t.wall / n}
    solver("a2", lambda: ordinary_ce(z0, m_new, base, seed))
    if pid in cfg["kappa_curve_pairs"]:
        for kap in cfg["kappa_curve"]:
            solver(f"dce_N_k{kap}", lambda kap=kap: delta_ce(z0, m_old, m_new, replace(base, kappa=float(kap)), seed, kind="N"))
    check_outputs(outs, n, z0.shape[1])
    extra["grad_old"] = old_margin_gradients(f_old, z0.numpy(), y[keep])
    arrays = {"point_ids": keep, "y": y[keep], "z0": z0.numpy(), **extra}
    for k, o in outs.items():
        arrays[f"{k}__found"], arrays[f"{k}__dphi"] = o.found, o.dphi
    row = {**p, "alpha_hat": qual["alpha_hat"], "t_hat": qual["t_hat"], "n_points": n, "data_seed": cfg["data_seed_base"] + pid,
           "data_sha256": digest, "runtime": rt, "checkpoints": entries}
    return arrays, row


def qualification(cfg) -> dict:
    q = json.loads((Path(cfg["models_run"]) / "qualification.json").read_text())
    return {r["pair_id"]: r for r in q["pairs"]}, q["grid"]


def cmd_run(cfg, out: Path) -> int:
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", 8))
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "experiment": cfg["experiment_id"], "decision": "D-036, D-037, D-038, D-039", "git_runs": [],
        "python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(), "device": "cpu (8 threads)"}
    meta["git_runs"].append({"git": git_state(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    meta["status"] = "RUNNING"
    meta_path.write_text(json.dumps(meta, indent=2))
    qual, _ = qualification(cfg)
    by_id = {p["pair_id"]: p for p in grid_pairs()}
    cache, start = {}, time.time()
    for pid in cfg["pair_ids"]:
        npz, js = out / "pairs" / f"pair_{pid}.npz", out / "pairs" / f"pair_{pid}.json"
        if npz.exists() and js.exists():
            print(f"pair {pid}: already done, skipped", flush=True)
            continue
        with Timer() as t:
            arrays, row = run_pair(cfg, by_id[pid], qual[pid], cache)
        np.savez_compressed(npz, **arrays)
        row["wall_s_pair"] = round(t.wall, 1)
        js.write_text(json.dumps(row, indent=2))
        print(f"pair {pid} done in {t.wall / 60:.1f} min (total {(time.time() - start) / 3600:.2f} h)", flush=True)
    meta["status"] = "RUN_COMPLETE" if all((out / "pairs" / f"pair_{p}.json").exists() for p in by_id) else "PARTIAL"
    meta_path.write_text(json.dumps(meta, indent=2))
    return 0


def load_pairs(out: Path, ids) -> dict:
    pairs = {}
    for pid in ids:
        row = json.loads((out / "pairs" / f"pair_{pid}.json").read_text())
        pairs[pid] = (row, dict(np.load(out / "pairs" / f"pair_{pid}.npz")))
    return pairs


def cmd_consistency(cfg, out: Path) -> int:
    path = out / "consistency.json"
    if path.exists():
        raise SystemExit("refusing to overwrite consistency.json")
    res = consistency(load_pairs(out, cfg["consistency_pairs"]))
    res["git"] = git_state()
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


def cmd_analyze(cfg, out: Path) -> int:
    con = json.loads((out / "consistency.json").read_text())
    if not con["pass"]:
        raise SystemExit("§5.3 consistency check failed: analysis stops (v1.9 §5.3)")
    meta = json.loads((out / "metadata.json").read_text())
    if meta.get("status") != "RUN_COMPLETE":
        raise SystemExit("run stage not complete")
    if (out / "results.json").exists():
        raise SystemExit("refusing to overwrite results.json")
    _, grid = qualification(cfg)
    pairs = load_pairs(out, [p["pair_id"] for p in grid_pairs()])
    res = {"label": "D-036 Stage A (v1.9, v1.9.1, D-038, D-039)", "exclusions": grid["exclusions"],
           "order_violations": grid["order_violations"], "consistency": con,
           "H5": h5(pairs, grid["exclusions"], B=cfg["bootstrap_resamples"]), "descriptive": descriptive(pairs)}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    meta.update(status="ANALYZED", analysis_git=git_state())
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"results written to {out / 'results.json'}")
    return 0


def cmd_figure(cfg, out: Path) -> int:
    from dce.analysis.v19_figure import make_map

    rows = json.loads((out / "results.json").read_text())["descriptive"]["pairs"]
    make_map(rows, out / "map_v19.pdf")
    print(f"figure written to {out / 'map_v19.pdf'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("run", "consistency", "analyze", "figure"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    return {"run": cmd_run, "consistency": cmd_consistency, "analyze": cmd_analyze, "figure": cmd_figure}[args.stage](cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
