"""D-054: calibrate κ on validation null updates, per architecture, then test the calibrated κ on fresh data.

Rule (fixed in D-054 before any calibration number): κ* is the smallest κ of the grid whose null witness share
(type N or O) on validation queries of the architecture's null pairs is ≤ the target (0.05). The grid is walked in
ascending order and stops at the first κ that passes; if none passes, κ* = None and no test is run. The test stage
reads κ* from calibration_<arch>.json (never from a hand-edited config) and runs the X4 pipeline unchanged
(`x4_resnet.run-chunked`) on fresh split-E data (data_seed_base from the config) with κ = κ*.

    python -m dce.cli.kappa_cal calibrate --arch cnn    --config configs/experiments/exp_0054.yaml --out runs/experiments/exp_0054
    python -m dce.cli.kappa_cal calibrate --arch resnet --config configs/experiments/exp_0054.yaml --out runs/experiments/exp_0054
    python -m dce.cli.kappa_cal test                    --config configs/experiments/exp_0054.yaml --out runs/experiments/exp_0054

Every stage resumes: finished κ levels, pairs and chunks are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from dce.cli import split_d, x4_resnet
from dce.cli.g0_tier0 import git_state, load_config, sha256


def _cfg_for(cfg: dict, arch: str, kappa: float) -> dict:
    a = cfg["calibration"][arch]
    base = load_config(Path(a["base_config"]))
    return base | {"kappa": float(kappa), "data": "val", "n_points": a["n_per_pair"], "pair_ids": a["null_pairs"],
                   "threads": cfg.get("threads", 8)}


def _null_share(arrays: dict) -> tuple[int, int]:
    w = arrays["dce_N__found"] | arrays["dce_O__found"]
    return int(w.sum()), int(w.size)


def cmd_calibrate(cfg: dict, out: Path, arch: str) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    root = out / f"calibration_{arch}"
    root.mkdir(parents=True, exist_ok=True)
    target, levels = cfg["calibration"]["target"], []
    kappa_star = None
    for kappa in cfg["calibration"]["grid"]:
        c = _cfg_for(cfg, arch, kappa)
        kdir = root / f"k{kappa:.2f}"
        if arch == "resnet":
            x4_resnet.cmd_run_chunked(c | {"models_run": cfg["calibration"]["resnet"]["models_run"]}, kdir, chunk=25)
            arrays = {pid: dict(np.load(kdir / "pairs" / f"pair_{pid}.npz")) for pid in c["pair_ids"]}
        else:
            (kdir / "pairs").mkdir(parents=True, exist_ok=True)
            arrays, cache = {}, {}
            for pid in c["pair_ids"]:
                f = kdir / "pairs" / f"pair_{pid}.npz"
                if not f.exists():
                    a, row = split_d.d_pair(c, split_d.BY_ID[pid], cache)
                    np.savez_compressed(f, **a)
                    (kdir / "pairs" / f"pair_{pid}.json").write_text(json.dumps(row, indent=2, default=float))
                arrays[pid] = dict(np.load(f))
        per_pair = {pid: _null_share(a) for pid, a in arrays.items()}
        k = sum(v[0] for v in per_pair.values())
        n = sum(v[1] for v in per_pair.values())
        level = {"kappa": kappa, "witnesses": k, "queries": n, "share": k / n,
                 "per_pair": {pid: v[0] / v[1] for pid, v in per_pair.items()}, "passes": k / n <= target}
        levels.append(level)
        print(json.dumps(level), flush=True)
        if level["passes"]:
            kappa_star = kappa
            break
    res = {"label": f"D-054 κ calibration, {arch}, validation nulls", "rule": cfg["calibration"]["rule"],
           "target": target, "grid": cfg["calibration"]["grid"], "levels": levels, "kappa_star": kappa_star,
           "config_sha256": sha256(Path(cfg["_path"])), "git": git_state(),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / f"calibration_{arch}.json").write_text(json.dumps(res, indent=2, default=float))
    print("kappa_star", kappa_star, flush=True)
    return 0


def cmd_test(cfg: dict, out: Path) -> int:
    cal = json.loads((out / "calibration_resnet.json").read_text())
    kappa_star = cal["kappa_star"]
    if kappa_star is None:
        raise SystemExit("no κ of the grid passes on validation: the test is not run (D-054)")
    t = cfg["test"]
    base = load_config(Path(t["base_config"]))
    c = base | {"kappa": float(kappa_star), "data_seed_base": t["data_seed_base"], "n_points": t["n_points"],
                "pair_ids": t["pair_ids"], "out_run": str(out / "test"), "threads": cfg.get("threads", 8)}
    tdir = out / "test"
    tdir.mkdir(parents=True, exist_ok=True)
    q = Path(base["models_run"]) / "qualification.json"       # validation qualification of exp_0050 (κ-free flags)
    if not (tdir / "qualification.json").exists():
        shutil.copy(q, tdir / "qualification.json")
    (tdir / "kappa_star.json").write_text(json.dumps({"kappa_star": kappa_star, "from": "calibration_resnet.json",
                                                       "calibration_sha256": sha256(out / "calibration_resnet.json")}))
    x4_resnet.cmd_run_chunked(c, tdir, chunk=20)
    res = x4_resnet.analyze(c)
    res["kappa"] = kappa_star
    res["git"] = git_state()
    (tdir / "results_x4_kstar.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("k_", "null"))}, default=float), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("calibrate", "test"))
    ap.add_argument("--arch", choices=("cnn", "resnet"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config) | {"_path": str(args.config)}
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    if args.stage == "calibrate":
        return cmd_calibrate(cfg, args.out, args.arch)
    return cmd_test(cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
