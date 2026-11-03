"""D-052 machine-sensitivity check (descriptive, declared before any X1–X3 number): the first n queries of some
exp_0049 pairs are re-run locally with `split_d.d_pair` unchanged and compared with the saved Kaggle arrays.

Phase-1 restart noise is drawn once for the whole batch (`search._restart_noise`, shape (R, B, p), sequential draws),
so a run on the first n points alone would get different noise for restarts ≥ 2 than those points had in the
300-point Kaggle run. The check therefore draws the noise for the full batch size and keeps its first n rows, which
reproduces the Kaggle noise exactly; nothing else is changed. Point ids, y and z0 must equal the saved prefix.

    python -m dce.cli.machine_check --config configs/experiments/exp_0049.yaml --out <dir> --pairs 110,115,100,105 --n 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dce.cli import split_d
from dce.cli.g0_tier0 import load_config
from dce.solvers import search


def full_batch_noise(n_full: int):
    orig = search._restart_noise

    def noise(z0, cfg, restart_seed):
        if z0.shape[0] > n_full:
            raise SystemExit(f"batch {z0.shape[0]} larger than the full batch {n_full}")
        full = torch.zeros((n_full, z0.shape[1]), dtype=z0.dtype)
        return orig(full, cfg, restart_seed)[:, : z0.shape[0]].to(z0.device)

    return noise


def compare(local: dict, saved: dict, n: int) -> dict:
    for k in ("point_ids", "y"):
        if not np.array_equal(local[k], saved[k][:n]):
            raise SystemExit(f"{k} differs from the saved prefix")
    if not np.array_equal(local["z0"], saved["z0"][:n]):
        raise SystemExit("z0 differs from the saved prefix")
    row = {}
    for k in ("dce_N__found", "dce_O__found", "b1_det__found", "b3__found"):
        if k in local:
            row[k + "_agree"] = float(np.mean(local[k] == saved[k][:n]))
    row["s_M_max_abs_diff"] = float(np.max(np.abs(local["s_M"] - saved["s_M"][:n])))
    cl, cs = split_d.cfdiff(local), split_d.cfdiff(saved)[:n]
    row["cfdiff_norm_local"] = [round(float(x), 4) for x in cl]
    row["cfdiff_norm_kaggle"] = [round(float(x), 4) for x in cs]
    row["cfdiff_norm_max_abs_diff"] = float(np.max(np.abs(cl - cs)))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    torch.set_num_threads(int(cfg.get("threads", 8)))
    search._restart_noise = full_batch_noise(int(cfg["n_points"]))
    run = Path(cfg["out_run"])
    res, cache = {"label": "D-052 machine-sensitivity check", "n": args.n, "pairs": {}}, {}
    for pid in (int(x) for x in args.pairs.split(",")):
        saved = dict(np.load(run / "pairs" / f"pair_{pid}.npz"))
        local, _ = split_d.d_pair(cfg | {"n_points": args.n}, split_d.BY_ID[pid], cache)
        res["pairs"][pid] = compare(local, saved, args.n)
        print(pid, {k: v for k, v in res["pairs"][pid].items() if not k.startswith("cfdiff_norm_l")}, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "machine_check.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
