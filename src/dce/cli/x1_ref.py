"""D-053 diagnostic: does IG-diff see an offset through its reference? (post hoc, after X1).

X1 found IG-diff recovering aspect on Threshold (0.73–0.75) where Grad-diff is blind (0.07). Hypothesis: CNN margins
saturate away from the boundary, so h = m_n − m_o varies with aspect only near the old and new boundaries; valid
queries (margins ≥ κ) sit where h is flat (Grad-diff blind), while the mean reference z̄ (aspect ≈ 0.5) sits on the
old boundary, so the path z̄ → z0 crosses the region where h changes. Prediction: with a reference on the query's
side, away from both boundaries (z̄ with aspect = 0.1 for y = 0 and 0.9 for y = 1), IG-diff Recovery on offsets falls
to the Grad-diff level, while on Shortcut (colour) it stays high.

Per pair: Recovery of IG-diff with the mean reference (64 and 256 steps; the saved X1 values use 64) and with the
query-side reference (64 steps), Grad-diff from the same points, and the completeness error |Σ IG-diff − (h(z0) −
h(ref))|. Points are processed in chunks (memory).

    python -m dce.cli.x1_ref split-a --out runs/experiments/exp_0051            # split A, pairs 0–8 (exploratory)
    python -m dce.cli.x1_ref split-d --out runs/experiments/exp_0051            # split D offsets 100–108 (test)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dce.cli.g0_tier0 import git_state
from dce.cli.review_0211 import integrated_gradients, rank_metrics
from dce.data.drift import clean_splits
from dce.eval.tier0_margins import Tier0Margin

ASPECT = 0
SIDE = {0: 0.1, 1: 0.9}


def ig_diff(f_old, f_new, z0: torch.Tensor, y: np.ndarray, base: torch.Tensor, steps: int, chunk: int = 50):
    out = []
    for i in range(0, len(y), chunk):
        j = slice(i, i + chunk)
        mn, mo = Tier0Margin(f_new, y[j]), Tier0Margin(f_old, y[j])
        out.append(integrated_gradients(mn, z0[j], base[j], steps) - integrated_gradients(mo, z0[j], base[j], steps))
    return np.concatenate(out)


@torch.no_grad()
def h_of(f_old, f_new, z: torch.Tensor, y: np.ndarray) -> np.ndarray:
    idx = torch.arange(len(y))
    return (Tier0Margin(f_new, y)(z, idx) - Tier0Margin(f_old, y)(z, idx)).numpy()


def side_reference(zbar: torch.Tensor, y: np.ndarray) -> torch.Tensor:
    b = zbar.expand(len(y), -1).clone()
    b[:, ASPECT] = torch.as_tensor([SIDE[int(c)] for c in y], dtype=b.dtype)
    return b


def pair_row(f_old, f_new, z0: np.ndarray, y: np.ndarray, k: int, zbar: torch.Tensor, grad_found, grad_dphi) -> dict:
    z = torch.as_tensor(z0, dtype=torch.float32)
    ones = np.ones(len(y), bool)
    rec = lambda d, s=slice(None): rank_metrics(ones[s], d[s], k)["recovery"]
    row = {}
    for name, base, steps in (("mean64", zbar.expand(len(y), -1), 64), ("mean256", zbar.expand(len(y), -1), 256),
                              ("side64", side_reference(zbar, y), 64)):
        d = ig_diff(f_old, f_new, z, y, base, steps)
        comp = np.abs(d.sum(1) - (h_of(f_old, f_new, z, y) - h_of(f_old, f_new, base, y)))
        row[f"ig_{name}"] = rec(d)
        row[f"ig_{name}_by_class"] = {c: rec(d, y == c) for c in (0, 1)}
        row[f"ig_{name}_completeness_err_max"] = float(comp.max())
    row["graddiff"] = rank_metrics(grad_found, grad_dphi, k)["recovery"]
    return row


def run_split_a(out: Path) -> dict:
    from dce.eval.pairs import load_pair
    sp = clean_splits(seed=1234)
    zbar = torch.as_tensor(sp["train"].phi.mean(0), dtype=torch.float32)
    res = {}
    for pid in range(9):
        a = dict(np.load(Path("runs/experiments/exp_0017/pairs") / f"pair_{pid:02d}.npz"))
        pair, f_old, f_new, _ = load_pair(pid, cache={})
        y = sp["test"].y_clean[a["point_ids"]]
        k = {"T1": 2, "T3": 1, "T4": 0}[pair["condition"]]
        res[pid] = {"condition": pair["condition"], **pair_row(f_old, f_new, a["z0"], y, k, zbar, a["b3__found"], a["b3__dphi"])}
        print(pid, {q: v for q, v in res[pid].items() if not q.endswith("by_class")}, flush=True)
    return res


def run_split_d(out: Path) -> dict:
    from dce.cli import split_d
    from dce.cli.g0_tier0 import load_config
    cfg = load_config(Path("configs/experiments/exp_0049.yaml"))
    sp = clean_splits(seed=cfg["data_seed"])
    zbar = torch.as_tensor(sp["train"].phi.mean(0), dtype=torch.float32)
    qual = {r["pair_id"]: r for r in json.loads(Path("runs/experiments/exp_0049/qualification.json").read_text())["pairs"]}
    aspect_rate = lambda d: float((np.argmax(np.abs(d), 1) == ASPECT).mean())
    res = {}
    for pid in [*range(110, 118), *range(100, 109)]:      # D-053 prediction: nulls and offsets, IG mean reference only
        p = split_d.BY_ID[pid]
        a = dict(np.load(Path("runs/experiments/exp_0049/pairs") / f"pair_{pid}.npz"))
        f_old, f_new, _ = split_d.load_d_models(cfg, p, {})
        d = ig_diff(f_old, f_new, torch.as_tensor(a["z0"], dtype=torch.float32), a["y"], zbar.expand(len(a["y"]), -1), 64)
        res[pid] = {"kind": p["kind"], "seed": p["seed"], "qualified": bool(qual[pid].get("qualified", p["kind"] == "null")),
                    "n": len(a["y"]), "ig_aspect_rate": aspect_rate(d), "graddiff_aspect_rate": aspect_rate(a["b3__dphi"]),
                    "ig_norm_mean": float(np.linalg.norm(d, axis=1).mean())}
        print(pid, res[pid], flush=True)
    nulls = [r for r in res.values() if r["kind"] == "null"]
    offs = [r for r in res.values() if r["kind"] == "offset" and r["qualified"]]
    lo, hi = min(r["ig_aspect_rate"] for r in nulls), max(r["ig_aspect_rate"] for r in nulls)
    pooled = lambda key: sum(r[key] * r["n"] for r in nulls) / sum(r["n"] for r in nulls)
    res["summary"] = {"null_ig_aspect_rate_pooled": pooled("ig_aspect_rate"),
                      "null_graddiff_aspect_rate_pooled": pooled("graddiff_aspect_rate"),
                      "null_ig_aspect_rate_range": [lo, hi],
                      "offsets_within_null_range": f"{sum(lo <= r['ig_aspect_rate'] <= hi for r in offs)} of {len(offs)}"}
    s = res["summary"]
    s["prediction_met"] = bool(s["null_ig_aspect_rate_pooled"] >= 0.30 and s["null_graddiff_aspect_rate_pooled"] <= 0.15
                               and int(s["offsets_within_null_range"].split()[0]) >= 6)
    print(s, flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("split-a", "split-d"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    res = run_split_a(args.out) if args.stage == "split-a" else run_split_d(args.out)
    path = args.out / f"x1_ref_{args.stage.replace('-', '_')}.json"
    path.write_text(json.dumps({"label": f"D-053 IG reference diagnostic, {args.stage}", "pairs": res, "git": git_state()},
                               indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
