"""D-052 (advisor review 2026-11-02): additional analyses, rules fixed in research/DECISIONS.md before any number.

m-analyze    — exp_0046 (saved z0 plus hash-verified checkpoints, no solver run): margin control for detection.
               s_M(z0) = |h(z0)|, h = Δℓ_new − Δℓ_old, Δℓ = ℓ_1 − ℓ_0; s_M,r = max |h| over 256 uniform points in the
               r-ball ∩ [0, 1]^5 (plus z0). Thresholds on the exp_0017 nulls 9–16; readout on the exp_0032 logit
               offsets (per |b|) and the 7 fine-tuned offset pairs of exp_0044, next to Δ-CE type N and CF-diff.

Usage:
    python -m dce.cli.review_0211 m-analyze --config configs/experiments/exp_0046.yaml --out runs/experiments/exp_0046
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dce.cli.g0_reeval import load_verified
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.cli.review_d048 import ASPECT, cfdiff_score, cmd_json
from dce.cli.v19_grid import grid_pairs
from dce.cli.v19_test import load_models
from dce.data.renderer import render
from dce.eval.pairs import load_pair
from dce.models.label_shift import LogitShift


@torch.no_grad()
def logit_gap(f, z: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Δℓ = ℓ_1 − ℓ_0 of f at the factor points z."""
    out = []
    for i in range(0, len(z), batch):
        lg = f(render(torch.as_tensor(z[i:i + batch], dtype=torch.float32)))
        out.append((lg[:, 1] - lg[:, 0]).numpy().astype(float))
    return np.concatenate(out)


def ball_samples(z0: np.ndarray, r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n points uniform in {‖z − z0‖ ≤ r} ∩ [0, 1]^d by rejection."""
    d, got = z0.shape[0], []
    while sum(len(g) for g in got) < n:
        u = rng.standard_normal((4 * n, d))
        u *= (r * rng.random(4 * n) ** (1 / d) / np.linalg.norm(u, axis=1))[:, None]
        z = z0 + u
        got.append(z[((z >= 0) & (z <= 1)).all(1)])
    return np.concatenate(got)[:n]


def margin_scores(f_old, f_new, z0: np.ndarray, y: np.ndarray | None, kappa: float, r: float, n_ball: int,
                  seed: int) -> dict[str, np.ndarray]:
    g_o, g_n = logit_gap(f_old, z0), logit_gap(f_new, z0)
    # identity check: every saved query must be valid for both models (margin ≥ κ in its class)
    yy = (g_o > 0).astype(int) if y is None else y
    sgn = np.where(yy == 1, 1.0, -1.0)
    if (sgn * g_o < kappa - 1e-4).any() or (sgn * g_n < kappa - 1e-4).any():
        raise SystemExit("saved z0 are not valid queries for these checkpoints (wrong models or points)")
    rng = np.random.default_rng(seed)
    balls = np.stack([ball_samples(z, r, n_ball, rng) for z in z0])
    flat = balls.reshape(-1, z0.shape[1])
    h_ball = (logit_gap(f_new, flat) - logit_gap(f_old, flat)).reshape(len(z0), n_ball)
    h0 = g_n - g_o
    return {"s_M": np.abs(h0), "s_Mr": np.maximum(np.abs(h0), np.abs(h_ball).max(1)), "y": yy}


def thresholds(null: np.ndarray) -> dict[str, float]:
    return {"fpr0": float(null.max()), "fpr0.05": float(np.quantile(null, 0.95))}


def m_analyze(cfg) -> dict:
    kappa, r, nb, base = cfg["kappa"], cfg["r"], cfg["n_ball"], cfg["ball_seed_base"]
    cache, grid = {}, {p["pair_id"]: p for p in grid_pairs()}

    def scores(f_old, f_new, a, pid, y):
        return margin_scores(f_old, f_new, a["z0"], y, kappa, r, nb, base + pid)

    null = {"s_M": [], "s_Mr": [], "cfdiff": []}
    for pid in cfg["null_pairs"]:
        a = dict(np.load(Path(cfg["null_run"]) / "pairs" / f"pair_{pid:02d}.npz"))
        _, f_old, f_new, _ = load_pair(pid, cache=cache)
        s = scores(f_old, f_new, a, pid, None)
        null["s_M"].append(s["s_M"]), null["s_Mr"].append(s["s_Mr"]), null["cfdiff"].append(cfdiff_score(a))
    null = {k: np.concatenate(v) for k, v in null.items()}
    taus = {k: thresholds(v) for k, v in null.items()}
    res = {"label": "D-052 M (descriptive)", "n_null_queries": int(null["s_M"].size), "tau": taus,
           "null_witness_share_dce": 0.0, "by_b": {}, "pairs": []}

    def readout(sc: dict, low: np.ndarray, dce: np.ndarray) -> dict:
        row = {"n_lowered": int(low.sum()), "dce_lowered": float(dce[low].mean()), "dce_all": float(dce.mean())}
        for name, s in sc.items():
            for k, t in taus[name].items():
                row[f"{name}_lowered_{k}"] = float((s[low] > t).mean())
                row[f"{name}_all_{k}"] = float((s > t).mean())
        return row

    rows_b = {}
    for npz in sorted((Path(cfg["h7_run"]) / "pairs").glob("pair_*.npz")):
        meta, a = json.loads(npz.with_suffix(".json").read_text()), dict(np.load(npz))
        f_old, _ = load_verified(Path(cfg["f_old_source"]), f"f_old_s{meta['seed']}", cache)
        s = scores(f_old, LogitShift(f_old, meta["b"]).eval(), a, meta["pair_id"], a["y"])
        if not np.allclose(s["s_M"], abs(meta["b"]), atol=1e-3):
            raise SystemExit(f"pair {meta['pair_id']}: |h| ≠ |b| for a logit shift")
        rows_b.setdefault(abs(meta["b"]), []).append(({"s_M": s["s_M"], "s_Mr": s["s_Mr"], "cfdiff": cfdiff_score(a)},
                                                     a["y"] == meta["lowered_class"], a["dce_N__found"]))
    for ab, rs in sorted(rows_b.items()):
        cat = {k: np.concatenate([x[0][k] for x in rs]) for k in ("s_M", "s_Mr", "cfdiff")}
        res["by_b"][str(ab)] = {"n_pairs": len(rs), **readout(cat, np.concatenate([x[1] for x in rs]),
                                                               np.concatenate([x[2] for x in rs]))}

    for run, pids in cfg["offset_runs"].items():
        for pid in pids:
            a = dict(np.load(Path(run) / "pairs" / f"pair_{pid}.npz"))
            if not np.array_equal(a["y"], (a["z0"][:, ASPECT] > 0.5).astype(a["y"].dtype)):
                raise SystemExit(f"{run} pair {pid}: saved y is not 1[aspect > 0.5]")
            if pid in grid and "exp_0026" in run:
                f_old, f_new, _ = load_models(cfg, grid[pid], cache)
            else:
                _, f_old, f_new, _ = load_pair(pid, cache=cache)
            s = scores(f_old, f_new, a, pid, a["y"])
            row = {"run": run, "pair_id": pid,
                   **readout({"s_M": s["s_M"], "s_Mr": s["s_Mr"], "cfdiff": cfdiff_score(a)}, a["y"] == 1,
                             a["dce_N__found"])}
            row["sM_minus_dce_fpr0"] = row["s_M_lowered_fpr0"] - row["dce_lowered"]
            res["pairs"].append(row)

    ps = res["pairs"]
    n_cheap = sum(p["s_M_lowered_fpr0"] >= p["dce_lowered"] - 0.05 for p in ps)
    witness_better = all(p["dce_lowered"] >= p["s_M_lowered_fpr0"] + 0.20 for p in ps)
    res["n_pairs_sM_ge_dce_minus_0.05"] = int(n_cheap)
    res["branch"] = ("whether is cheap" if n_cheap >= 5 else
                     "witness more sensitive" if witness_better else "report only (framing as whether is cheap)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("m-analyze",))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    torch.set_num_threads(int(cfg.get("threads", 8)))
    return cmd_json(m_analyze, "results_m.json")(cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
