"""v1.10 development runs on VALIDATION only (clean_splits(data_seed)["val"], clean labels of the original rule).

h8       — per grid pair: N valid points, Δ-CE N, B3 at z0 and at the witness z*, ∇m_o at z0 and z*; the H8 2×2 gaps
           on the Δ-CE support, support size, locality descriptives and runtime. Resumable per pair.
tc-cost  — T-C cost: per pair, Δ-CE on Q points; for each NOT_FOUND query, max of the slack s over B(z0, r) ∩ [0,1]^5
           by uniform sampling (M per query) and multi-start projected ascent on s (no proximity term); wall time.
No test split is read: the data come only from the validation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dce.analysis.projection import old_margin_gradients
from dce.analysis.v110 import CELLS, boot_h8, h8_indicators, locality, slack
from dce.baselines.b3_grad import b3_grad
from dce.cli.calibrate_r import solver_config
from dce.cli.final_tier0 import Timer
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.cli.v19_grid import grid_pairs
from dce.cli.v19_test import load_models
from dce.data.drift import clean_splits
from dce.eval.records import from_direction, from_points
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.solvers.dce import delta_ce

K_INDEX = {"aspect": 0, "color": 2}


def setup(cfg, p, cache, n):
    f_old, f_new, entries = load_models(cfg, p, cache)
    val = clean_splits(seed=cfg["data_seed"])["val"]
    y = val.y_clean.astype(np.int64)
    keep = valid_points(val.phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), n,
                        cfg["seeds"]["points_base"] + p["pair_id"], cfg["kappa"])
    z0 = torch.as_tensor(val.phi[keep], dtype=torch.float32)
    return f_old, f_new, entries, keep, z0, y[keep]


def h8_pair(cfg, p, cache):
    n = cfg["n_points"]
    f_old, f_new, entries, keep, z0, y = setup(cfg, p, cache, n)
    m_old, m_new = Tier0Margin(f_old, y), Tier0Margin(f_new, y)
    base = solver_config(cfg, r=float(cfg["r"]))
    with Timer() as t_dce:
        res = delta_ce(z0, m_old, m_new, base, cfg["seeds"]["restart_base"] + p["pair_id"], kind="N")
    dce = from_points(res.found, res.z, z0)
    zs = torch.where(res.found[:, None], res.z, z0)          # z* on S, z0 elsewhere (rows outside S are not scored)
    with Timer() as t_rest:
        b3 = {}
        for key, z in (("z0", z0), ("zs", zs)):
            v, d = b3_grad(z, m_old, m_new)
            b3[key] = from_direction(v, d)
        g = {"z0": old_margin_gradients(f_old, z0.numpy(), y), "zs": old_margin_gradients(f_old, zs.numpy(), y)}
    arrays = {"point_ids": keep, "y": y, "z0": z0.numpy(), "zs": zs.numpy(), "dce__found": dce.found,
              "dce__dphi": dce.dphi, "grad_z0": g["z0"], "grad_zs": g["zs"]}
    for key, o in b3.items():
        arrays[f"b3_{key}__found"], arrays[f"b3_{key}__dphi"] = o.found, o.dphi
    row = {**p, "n_points": n, "support": int(dce.found.sum()), "found_rate": float(dce.found.mean()),
           "wall_dce_s": round(t_dce.wall, 1), "wall_rest_s": round(t_rest.wall, 1), "checkpoints": entries}
    return arrays, row


def h8_summary(out: Path, ids) -> dict:
    per, inds = [], []
    for pid in ids:
        row = json.loads((out / "pairs" / f"pair_{pid}.json").read_text())
        a = dict(np.load(out / "pairs" / f"pair_{pid}.npz"))
        from dce.eval.metrics import MethodOutput
        dce = MethodOutput(a["dce__found"], a["dce__dphi"])
        b3 = {k: MethodOutput(a[f"b3_{k}__found"], a[f"b3_{k}__dphi"]) for k in ("z0", "zs")}
        ind = h8_indicators(dce, b3, {"z0": a["grad_z0"], "zs": a["grad_zs"]}, K_INDEX[row["k_star"]])
        inds.append(ind)
        s = dce.found
        loc = locality(a["z0"][s], a["zs"][s], a["grad_z0"][s], a["grad_zs"][s], a["dce__dphi"][s])
        per.append({"pair_id": pid, "alpha": row["alpha"], "t": row["t"], "support": row["support"],
                    "found_rate": row["found_rate"], "wall_dce_s": row["wall_dce_s"],
                    "gap": {f"{c[0]},{c[1]}": float(ind[c][0].mean() - ind[c][1].mean()) if s.any() else None for c in CELLS},
                    "median_dist": float(np.median(loc["dist"])) if s.any() else None,
                    "median_one_minus_cos": float(np.median(loc["one_minus_cos"])) if s.any() else None})
    pooled = boot_h8(inds, np.random.default_rng(0), 2000)
    pooled["gap"] = {f"{c[0]},{c[1]}": v for c, v in pooled["gap"].items()}
    pooled["gap_ci90"] = {f"{c[0]},{c[1]}": v for c, v in pooled["gap_ci90"].items()}
    return {"label": "v1.10 H8 development (validation, descriptive)", "pairs": per, "pooled": pooled}


def cmd_h8(cfg, out: Path) -> int:
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", 8))
    by_id, cache = {p["pair_id"]: p for p in grid_pairs()}, {}
    for pid in cfg["pair_ids"]:
        js = out / "pairs" / f"pair_{pid}.json"
        if js.exists():
            continue
        arrays, row = h8_pair(cfg, by_id[pid], cache)
        np.savez_compressed(out / "pairs" / f"pair_{pid}.npz", **arrays)
        js.write_text(json.dumps(row, indent=2))
        print(f"pair {pid}: support {row['support']}/{row['n_points']}, Δ-CE {row['wall_dce_s'] / 60:.1f} min", flush=True)
    summ = h8_summary(out, cfg["pair_ids"])
    summ["git"] = git_state()
    (out / "h8_dev.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ["pooled"], indent=2))
    return 0


def sample_ball(z0: torch.Tensor, r: float, m: int, gen: torch.Generator, chunk: int = 200_000) -> tuple[torch.Tensor, int]:
    """Exactly m uniform samples in B(z0, r) ∩ [0,1]^p: draws in the ball, rejection outside the cube, repeated until m
    are accepted (v1.10 §4). Returns (samples, number of draws)."""
    p, kept, draws = z0.shape[0], [], 0
    while sum(len(k) for k in kept) < m:
        d = torch.randn(chunk, p, generator=gen)
        d = d / d.norm(dim=1, keepdim=True) * r * torch.rand(chunk, 1, generator=gen) ** (1.0 / p)
        z = z0 + d
        kept.append(z[((z >= 0) & (z <= 1)).all(1)])
        draws += chunk
    return torch.cat(kept)[:m], draws


def project_ball_box(z: torch.Tensor, z0: torch.Tensor, r: float, iters: int = 5) -> torch.Tensor:
    for _ in range(iters):   # alternating projections onto the cube and the ball
        z = z.clamp(0, 1)
        d = z - z0
        n = d.norm(dim=1, keepdim=True)
        z = z0 + d * torch.clamp(r / n.clamp_min(1e-12), max=1.0)
    return z.clamp(0, 1)


def max_slack(z0, i, m_old, m_new, kappa, r, M, starts, steps, gen, batch=4096):
    """max over B(z0, r) ∩ cube of s, by uniform sampling then projected Adam ascent from the best `starts` samples."""
    zs, draws = sample_ball(z0, r, M, gen)
    vals = []
    with torch.no_grad():
        for j in range(0, len(zs), batch):
            zb = zs[j : j + batch]
            idx = torch.full((len(zb),), i, dtype=torch.long)
            vals.append(slack(m_old(zb, idx), m_new(zb, idx), kappa))
    vals = torch.cat(vals)
    s_sample = float(vals.max())
    top = zs[vals.topk(min(starts, len(zs))).indices].clone().requires_grad_(True)
    opt = torch.optim.Adam([top], lr=0.01)
    idx = torch.full((len(top),), i, dtype=torch.long)
    for _ in range(steps):
        opt.zero_grad()
        (-slack(m_old(top, idx), m_new(top, idx), kappa).sum()).backward()
        opt.step()
        with torch.no_grad():
            top.copy_(project_ball_box(top, z0, r))
    with torch.no_grad():
        s_fin = slack(m_old(top, idx), m_new(top, idx), kappa)
    k = int(s_fin.argmax())
    # the ascent can end below its start (Adam step + projection): the estimate is the max of both stages
    return {"s_sample": s_sample, "s_ascent": float(s_fin.max()), "s_max": max(s_sample, float(s_fin.max())),
            "n_samples": len(zs), "n_draws": draws, "dist_best": float((top[k] - z0).norm())}


def cmd_tc_cost(cfg, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", 8))
    by_id, cache, rows = {p["pair_id"]: p for p in grid_pairs()}, {}, []
    r, kappa = float(cfg["r"]), cfg["kappa"]
    for pid in cfg["tc_pair_ids"]:
        p = by_id[pid]
        f_old, f_new, _, keep, z0, y = setup(cfg, p, cache, cfg["tc_queries"])
        m_old, m_new = Tier0Margin(f_old, y), Tier0Margin(f_new, y)
        with Timer() as t:
            res = delta_ce(z0, m_old, m_new, solver_config(cfg, r=r), cfg["seeds"]["restart_base"] + pid, kind="N")
        nf = np.flatnonzero(~res.found.numpy())
        for M in cfg["tc_M"]:
            gen = torch.Generator().manual_seed(cfg["tc_seed"] + pid)
            for i in nf[: cfg["tc_max_notfound"]]:
                with Timer() as tq:
                    q = max_slack(z0[i], int(i), m_old, m_new, kappa, r, int(M), cfg["tc_starts"], cfg["tc_steps"], gen)
                rows.append({"pair_id": pid, "alpha": p["alpha"], "t": p["t"], "M": int(M), "query": int(i), **q,
                             "wall_s": round(tq.wall, 2)})
                print(json.dumps(rows[-1]), flush=True)
        rows.append({"pair_id": pid, "dce_found": int(res.found.sum()), "queries": len(keep), "wall_dce_s": round(t.wall, 1)})
    (out / "tc_cost.json").write_text(json.dumps({"label": "v1.10 T-C cost (validation)", "git": git_state(), "rows": rows}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("h8", "tc-cost"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    return {"h8": cmd_h8, "tc-cost": cmd_tc_cost}[args.stage](cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
