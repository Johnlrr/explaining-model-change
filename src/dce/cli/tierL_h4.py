"""H-004 on Tier L (spec §8 H4, amendment v1.4 E3–E4): solver versus closed forms.

Usage:
    python -m dce.cli.tierL_h4 --config configs/experiments/exp_0006.yaml --out runs/experiments/exp_0006

One solver (configs/tiers/tierL.yaml `solver`) on every seed set of `seed_sets` (v1.5 F3: seeds
0–999 and 1000–1999) for every dim; H4 must pass on every set.
Checks, each with the spec §8 tolerances (share ≥ 0.99 of pairs in every dim; a pair without a
solution fails):
  delta_ce:   rel. error of δ_Δ (Proposition 3) and |cos(P u, P Δw)| (Proposition 4);
  ce_old/new: rel. error of the ordinary-CE distance and |cos(u, u*)| against u* = −(m(z0)+κ)/‖w‖² w.
(exp_0005 ran the original spec solver and its variants from commit 157320b; exp_0006 the v1.4
solver from commit f45d5ba.)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from dce.cli.g0_tier0 import git_state
from dce.data.tierL import abs_cos, ce_closed_form, closed_form, project_out, sample_pair
from dce.solvers.dce import delta_ce
from dce.solvers.ordinary_ce import ordinary_ce
from dce.solvers.search import SolverConfig


def summarise(found, rel, cos, tol) -> dict:
    rel = np.where(found, rel, np.inf)
    cos = np.where(found, cos, 0.0)
    fin = rel[np.isfinite(rel)]
    out = {"found_rate": float(found.mean()),
           "share_rel_lt_tol": float((rel < tol["rel_error"]).mean()),
           "share_cos_gt_tol": float((cos > tol["abs_cos"]).mean()),
           "rel_median": float(np.median(fin)) if len(fin) else None,
           "rel_max": float(fin.max()) if len(fin) else None,
           "cos_min_found": float(cos[found].min()) if found.any() else None}
    out["pass"] = out["share_rel_lt_tol"] >= tol["share"] and out["share_cos_gt_tol"] >= tol["share"]
    return out


def run_dim(pairs, cfg: SolverConfig, restart_seed: int, kappa: float, tol: dict) -> dict:
    # float32 evaluation models (v1.5 F2); SLSQP promotes to float64 internally
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    w_o, w_n = t([q.w_o for q in pairs]), t([q.w_n for q in pairs])
    b_o, b_n, z0 = t([q.b_o for q in pairs]), t([q.b_n for q in pairs]), t([q.z0 for q in pairs])
    m_o = lambda z, idx: (w_o[idx] * z).sum(1) + b_o[idx]
    m_n = lambda z, idx: (w_n[idx] * z).sum(1) + b_n[idx]
    out = {}

    res = delta_ce(z0, m_o, m_n, cfg, restart_seed)
    cf = [closed_form(q, kappa) for q in pairs]
    u = res.z.double().numpy() - np.stack([q.z0 for q in pairs])
    found = res.found.numpy()
    rel = np.array([abs(res.distance[i].item() - c["delta"]) / c["delta"] for i, c in enumerate(cf)])
    cos = np.array([abs_cos(project_out(u[i], q.w_o), project_out(q.dw, q.w_o)) if found[i] else 0.0
                    for i, q in enumerate(pairs)])
    out["delta_ce"] = {**summarise(found, rel, cos, tol), "share_case_b": float(np.mean([c["case"] == "B" for c in cf])),
                       "seconds": round(res.seconds, 2), "polish_feasible": res.n_polished,
                       "polish_runs": res.n_polish_runs, "answers_from_phase2": res.n_from_phase2}

    for name, m, key in (("ce_old", m_o, "w_o"), ("ce_new", m_n, "w_n")):
        r = ordinary_ce(z0, m, cfg, restart_seed)
        ref = [ce_closed_form(getattr(q, key), q.b_o if key == "w_o" else q.b_n, q.z0, kappa) for q in pairs]
        u = r.z.double().numpy() - np.stack([q.z0 for q in pairs])
        f = r.found.numpy()
        rel = np.array([abs(r.distance[i].item() - c["delta"]) / c["delta"] for i, c in enumerate(ref)])
        cos = np.array([abs_cos(u[i], c["u"]) if f[i] else 0.0 for i, c in enumerate(ref)])
        out[name] = {**summarise(f, rel, cos, tol), "seconds": round(r.seconds, 2),
                     "polish_feasible": r.n_polished, "polish_runs": r.n_polish_runs,
                     "answers_from_phase2": r.n_from_phase2}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    cfg = {**yaml.safe_load(Path(cfg["tier_config"]).read_text()), **cfg}
    out = args.out
    if (out / "metrics.json").exists():
        print(f"refusing to overwrite completed run {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.set_num_threads(cfg.get("threads", 8))
    meta = {"experiment": cfg["experiment_id"], "gate": "G1", "tier": "L", "git": git_state(),
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": "cpu", "status": "RUNNING"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    start = time.time()

    solver = SolverConfig(**{**cfg["solver"], "kappa": cfg["kappa"], "r": float(cfg["r"])})
    tol = cfg["tolerance"]
    results = {"tolerance": tol, "solver": solver.__dict__, "sets": {}}
    for name, (lo, hi) in cfg["seed_sets"].items():
        res_set = {}
        for p in cfg["dims"]:
            pairs = [sample_pair(s, p, cfg["kappa"]) for s in range(lo, hi)]
            res_set[str(p)] = run_dim(pairs, solver, cfg["restart_base"] + lo, cfg["kappa"], tol)
            print(name, p, json.dumps({k: {m: v[m] for m in ("found_rate", "share_rel_lt_tol", "share_cos_gt_tol", "pass")}
                                       for k, v in res_set[str(p)].items()}), flush=True)
        results["sets"][name] = {"seeds": [lo, hi - 1], "dims": res_set}
    results["h4_pass"] = all(v["pass"] for st in results["sets"].values() for d in st["dims"].values()
                             for v in d.values())
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    meta.update(status="SUCCESS", seconds=round(time.time() - start, 1))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"h4_pass": results["h4_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
