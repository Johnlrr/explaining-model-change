"""H-004 on Tier L: run the Δ-CE solver on random linear pairs and compare with Proposition 3–4.

Usage:
    python -m dce.cli.tierL_h4 --config configs/experiments/exp_0005.yaml --out runs/experiments/exp_0005

The config lists solver variants (`variants: {name: {SolverConfig overrides}}`); each is run on the
same pairs (seeds 0..n_pairs−1, both dims). H4 passes for a variant iff, in every dim, the share of
pairs with relative error of δ_Δ < rel_error and the share with |cos(Pu, PΔw)| > abs_cos are both
≥ share (spec §8 H4); a pair without a witness fails both.
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
from dce.data.tierL import abs_cos, closed_form, project_out, sample_pair
from dce.solvers.dce import delta_ce
from dce.solvers.search import SolverConfig


def run_variant(pairs, overrides: dict, base: dict, restart_seed: int) -> dict:
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)  # noqa: E731
    w_o, w_n = t([q.w_o for q in pairs]), t([q.w_n for q in pairs])
    b_o, b_n, z0 = t([q.b_o for q in pairs]), t([q.b_n for q in pairs]), t([q.z0 for q in pairs])
    cfg = SolverConfig(**{**base, **overrides})
    res = delta_ce(z0, lambda z: (w_o * z).sum(1) + b_o, lambda z: (w_n * z).sum(1) + b_n, cfg, restart_seed)
    cf = [closed_form(q) for q in pairs]
    ref = np.array([c["delta"] for c in cf])
    found = res.found.numpy()
    rel = np.where(found, np.abs(res.distance.double().numpy() - ref) / ref, np.inf)
    cos = np.array([abs_cos(project_out(res.z[i].double().numpy() - q.z0, q.w_o), project_out(q.dw, q.w_o))
                    if found[i] else 0.0 for i, q in enumerate(pairs)])
    return {"found": found, "rel": rel, "cos": cos, "case_b": np.array([c["case"] == "B" for c in cf]),
            "seconds": res.seconds}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    tier = yaml.safe_load(Path(cfg["tier_config"]).read_text())
    cfg = {**tier, **cfg}
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

    base = {**cfg["solver"], "kappa": cfg["kappa"], "r": float(cfg["r"])}
    tol = cfg["tolerance"]
    pairs = {p: [sample_pair(s, p, cfg["kappa"]) for s in range(cfg["n_pairs"])] for p in cfg["dims"]}
    results = {"tolerance": tol, "n_pairs": cfg["n_pairs"], "variants": {}}
    for name, overrides in cfg["variants"].items():
        row = {"overrides": overrides or {}, "dims": {}}
        for p, ps in pairs.items():
            r = run_variant(ps, overrides or {}, base, cfg["restart_base"])
            fin = r["rel"][np.isfinite(r["rel"])]
            row["dims"][str(p)] = {
                "found_rate": float(r["found"].mean()), "share_case_b": float(r["case_b"].mean()),
                "share_rel_lt_tol": float((r["rel"] < tol["rel_error"]).mean()),
                "share_rel_lt_1e-2": float((r["rel"] < 1e-2).mean()),
                "share_cos_gt_tol": float((r["cos"] > tol["abs_cos"]).mean()),
                "share_cos_gt_0.99": float((r["cos"] > 0.99).mean()),
                "rel_median_found": float(np.median(fin)) if len(fin) else None,
                "rel_q99_found": float(np.quantile(fin, 0.99)) if len(fin) else None,
                "seconds": round(r["seconds"], 2)}
        d = row["dims"].values()
        row["h4_pass"] = all(v["share_rel_lt_tol"] >= tol["share"] and v["share_cos_gt_tol"] >= tol["share"] for v in d)
        results["variants"][name] = row
        print(name, json.dumps({p: {k: v[k] for k in ("found_rate", "share_rel_lt_tol", "share_cos_gt_tol")}
                                for p, v in row["dims"].items()}), "H4", row["h4_pass"], flush=True)
    results["h4_pass_spec_solver"] = results["variants"].get("spec", {}).get("h4_pass")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    meta.update(status="SUCCESS", seconds=round(time.time() - start, 1))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
