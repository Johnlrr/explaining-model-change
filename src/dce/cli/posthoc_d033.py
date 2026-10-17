"""D-033: exploratory analyses of exp_0017 from its saved per-point outputs (no method is rerun). Post hoc.

(a) Recovery|reach and share of k* by condition × method; (b) projected Recovery (dce.analysis.projection);
(c) Δ-CE − B1 on T3 with the paired bootstrap (seed 6100), outside Holm; (d) SAR of B1, B1-det on N0/N1;
(e) Δ-CE NOT_FOUND split into "feasible but d > r" and "no feasible point".

Usage: python -m dce.cli.posthoc_d033 --run runs/experiments/exp_0017 --out runs/experiments/exp_0017_posthoc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from dce.analysis.projection import old_margin_gradients, project
from dce.cli.final_tier0 import NULL, load_pairs
from dce.cli.g0_tier0 import git_state, load_config
from dce.data.drift import clean_splits
from dce.eval.metrics import recovery_at_r, recovery_indicator
from dce.eval.pairs import load_pair
from dce.eval.rules import PairRecord, compare

D033_SEED = 6100


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if (args.out / "d033.json").exists():
        raise SystemExit("refusing to overwrite d033.json")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path(yaml.safe_load((args.run / "config.yaml").read_text())["tier_config"]).parent.parent / "experiments" / "exp_0017.yaml")
    results = json.loads((args.run / "results.json").read_text())
    pairs = load_pairs(args.run, cfg["pair_ids"])
    test = clean_splits(seed=cfg["data_seed"])["test"]
    r = float(cfg["r"])
    out = {"label": "EXPLORATORY, post hoc (D-033)", "source_run": str(args.run), "git": git_state()}

    by = {}
    for pid, (row, outs) in pairs.items():
        by.setdefault(row["condition"], []).append((pid, row, outs))

    # (a) reach and share
    a = {}
    for c, items in by.items():
        if c in NULL:
            continue
        pp = [results["per_pair"][str(pid)] for pid, _, _ in items]
        methods = pp[0]["methods"].keys()
        a[c] = {m: {"recovery_given_reach": float(np.mean([p["recovery_given_reach"][m] for p in pp])),
                    "share_k_star": float(np.nanmean([p["methods"][m]["share_k_star"] for p in pp]))} for m in methods}
        a[c]["dce_reach_share"] = float(np.mean([p["dce_reach_count"] / 300 for p in pp]))
    out["a_reach_and_share"] = a

    # (b) projected Recovery
    b = {}
    for c, items in by.items():
        if c in NULL:
            continue
        acc = {}
        for pid, row, outs in items:
            _, f_old, _, _ = load_pair(pid)
            z0 = np.load(args.run / "pairs" / f"pair_{pid:02d}.npz")
            g = old_margin_gradients(f_old, z0["z0"], test.y_clean[z0["point_ids"]])
            for m, o in outs.items():
                acc.setdefault(m, {"unprojected": [], "projected": []})
                acc[m]["unprojected"].append(recovery_at_r(o, row["k_star"]))
                acc[m]["projected"].append(recovery_at_r(project(o, g), row["k_star"]))
        b[c] = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}
    out["b_projected_recovery"] = b

    # (c) Δ-CE − B1 on T3, descriptive
    recs = [PairRecord(pid, row["seed"], {m: recovery_indicator(outs[m], row["k_star"]) for m in ("dce_N", "b1")})
            for pid, row, outs in by["T3"]]
    cmp = compare("dce_vs_b1_T3", recs, "dce_N", "b1", np.random.default_rng(D033_SEED), B=10_000)
    out["c_dce_minus_b1_T3"] = {"diff": cmp.boot.diff, "ci95": [cmp.boot.ci_lo, cmp.boot.ci_hi], "p": cmp.boot.p,
                                "seed_diffs": cmp.seed_diffs, "note": "descriptive, outside Holm"}

    # (d) SAR of B1, B1-det on null pairs
    out["d_sar_null"] = {c: {m: results["by_condition"][c][m]["sar_share_of_pairs"] for m in ("b1", "b1_det", "dce_N", "dce_O")}
                         for c in NULL}

    # (e) NOT_FOUND split
    e = {}
    for c, items in by.items():
        rows = {}
        for typ in ("dce_N", "dce_O"):
            fins, infs, founds = [], [], []
            for pid, _, _ in items:
                z = np.load(args.run / "pairs" / f"pair_{pid:02d}.npz")
                if f"{typ}_distance" not in z.files:
                    continue
                d = z[f"{typ}_distance"]
                founds.append(np.mean(np.isfinite(d) & (d <= r)))
                fins.append(np.mean(np.isfinite(d) & (d > r)))
                infs.append(np.mean(~np.isfinite(d)))
            if founds:
                rows[typ] = {"found_within_r": float(np.mean(founds)), "feasible_beyond_r": float(np.mean(fins)),
                             "no_feasible_point": float(np.mean(infs))}
        e[c] = rows
    out["e_not_found_split"] = e
    (args.out / "d033.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"written {args.out / 'd033.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
