"""Final Tier 0 procedure (EVALUATION §8, §19; Route A = H4 ∧ H2a since v1.8 I6; D-028).

Two stages, one code path:
  run      — every method on the N valid points of each pair; per-point outputs saved per pair (resumable: a pair
             whose file exists is skipped). Prints progress and timing only, never a metric.
  analyze  — metrics, statistics and decision rules from the saved outputs; refuses an incomplete run.

The test split is used only with `run_kind: final` in the config and `--final` on the command line; every other
run uses validation (smoke). Methods: dce_N, dce_O (N0, N1), b1, b1_det, b3, b4, b5, a1, a2, a3_k0, a3_k1.

Usage:
    python -m dce.cli.final_tier0 run --config configs/experiments/exp_0017.yaml --out runs/experiments/exp_0017 --final
    python -m dce.cli.final_tier0 analyze --config configs/experiments/exp_0017.yaml --out runs/experiments/exp_0017
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
import yaml

from dce.baselines.b1_diff import b1_diff
from dce.baselines.b3_grad import b3_grad
from dce.baselines.b4_perm import b4_perm
from dce.baselines.b5_line import b5_line
from dce.cli.calibrate_r import solver_config
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.data.drift import clean_splits
from dce.data.renderer import render
from dce.eval.metrics import (
    MethodOutput,
    auroc,
    found_rate,
    median_norm,
    pooled_found_rate,
    recovery_at_r,
    recovery_given_reach,
    sar_pair,
    share_of_kstar,
)
from dce.eval.pairs import load_pair
from dce.eval.records import (
    b0_disagreement,
    check_outputs,
    check_run_counts,
    from_direction,
    from_points,
    pair_record,
    runtime_row,
)
from dce.eval.rules import (
    BOOTSTRAP_SEEDS,
    b5_outcome,
    bootstrap_rng,
    compare,
    holm_family,
    p_tie_contradicted,
    route_a,
)
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.models.cnn import predict_logits
from dce.solvers.dce import delta_ce
from dce.solvers.ordinary_ce import ordinary_ce

PER_POINT = ("dce_N", "dce_O", "b1", "b1_det", "b3", "b4", "b5", "a1", "a2", "a3_k0", "a3_k1")
NULL = ("N0", "N1")


class Timer:
    def __enter__(self):
        self.w, self.c = time.time(), time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall, self.cpu = time.time() - self.w, time.process_time() - self.c


def _split(cfg: dict, final: bool):
    if cfg["run_kind"] == "final" and not final:
        raise SystemExit("a final config needs --final")
    if final and cfg["run_kind"] != "final":
        raise SystemExit("--final needs run_kind: final in the config")
    splits = clean_splits(seed=cfg["data_seed"])
    return splits["test" if final else "val"], splits["train"], "test" if final else "validation"


def run_pair(cfg: dict, pair_id: int, split, train, n: int) -> tuple[dict, dict]:
    """All methods on one pair. Returns (per-point arrays, json row with runtime and counts)."""
    r, kappa = float(cfg["r"]), cfg["kappa"]
    base = solver_config(cfg, r=r)
    pair, f_old, f_new, entries = load_pair(pair_id)
    cond = pair["condition"]
    mo_all, mn_all = Tier0Margin(f_old, split.y_clean), Tier0Margin(f_new, split.y_clean)
    keep = valid_points(split.phi, split.y_clean, mo_all, mn_all, n, cfg["seeds"]["points_base"] + pair_id, kappa)
    z0 = torch.as_tensor(split.phi[keep], dtype=torch.float32)
    m_old, m_new = Tier0Margin(f_old, split.y_clean[keep]), Tier0Margin(f_new, split.y_clean[keep])
    seed = cfg["seeds"]["restart_base"] + pair_id
    outs, rt = {}, {}

    def solver(name, fn):
        with Timer() as t:
            res = fn()
        outs[name] = from_points(res.found, res.z, z0)
        rt[name] = runtime_row(res, t.wall, n)
        outs[name + "_distance"] = res.distance.numpy()

    solver("dce_N", lambda: delta_ce(z0, m_old, m_new, base, seed, kind="N"))
    if cond in NULL:
        solver("dce_O", lambda: delta_ce(z0, m_old, m_new, base, seed, kind="O"))
    for name, det in (("b1", False), ("b1_det", True)):
        with Timer() as t:
            valid, _, ce_o, ce_n = b1_diff(z0, m_old, m_new, base, seed, deterministic=det)
        outs[name] = from_points(valid, ce_n.z, ce_o.z)
        rt[name] = {"wall_s_per_point": t.wall / n, "cpu_s_per_point": (ce_o.cpu_seconds + ce_n.cpu_seconds) / n}
    with Timer() as t:
        valid, d = b3_grad(z0, m_old, m_new)
    outs["b3"], rt["b3"] = from_direction(valid, d), {"wall_s_per_point": t.wall / n}
    with Timer() as t:
        pred = lambda f: (lambda phi: predict_logits(f, phi, lambda p: render(torch.as_tensor(p, dtype=torch.float32))).argmax(1).numpy())
        b4_phi, b4_y = split.phi[: cfg["b4_points"]], split.y_clean[: cfg["b4_points"]]
        delta = b4_perm(pred(f_old), pred(f_new), b4_phi, b4_y)
        b0 = b0_disagreement(pred(f_old)(split.phi), pred(f_new)(split.phi))   # all samples of the split, unfiltered
    outs["b4"] = MethodOutput(np.ones(n, bool), np.tile(delta, (n, 1)))        # one global vector for every point
    rt["b4"] = {"wall_s_total": t.wall}
    with Timer() as t:
        found, d, _ = b5_line(z0, torch.as_tensor(train.phi, dtype=torch.float32), m_old, m_new, kappa, r)
    outs["b5"], rt["b5"] = from_direction(found, d), {"wall_s_per_point": t.wall / n}
    solver("a1", lambda: delta_ce(z0, m_old, m_new, replace(base, lam0=0.0, polish=False), seed, kind="N"))
    solver("a2", lambda: ordinary_ce(z0, m_new, base, seed))
    solver("a3_k0", lambda: delta_ce(z0, m_old, m_new, replace(base, kappa=0.0), seed, kind="N"))
    solver("a3_k1", lambda: delta_ce(z0, m_old, m_new, replace(base, kappa=1.0), seed, kind="N"))

    check_outputs({k: v for k, v in outs.items() if isinstance(v, MethodOutput)}, n, z0.shape[1])
    arrays = {"point_ids": keep, "z0": z0.numpy()}
    for k, v in outs.items():
        if isinstance(v, MethodOutput):
            arrays[f"{k}__found"], arrays[f"{k}__dphi"] = v.found, v.dphi
        else:
            arrays[k] = v
    row = {**pair, "n_points": n, "b0": b0, "runtime": rt, "checkpoints": entries}
    return arrays, row


def cmd_run(cfg: dict, out: Path, final: bool) -> int:
    split, train, split_name = _split(cfg, final)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.set_num_threads(cfg.get("threads", 8))
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "experiment": cfg["experiment_id"], "run_kind": cfg["run_kind"], "split": split_name, "git_runs": [],
        "python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(), "device": "cpu (8 threads)"}
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
            arrays, row = run_pair(cfg, pid, split, train, cfg["n_points"])
        np.savez_compressed(npz, **arrays)
        row["wall_s_pair"] = round(t.wall, 1)
        js.write_text(json.dumps(row, indent=2))
        print(f"pair {pid} ({row['condition']} s{row['seed']}) done in {t.wall / 60:.1f} min "
              f"(total {(time.time() - start) / 3600:.2f} h)", flush=True)
    meta.update(status="RUN_COMPLETE", run_seconds_last=round(time.time() - start, 1))
    meta_path.write_text(json.dumps(meta, indent=2))
    return 0


def load_pairs(out: Path, pair_ids) -> dict[int, tuple[dict, dict[str, MethodOutput]]]:
    res = {}
    for pid in pair_ids:
        row = json.loads((out / "pairs" / f"pair_{pid:02d}.json").read_text())
        z = np.load(out / "pairs" / f"pair_{pid:02d}.npz")
        outs = {m: MethodOutput(z[f"{m}__found"], z[f"{m}__dphi"]) for m in PER_POINT if f"{m}__found" in z.files}
        res[pid] = (row, outs)
    return res


def analyze(pairs: dict[int, tuple[dict, dict[str, MethodOutput]]], cfg: dict, expect_full: bool = True) -> dict:
    """Metrics, statistics and decisions (EVALUATION §2, §3, §10, §11, §15). Pure: reads only `pairs`."""
    if expect_full:
        check_run_counts(0, {pid: row["n_points"] for pid, (row, _) in pairs.items()})
    B = cfg["bootstrap_resamples"]
    for name, seed in cfg.get("bootstrap_seeds", {}).items():
        if BOOTSTRAP_SEEDS[name] != seed:
            raise ValueError(f"bootstrap seed of {name} in the config ({seed}) differs from the code ({BOOTSTRAP_SEEDS[name]})")
    per_pair, recs = {}, {}
    for pid, (row, outs) in pairs.items():
        k = row["k_star"]
        m = {}
        for name, o in outs.items():
            m[name] = {"found_rate": found_rate(o)}
            if k is not None:
                m[name].update(recovery=recovery_at_r(o, k), share_k_star=share_of_kstar(o, k))
            if row["condition"] in NULL:
                m[name]["sar"] = sar_pair(o)
        s_dce = pooled_found_rate([outs["dce_N"], outs["dce_O"]]) if "dce_O" in outs else found_rate(outs["dce_N"])
        scores = {"dce": s_dce, "b0": row["b0"], "b1": median_norm(outs["b1"]), "b1_det": median_norm(outs["b1_det"]),
                  "b3": median_norm(outs["b3"])}
        reach = recovery_given_reach(outs["dce_N"].found, outs, k) if k is not None else None
        per_pair[pid] = {"condition": row["condition"], "seed": row["seed"], "k_star": k, "methods": m, "S": scores,
                         "recovery_given_reach": reach, "dce_reach_count": int(outs["dce_N"].found.sum()),
                         "runtime": row["runtime"], "wall_s_pair": row.get("wall_s_pair")}
        if k is not None:
            recs[pid] = pair_record(pid, row["seed"], outs, k)

    def cond_ids(c):
        return [pid for pid, v in per_pair.items() if v["condition"] == c]

    by_cond = {}
    for c in ("T1", "T3", "T4", "N0", "N1"):
        ids = cond_ids(c)
        if not ids:
            continue
        methods = per_pair[ids[0]]["methods"].keys()
        by_cond[c] = {mth: {key: float(np.mean([per_pair[i]["methods"][mth][key] for i in ids if mth in per_pair[i]["methods"]]))
                            for key in ("found_rate", "recovery") if key in per_pair[ids[0]]["methods"][mth]}
                      for mth in methods}
        if c in NULL:
            for mth in methods:
                by_cond[c][mth]["sar_share_of_pairs"] = float(np.mean([per_pair[i]["methods"][mth]["sar"] for i in ids]))

    t3 = [recs[i] for i in cond_ids("T3") if i in recs]
    res = {"per_pair": per_pair, "by_condition": by_cond}
    if t3:
        h2a = compare("H2a", t3, "dce_N", "b3", bootstrap_rng("H2a"), B=B)
        supported = holm_family({"H2a": h2a})
        b5 = compare("B5-T0", t3, "dce_N", "b5", bootstrap_rng("B5-T0"), B=B)
        per_tier, label = b5_outcome({0: b5})
        res["H2a"] = {"diff": h2a.boot.diff, "ci95": [h2a.boot.ci_lo, h2a.boot.ci_hi], "p_two_sided": h2a.boot.p,
                      "holm_m": 1, "seed_diffs": h2a.seed_diffs, "supported": supported["H2a"]}
        res["route_A"] = route_a(cfg["h4_pass"], supported)
        res["B5_rule"] = {"diff": b5.boot.diff, "ci95": [b5.boot.ci_lo, b5.boot.ci_hi], "per_tier": per_tier,
                          "outcome": label, "note": "Tier 1 dropped: (a) cannot be met; (c) applies to Tier 0"}
    ptie = {}
    for c in ("T1", "T4"):
        ids = [i for i in cond_ids(c) if i in recs]
        for base in ("b3", "b1_det"):
            if ids:
                d = float(np.mean([recs[i].indicators["dce_N"].mean() - recs[i].indicators[base].mean() for i in ids]))
                ptie[f"{c}_vs_{base}"] = {"diff": d, "contradicts_prop4": p_tie_contradicted(d)}
    res["P_tie"] = ptie
    drift = [pid for pid, v in per_pair.items() if v["condition"] not in NULL]
    null = [pid for pid, v in per_pair.items() if v["condition"] in NULL]
    if drift and null:
        res["H3_auroc_descriptive"] = {s: auroc([per_pair[i]["S"][s] for i in drift], [per_pair[i]["S"][s] for i in null])
                                       for s in ("dce", "b0", "b1", "b3")}
        res["H3_note"] = "conditional on G0-qualified pairs; B0 is global, Δ-CE found-rate is local (A7)"
    return res


def cmd_analyze(cfg: dict, out: Path) -> int:
    meta = json.loads((out / "metadata.json").read_text())
    if meta.get("status") != "RUN_COMPLETE":
        raise SystemExit("run stage not complete")
    if (out / "results.json").exists():
        print("refusing to overwrite results.json", file=sys.stderr)
        return 2
    res = analyze(load_pairs(out, cfg["pair_ids"]), cfg, expect_full=cfg["run_kind"] == "final")
    res["split"] = meta["split"]
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    meta.update(status="ANALYZED", analysis_git=git_state())
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"results written to {out / 'results.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("run", "analyze"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if cfg.get("config_sha256_check", True) and cfg["run_kind"] == "final":
        print(f"config sha256 {sha256(args.config)}", flush=True)
    if args.stage == "run":
        return cmd_run(cfg, args.out, args.final)
    return cmd_analyze(cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
