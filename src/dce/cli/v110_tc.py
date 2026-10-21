"""v1.10 (D-043) descriptive runs: T-C abstention recall (exp_0034) and T-D found-rate against κ (exp_0035).

Pair kinds: Tier 0 pairs 6–16 (exp_0002/exp_0003 checkpoints), grid pairs 20–27 (exp_0025), H7 pairs 60–89
(LogitShift of exp_0002 f_old). Data: new_test_set(9000 + pair_id, 5 000) for Tier 0 and grid pairs; the H7 pairs reuse
the exp_0032 points (allowed after H7c is computed, v1.10 §6.1).

tc-run      — T-C (§4): queries = NOT_FOUND points of Δ-CE among the first 100 valid points (G1 null 9–16, G3 offsets
              20–27), restricted to class 0 for G2 (T4 pairs 6–8), and to the lowered class for G4 (H7 pairs with
              |b| ∈ {1.1, 1.5}; the exp_0032 Δ-CE run, all 300 points); up to 50 per pair; max slack per query.
tc-analyze  — false-negative rate per group with Clopper–Pearson 95% intervals; distribution of max s.
td-run      — T-D (§5): Δ-CE type N at each κ on the first 100 κ = 0.5-valid points; found-rate per class.
td-figure   — found-rate against b per κ with the Lemma 1 lines |b| = 2κ.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from dce.cli.calibrate_r import solver_config
from dce.cli.final_tier0 import Timer
from dce.cli.g0_reeval import load_verified
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.cli.v19_grid import grid_pairs
from dce.cli.v19_test import load_models
from dce.cli.v110_dev import max_slack
from dce.cli.v110_test import data_for, h7_pairs, run_pairs
from dce.eval.pairs import load_pair
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.models.label_shift import LogitShift
from dce.solvers.dce import delta_ce

GROUPS = {"G1_null": range(9, 17), "G2_t4_class0": range(6, 9), "G3_offsets": range(20, 28),
          "G4_h7": (61, 62, 67, 68, 71, 72, 77, 78, 81, 82, 87, 88)}
N_FIRST, Q_MAX = 100, 50   # v1.10 §4–§5; a config may lower them for a smoke run only (tc_points, tc_queries)


def group_of(pid: int) -> str:
    return next(g for g, ids in GROUPS.items() if pid in ids)


def models_for(cfg, pid, cache):
    """(f_old, f_new, info) for any v1.10 pair id."""
    if 6 <= pid <= 16:
        p, f_old, f_new, entries = load_pair(pid, cache=cache)
        return f_old, f_new, {"kind": "tier0", "condition": p["condition"], "seed": p["seed"], "checkpoints": entries}
    if 20 <= pid <= 51:
        p = next(q for q in grid_pairs() if q["pair_id"] == pid)
        f_old, f_new, entries = load_models(cfg, p, cache)
        return f_old, f_new, {"kind": "grid", "alpha": p["alpha"], "t": p["t"], "seed": p["seed"], "checkpoints": entries}
    p = next(q for q in h7_pairs() if q["pair_id"] == pid)
    f_old, entry = load_verified(Path(cfg["f_old_source"]), f"f_old_s{p['seed']}", cache)
    return f_old, LogitShift(f_old, p["b"]).eval(), {"kind": "h7", **p, "checkpoints": [entry]}


def points_for(cfg, pid, f_old, f_new, n):
    """z0 (n, 5), y (n,), digest: the first n valid points. H7 pairs: the exp_0032 points (same order)."""
    if pid >= 60:
        a = np.load(Path(cfg["h7_run"]) / "pairs" / f"pair_{pid}.npz")
        return torch.as_tensor(a["z0"][:n]), a["y"][:n], f"exp_0032 pair {pid}"
    phi, y, digest = data_for(cfg, pid)
    keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), n, cfg["seeds"]["points_base"] + pid, cfg["kappa"])
    return torch.as_tensor(phi[keep], dtype=torch.float32), y[keep], digest


class Deadline(Exception):
    """Raised between T-C queries when --deadline-h is reached (a Kaggle session limit); the pair stays partial."""


class Partial:
    """Per-pair checkpoint inside a pair (the local session can end and stop a run): arrays in
    pairs/pair_<id>.partial.npz, rows in pairs/pair_<id>.partial.json. Removed when the pair is complete."""

    def __init__(self, cfg, pid):
        d = Path(cfg["_out"]) / "pairs"
        self.npz, self.js = d / f"pair_{pid}.partial.npz", d / f"pair_{pid}.partial.json"

    def arrays(self):
        return dict(np.load(self.npz)) if self.npz.exists() else None

    def save_arrays(self, **a):
        tmp = self.npz.with_name(self.npz.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **a)
        os.replace(tmp, self.npz)

    def rows(self):
        return json.loads(self.js.read_text()) if self.js.exists() else {}

    def save_rows(self, rows):
        tmp = self.js.with_name(self.js.name + ".tmp")
        tmp.write_text(json.dumps(rows))
        os.replace(tmp, self.js)

    def clear(self):
        for f in (self.npz, self.js):
            f.unlink(missing_ok=True)


def tc_pair(cfg, p, cache):
    pid = p["pair_id"]
    f_old, f_new, info = models_for(cfg, pid, cache)
    base = solver_config(cfg, r=float(cfg["r"]))
    part = Partial(cfg, pid)
    if part.arrays() is not None:                              # resume: Δ-CE already solved for this pair
        a = part.arrays()
        z0, y, found, eligible = torch.as_tensor(a["z0"]), a["y"], a["dce_found"], a["eligible"]
        digest = part.rows()["digest"]
    elif pid >= 60:                                              # G4: reuse the exp_0032 Δ-CE run on all 300 points
        a = np.load(Path(cfg["h7_run"]) / "pairs" / f"pair_{pid}.npz")
        z0, y, digest, found = torch.as_tensor(a["z0"]), a["y"], f"exp_0032 pair {pid}", a["dce_N__found"].astype(bool)
        eligible = ~found & (y == info["lowered_class"])
    else:
        z0, y, digest = points_for(cfg, pid, f_old, f_new, cfg.get("tc_points", N_FIRST))
        res = delta_ce(z0, Tier0Margin(f_old, y), Tier0Margin(f_new, y), base, cfg["seeds"]["restart_base"] + pid, kind="N")
        found = res.found.numpy()
        eligible = ~found & ((y == 0) if group_of(pid) == "G2_t4_class0" else True)
    if part.arrays() is None:
        part.save_arrays(z0=z0.numpy(), y=y, dce_found=found, eligible=eligible)
        part.save_rows({"digest": digest, "queries": []})
    m_old, m_new = Tier0Margin(f_old, y), Tier0Margin(f_new, y)
    state = part.rows()
    rows = state["queries"]
    done = {r["query"] for r in rows}
    for i in np.flatnonzero(eligible)[: cfg.get("tc_queries", Q_MAX)]:
        if int(i) in done:
            continue
        if time.time() > cfg.get("_deadline", float("inf")):   # every finished query is in the partial checkpoint
            raise Deadline(pid)
        # one generator per query, derived from the registered seed 9500 + pair_id, so a resumed run gives the same values
        gen = torch.Generator().manual_seed((cfg["tc_seed_base"] + pid) * 10_000 + int(i))
        with Timer() as t:
            q = max_slack(z0[i], int(i), m_old, m_new, cfg["kappa"], float(cfg["r"]), cfg["tc_M"], cfg["tc_starts"],
                          cfg["tc_steps"], gen)
        rows.append({"query": int(i), "y": int(y[i]), **q, "wall_s": round(t.wall, 2)})
        part.save_rows(state)
    rows.sort(key=lambda r: r["query"])
    s_max = np.array([r["s_max"] for r in rows]) if rows else np.zeros(0)
    arrays = {"z0": z0.numpy(), "y": y, "dce_found": found, "queries": np.array([r["query"] for r in rows], dtype=int),
              "s_max": s_max}
    row = {"pair_id": pid, "group": group_of(pid), **{k: v for k, v in info.items() if k != "checkpoints"},
           "data_sha256": digest, "n_points": len(y), "dce_found": int(found.sum()), "n_eligible": int(eligible.sum()),
           "queries": rows, "checkpoints": info["checkpoints"]}
    part.clear()
    return arrays, row


def tc_recheck(cfg, pid: int, queries: list[int], ref: dict) -> list[dict]:
    """Re-run T-C queries of a pair on this machine (D-044 guard): the points are regenerated and must match the
    reference run (digest and z0); each query uses the registered per-query generator."""
    f_old, f_new, _ = models_for(cfg, pid, {})
    torch.set_num_threads(cfg.get("threads", 8))
    z0, y, digest = points_for(cfg, pid, f_old, f_new, cfg.get("tc_points", N_FIRST))
    if digest != ref["data_sha256"] or not np.array_equal(z0.numpy(), ref["z0"]):
        raise SystemExit(f"pair {pid}: regenerated points differ from the reference run")
    m_old, m_new = Tier0Margin(f_old, y), Tier0Margin(f_new, y)
    out = []
    for i in queries:
        gen = torch.Generator().manual_seed((cfg["tc_seed_base"] + pid) * 10_000 + i)
        with Timer() as t:
            q = max_slack(z0[i], i, m_old, m_new, cfg["kappa"], float(cfg["r"]), cfg["tc_M"], cfg["tc_starts"],
                          cfg["tc_steps"], gen)
        out.append({"pair_id": pid, "query": i, "y": int(y[i]), **q, "wall_s": round(t.wall, 2)})
        print(json.dumps(out[-1]), flush=True)
    return out


def clopper_pearson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def tc_analyze(rows: list[dict]) -> dict:
    out = {}
    for g in GROUPS:
        s = np.array([q["s_max"] for r in rows if r["group"] == g for q in r["queries"]])
        k, n = int((s >= 0).sum()), int(s.size)
        out[g] = {"pairs": [r["pair_id"] for r in rows if r["group"] == g], "queries": n, "false_negatives": k,
                  "rate": k / n if n else None, "cp95": clopper_pearson(k, n),
                  "max_s_quantiles": np.quantile(s, [0, 0.1, 0.5, 0.9, 1]).tolist() if n else None,
                  "accepted_min": min((q["n_samples"] for r in rows if r["group"] == g for q in r["queries"]), default=None)}
    return {"label": "v1.10 T-C (D-043, descriptive)", "groups": out}


def td_pair(cfg, p, cache):
    pid = p["pair_id"]
    f_old, f_new, info = models_for(cfg, pid, cache)
    z0, y, digest = points_for(cfg, pid, f_old, f_new, cfg.get("tc_points", N_FIRST))
    m_old, m_new = Tier0Margin(f_old, y), Tier0Margin(f_new, y)
    base = solver_config(cfg, r=float(cfg["r"]))
    part = Partial(cfg, pid)
    arrays, per = (part.arrays() or {"z0": z0.numpy(), "y": y}), {}
    for kap in cfg["kappa_curve"]:
        if f"found_k{kap}" not in arrays:
            res = delta_ce(z0, m_old, m_new, replace(base, kappa=float(kap)), cfg["seeds"]["restart_base"] + pid, kind="N")
            arrays[f"found_k{kap}"] = res.found.numpy()
            part.save_arrays(**arrays)
        f = arrays[f"found_k{kap}"]
        per[str(kap)] = {"found": float(f.mean()), "found_y0": float(f[y == 0].mean()) if (y == 0).any() else None,
                         "found_y1": float(f[y == 1].mean()) if (y == 1).any() else None}
    row = {"pair_id": pid, **{k: v for k, v in info.items() if k != "checkpoints"}, "data_sha256": digest,
           "n_points": len(y), "found_by_kappa": per, "checkpoints": info["checkpoints"]}
    part.clear()
    return arrays, row


def td_figure(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h7 = sorted((r for r in rows if r["kind"] == "h7"), key=lambda r: r["b"])
    kappas = list(h7[0]["found_by_kappa"]) if h7 else []
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    for kap in kappas:
        line = ax.plot([r["b"] for r in h7], [r["found_by_kappa"][kap]["found"] for r in h7], marker="o", ms=3,
                       label=f"κ = {kap}")[0]
        if float(kap) > 0:
            for sgn in (-1, 1):
                ax.axvline(sgn * 2 * float(kap), color=line.get_color(), lw=0.6, ls=":")
    ax.set_xlabel("logit offset b"), ax.set_ylabel("Δ-CE found-rate")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("tc-run", "tc-analyze", "td-run", "td-figure", "tc-recheck"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1", help="i/k for a static split, or 'dyn' to claim pairs one at a time")
    ap.add_argument("--threads", type=int, help="override cfg threads (recorded in the metadata)")
    ap.add_argument("--pairs", help="comma-separated subset of cfg pair_ids to run (a run stage only)")
    ap.add_argument("--deadline-h", type=float, help="tc-run: stop between queries after this many hours")
    ap.add_argument("--queries", help="tc-recheck: pair:query,... to re-run against the pair files in --out/pairs")
    ap.add_argument("--tag", default="recheck", help="tc-recheck: output name tc_<tag>.json")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    shard = None if args.shard == "dyn" else tuple(map(int, args.shard.split("/")))
    if args.threads:
        cfg["threads"] = args.threads
    by_id = {pid: {"pair_id": pid} for pid in cfg["pair_ids"]}
    cfg["_out"] = str(args.out)
    if args.stage in ("tc-run", "td-run"):
        if args.pairs:
            sub = [int(x) for x in args.pairs.split(",")]
            if not set(sub) <= set(cfg["pair_ids"]):
                raise SystemExit(f"--pairs not in the config: {sorted(set(sub) - set(cfg['pair_ids']))}")
            cfg["pair_ids"] = sub
        if args.deadline_h:
            cfg["_deadline"] = time.time() + 3600 * args.deadline_h
        fn = tc_pair if args.stage == "tc-run" else td_pair
        try:
            return run_pairs(cfg, args.out, shard, by_id, fn, f"D-043 (v1.10 {'§4' if fn is tc_pair else '§5'})")
        except Deadline as e:
            (args.out / "pairs" / f"pair_{e.args[0]}.claim").unlink(missing_ok=True)
            print(f"deadline reached during pair {e.args[0]}; its finished queries are in the partial checkpoint", flush=True)
            return 3
    if args.stage == "tc-recheck":
        todo: dict[int, list[int]] = {}
        for item in args.queries.split(","):
            pid, q = map(int, item.split(":"))
            todo.setdefault(pid, []).append(q)
        res = []
        for pid, qs in todo.items():
            ref = {**json.loads((args.out / "pairs" / f"pair_{pid}.json").read_text()),
                   "z0": np.load(args.out / "pairs" / f"pair_{pid}.npz")["z0"]}
            res += tc_recheck(cfg, pid, qs, ref)
        (args.out / f"tc_{args.tag}.json").write_text(json.dumps({"git": git_state(), "rows": res}, indent=2))
        return 0
    rows = [json.loads((args.out / "pairs" / f"pair_{p}.json").read_text()) for p in cfg["pair_ids"]]
    if args.stage == "tc-analyze":
        res = tc_analyze(rows)
        res["git"] = git_state()
        (args.out / "results_tc.json").write_text(json.dumps(res, indent=2, default=float))
        print(json.dumps(res["groups"], indent=1, default=float))
    else:
        td_figure(rows, args.out / "found_vs_b.pdf")
        (args.out / "results_td.json").write_text(json.dumps({"label": "v1.10 T-D (descriptive)", "pairs": rows,
                                                              "git": git_state()}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
