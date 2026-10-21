"""v1.10 (D-043) test runs on fresh renderer data, local CPU, locked solver.

ta-run      — T-A (exp_0032): H7 pairs 60–89 = f_old seed s ∈ {0, 1, 2} × b (v1.10 §2.3), f_new = LogitShift(f_old, b).
              Per pair (resumable; --shard i/k splits the pairs over processes): data new_test_set(9000 + pair_id, 5 000)
              (or the validation split when `data: val`, for the smoke run), 300 valid points (κ = 0.5, order seed
              5000 + pair_id), Δ-CE N, B1-det, B3, restarts 10000 + pair_id. No metric is printed.
ta-analyze  — U1–U3 checks (a failure is a code error and stops the analysis), H7c (unadjusted p; Holm with H8 later)
              and the descriptive table of v1.10 §2.6.
tb-run      — T-B (exp_0033): grid pairs (exp_0025 f_new, exp_0002 f_old), data new_test_set(9000 + pair_id, 5 000),
              300 valid points, Δ-CE N; B3 at z0 and at z*; ∇m_o at z0 and z*. Resumable, shardable.
tb-analyze  — H8 on the H5d pairs 36, 37, 44, 45 (registered cell (z*, z0), v1.10 §3.3); descriptive cells, locality
              and the other α > 0 pairs.
holm        — Holm (m = 2) over {H7c, H8} once both results exist.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from dce.analysis.projection import old_margin_gradients
from dce.analysis.v19 import boot_mean
from dce.analysis.v110 import CELLS, REFERENCE, boot_h8, check_fresh_seed, h8_indicators, locality
from dce.baselines.b1_diff import b1_diff
from dce.baselines.b3_grad import b3_grad
from dce.cli.calibrate_r import solver_config
from dce.cli.confirm_tier0 import new_test_set
from dce.cli.final_tier0 import Timer
from dce.cli.g0_reeval import load_verified
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.cli.v19_grid import grid_pairs
from dce.cli.v19_test import load_models
from dce.data.drift import clean_splits, tier0_pairs
from dce.eval.metrics import MethodOutput, recovery_indicator
from dce.eval.pairs import load_pair
from dce.eval.records import check_outputs, from_direction, from_points, runtime_row
from dce.eval.stats import holm
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.models.label_shift import LogitShift, lowered_class
from dce.solvers.dce import delta_ce

H7_SEEDS = (0, 1, 2)
H7_B = (-2.0, -1.5, -1.1, -0.9, -0.5, 0.5, 0.9, 1.1, 1.5, 2.0)
H7C_MIN = 0.90
H7C_SEED = 6600
H8_SEED = 6601
H8_PAIRS = (36, 37, 44, 45)
H8_GUARD = 0.50
ASPECT, COLOUR = 0, 2


def h7_pairs() -> list[dict]:
    """Pair id = 60 + 10·s + j (v1.10 §2.3)."""
    return [{"pair_id": 60 + 10 * s + j, "seed": s, "b": b, "lowered_class": lowered_class(b)}
            for s in H7_SEEDS for j, b in enumerate(H7_B)]


def data_for(cfg, pid):
    if cfg.get("data") == "val":           # smoke only: same path on the validation split
        val = clean_splits(seed=cfg["data_seed"])["val"]
        return val.phi, val.y_clean.astype(np.int64), "validation"
    phi, y, digest = new_test_set(check_fresh_seed(cfg["data_seed_base"] + pid), cfg["n_samples"])
    return phi, y, digest


def ta_pair(cfg, p, cache):
    pid, n, kappa = p["pair_id"], cfg["n_points"], cfg["kappa"]
    base = solver_config(cfg, r=float(cfg["r"]))
    f_old, entry = load_verified(Path(cfg["f_old_source"]), f"f_old_s{p['seed']}", cache)
    f_new = LogitShift(f_old, p["b"]).eval()
    phi, y, digest = data_for(cfg, pid)
    keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), n, cfg["seeds"]["points_base"] + pid, kappa)
    z0 = torch.as_tensor(phi[keep], dtype=torch.float32)
    yk = y[keep]
    m_old, m_new = Tier0Margin(f_old, yk), Tier0Margin(f_new, yk)
    seed = cfg["seeds"]["restart_base"] + pid
    outs, rt = {}, {}
    with Timer() as t:
        res = delta_ce(z0, m_old, m_new, base, seed, kind="N")
    outs["dce_N"], rt["dce_N"] = from_points(res.found, res.z, z0), runtime_row(res, t.wall, n)
    with Timer() as t:
        valid, _, ce_o, ce_n = b1_diff(z0, m_old, m_new, base, seed, deterministic=True)
    outs["b1_det"], rt["b1_det"] = from_points(valid, ce_n.z, ce_o.z), {"wall_s_per_point": t.wall / n}
    v, d = b3_grad(z0, m_old, m_new)
    outs["b3"] = from_direction(v, d)
    check_outputs(outs, n, z0.shape[1])
    idx = torch.arange(n)
    with torch.no_grad():
        mo_star, mn_star = m_old(res.z, idx).numpy(), m_new(res.z, idx).numpy()
    arrays = {"point_ids": keep, "y": yk, "z0": z0.numpy(), "dce_N_distance": res.distance.numpy(),
              "dce_N_m_old": mo_star, "dce_N_m_new": mn_star}
    for k, o in outs.items():
        arrays[f"{k}__found"], arrays[f"{k}__dphi"] = o.found, o.dphi
    row = {**p, "n_points": n, "data_seed": cfg.get("data_seed_base", 0) + pid if cfg.get("data") != "val" else None,
           "data_sha256": digest, "runtime": rt, "checkpoints": [entry]}
    return arrays, row


def cmd_ta_run(cfg, out: Path, shard: tuple[int, int]) -> int:
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", 8))
    meta_path = out / f"metadata_shard{shard[0]}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "experiment": cfg["experiment_id"], "decision": "D-043 (v1.10 §2)", "shard": list(shard), "git_runs": [],
        "python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform()}
    meta["git_runs"].append({"git": git_state(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    meta_path.write_text(json.dumps(meta, indent=2))
    by_id, cache, start = {p["pair_id"]: p for p in h7_pairs()}, {}, time.time()
    ids = [pid for k, pid in enumerate(cfg["pair_ids"]) if k % shard[1] == shard[0]]
    for pid in ids:
        npz, js = out / "pairs" / f"pair_{pid}.npz", out / "pairs" / f"pair_{pid}.json"
        if npz.exists() and js.exists():
            print(f"pair {pid}: already done, skipped", flush=True)
            continue
        with Timer() as t:
            arrays, row = ta_pair(cfg, by_id[pid], cache)
        np.savez_compressed(npz, **arrays)
        row["wall_s_pair"] = round(t.wall, 1)
        js.write_text(json.dumps(row, indent=2))
        print(f"pair {pid} done in {t.wall / 60:.1f} min (total {(time.time() - start) / 3600:.2f} h)", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta_path.write_text(json.dumps(meta, indent=2))
    return 0


def outputs(a, m):
    return MethodOutput(a[f"{m}__found"], a[f"{m}__dphi"])


def ta_analyze(pairs: dict, kappa: float, B: int = 10_000) -> dict:
    """pairs: pair_id → (row, arrays). U1–U3 (v1.10 §2.2) must hold, else SystemExit; H7c (§2.5); descriptive (§2.6)."""
    rows, errors = [], []
    for pid in sorted(pairs):
        r, a = pairs[pid]
        b, f = r["b"], a["dce_N__found"].astype(bool)
        if a["b3__found"].any():
            errors.append(f"U1: B3 valid on pair {pid}")
        if abs(b) < 2 * kappa and f.any():
            errors.append(f"U2: {int(f.sum())} witnesses on pair {pid} (|b| < 2κ)")
        if f.any():
            if (a["y"][f] != r["lowered_class"]).any():
                errors.append(f"U3: witness of the raised class on pair {pid}")
            mo = a["dce_N_m_old"][f]
            if (mo < kappa - 1e-5).any() or (mo > abs(b) - kappa + 1e-5).any():
                errors.append(f"U3: m_old(z*) outside [κ, |b| − κ] on pair {pid}")
        low = a["y"] == r["lowered_class"]
        rows.append({"pair_id": pid, "seed": r["seed"], "b": b, "found_dce": float(f.mean()),
                     "found_dce_lowered": float(f[low].mean()) if low.any() else None, "n_lowered": int(low.sum()),
                     "found_b1_det": float(a["b1_det__found"].mean()),
                     "rec_dce_aspect": float(recovery_indicator(outputs(a, "dce_N"), ASPECT).mean()),
                     "rec_b1_det_aspect": float(recovery_indicator(outputs(a, "b1_det"), ASPECT).mean())})
    if errors:
        raise SystemExit("v1.10 U1–U3 failed (code error, analysis stops):\n" + "\n".join(errors))
    sub = [pairs[p][1]["b1_det__found"].astype(float) for p in sorted(pairs) if abs(pairs[p][0]["b"]) < 2 * kappa]
    point, stat = boot_mean(sub, np.random.default_rng(H7C_SEED), B)
    lo = float(np.quantile(stat, 0.025))
    per_pair = [float(g.mean()) for g in sub]
    h7c = {"n_pairs": len(sub), "estimate": point, "ci95": [lo, float(np.quantile(stat, 0.975))],
           "p_one_sided": float(max((stat <= H7C_MIN).mean(), 1.0 / B)), "per_pair_found_b1_det": per_pair,
           "min_per_pair": min(per_pair),
           "criterion": "ci95 lower > 0.90 and p_holm < 0.05 (Holm with H8) and every pair >= 0.90",
           "pass_before_holm": bool(lo > H7C_MIN and min(per_pair) >= H7C_MIN)}
    return {"label": "v1.10 T-A (D-043)", "checks_U1_U3": "passed", "H7c": h7c, "descriptive": rows}


def cmd_ta_analyze(cfg, out: Path) -> int:
    path = out / "results_h7.json"
    if path.exists():
        raise SystemExit("refusing to overwrite results_h7.json")
    missing = [p for p in cfg["pair_ids"] if not (out / "pairs" / f"pair_{p}.json").exists()]
    if missing:
        raise SystemExit(f"run stage not complete: {missing}")
    pairs = {p: (json.loads((out / "pairs" / f"pair_{p}.json").read_text()), dict(np.load(out / "pairs" / f"pair_{p}.npz")))
             for p in cfg["pair_ids"]}
    res = ta_analyze(pairs, cfg["kappa"], cfg["bootstrap_resamples"])
    res["git"] = git_state()
    path.write_text(json.dumps(res, indent=2, default=float))
    print(f"results written to {path}")
    return 0


def tb_pair(cfg, p, cache):
    pid, n = p["pair_id"], cfg["n_points"]
    base = solver_config(cfg, r=float(cfg["r"]))
    if "condition" in p:                                       # Tier 0 pair (T1 2×2, D-045)
        _, f_old, f_new, entries = load_pair(pid, cache=cache)
    else:
        f_old, f_new, entries = load_models(cfg, p, cache)
    phi, y, digest = data_for(cfg, pid)
    keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), n, cfg["seeds"]["points_base"] + pid, cfg["kappa"])
    z0 = torch.as_tensor(phi[keep], dtype=torch.float32)
    yk = y[keep]
    m_old, m_new = Tier0Margin(f_old, yk), Tier0Margin(f_new, yk)
    with Timer() as t:
        res = delta_ce(z0, m_old, m_new, base, cfg["seeds"]["restart_base"] + pid, kind="N")
    dce = from_points(res.found, res.z, z0)
    zs = torch.where(res.found[:, None], res.z, z0)           # z* on the support, z0 elsewhere (not scored)
    arrays = {"point_ids": keep, "y": yk, "z0": z0.numpy(), "zs": zs.numpy(), "dce__found": dce.found,
              "dce__dphi": dce.dphi, "dce_distance": res.distance.numpy(),
              "grad_z0": old_margin_gradients(f_old, z0.numpy(), yk), "grad_zs": old_margin_gradients(f_old, zs.numpy(), yk)}
    for key, z in (("z0", z0), ("zs", zs)):
        v, d = b3_grad(z, m_old, m_new)
        o = from_direction(v, d)
        arrays[f"b3_{key}__found"], arrays[f"b3_{key}__dphi"] = o.found, o.dphi
    row = {**p, "n_points": n, "data_seed": cfg.get("data_seed_base", 0) + pid if cfg.get("data") != "val" else None,
           "data_sha256": digest, "support": int(dce.found.sum()), "runtime": {"dce_N": runtime_row(res, t.wall, n)},
           "checkpoints": entries}
    return arrays, row


def claim(path: Path) -> bool:
    """Create the claim file atomically; take over a claim whose owner process is gone."""
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                os.kill(int(path.read_text() or 0), 0)
                return False
            except (ProcessLookupError, ValueError, FileNotFoundError):
                path.unlink(missing_ok=True)
                continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    return False


def run_pairs(cfg, out: Path, shard, by_id, fn, label) -> int:
    """shard = (i, k) runs pairs k-th apart; shard = None claims pairs one at a time (any number of workers)."""
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", 8))
    tag = f"shard{shard[0]}" if shard else f"worker{os.getpid()}"
    meta_path = out / f"metadata_{tag}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "experiment": cfg["experiment_id"], "decision": label, "shard": list(shard) if shard else "dyn", "git_runs": [],
        "python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform()}
    meta["git_runs"].append({"git": git_state(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "threads": torch.get_num_threads()})
    meta_path.write_text(json.dumps(meta, indent=2))
    cache, start = {}, time.time()
    ids = cfg["pair_ids"] if shard is None else [p for k, p in enumerate(cfg["pair_ids"]) if k % shard[1] == shard[0]]
    for pid in ids:
        npz, js = out / "pairs" / f"pair_{pid}.npz", out / "pairs" / f"pair_{pid}.json"
        if npz.exists() and js.exists():
            print(f"pair {pid}: already done, skipped", flush=True)
            continue
        lock = out / "pairs" / f"pair_{pid}.claim"
        if shard is None and not claim(lock):
            continue
        with Timer() as t:
            arrays, row = fn(cfg, by_id[pid], cache)
        np.savez_compressed(npz, **arrays)
        row["wall_s_pair"] = round(t.wall, 1)
        js.write_text(json.dumps(row, indent=2))
        lock.unlink(missing_ok=True)
        meta.setdefault("pairs", []).append(pid)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"pair {pid} done in {t.wall / 60:.1f} min (total {(time.time() - start) / 3600:.2f} h)", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta_path.write_text(json.dumps(meta, indent=2))
    return 0


def h8_inputs(a: dict, k_star: int):
    dce = MethodOutput(a["dce__found"], a["dce__dphi"])
    b3 = {k: MethodOutput(a[f"b3_{k}__found"], a[f"b3_{k}__dphi"]) for k in ("z0", "zs")}
    return dce, h8_indicators(dce, b3, {"z0": a["grad_z0"], "zs": a["grad_zs"]}, k_star)


def tb_analyze(pairs: dict, B: int = 10_000) -> dict:
    """pairs: pair_id → (row, arrays) of the grid α > 0 pairs. H8 on H8_PAIRS (v1.10 §3.3); descriptive (§3.4)."""
    key = lambda c: f"{c[0]},{c[1]}"
    conf = [h8_inputs(pairs[p][1], COLOUR)[1] for p in H8_PAIRS]
    res = boot_h8(conf, np.random.default_rng(H8_SEED), B)
    ref_ci90 = res["gap_ci90"][REFERENCE]
    testable = not (ref_ci90[0] <= 0 <= ref_ci90[1])
    h8 = {"pairs": list(H8_PAIRS), "cell": res["cell"], "support_sizes": res["support_sizes"],
          "gap": {key(c): v for c, v in res["gap"].items()}, "gap_ci90": {key(c): v for c, v in res["gap_ci90"].items()},
          "T": res["T"], "T_ci_lower95": res["T_ci_lower95"], "p_one_sided": res["p_one_sided"],
          "guard_rec_b3": res["guard_rec_b3"], "testable": testable,
          "criterion": "T ci95 lower > 0 and p_holm < 0.05 and guard_rec_b3 >= 0.50; not testable if gap(z0,z0) ci90 contains 0",
          "pass_before_holm": bool(testable and res["T_ci_lower95"] > 0 and res["guard_rec_b3"] >= H8_GUARD)}
    rows, loc_all = [], {"dist": [], "one_minus_cos": [], "proj_dev": []}
    for pid in sorted(pairs):
        r, a = pairs[pid]
        dce, ind = h8_inputs(a, COLOUR)
        s = dce.found
        loc = locality(a["z0"][s], a["zs"][s], a["grad_z0"][s], a["grad_zs"][s], a["dce__dphi"][s])
        if pid in H8_PAIRS:
            for k, acc in loc_all.items():
                acc.append(loc[k])
        rows.append({"pair_id": pid, "alpha": r["alpha"], "t": r["t"], "seed": r["seed"], "support": int(s.sum()),
                     "found_rate": float(s.mean()), "confirmatory": pid in H8_PAIRS,
                     "rec": {key(c): [float(ind[c][0].mean()), float(ind[c][1].mean())] if s.any() else None for c in CELLS},
                     "rec_unprojected_all_points": [float(recovery_indicator(dce, COLOUR).mean()),
                                                    float(recovery_indicator(MethodOutput(a["b3_z0__found"], a["b3_z0__dphi"]), COLOUR).mean())],
                     "median_dist": float(np.median(loc["dist"])) if s.any() else None,
                     "median_one_minus_cos": float(np.median(loc["one_minus_cos"])) if s.any() else None})
    cat = {k: np.concatenate(v) for k, v in loc_all.items()}
    spear = {f"proj_dev~{k}": float(spearmanr(cat["proj_dev"], cat[k]).statistic) for k in ("dist", "one_minus_cos")}
    return {"label": "v1.10 T-B (D-043)", "H8": h8, "descriptive": {"pairs": rows, "spearman_h8_pairs": spear}}


def tb_describe(pairs: dict, seed: int, B: int = 10_000) -> dict:
    """Descriptive 2×2 (D-045): pooled gaps of every cell with CI90 (paired bootstrap stratified by pair, as H8),
    T for the cell (z*, z0), per-pair recoveries, and locality. k* from the pair (Tier 0) or colour (grid)."""
    key = lambda c: f"{c[0]},{c[1]}"
    ks = {pid: COLOUR if r.get("k_star") is None else r["k_star"] for pid, (r, _) in pairs.items()}
    ind = {pid: h8_inputs(a, ks[pid]) for pid, (_, a) in pairs.items()}
    res = boot_h8([ind[p][1] for p in sorted(pairs)], np.random.default_rng(seed), B)
    rows, loc_all = [], {"dist": [], "one_minus_cos": [], "proj_dev": []}
    for pid in sorted(pairs):
        a = pairs[pid][1]
        dce, i = ind[pid]
        s = dce.found
        loc = locality(a["z0"][s], a["zs"][s], a["grad_z0"][s], a["grad_zs"][s], a["dce__dphi"][s])
        for k, acc in loc_all.items():
            acc.append(loc[k])
        rows.append({"pair_id": pid, "k_star": ks[pid], "support": int(s.sum()), "found_rate": float(s.mean()),
                     "rec": {key(c): [float(i[c][0].mean()), float(i[c][1].mean())] if s.any() else None for c in CELLS},
                     "median_dist": float(np.median(loc["dist"])) if s.any() else None,
                     "median_one_minus_cos": float(np.median(loc["one_minus_cos"])) if s.any() else None})
    cat = {k: np.concatenate(v) for k, v in loc_all.items()}
    return {"label": "v1.10 T1 2×2 (D-045, descriptive)", "bootstrap_seed": seed, "support_sizes": res["support_sizes"],
            "gap": {key(c): v for c, v in res["gap"].items()}, "gap_ci90": {key(c): v for c, v in res["gap_ci90"].items()},
            "T_zs_z0": res["T"], "T_ci_lower95": res["T_ci_lower95"], "rec_b3_zs_z0": res["guard_rec_b3"],
            "pairs": rows, "spearman": {"one_minus_cos~dist": float(spearmanr(cat["one_minus_cos"], cat["dist"]).statistic),
                                        "proj_dev~dist": float(spearmanr(cat["proj_dev"], cat["dist"]).statistic)}}


def cmd_tb_describe(cfg, out: Path) -> int:
    path = out / "results_2x2.json"
    if path.exists():
        raise SystemExit("refusing to overwrite results_2x2.json")
    pairs = {p: (json.loads((out / "pairs" / f"pair_{p}.json").read_text()), dict(np.load(out / "pairs" / f"pair_{p}.npz")))
             for p in cfg["pair_ids"]}
    res = tb_describe(pairs, cfg["bootstrap_seed"], cfg["bootstrap_resamples"])
    res["git"] = git_state()
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("support_sizes", "gap", "gap_ci90", "T_zs_z0", "rec_b3_zs_z0", "spearman")}, indent=1))
    return 0


def cmd_tb_analyze(cfg, out: Path) -> int:
    path = out / "results_h8.json"
    if path.exists():
        raise SystemExit("refusing to overwrite results_h8.json")
    missing = [p for p in cfg["pair_ids"] if not (out / "pairs" / f"pair_{p}.json").exists()]
    if missing:
        raise SystemExit(f"run stage not complete: {missing}")
    pairs = {p: (json.loads((out / "pairs" / f"pair_{p}.json").read_text()), dict(np.load(out / "pairs" / f"pair_{p}.npz")))
             for p in cfg["pair_ids"]}
    res = tb_analyze(pairs, cfg["bootstrap_resamples"])
    res["git"] = git_state()
    path.write_text(json.dumps(res, indent=2, default=float))
    print(f"results written to {path}")
    return 0


def cmd_holm(h7_path: Path, h8_path: Path, out: Path) -> int:
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    h7c = json.loads(h7_path.read_text())["H7c"]
    h8 = json.loads(h8_path.read_text())["H8"]
    pv = {"H7c": h7c["p_one_sided"]}
    if h8["testable"]:
        pv["H8"] = h8["p_one_sided"]
    adj = holm(pv)
    res = {"holm_m": len(pv), "H7c": {"p_holm": adj["H7c"][0], "confirmed": bool(h7c["pass_before_holm"] and adj["H7c"][0] < 0.05)}}
    res["H8"] = ({"p_holm": adj["H8"][0], "confirmed": bool(h8["pass_before_holm"] and adj["H8"][0] < 0.05)}
                 if h8["testable"] else {"status": "not testable (gap(z0,z0) CI90 contains 0)"})
    res["git"] = git_state()
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("ta-run", "ta-analyze", "tb-run", "tb-analyze", "tb-describe", "holm"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1", help="i/k: run the pairs whose position mod k is i")
    ap.add_argument("--h7", type=Path, help="holm: results_h7.json")
    ap.add_argument("--h8", type=Path, help="holm: results_h8.json")
    args = ap.parse_args(argv)
    if args.stage == "holm":
        return cmd_holm(args.h7, args.h8, args.out)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    shard = tuple(map(int, args.shard.split("/")))
    if args.stage == "ta-run":
        return cmd_ta_run(cfg, args.out, shard)
    if args.stage == "tb-run":
        by_id = {p["pair_id"]: p for p in [*tier0_pairs(), *grid_pairs()]}
        return run_pairs(cfg, args.out, shard, by_id, tb_pair, cfg.get("decision", "D-043 (v1.10 §3)"))
    return {"ta-analyze": cmd_ta_analyze, "tb-analyze": cmd_tb_analyze, "tb-describe": cmd_tb_describe}[args.stage](cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
