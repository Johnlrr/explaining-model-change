"""D-052 X2 + X3: split D (data seeds 9500 + pair_id), new pairs on the exp_0002 f_old checkpoints.

Pairs: X2 offsets 100–108 = α 0 × t ∈ {0.53, 0.56, 0.60} × f_old seeds {2, 3, 4} (grid recipe of exp_0025: v1.9
adaptation set, 10 epochs, lr 1e-3); X3 nulls 110–117 = N0 × seeds 0–4, N1 × seeds 0–2 (null recipe of exp_0002:
3 epochs, lr 3e-4, adaptation label noise 5%). Every per-pair seed is a tier0.yaml base + pair_id.

train     — fine-tune every f_new; manifest with sha256.
qualify   — validation only. Offsets: v1.9.1 checks (1), (3), (2a) against t̂_old of the same seed, (2b) order.
            Nulls: decisive disagreement recorded, NOT used to admit (X3: all nulls admitted).
run       — split D: 300 valid points per pair; Δ-CE type N (all pairs) and type O (nulls), CF-diff (B1-det),
            Grad-diff (B3), margin scores s_M, s_M,r (as exp_0046). Locked solver.
analyze   — X3 (null witness share; thresholds from these nulls) and X2 (Grad-diff < 0.2 per qualified offset pair,
            exact sign test if ≥ 6 qualify; detection on the lowered class at the X3 thresholds; M's branch rule).

Usage:
    python -m dce.cli.split_d {train,qualify,run,analyze} --config configs/experiments/exp_0049.yaml --out runs/experiments/exp_0049
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dce.analysis.qualification import grid_checks_v191, qualification_row
from dce.baselines.b1_diff import b1_diff
from dce.baselines.b3_grad import b3_grad
from dce.cli.calibrate_r import solver_config
from dce.cli.final_tier0 import Timer
from dce.cli.g0_reeval import load_verified
from dce.cli.g0_tier0 import git_state, load_config, make_batch_fn, sha256
from dce.cli.review_0211 import margin_scores, rank_metrics, thresholds
from dce.cli.review_d048 import cmd_json
from dce.cli.v19_grid import logits_np
from dce.cli.v110_test import data_for, run_pairs
from dce.data.drift import adaptation_set, clean_splits, ro_adaptation_set, ro_label
from dce.eval.g0 import decisive_disagreement
from dce.eval.records import from_direction, from_points, runtime_row
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.models.cnn import SmallCNN, train
from dce.solvers.dce import delta_ce

ASPECT = 0


def d_pairs() -> list[dict]:
    pairs = []
    for i, (t, s) in enumerate((t, s) for s in (2, 3, 4) for t in (0.53, 0.56, 0.60)):
        pairs.append({"pair_id": 100 + i, "kind": "offset", "alpha": 0.0, "t": t, "seed": s, "k_star": "aspect"})
    for i, (c, s) in enumerate([("N0", s) for s in range(5)] + [("N1", s) for s in range(3)]):
        pairs.append({"pair_id": 110 + i, "kind": "null", "condition": c, "seed": s, "k_star": None})
    return pairs


BY_ID = {p["pair_id"]: p for p in d_pairs()}


def name_of(p: dict) -> str:
    return f"f_new_p{p['pair_id']}_" + (f"a0.00_t{p['t']:.2f}" if p["kind"] == "offset" else p["condition"]) + f"_s{p['seed']}"


def load_d_models(cfg, p, cache):
    f_old, e_old = load_verified(Path(cfg["f_old_source"]), f"f_old_s{p['seed']}", cache)
    out = Path(cfg["models_run"])
    man = {a["name"]: a for a in json.loads((out / "artifact_manifest.json").read_text())["artifacts"]}
    entry = man[name_of(p)]
    path = out / entry["path"]
    if sha256(path) != entry["sha256"]:
        raise SystemExit(f"hash mismatch for {path}")
    f_new = SmallCNN()
    f_new.load_state_dict(torch.load(path))
    f_new.eval()
    return f_old, f_new, [e_old, {"name": entry["name"], "path": str(path), "sha256": entry["sha256"]}]


def cmd_train(cfg, out: Path) -> int:
    (out / "models").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", 8))
    sd, n_adapt = cfg["seeds"], cfg["n_adapt"]
    man_path = out / "artifact_manifest.json"
    manifest = json.loads(man_path.read_text())["artifacts"] if man_path.exists() else []
    done, cache, rows = {a["name"] for a in manifest}, {}, []
    for pid in cfg["pair_ids"]:
        p = BY_ID[pid]
        if name_of(p) in done:
            continue
        f_old, _ = load_verified(Path(cfg["f_old_source"]), f"f_old_s{p['seed']}", cache)
        m = SmallCNN()
        m.load_state_dict(f_old.state_dict())
        if p["kind"] == "offset":
            phi_a, y_a = ro_adaptation_set(0.0, p["t"], sd["adapt_data_base"] + pid, n=n_adapt,
                                           label_noise=cfg["adapt_label_noise"])
            rc = cfg["recipe_offset"]
        else:
            phi_a, y_a = adaptation_set(p["condition"], seed=sd["adapt_data_base"] + pid, n=n_adapt,
                                        label_noise=cfg["adapt_label_noise"])
            rc = cfg["recipe_null"]
        t0 = time.time()
        losses = train(m, phi_a, y_a, make_batch_fn(True), epochs=rc["epochs"], lr=rc["lr"], batch_size=rc["batch_size"],
                       order_seed=sd["adapt_order_base"] + pid, noise_seed=sd["adapt_noise_base"] + pid)
        path = out / "models" / f"{name_of(p)}.pt"
        torch.save(m.state_dict(), path)
        manifest.append({"name": name_of(p), "kind": "checkpoint", "path": f"models/{path.name}", "sha256": sha256(path)})
        man_path.write_text(json.dumps({"schema_version": 1, "artifacts": manifest}, indent=2))
        rows.append({**p, "loss": losses, "wall_s": round(time.time() - t0, 1)})
        print(f"trained {name_of(p)} in {rows[-1]['wall_s']} s, final loss {losses[-1]:.4f}", flush=True)
    log = out / "train_log.json"
    log.write_text(json.dumps((json.loads(log.read_text()) if log.exists() else []) + rows, indent=2))
    return 0


def cmd_qualify(cfg, out: Path) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    val = clean_splits(seed=cfg["data_seed"])["val"]
    phi, g0_phi = val.phi[: cfg["n_qualify"]], val.phi[: cfg["n_g0_points"]]
    cache, rows, t_old = {}, [], {}
    for pid in cfg["pair_ids"]:
        p = BY_ID[pid]
        f_old, f_new, _ = load_d_models(cfg | {"models_run": str(out)}, p, cache)
        dd = float(decisive_disagreement(logits_np(f_old, g0_phi), logits_np(f_new, g0_phi), cfg["kappa"]).mean())
        if p["kind"] == "null":
            rows.append({"pair_id": pid, "kind": "null", "condition": p["condition"], "seed": p["seed"], "dd": dd,
                         "admitted": True})
            continue
        if p["seed"] not in t_old:
            lo = logits_np(f_old, phi)
            t_old[p["seed"]] = qualification_row(phi, (lo[:, 1] - lo[:, 0] > 0).astype(np.int64),
                                                 ro_label(phi, 0.0, 0.5), 0.0, 0.5)["t_hat"]
        ln = logits_np(f_new, phi)
        row = qualification_row(phi, (ln[:, 1] - ln[:, 0] > 0).astype(np.int64), ro_label(phi, 0.0, p["t"]), 0.0, p["t"])
        row.update(pair_id=pid, kind="offset", alpha=0.0, t=p["t"], seed=p["seed"], dd=dd, dt=row["t_hat"] - t_old[p["seed"]])
        for k in ("t_ok", "qualified"):
            row.pop(k, None)
        rows.append(row)
    off = [r for r in rows if r["kind"] == "offset"]
    checks = grid_checks_v191(off) if off else None
    if checks:
        bad = {p for v in checks["order_violations"] for p in v["pairs"]}
        for r in off:
            r["qualified"] = bool(checks["per_pair"][r["pair_id"]]["base_qualified"] and r["pair_id"] not in bad)
    res = {"split": "validation", "criteria": "v1.9.1 (offsets); nulls all admitted (D-052 X3)", "pairs": rows,
           "checks": checks, "t_hat_old": t_old, "git": git_state()}
    (out / "qualification.json").write_text(json.dumps(res, indent=2, default=float))
    for r in rows:
        print({k: r[k] for k in ("pair_id", "kind", "dd", "qualified", "t_hat", "dt") if k in r}, flush=True)
    return 0


def d_pair(cfg, p, cache):
    pid, n = p["pair_id"], cfg["n_points"]
    base = solver_config(cfg, r=float(cfg["r"]))
    f_old, f_new, entries = load_d_models(cfg, p, cache)
    phi, y, digest = data_for(cfg, pid)
    keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), n, cfg["seeds"]["points_base"] + pid, cfg["kappa"])
    z0 = torch.as_tensor(phi[keep], dtype=torch.float32)
    yk = y[keep]
    m_old, m_new = Tier0Margin(f_old, yk), Tier0Margin(f_new, yk)
    seed = cfg["seeds"]["restart_base"] + pid
    arrays, rt = {"point_ids": keep, "y": yk, "z0": z0.numpy()}, {}
    for kind in ("N", "O") if p["kind"] == "null" else ("N",):
        with Timer() as t:
            res = delta_ce(z0, m_old, m_new, base, seed, kind=kind)
        rt[f"dce_{kind}"] = runtime_row(res, t.wall, n)
        o = from_points(res.found, res.z, z0)
        arrays[f"dce_{kind}__found"], arrays[f"dce_{kind}__dphi"] = o.found, o.dphi
        arrays[f"dce_{kind}_distance"] = res.distance.numpy()
    valid, _, ce_o, ce_n = b1_diff(z0, m_old, m_new, base, seed, deterministic=True)
    o = from_points(valid, ce_n.z, ce_o.z)
    arrays["b1_det__found"], arrays["b1_det__dphi"] = o.found, o.dphi
    valid, d = b3_grad(z0, m_old, m_new)
    o = from_direction(valid, d)
    arrays["b3__found"], arrays["b3__dphi"] = o.found, o.dphi
    s = margin_scores(f_old, f_new, z0.numpy(), yk, cfg["kappa"], float(cfg["r"]), cfg["n_ball"], cfg["ball_seed_base"] + pid)
    arrays["s_M"], arrays["s_Mr"] = s["s_M"], s["s_Mr"]
    row = {**p, "n_points": n, "data_seed": cfg["data_seed_base"] + pid, "data_sha256": digest, "runtime": rt,
           "checkpoints": entries}
    return arrays, row


def cfdiff(a) -> np.ndarray:
    return np.where(a["b1_det__found"], np.linalg.norm(a["b1_det__dphi"], axis=1), 0.0)


def analyze(cfg) -> dict:
    out = Path(cfg["out_run"])
    qual = {r["pair_id"]: r for r in json.loads((out / "qualification.json").read_text())["pairs"]}
    load = lambda pid: dict(np.load(out / "pairs" / f"pair_{pid}.npz"))
    nulls = [pid for pid in cfg["pair_ids"] if BY_ID[pid]["kind"] == "null"]
    offs = [pid for pid in cfg["pair_ids"] if BY_ID[pid]["kind"] == "offset"]
    res = {"label": "D-052 X2 + X3", "x3": {"pairs": []}, "x2": {"pairs": []}}

    # X3: null witness share and thresholds
    sc = {"s_M": [], "s_Mr": [], "cfdiff": []}
    wit = []
    for pid in nulls:
        a = load(pid)
        w = a["dce_N__found"] | a["dce_O__found"]
        wit.append(w)
        res["x3"]["pairs"].append({"pair_id": pid, "condition": BY_ID[pid]["condition"], "seed": BY_ID[pid]["seed"],
                                   "dd_validation": qual[pid]["dd"], "witness_share_N_or_O": float(w.mean()),
                                   "witness_share_N": float(a["dce_N__found"].mean()),
                                   "cfdiff_attributes": float(a["b1_det__found"].mean())})
        sc["s_M"].append(a["s_M"]), sc["s_Mr"].append(a["s_Mr"]), sc["cfdiff"].append(cfdiff(a))
    sc = {k: np.concatenate(v) for k, v in sc.items()}
    w = np.concatenate(wit)
    taus = {k: thresholds(v) for k, v in sc.items()}
    res["x3"].update(n_null_queries=int(w.size), witness_share_pooled=float(w.mean()),
                     max_witness_share_pair=max(p["witness_share_N_or_O"] for p in res["x3"]["pairs"]), tau=taus,
                     prediction_le_005=bool(w.mean() <= 0.05 and all(p["witness_share_N_or_O"] <= 0.05 for p in res["x3"]["pairs"])))

    # X2: Grad-diff < 0.2 per qualified offset pair; detection on the lowered class (class 1) at X3 thresholds
    for pid in offs:
        a = load(pid)
        if not np.array_equal(a["y"], (a["z0"][:, ASPECT] > 0.5).astype(a["y"].dtype)):
            raise SystemExit(f"pair {pid}: saved y is not 1[aspect > 0.5]")
        low = a["y"] == 1
        row = {"pair_id": pid, "t": BY_ID[pid]["t"], "seed": BY_ID[pid]["seed"], "qualified": bool(qual[pid].get("qualified")),
               "graddiff_recovery": rank_metrics(a["b3__found"], a["b3__dphi"], ASPECT)["recovery"],
               "dce_recovery": rank_metrics(a["dce_N__found"], a["dce_N__dphi"], ASPECT)["recovery"],
               "n_lowered": int(low.sum()), "dce_lowered": float(a["dce_N__found"][low].mean())}
        for name, s in (("s_M", a["s_M"]), ("s_Mr", a["s_Mr"]), ("cfdiff", cfdiff(a))):
            for k, t in taus[name].items():
                row[f"{name}_lowered_{k}"] = float((s[low] > t).mean())
        res["x2"]["pairs"].append(row)
    rows = [r for r in res["x2"]["pairs"] if r["qualified"]]
    k = sum(r["graddiff_recovery"] < 0.2 for r in rows)
    n = len(rows)
    from scipy.stats import binomtest
    res["x2"].update(n_qualified=n, k_graddiff_below_02=k,
                     confirmatory=n >= 6,
                     sign_test_p=float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else None)
    n_cheap = sum(r["s_M_lowered_fpr0"] >= r["dce_lowered"] - 0.05 for r in rows)
    res["x2"]["n_pairs_sM_ge_dce_minus_0.05"] = int(n_cheap)
    res["x2"]["branch"] = ("whether is cheap" if n and n_cheap >= (5 / 7) * n else
                           "witness more sensitive" if rows and all(r["dce_lowered"] >= r["s_M_lowered_fpr0"] + 0.20 for r in rows)
                           else "report only (framing as whether is cheap)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("train", "qualify", "run", "analyze"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    if args.stage == "train":
        return cmd_train(cfg, args.out)
    if args.stage == "qualify":
        return cmd_qualify(cfg, args.out)
    if args.stage == "run":
        return run_pairs(cfg, args.out, tuple(map(int, args.shard.split("/"))), BY_ID, d_pair, "D-052")
    return cmd_json(analyze, "results_d.json")(cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
