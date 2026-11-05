"""D-057 X6: an update with two independent local changes (colour shortcut everywhere + label flip for size > 0.75);
does the nearest witness point to the change near the query? Registered on OSF (see D-057).

Pairs 130–137: f_old (exp_0002) seeds 0–4 (130–134) and 0–2 with a second adaptation draw (135–137); every per-pair
seed is a tier0.yaml base + pair_id. Split G: data seed 13000 + pair_id.

train     — fine-tune every f_new on `ss_adaptation_set` (10 epochs, lr 1e-3, batch 256); manifest with sha256.
qualify   — validation: ΔFlip_colour ≥ 0.15, ΔFlip_size ≥ 0.15, max |ΔFlip| over bg, posx ≤ 0.05 (G0 flip rates).
run       — split G, 300 valid queries per pair: Δ-CE N, CF-diff (B1-det), Grad-diff (B3), CF-new, Imp-diff (test data).
analyze   — locality L (near: size ≥ 0.60, far: size ≤ 0.45) for Δ-CE (registered), Grad-diff, CF-diff, Imp-diff;
            sign test over qualified pairs if ≥ 6 qualify; set-recovery {colour, size}; MRR of colour and size.

    python -m dce.cli.x6_multi {train,qualify,run,analyze} --config configs/experiments/exp_0060.yaml --out runs/experiments/exp_0060
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dce.baselines.b1_diff import b1_diff
from dce.baselines.b3_grad import b3_grad
from dce.baselines.b4_perm import b4_perm
from dce.cli.calibrate_r import solver_config
from dce.cli.final_tier0 import Timer
from dce.cli.g0_reeval import load_verified
from dce.cli.g0_tier0 import git_state, load_config, make_batch_fn, sha256
from dce.cli.review_0211 import kstar_rank
from dce.cli.review_d048 import cmd_json
from dce.cli.v19_grid import logits_np
from dce.cli.v110_test import data_for, run_pairs
from dce.data.drift import clean_splits, ss_adaptation_set
from dce.eval.g0 import flip_rates
from dce.eval.records import MethodOutput, from_direction, from_points, runtime_row
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.models.cnn import SmallCNN, train
from dce.solvers.dce import delta_ce
from dce.solvers.ordinary_ce import ordinary_ce

ASPECT, SIZE, COLOUR, BG, POSX = range(5)
METHODS = ("dce_N", "b1_det", "b3", "a2", "b4")


def x6_pairs() -> list[dict]:
    seeds = [0, 1, 2, 3, 4, 0, 1, 2]
    return [{"pair_id": 130 + i, "kind": "SS", "seed": s, "k_star": ["colour", "size"]} for i, s in enumerate(seeds)]


BY_ID = {p["pair_id"]: p for p in x6_pairs()}


def name_of(p: dict) -> str:
    return f"f_new_p{p['pair_id']}_SS_s{p['seed']}"


def load_models(cfg, p, cache):
    f_old, e_old = load_verified(Path(cfg["f_old_source"]), f"f_old_s{p['seed']}", cache)
    run = Path(cfg["models_run"])
    man = {a["name"]: a for a in json.loads((run / "artifact_manifest.json").read_text())["artifacts"]}
    entry = man[name_of(p)]
    path = run / entry["path"]
    if sha256(path) != entry["sha256"]:
        raise SystemExit(f"hash mismatch for {path}")
    f_new = SmallCNN()
    f_new.load_state_dict(torch.load(path))
    f_new.eval()
    return f_old, f_new, [e_old, {"name": entry["name"], "path": str(path), "sha256": entry["sha256"]}]


def cmd_train(cfg, out: Path) -> int:
    (out / "models").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", 8))
    sd, rc = cfg["seeds"], cfg["recipe"]
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
        phi_a, y_a = ss_adaptation_set(sd["adapt_data_base"] + pid, n=cfg["n_adapt"], t1_correlation=cfg["t1_correlation"],
                                       size_threshold=cfg["size_threshold"], label_noise=cfg["adapt_label_noise"])
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


def qualification(flip_old: np.ndarray, flip_new: np.ndarray, q: dict) -> dict:
    d = flip_new - flip_old
    other = float(np.abs(d[[BG, POSX]]).max())
    return {"delta_flip": d.tolist(), "delta_flip_colour": float(d[COLOUR]), "delta_flip_size": float(d[SIZE]),
            "max_abs_delta_flip_bg_posx": other,
            "qualified": bool(d[COLOUR] >= q["min_delta"] and d[SIZE] >= q["min_delta"] and other <= q["max_other"])}


def cmd_qualify(cfg, out: Path) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    g0_phi = clean_splits(seed=cfg["data_seed"])["val"].phi[: cfg["n_g0_points"]]
    cache, rows = {}, []
    for pid in cfg["pair_ids"]:
        p = BY_ID[pid]
        f_old, f_new, _ = load_models(cfg | {"models_run": str(out)}, p, cache)
        fo = flip_rates(lambda z, f=f_old: logits_np(f, z).argmax(1), g0_phi)
        fn = flip_rates(lambda z, f=f_new: logits_np(f, z).argmax(1), g0_phi)
        rows.append({"pair_id": pid, "seed": p["seed"], "flip_old": fo.tolist(), "flip_new": fn.tolist(),
                     **qualification(fo, fn, cfg["qualify"])})
        print({k: rows[-1][k] for k in ("pair_id", "delta_flip_colour", "delta_flip_size", "max_abs_delta_flip_bg_posx",
                                        "qualified")}, flush=True)
    res = {"split": "validation", "criteria": "D-057", "pairs": rows,
           "n_qualified": sum(r["qualified"] for r in rows), "git": git_state()}
    (out / "qualification.json").write_text(json.dumps(res, indent=2))
    return 0


def x6_pair(cfg, p, cache):
    pid, n, kappa = p["pair_id"], cfg["n_points"], cfg["kappa"]
    base = solver_config(cfg, r=float(cfg["r"]))
    f_old, f_new, entries = load_models(cfg, p, cache)
    phi, y, digest = data_for(cfg, pid)
    keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), n, cfg["seeds"]["points_base"] + pid, kappa)
    z0 = torch.as_tensor(phi[keep], dtype=torch.float32)
    yk = y[keep]
    m_old, m_new = Tier0Margin(f_old, yk), Tier0Margin(f_new, yk)
    seed = cfg["seeds"]["restart_base"] + pid
    arrays, rt, outs = {"point_ids": keep, "y": yk, "z0": z0.numpy()}, {}, {}
    with Timer() as t:
        res = delta_ce(z0, m_old, m_new, base, seed, kind="N")
    outs["dce_N"], rt["dce_N"] = from_points(res.found, res.z, z0), runtime_row(res, t.wall, n)
    arrays["dce_N_distance"] = res.distance.numpy()
    with Timer() as t:
        valid, _, ce_o, ce_n = b1_diff(z0, m_old, m_new, base, seed, deterministic=True)
    outs["b1_det"], rt["b1_det"] = from_points(valid, ce_n.z, ce_o.z), {"wall_s_per_point": t.wall / n}
    with Timer() as t:
        v, d = b3_grad(z0, m_old, m_new)
    outs["b3"], rt["b3"] = from_direction(v, d), {"wall_s_per_point": t.wall / n}
    with Timer() as t:
        res = ordinary_ce(z0, m_new, base, seed)
    outs["a2"], rt["a2"] = from_points(res.found, res.z, z0), runtime_row(res, t.wall, n)
    with Timer() as t:
        pred = lambda f: (lambda q: logits_np(f, q).argmax(1))
        delta = b4_perm(pred(f_old), pred(f_new), phi[: cfg["b4_points"]], y[: cfg["b4_points"]])
    outs["b4"], rt["b4"] = MethodOutput(np.ones(n, bool), np.tile(delta, (n, 1))), {"wall_s_total": t.wall}
    for k, o in outs.items():
        arrays[f"{k}__found"], arrays[f"{k}__dphi"] = o.found, o.dphi
    row = {**p, "n_points": n, "data_seed": cfg["data_seed_base"] + pid, "data_sha256": digest, "runtime": rt,
           "checkpoints": entries}
    return arrays, row


def locality(found: np.ndarray, dphi: np.ndarray, size0: np.ndarray, near: float, far: float) -> dict:
    """L = share of outputs whose largest |Δφ| is size (ties: not size), near minus far; NaN if a group is empty."""
    top_size = kstar_rank(found, dphi, SIZE) == 1
    nr, fr = found & (size0 >= near), found & (size0 <= far)
    sn = float(top_size[nr].mean()) if nr.any() else float("nan")
    sf = float(top_size[fr].mean()) if fr.any() else float("nan")
    return {"n_near": int(nr.sum()), "n_far": int(fr.sum()), "share_size_near": sn, "share_size_far": sf,
            "L": sn - sf, "n_size_near": int(top_size[nr].sum()), "n_size_far": int(top_size[fr].sum())}


def set_metrics(found: np.ndarray, dphi: np.ndarray) -> dict:
    rc, rs = kstar_rank(found, dphi, COLOUR), kstar_rank(found, dphi, SIZE)
    mrr = lambda r: float(np.where(r > 0, 1.0 / np.maximum(r, 1), 0.0).mean())
    return {"set_recovery": float(((rc > 0) & (rc <= 2) & (rs > 0) & (rs <= 2)).mean()), "mrr_colour": mrr(rc),
            "mrr_size": mrr(rs), "found": float(found.mean()),
            "top1_colour": float((rc == 1).mean()), "top1_size": float((rs == 1).mean())}


def analyze(cfg) -> dict:
    out = Path(cfg["out_run"])
    qual = {r["pair_id"]: r for r in json.loads((out / "qualification.json").read_text())["pairs"]}
    lc = cfg["locality"]
    res = {"label": "D-057 X6", "pairs": []}
    for pid in cfg["pair_ids"]:
        a = dict(np.load(out / "pairs" / f"pair_{pid}.npz"))
        row = {"pair_id": pid, "seed": BY_ID[pid]["seed"], "qualified": qual[pid]["qualified"]}
        for m in METHODS:
            row[m] = {**locality(a[f"{m}__found"], a[f"{m}__dphi"], a["z0"][:, SIZE], lc["near"], lc["far"]),
                      **set_metrics(a[f"{m}__found"], a[f"{m}__dphi"])}
        res["pairs"].append(row)
    q = [r for r in res["pairs"] if r["qualified"]]
    n = len(q)
    L = [r["dce_N"]["L"] for r in q]
    k = sum(1 for x in L if np.isfinite(x) and x > 0)
    pooled = {}
    for m in METHODS:
        nn_ = sum(r[m]["n_near"] for r in q)
        nf = sum(r[m]["n_far"] for r in q)
        pooled[m] = (sum(r[m]["n_size_near"] for r in q) / nn_ if nn_ else float("nan")) - \
                    (sum(r[m]["n_size_far"] for r in q) / nf if nf else float("nan"))
    from scipy.stats import binomtest
    res["primary"] = {"n_qualified": n, "confirmatory": n >= 6, "k_L_positive": k, "L_per_pair": L,
                      "L_pooled": pooled["dce_N"],
                      "sign_test_p": float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else None,
                      "prediction_met": bool(n and k == n and pooled["dce_N"] >= 0.30)}
    res["L_pooled_by_method"] = pooled
    res["imp_diff_neither_first"] = {r["pair_id"]: bool(r["b4"]["top1_colour"] == 0 and r["b4"]["top1_size"] == 0)
                                     for r in res["pairs"]}
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
        return run_pairs(cfg, args.out, tuple(map(int, args.shard.split("/"))), BY_ID, x6_pair, "D-057")
    return cmd_json(analyze, "results_x6.json")(cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
