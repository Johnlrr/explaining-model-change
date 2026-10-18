"""D-035: T5 label shift on a second new Tier 0 test set (C4, C5, C6). Post hoc origin (exp_0017, P5).

train    (exp_0021) — f_new for pairs 17–19 (seeds 0–2) from the exp_0017 f_old; G0-T5 on the 2 000 validation G0 points
                      (DD ≥ 0.02, max_{k≠aspect} |ΔFlip_k| ≤ 0.05 at every seed); one knob (10 epochs, lr 1e-3) if needed;
                      if it still fails, D-035 stops.
run      (exp_0022) — test set default_rng(2028); Δ-CE N, B1-det, B3, B1 on 300 valid points per pair (seed 8100 + pair_id).
analyze             — C4 (B1-det − B3, seed 6400), C5 (Δ-CE − B3, seed 6401), Holm m = 2; C6 (Δ-CE found-rate on y = 1 ≤ 0.05).
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
from dce.cli.calibrate_r import solver_config
from dce.cli.confirm_tier0 import new_test_set
from dce.cli.final_tier0 import Timer
from dce.cli.g0_reeval import load_verified
from dce.cli.g0_tier0 import git_state, load_config, make_batch_fn, sha256
from dce.data.drift import clean_splits
from dce.data.renderer import sample_factors, task_label
from dce.eval.g0 import decisive_disagreement, flip_rates
from dce.eval.metrics import MethodOutput, recovery_indicator
from dce.eval.records import from_direction, from_points, runtime_row
from dce.eval.rules import SEEDS, PairRecord, compare
from dce.eval.stats import holm
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.models.cnn import SmallCNN, predict_logits, train
from dce.solvers.dce import delta_ce

PAIRS = {17: 0, 18: 1, 19: 2}   # pair_id → seed
K_STAR = 0                       # aspect
RECIPES = [("default", {"epochs": 3, "lr": 3.0e-4}), ("knob1", {"epochs": 10, "lr": 1.0e-3})]


def t5_adaptation(pid: int, n: int = 10_000, p1: float = 0.8, noise: float = 0.05):
    rng = np.random.default_rng(2000 + pid)
    need1 = round(n * p1)
    need0 = n - need1
    ones, zeros = [], []
    while sum(len(a) for a in ones) < need1 or sum(len(a) for a in zeros) < need0:
        phi = sample_factors(n, rng)
        y = task_label(phi)
        ones.append(phi[y == 1])
        zeros.append(phi[y == 0])
    phi = np.concatenate([np.concatenate(ones)[:need1], np.concatenate(zeros)[:need0]])
    y = np.r_[np.ones(need1, dtype=np.int64), np.zeros(need0, dtype=np.int64)]
    perm = rng.permutation(n)
    phi, y = phi[perm], y[perm]
    flip = rng.random(n) < noise
    return phi, np.where(flip, 1 - y, y)


def logits_fn(model):
    clean = make_batch_fn(False)
    return lambda phi: predict_logits(model, phi, clean).numpy()


def cmd_train(cfg, out: Path) -> int:
    (out / "models").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", 8))
    val = clean_splits(seed=cfg["data_seed"])["val"]
    g0_phi = val.phi[:2000]
    noisy = make_batch_fn(True)
    cache, manifest, res = {}, [], {"g0_points": 2000, "attempts": []}
    f_old = {s: load_verified(Path(cfg["f_old_source"]), f"f_old_s{s}", cache)[0] for s in PAIRS.values()}
    for name, recipe in RECIPES:
        rows = []
        for pid, s in PAIRS.items():
            m = SmallCNN()
            m.load_state_dict(f_old[s].state_dict())
            phi_a, y_a = t5_adaptation(pid)
            losses = train(m, phi_a, y_a, noisy, epochs=recipe["epochs"], lr=recipe["lr"], batch_size=256,
                           order_seed=3000 + pid, noise_seed=4000 + pid)
            path = out / "models" / f"f_new_p{pid}_T5_s{s}_{name}.pt"
            torch.save(m.state_dict(), path)
            manifest.append({"name": path.stem, "path": f"models/{path.name}", "sha256": sha256(path)})
            lo, ln = logits_fn(f_old[s])(g0_phi), logits_fn(m)(g0_phi)
            fo = flip_rates(lambda p, f=f_old[s]: logits_fn(f)(p).argmax(1), g0_phi)
            fn = flip_rates(lambda p, f=m: logits_fn(f)(p).argmax(1), g0_phi)
            delta = fn - fo
            dd = float(decisive_disagreement(lo, ln, cfg["kappa"]).mean())
            other = float(np.abs(np.delete(delta, K_STAR)).max())
            rows.append({"pair_id": pid, "seed": s, "recipe": name, "loss": losses, "dd": dd, "delta_flip": delta.tolist(),
                         "max_abs_delta_flip_other": other, "pass": bool(dd >= 0.02 and other <= 0.05),
                         "share_pred_1_old": float((lo.argmax(1) == 1).mean()), "share_pred_1_new": float((ln.argmax(1) == 1).mean())})
            print(f"{name} pair {pid}: pass={rows[-1]['pass']}", flush=True)
        res["attempts"].append({"recipe": name, **recipe, "pairs": rows, "pass": all(r["pass"] for r in rows)})
        if all(r["pass"] for r in rows):
            res["chosen_recipe"] = name
            break
    res.setdefault("chosen_recipe", None)
    res["d035_stops"] = res["chosen_recipe"] is None
    (out / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "artifacts": manifest}, indent=2))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "metadata.json").write_text(json.dumps({"experiment": cfg["train_experiment"], "git": git_state(), "split": "validation"}, indent=2))
    print(json.dumps({"chosen_recipe": res["chosen_recipe"], "d035_stops": res["d035_stops"]}))
    return 0


def cmd_run(cfg, out: Path) -> int:
    src = Path(cfg["train_run"])
    tr = json.loads((src / "metrics.json").read_text())
    if tr["chosen_recipe"] is None:
        raise SystemExit("G0-T5 failed: D-035 stops; the test set is not generated")
    man = {a["name"]: a["sha256"] for a in json.loads((src / "artifact_manifest.json").read_text())["artifacts"]}
    torch.set_num_threads(cfg.get("threads", 8))
    phi, y, digest = new_test_set(cfg["data_seed_new"], cfg["n_samples"])
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "experiment": cfg["experiment_id"], "decision": "D-035", "data_seed_new": cfg["data_seed_new"], "data_sha256": digest,
        "recipe": tr["chosen_recipe"], "git_runs": []}
    if meta["data_sha256"] != digest:
        raise SystemExit("regenerated data differ from the recorded hash")
    meta["git_runs"].append({"git": git_state(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    meta["status"] = "RUNNING"
    meta_path.write_text(json.dumps(meta, indent=2))
    base = solver_config(cfg, r=float(cfg["r"]))
    cache = {}
    for pid, s in PAIRS.items():
        npz = out / "pairs" / f"pair_{pid}.npz"
        if npz.exists():
            continue
        f_old = load_verified(Path(cfg["f_old_source"]), f"f_old_s{s}", cache)[0]
        name = f"f_new_p{pid}_T5_s{s}_{tr['chosen_recipe']}"
        path = src / "models" / f"{name}.pt"
        if sha256(path) != man[name]:
            raise SystemExit(f"hash mismatch for {path}")
        f_new = SmallCNN()
        f_new.load_state_dict(torch.load(path))
        f_new.eval()
        n = cfg["n_points"]
        with Timer() as tp:
            keep = valid_points(phi, y, Tier0Margin(f_old, y), Tier0Margin(f_new, y), n, cfg["points_seed_base"] + pid, cfg["kappa"])
            z0 = torch.as_tensor(phi[keep], dtype=torch.float32)
            m_old, m_new = Tier0Margin(f_old, y[keep]), Tier0Margin(f_new, y[keep])
            seed = cfg["seeds"]["restart_base"] + pid
            outs, rt, arrays = {}, {}, {"point_ids": keep, "y": y[keep], "z0": z0.numpy()}
            with Timer() as t:
                res = delta_ce(z0, m_old, m_new, base, seed, kind="N")
            outs["dce_N"], rt["dce_N"] = from_points(res.found, res.z, z0), runtime_row(res, t.wall, n)
            arrays["dce_N_distance"] = res.distance.numpy()
            for nm, det in (("b1_det", True), ("b1", False)):
                valid, _, ce_o, ce_n = b1_diff(z0, m_old, m_new, base, seed, deterministic=det)
                outs[nm] = from_points(valid, ce_n.z, ce_o.z)
            v, d = b3_grad(z0, m_old, m_new)
            outs["b3"] = from_direction(v, d)
            for k, o in outs.items():
                arrays[f"{k}__found"], arrays[f"{k}__dphi"] = o.found, o.dphi
        np.savez_compressed(npz, **arrays)
        (out / "pairs" / f"pair_{pid}.json").write_text(json.dumps({"pair_id": pid, "seed": s, "n_points": n, "runtime": rt,
                                                                     "wall_s_pair": round(tp.wall, 1)}, indent=2))
        print(f"pair {pid} done in {tp.wall / 60:.1f} min", flush=True)
    meta["status"] = "RUN_COMPLETE"
    meta_path.write_text(json.dumps(meta, indent=2))
    return 0


def analyze(pairs: dict, cfg: dict) -> dict:
    B = cfg["bootstrap_resamples"]
    recs = []
    for pid, arr in pairs.items():
        ind = {m: recovery_indicator(MethodOutput(arr[f"{m}__found"], arr[f"{m}__dphi"]), K_STAR) for m in ("dce_N", "b1_det", "b3", "b1")}
        recs.append(PairRecord(pid, PAIRS[pid], ind))
    c4 = compare("C4", recs, "b1_det", "b3", np.random.default_rng(cfg["seeds_confirm"]["C4"]), B=B)
    c5 = compare("C5", recs, "dce_N", "b3", np.random.default_rng(cfg["seeds_confirm"]["C5"]), B=B)
    adj = holm({"C4": c4.boot.p, "C5": c5.boot.p})
    out = {"label": "D-035 (post-hoc origin, tested on fresh data)", "holm_m": 2}
    for name, c in (("C4", c4), ("C5", c5)):
        out[name] = {"diff": c.boot.diff, "ci95": [c.boot.ci_lo, c.boot.ci_hi], "p_two_sided": c.boot.p, "p_holm": adj[name][0],
                     "seed_diffs": c.seed_diffs,
                     "pass": bool(adj[name][0] < 0.05 and c.boot.diff > 0 and all(c.seed_diffs.get(s, -1) > 0 for s in SEEDS))}
    c6 = {}
    for pid, arr in pairs.items():
        y1 = arr["y"] == 1
        c6[pid] = {"n_y1": int(y1.sum()), "dce_found_rate_y1": float(arr["dce_N__found"][y1].mean()),
                   "dce_found_rate_y0": float(arr["dce_N__found"][~y1].mean())}
    out["C6"] = {"pairs": c6, "pass": all(v["dce_found_rate_y1"] <= 0.05 for v in c6.values())}
    out["recovery_means"] = {m: float(np.mean([r.indicators[m].mean() for r in recs])) for m in ("dce_N", "b1_det", "b3", "b1")}
    out["found_rate_means"] = {m: float(np.mean([pairs[p][f"{m}__found"].mean() for p in pairs])) for m in ("dce_N", "b1_det", "b3", "b1")}
    return out


def cmd_analyze(cfg, out: Path) -> int:
    meta = json.loads((out / "metadata.json").read_text())
    if meta.get("status") != "RUN_COMPLETE" or (out / "results.json").exists():
        raise SystemExit("run incomplete or already analyzed")
    pairs = {pid: dict(np.load(out / "pairs" / f"pair_{pid}.npz")) for pid in PAIRS}
    res = analyze(pairs, cfg)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    meta["status"] = "ANALYZED"
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"results written to {out / 'results.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("train", "run", "analyze"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    return {"train": cmd_train, "run": cmd_run, "analyze": cmd_analyze}[args.stage](cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
