"""D-036 Stage B (v1.9 §4, D-040): null pairs on Adult; validation only until the author confirms.

train     — 5 f_old (MLP, 20 epochs, lr 1e-3) and 5 f_new (fine-tune on a 10 000-sample subset of train drawn without
            replacement, 3 epochs, lr 3e-4): pair B_s = seed s.
qualify   — DD ≤ 0.02 on the first 2 000 validation points (κ = 0.5) and f_old validation accuracy ≥ 0.80, reported with
            the validation majority-class rate.
calibrate — r = 2 × median CE(f_new) distance, r = ∞, on 300 valid validation points of pair B_0 (spec §6.3); once.
No test-split value is read by any stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from dce.cli.calibrate_r import solver_config
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.data.adult import adult_splits
from dce.eval.g0 import decisive_disagreement
from dce.eval.tier0_margins import valid_points
from dce.models.cnn import train
from dce.models.mlp import MLP, TabMargin, as_batch
from dce.solvers.ordinary_ce import ordinary_ce


def logits_np(model, x) -> np.ndarray:
    with torch.no_grad():
        return model.eval()(torch.as_tensor(x, dtype=torch.float32)).numpy()


def load(out: Path, name: str) -> MLP:
    man = {a["name"]: a for a in json.loads((out / "artifact_manifest.json").read_text())["artifacts"]}
    path = out / man[name]["path"]
    if sha256(path) != man[name]["sha256"]:
        raise SystemExit(f"hash mismatch for {path}")
    m = MLP()
    m.load_state_dict(torch.load(path))
    return m.eval()


def cmd_train(cfg, out: Path) -> int:
    (out / "models").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg["threads"])
    data = adult_splits(Path(cfg["data_path"]), seed=cfg["split_seed"])
    xtr, ytr = data.x["train"], data.y["train"]
    sd, fo, fn = cfg["seeds"], cfg["f_old"], cfg["f_new"]
    manifest, log = [], []

    def save(model, name):
        path = out / "models" / f"{name}.pt"
        torch.save(model.state_dict(), path)
        manifest.append({"name": name, "kind": "checkpoint", "path": f"models/{path.name}", "sha256": sha256(path)})

    for s in cfg["seeds_list"]:
        torch.manual_seed(sd["init_base"] + s)
        f_old = MLP()
        lo = train(f_old, xtr, ytr, as_batch, epochs=fo["epochs"], lr=fo["lr"], batch_size=fo["batch_size"],
                   order_seed=sd["order_base"] + s, noise_seed=sd["noise_base"] + s)
        save(f_old, f"f_old_s{s}")
        sub = np.random.default_rng(sd["subset_base"] + s).choice(len(ytr), size=fn["n_subset"], replace=False)
        f_new = MLP()
        f_new.load_state_dict(f_old.state_dict())
        ln = train(f_new, xtr[sub], ytr[sub], as_batch, epochs=fn["epochs"], lr=fn["lr"], batch_size=fn["batch_size"],
                   order_seed=sd["ft_order_base"] + s, noise_seed=sd["ft_noise_base"] + s)
        save(f_new, f"f_new_B{s}")
        log.append({"seed": s, "loss_old": lo, "loss_new": ln, "subset_sha256": hashlib.sha256(np.sort(sub).tobytes()).hexdigest()})
        print(f"seed {s}: f_old loss {lo[-1]:.4f}, f_new loss {ln[-1]:.4f}", flush=True)
    (out / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "artifacts": manifest}, indent=2))
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    return 0


def cmd_qualify(cfg, out: Path) -> int:
    data = adult_splits(Path(cfg["data_path"]), seed=cfg["split_seed"])
    xv, yv = data.x["val"], data.y["val"]
    g0 = xv[: cfg["n_g0_points"]]
    majority = float(max(yv.mean(), 1 - yv.mean()))
    rows = []
    for s in cfg["seeds_list"]:
        f_old, f_new = load(out, f"f_old_s{s}"), load(out, f"f_new_B{s}")
        lo, ln = logits_np(f_old, xv), logits_np(f_new, xv)
        dd = float(decisive_disagreement(logits_np(f_old, g0), logits_np(f_new, g0), cfg["kappa"]).mean())
        acc_o, acc_n = float((lo.argmax(1) == yv).mean()), float((ln.argmax(1) == yv).mean())
        rows.append({"pair": f"B{s}", "seed": s, "dd": dd, "acc_old": acc_o, "acc_new": acc_n,
                     "disagreement_all_val": float((lo.argmax(1) != ln.argmax(1)).mean()),
                     "dd_ok": dd <= 0.02, "acc_ok": acc_o >= 0.80, "qualified": bool(dd <= 0.02 and acc_o >= 0.80)})
        print(json.dumps(rows[-1]), flush=True)
    res = {"split": "validation", "n_val": len(yv), "n_g0_points": len(g0), "majority_class_rate_val": majority,
           "pairs": rows, "git": git_state()}
    (out / "qualification.json").write_text(json.dumps(res, indent=2))
    print(f"majority-class rate (validation) {majority:.4f}")
    return 0


def cmd_calibrate(cfg, out: Path) -> int:
    path = out / "r_calibration.json"
    if path.exists():
        raise SystemExit("refusing to overwrite r_calibration.json (r is calibrated once)")
    torch.set_num_threads(cfg["threads"])
    data = adult_splits(Path(cfg["data_path"]), seed=cfg["split_seed"])
    xv, yv = data.x["val"], data.y["val"]
    s = cfg["seeds_list"][0]
    f_old, f_new = load(out, f"f_old_s{s}"), load(out, f"f_new_B{s}")
    keep = valid_points(xv, yv, TabMargin(f_old, yv), TabMargin(f_new, yv), cfg["n_points"],
                        cfg["seeds"]["points_base"] + s, cfg["kappa"])
    z0 = torch.as_tensor(xv[keep], dtype=torch.float32)
    start = time.time()
    res = ordinary_ce(z0, TabMargin(f_new, yv[keep]), solver_config(cfg, r=float("inf")), cfg["seeds"]["restart_base"] + s)
    d = res.distance[res.found].double().numpy()
    median = float(np.median(d))
    metrics = {"pair": f"B{s}", "split": "validation", "n_points": len(keep), "ce_found_share": float(res.found.float().mean()),
               "median_ce_distance": median, "r": 2 * median, "wall_seconds": round(time.time() - start, 1),
               "git": git_state()}
    path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("train", "qualify", "calibrate"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    return {"train": cmd_train, "qualify": cmd_qualify, "calibrate": cmd_calibrate}[args.stage](cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
