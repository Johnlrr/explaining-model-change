"""Gate G0 on Tier 0: train f_old per seed, fine-tune f_new per pair, check the phenomenon.

Usage:
    python -m dce.cli.g0_tier0 --config configs/experiments/exp_0001.yaml --out runs/experiments/exp_0001

Smoke and full runs use this same path and differ only in the `scale` block of the config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from dce.data.drift import DriftParams, adaptation_set, clean_splits, drift_label, tier0_pairs
from dce.data.renderer import FACTOR_NAMES, add_train_noise, render
from dce.eval.g0 import flip_rates, g0_criteria
from dce.models.cnn import SmallCNN, predict_logits, train


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", *args], check=False, capture_output=True, text=True
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "src", "configs"))}


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    tier = yaml.safe_load(Path(cfg["tier_config"]).read_text())
    return {**tier, **cfg, "_tier": tier}


def make_batch_fn(noise: bool):
    def fn(phi, gen=None):
        images = render(torch.as_tensor(phi, dtype=torch.float32))
        return add_train_noise(images, gen) if noise else images

    return fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = args.out
    if (out / "metrics.json").exists():
        print(f"refusing to overwrite completed run {out}", file=sys.stderr)
        return 2
    (out / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    torch.set_num_threads(cfg.get("threads", 8))
    scale = cfg["scale"]
    params = replace(DriftParams(), **cfg.get("drift_params", {}))
    tr = cfg["training"]
    start = time.time()
    meta = {
        "experiment": cfg["experiment_id"], "gate": "G0", "tier": 0, "git": git_state(),
        "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "device": "cpu", "threads": torch.get_num_threads(),
        "drift_params": asdict(params), "scale": scale, "status": "RUNNING",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    splits = clean_splits(seed=cfg["data_seed"], n_train=scale["n_train"], n_val=scale["n_val"],
                          n_test=scale["n_test"], label_noise=cfg["label_noise"])
    val = splits["val"]
    g0_phi = val.phi[: scale["n_g0_points"]]
    noisy, clean = make_batch_fn(True), make_batch_fn(False)

    def predictor(model):
        return lambda phi: predict_logits(model, phi, clean).argmax(1).numpy()

    pairs = [p for p in tier0_pairs() if p["pair_id"] in set(cfg["pair_ids"])]
    seeds = sorted({p["seed"] for p in pairs})
    manifest, results, f_old = [], {"f_old": {}, "pairs": []}, {}

    for s in seeds:
        torch.manual_seed(cfg["seeds"]["init_base"] + s)
        model = SmallCNN()
        losses = train(model, splits["train"].phi, splits["train"].y, noisy,
                       epochs=scale["f_old_epochs"], lr=tr["f_old_lr"], batch_size=tr["batch_size"],
                       order_seed=cfg["seeds"]["order_base"] + s, noise_seed=cfg["seeds"]["noise_base"] + s)
        path = out / "artifacts" / "models" / f"f_old_s{s}.pt"
        torch.save(model.state_dict(), path)
        manifest.append({"name": f"f_old_s{s}", "kind": "checkpoint", "path": f"models/{path.name}",
                         "sha256": sha256(path)})
        pred = predictor(model)
        acc = float((pred(val.phi) == val.y_clean).mean())
        flips = flip_rates(pred, g0_phi)
        f_old[s] = (model, pred, flips)
        results["f_old"][s] = {"val_accuracy": acc, "meets_95": acc >= 0.95, "train_loss": losses,
                               "flip": dict(zip(FACTOR_NAMES, flips.tolist()))}
        print(f"f_old s{s}: val acc {acc:.4f}  ({time.time() - start:.0f}s)", flush=True)

    for p in pairs:
        base_model, pred_old, flip_old = f_old[p["seed"]]
        model = SmallCNN()
        model.load_state_dict(base_model.state_dict())
        phi_a, y_a = adaptation_set(p["condition"], seed=cfg["seeds"]["adapt_data_base"] + p["pair_id"],
                                    n=scale["n_adapt"], params=params)
        epochs = cfg.get("finetune_epochs_override", {}).get(p["condition"], scale["finetune_epochs"])
        losses = train(model, phi_a, y_a, noisy, epochs=epochs, lr=tr["finetune_lr"],
                       batch_size=tr["batch_size"], order_seed=cfg["seeds"]["adapt_order_base"] + p["pair_id"],
                       noise_seed=cfg["seeds"]["adapt_noise_base"] + p["pair_id"])
        path = out / "artifacts" / "models" / f"f_new_p{p['pair_id']:02d}_{p['condition']}_s{p['seed']}.pt"
        torch.save(model.state_dict(), path)
        manifest.append({"name": path.stem, "kind": "checkpoint", "path": f"models/{path.name}",
                         "sha256": sha256(path)})
        pred_new = predictor(model)
        flip_new = flip_rates(pred_new, g0_phi)
        y_new = drift_label(p["condition"], g0_phi, params)
        crit = g0_criteria(p["condition"], g0_phi, pred_old(g0_phi), pred_new(g0_phi), y_new,
                           flip_old, flip_new, params)
        row = {**p, "finetune_epochs": epochs, "finetune_loss": losses,
               "f_new_val_accuracy_clean_labels": float((pred_new(val.phi) == val.y_clean).mean()),
               "flip_old": dict(zip(FACTOR_NAMES, flip_old.tolist())),
               "flip_new": dict(zip(FACTOR_NAMES, flip_new.tolist())), **crit}
        results["pairs"].append(row)
        print(f"pair {p['pair_id']:2d} {p['condition']} s{p['seed']}: pass={crit['pass']}  "
              f"({time.time() - start:.0f}s)", flush=True)

    by_cond = {}
    for row in results["pairs"]:
        by_cond.setdefault(row["condition"], []).append(row["pass"])
    results["condition_pass"] = {c: all(v) for c, v in by_cond.items()}
    results["f_old_all_meet_95"] = all(v["meets_95"] for v in results["f_old"].values())
    results["g0_pass"] = results["f_old_all_meet_95"] and all(results["condition_pass"].values())
    results["n_g0_points"] = len(g0_phi)
    results["split"] = "validation"

    (out / "artifacts" / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": 1, "artifacts": manifest}, indent=2))
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    meta.update(status="SUCCESS", seconds=round(time.time() - start, 1))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"g0_pass": results["g0_pass"], "condition_pass": results["condition_pass"],
                      "f_old_all_meet_95": results["f_old_all_meet_95"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
