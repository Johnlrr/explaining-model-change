"""Re-evaluate G0 on saved Tier 0 checkpoints, without training (amendment v1.3).

Loads f_old and f_new checkpoints from earlier runs, verifies every SHA256 against the source
run's manifest (any mismatch is fatal), recomputes the G0 criteria on the same 2 000 validation
points, and writes a new run record. Source runs are never modified.

Usage:
    python -m dce.cli.g0_reeval --config configs/experiments/exp_0004.yaml --out runs/experiments/exp_0004
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from dce.cli.g0_tier0 import git_state, load_config, make_batch_fn, sha256
from dce.data.drift import DriftParams, clean_splits, drift_label, tier0_pairs
from dce.data.renderer import FACTOR_NAMES
from dce.eval.g0 import flip_rates, g0_criteria
from dce.models.cnn import SmallCNN, predict_logits


def load_verified(run: Path, name: str, cache: dict) -> tuple[SmallCNN, dict]:
    if run not in cache:
        manifest = json.loads((run / "artifacts" / "artifact_manifest.json").read_text())
        cache[run] = {a["name"]: a for a in manifest["artifacts"]}
    entry = cache[run][name]
    path = Path(entry["path"]) if entry["kind"] == "checkpoint_reference" else run / "artifacts" / entry["path"]
    digest = sha256(path)
    if digest != entry["sha256"]:
        raise SystemExit(f"hash mismatch for {path}: {digest} != {entry['sha256']}")
    model = SmallCNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, {"name": name, "kind": "checkpoint_reference", "path": str(path), "sha256": digest}


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
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    torch.set_num_threads(cfg.get("threads", 8))
    scale = cfg["scale"]
    params = replace(DriftParams(), **cfg.get("drift_params", {}))
    start = time.time()
    meta = {
        "experiment": cfg["experiment_id"], "gate": "G0", "tier": 0, "kind": "re-evaluation (no training)",
        "git": git_state(), "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__, "platform": platform.platform(), "device": "cpu",
        "threads": torch.get_num_threads(), "drift_params": asdict(params), "scale": scale,
        "t3_criteria": cfg["t3_criteria"], "sources": cfg["sources"], "status": "RUNNING",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    splits = clean_splits(seed=cfg["data_seed"], n_train=scale["n_train"], n_val=scale["n_val"],
                          n_test=scale["n_test"], label_noise=cfg["label_noise"])
    val = splits["val"]
    g0_phi = val.phi[: scale["n_g0_points"]]
    clean = make_batch_fn(False)

    def logits(model, phi):
        return predict_logits(model, phi, clean).numpy()

    def predictor(model):
        return lambda phi: predict_logits(model, phi, clean).argmax(1).numpy()

    sources = {k: Path(v) for k, v in cfg["sources"].items()}
    cache, manifest, results, f_old = {}, [], {"f_old": {}, "pairs": []}, {}
    pairs = [p for p in tier0_pairs() if p["pair_id"] in set(cfg["pair_ids"])]

    for s in sorted({p["seed"] for p in pairs}):
        model, entry = load_verified(sources["f_old"], f"f_old_s{s}", cache)
        manifest.append(entry)
        pred = predictor(model)
        acc = float((pred(val.phi) == val.y_clean).mean())
        flips = flip_rates(pred, g0_phi)
        f_old[s] = (model, flips)
        results["f_old"][s] = {"val_accuracy": acc, "meets_95": acc >= 0.95,
                               "flip": dict(zip(FACTOR_NAMES, flips.tolist())), "source": str(sources["f_old"])}

    for p in pairs:
        old, flip_old = f_old[p["seed"]]
        run = sources[p["condition"]]
        model, entry = load_verified(run, f"f_new_p{p['pair_id']:02d}_{p['condition']}_s{p['seed']}", cache)
        manifest.append(entry)
        pred_new = predictor(model)
        flip_new = flip_rates(pred_new, g0_phi)
        y_new = drift_label(p["condition"], g0_phi, params)
        crit = g0_criteria(p["condition"], g0_phi, logits(old, g0_phi), logits(model, g0_phi), y_new,
                           flip_old, flip_new, params, kappa=cfg["kappa"], t3_criteria=cfg["t3_criteria"])
        results["pairs"].append({**p, "source": str(run),
                                 "f_new_val_accuracy_clean_labels": float((pred_new(val.phi) == val.y_clean).mean()),
                                 "flip_old": dict(zip(FACTOR_NAMES, flip_old.tolist())),
                                 "flip_new": dict(zip(FACTOR_NAMES, flip_new.tolist())), **crit})
        print(f"pair {p['pair_id']:2d} {p['condition']} s{p['seed']}: pass={crit['pass']}", flush=True)

    by_cond = {}
    for row in results["pairs"]:
        by_cond.setdefault(row["condition"], []).append(row["pass"])
    results["condition_pass"] = {c: all(v) for c, v in by_cond.items()}
    results["f_old_all_meet_95"] = all(v["meets_95"] for v in results["f_old"].values())
    results["g0_pass"] = results["f_old_all_meet_95"] and all(results["condition_pass"].values())
    results["n_g0_points"] = len(g0_phi)
    results["split"] = "validation"
    results["t3_criteria"] = cfg["t3_criteria"]

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
