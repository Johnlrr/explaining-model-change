"""Gate G2 checks on Tier 1, local CPU, validation only (spec §9 G2; amendments A1, v1.3 C3, v1.6 G2; D-029).

Loads the checkpoints trained on Kaggle (hash-verified against the job's manifest), then computes:
oracle Spearman on D(μ(x)), shape-oracle accuracy on real and reconstructed validation images, f_old clean
accuracy, σ_k, G0-Tier 1 for every pair, and max |logit_device − logit_CPU|. The test split is not read.

Usage:
    python -m dce.cli.g2_tier1 --config configs/experiments/exp_0013.yaml --out runs/experiments/exp_0013
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from dce.cli.g0_tier0 import git_state, sha256
from dce.data.tier1 import T1Params, load_split, new_labels, subgroup, t3_threshold, tier1_pairs
from dce.eval.g0_tier1 import flip_rates, g0_criteria
from dce.models.cnn import SmallCNN, predict_logits
from dce.models.oracle import alpha_hat, new_shape_oracle, oracle_factors, shape_prob
from dce.models.vae import VAE

THRESHOLDS = {"color": 0.9, "bg": 0.9, "thick": 0.8}  # Spearman (spec §9 G2)
F_OLD_MIN_ACC = 0.97                                  # EVALUATION §16
SHAPE_REAL_MIN, SHAPE_RECON_MIN = 0.98, 0.95


def load_verified(src: Path, name: str, manifest: dict, cls):
    path = src / f"{name}.pt"
    digest = sha256(path)
    if digest != manifest[name]:
        raise SystemExit(f"hash mismatch for {path}: {digest} != {manifest[name]}")
    model = cls()
    model.load_state_dict(torch.load(path))
    return model.eval(), digest


def spearman(est: np.ndarray, truth: np.ndarray) -> dict:
    ok = np.isfinite(est) & np.isfinite(truth)
    return {"spearman": float(spearmanr(est[ok], truth[ok]).statistic), "n": int(ok.sum()),
            "invalid_share": float(1 - ok.mean())}


@torch.no_grad()
def reconstruct(vae: VAE, x: np.ndarray, batch: int = 500) -> np.ndarray:
    out = [vae.decoder(vae.encoder(torch.as_tensor(x[i : i + batch]))[0]).numpy() for i in range(0, len(x), batch)]
    return np.concatenate(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    tier = yaml.safe_load(Path(cfg["tier_config"]).read_text())
    out = args.out
    if (out / "metrics.json").exists():
        print(f"refusing to overwrite completed run {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.set_num_threads(cfg.get("threads", 8))
    start = time.time()
    src = Path(cfg["source_run"]) / "artifacts"
    manifest = {a["name"]: a["sha256"] for a in json.loads((src / "artifact_manifest.json").read_text())["artifacts"]}
    data = Path(cfg["data"])
    data_sha = sha256(data)
    if data_sha != cfg["data_sha256"]:
        raise SystemExit(f"tier1_data.npz hash mismatch: {data_sha}")
    meta = {"experiment": cfg["experiment_id"], "gate": "G2", "tier": 1, "git": git_state(), "source_run": cfg["source_run"],
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": "cpu", "data_sha256": data_sha, "status": "RUNNING",
            "checkpoints": []}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    limit = cfg.get("limit")  # smoke scale only; must equal the training job's `limit`
    train = load_split(data, "train").head(limit)  # the test split is refused by load_split
    val = load_split(data, "val").head(limit)
    x_val = val.images()
    vae, h = load_verified(src, "vae", manifest, VAE)
    meta["checkpoints"].append({"vae": h})
    shape, h = load_verified(src, "shape_oracle", manifest, new_shape_oracle)
    meta["checkpoints"].append({"shape_oracle": h})
    to_t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    results = {"split": "validation", "n_val": val.n}

    # oracle on D(μ(x)) vs the true factors (G2)
    recon = reconstruct(vae, x_val)
    est = oracle_factors(recon)
    truth = {"color": val.phi_c, "bg": val.phi_b, "thick": val.thick}
    results["oracle_on_decoded"] = {k: {**spearman(est[k], truth[k]), "threshold": THRESHOLDS[k]} for k in THRESHOLDS}
    for v in results["oracle_on_decoded"].values():
        v["pass"] = v["spearman"] >= v["threshold"]
    # the same oracle on the real images (identity check; reported)
    est_real = oracle_factors(x_val)
    results["oracle_on_real"] = {k: spearman(est_real[k], truth[k]) for k in THRESHOLDS}
    results["recon_mse"] = float(((recon - x_val) ** 2).mean())

    # shape oracle
    acc_real = float(((shape_prob(shape, alpha_hat(x_val)) > 0.5) == val.y).mean())
    acc_recon = float(((shape_prob(shape, est["alpha"]) > 0.5) == val.y).mean())
    results["shape_oracle"] = {"acc_real": acc_real, "acc_recon": acc_recon,
                               "pass": acc_real >= SHAPE_REAL_MIN and acc_recon >= SHAPE_RECON_MIN}

    # σ_k on clean validation D(μ(x)) (A1, D-011)
    sig = {k: float(np.nanstd(est[k])) for k in THRESHOLDS}
    sig["shape"] = float(np.std(shape_prob(shape, est["alpha"])))
    results["sigma_k"] = sig

    # classifiers: accuracy, G0-Tier 1, device gap
    dev = {k: np.asarray(v) for k, v in json.loads((src / "result.json").read_text())["val_logits"].items()}
    drift = T1Params(**tier["drift_params"])
    pairs = tier1_pairs()
    models, logits, gaps = {}, {}, {}

    def get(name):
        if name not in models:
            m, h = load_verified(src, name, manifest, SmallCNN)
            meta["checkpoints"].append({name: h})
            models[name] = m
            logits[name] = predict_logits(m, x_val, to_t).numpy()
            gaps[name] = float(np.abs(logits[name] - dev[name]).max())
        return models[name]

    predictor = lambda m: (lambda imgs: predict_logits(m, imgs, to_t).argmax(1).numpy())
    f_old = {}
    for s in sorted({p["seed"] for p in pairs}):
        m = get(f"f_old_s{s}")
        acc = float((logits[f"f_old_s{s}"].argmax(1) == val.y).mean())
        f_old[s] = {"val_accuracy": acc, "meets_97": acc >= F_OLD_MIN_ACC, "flip": flip_rates(predictor(m), val)}
    results["f_old"] = f_old
    thr = t3_threshold(train, drift)
    rows = []
    for p in pairs:
        name = f"f_new_p{p['pair_id']:02d}_{p['condition']}_s{p['seed']}"
        m = get(name)
        flip_new = flip_rates(predictor(m), val)
        y_new = new_labels(p["condition"], val, drift, thr)
        sub = subgroup(val, drift, thr) if p["condition"] == "T3" else None
        crit = g0_criteria(p["condition"], logits[f"f_old_s{p['seed']}"], logits[name], y_new, sub,
                           f_old[p["seed"]]["flip"], flip_new, tier["kappa"], k_star=drift.t3_factor)
        rows.append({**p, "name": name, "flip_new": flip_new,
                     "f_new_val_accuracy_clean_labels": float((logits[name].argmax(1) == val.y).mean()), **crit})
        print(f"pair {p['pair_id']:2d} {p['condition']} s{p['seed']}: pass={crit['pass']}", flush=True)
    results["pairs"] = rows
    results["condition_pass"] = {c: all(r["pass"] for r in rows if r["condition"] == c) for c in ("T1", "T3", "N0")}
    results["t3_threshold"] = thr
    results["device_gap_max_abs_logit"] = max(gaps.values())
    results["device_gap_by_model"] = gaps
    results["g2"] = {"oracle": all(v["pass"] for v in results["oracle_on_decoded"].values()),
                     "shape_oracle": results["shape_oracle"]["pass"],
                     "f_old": all(v["meets_97"] for v in f_old.values()),
                     "g0_tier1": all(results["condition_pass"].values())}
    results["g2"]["pass"] = all(results["g2"].values())
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    meta.update(status="SUCCESS", seconds=round(time.time() - start, 1))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"g2": results["g2"], "condition_pass": results["condition_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
