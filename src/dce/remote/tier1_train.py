"""Tier 1 training job (G2; spec §5, v1.6 G2; D-029). Kaggle T4, offline, `tier1_data.npz` via kernel_sources.

Trains, on the TRAIN split only (the test arrays are never read): the VAE (clean images), the shape oracle
(α of every valid ladder level), f_old for seeds 0–4 (5% symmetric label noise on the train labels, B2 reading),
and f_new for the 11 pairs (fine-tune of f_old on an adaptation set with 5% noise). Writes flat checkpoints,
a SHA256 manifest, and result.json (which carries the logits of every classifier on validation for the
device-gap check; Kaggle's output listing returns at most 20 files per call, so the job writes exactly 20).

`run(params, data_path, out_dir, device)` is the only code path: the local smoke and the full job differ in
`params` only (scale, epochs).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dce.data.tier1 import T1Params, adaptation_set, load_split, noisy_labels, tier1_pairs
from dce.models.cnn import SmallCNN, train
from dce.models.oracle import new_shape_oracle
from dce.models.vae import VAE, train_vae


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_data() -> Path:
    for root in (Path("/kaggle/input"), Path("in")):
        if root.exists():
            hits = sorted(root.rglob("tier1_data.npz"))
            if hits:
                return hits[0]
    raise FileNotFoundError("tier1_data.npz not found under /kaggle/input")


def verify_data(path: Path) -> str:
    """The attached data must match the manifest written by the immutable prep kernel."""
    manifest = path.parent / "artifact_manifest.json"
    digest = sha256(path)
    if manifest.exists():
        expected = {a["path"]: a["sha256"] for a in json.loads(manifest.read_text())["artifacts"]}[path.name]
        if digest != expected:
            raise RuntimeError(f"tier1_data.npz hash mismatch: {digest} != {expected}")
    return digest


def run(params: dict, data_path: Path, out: Path, device: torch.device, data_sha: str | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    sc, tr, seeds = params["scale"], params["training"], params["seeds"]
    train_split = load_split(data_path, "train").head(sc.get("limit"))
    val = load_split(data_path, "val").head(sc.get("limit"))
    x_train, y_clean = train_split.images(), train_split.y
    y_train = noisy_labels(y_clean, params["label_noise"], np.random.default_rng(seeds["label_noise"]))
    x_val = val.images()
    manifest, report = [], {"models": {}}

    def save(model, name):
        path = out / f"{name}.pt"
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
        manifest.append({"name": name, "kind": "checkpoint", "path": path.name, "sha256": sha256(path),
                         "bytes": path.stat().st_size})

    identity = lambda x, gen=None: torch.as_tensor(x, dtype=torch.float32)

    def val_logits(model):
        model.to(device)
        with torch.no_grad():
            lg = torch.cat([model(torch.as_tensor(x_val[i : i + 500]).to(device)).cpu()
                            for i in range(0, len(x_val), 500)])
        model.cpu()
        return lg.numpy()

    # VAE on clean train images (spec §5.2)
    torch.manual_seed(seeds["vae"])
    vae = VAE()
    losses = train_vae(vae, torch.as_tensor(x_train), epochs=sc["vae_epochs"], lr=tr["vae_lr"],
                       batch_size=tr["batch_size"], seed=seeds["vae"], device=device)
    save(vae, "vae")
    report["models"]["vae"] = {"loss": losses}

    # shape oracle: one-channel classifier on the α of every valid ladder level of the train images
    lad = train_split.ladder.astype(np.float32) / 255.0                     # (N, L, 28, 28)
    valid = np.isfinite(train_split.thick_all)
    oa = lad[valid][:, None]                                                 # (M, 1, 28, 28)
    oy = np.repeat(y_clean[:, None], lad.shape[1], 1)[valid]
    torch.manual_seed(seeds["oracle"])
    oracle = new_shape_oracle()
    losses = train(oracle, oa, oy, identity, epochs=sc["oracle_epochs"], lr=tr["oracle_lr"],
                   batch_size=tr["batch_size"], order_seed=seeds["oracle_order"], noise_seed=0, device=device)
    oracle.cpu()
    save(oracle, "shape_oracle")
    report["models"]["shape_oracle"] = {"loss": losses, "n_train": len(oy)}

    # f_old, one per seed
    logits_dump, f_old = {}, {}
    for s in sc["seeds"]:
        torch.manual_seed(seeds["init_base"] + s)
        m = SmallCNN()
        losses = train(m, x_train, y_train, identity, epochs=sc["f_old_epochs"], lr=tr["f_old_lr"],
                       batch_size=tr["batch_size"], order_seed=seeds["order_base"] + s, noise_seed=0,
                       cosine=tr["f_old_cosine"], device=device)
        m.cpu()
        f_old[s] = {k: v.clone() for k, v in m.state_dict().items()}
        lg = val_logits(m)
        logits_dump[f"f_old_s{s}"] = lg
        save(m, f"f_old_s{s}")
        report["models"][f"f_old_s{s}"] = {"loss": losses, "val_accuracy_clean_labels":
                                           float((lg.argmax(1) == val.y).mean())}

    # f_new, one per pair
    drift = T1Params(**params.get("drift_params", {}))
    for p in tier1_pairs():
        if p["pair_id"] not in sc["pair_ids"]:
            continue
        recipe = params["finetune"][p["condition"]]
        ad = adaptation_set(p["condition"], train_split, seeds["adapt_data_base"] + p["pair_id"], drift,
                            n=sc["n_adapt"], label_noise=params["label_noise"])
        m = SmallCNN()
        m.load_state_dict(f_old[p["seed"]])
        losses = train(m, ad["images"], ad["y"], identity, epochs=recipe["epochs"], lr=recipe["lr"],
                       batch_size=tr["batch_size"], order_seed=seeds["adapt_order_base"] + p["pair_id"],
                       noise_seed=0, device=device)
        m.cpu()
        name = f"f_new_p{p['pair_id']:02d}_{p['condition']}_s{p['seed']}"
        lg = val_logits(m)
        logits_dump[name] = lg
        save(m, name)
        report["models"][name] = {**p, "epochs": recipe["epochs"], "lr": recipe["lr"], "loss": losses,
                                  "val_accuracy_clean_labels": float((lg.argmax(1) == val.y).mean())}
        print(f"{name} done ({time.time() - start:.0f}s)", flush=True)

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    result = {"schema_version": 1, "kind": "tier1_train", "python": sys.version.split()[0],
              "platform": platform.platform(), "torch": torch.__version__, "device": str(device),
              "observed_hardware": gpus or ["cpu"], "bundle_tree_hash": os.environ.get("DCE_BUNDLE_TREE_HASH"),
              "data_sha256": data_sha, "n_train": train_split.n, "n_val": val.n, "params": params,
              "report": report, "val_logits": {k: v.tolist() for k, v in logits_dump.items()},
              "seconds": round(time.time() - start, 1),
              "work": {"processed": len(report["models"])}}
    (out / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "artifacts": manifest}, indent=2))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    params = json.loads(os.environ.get("RESEARCH_HARNESS_PARAMETERS", "{}"))
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("out")
    data = find_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = run(params, data, out, device, data_sha=verify_data(data))
    print(json.dumps({k: res[k] for k in ("observed_hardware", "seconds", "work")}))


if __name__ == "__main__":
    main()
