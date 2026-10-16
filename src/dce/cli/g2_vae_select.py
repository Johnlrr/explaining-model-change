"""Branch (b) of amendment v1.7 H3: VAE β ∈ {0.5, 0.1} on CPU (β = 1 is the exp_0012 VAE, already measured).

For each β: train the VAE (30 epochs, latent 16, seed of tier1.yaml), then on validation D(μ(x)): accuracy of the
diagnostic f_old (exp_0014, seed 0, 60 epochs), the three oracle Spearman criteria, the probe accuracy (descriptive).
Selection rule (declared before running): the LARGEST β with f_old accuracy on D(μ(x)) ≥ 0.95 and all three
Spearman criteria met; none → back to the author. Validation/train only.

Usage: python -m dce.cli.g2_vae_select --config configs/experiments/exp_0015.yaml --out runs/experiments/exp_0015
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from dce.cli.g0_tier0 import git_state, sha256
from dce.cli.g2_diagnose import acc_f, probe_accuracy
from dce.cli.g2_tier1 import THRESHOLDS, reconstruct, spearman
from dce.data.tier1 import load_split
from dce.models.cnn import SmallCNN
from dce.models.oracle import oracle_factors
from dce.models.vae import VAE, train_vae

F_OLD_DECODED_MIN = 0.95


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
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.set_num_threads(cfg.get("threads", 8))
    data = Path(cfg["data"])
    if sha256(data) != cfg["data_sha256"]:
        raise SystemExit("tier1_data.npz hash mismatch")
    f_path = Path(cfg["f_old_checkpoint"])
    if sha256(f_path) != cfg["f_old_sha256"]:
        raise SystemExit("diagnostic f_old hash mismatch")
    f_old = SmallCNN()
    f_old.load_state_dict(torch.load(f_path))
    f_old.eval()
    train_s, val = load_split(data, "train"), load_split(data, "val")
    x_tr, x_va = train_s.images(), val.images()
    truth = {"color": val.phi_c, "bg": val.phi_b, "thick": val.thick}
    rows, start = [], time.time()
    for beta in cfg["betas"]:
        t0 = time.time()
        torch.manual_seed(tier["seeds"]["vae"])
        vae = VAE()
        losses = train_vae(vae, torch.as_tensor(x_tr), epochs=tier["scale"]["vae_epochs"], lr=tier["training"]["vae_lr"],
                           batch_size=tier["training"]["batch_size"], seed=tier["seeds"]["vae"], beta=beta)
        path = out / "artifacts" / f"vae_beta{beta}.pt"
        torch.save({k: v.cpu() for k, v in vae.state_dict().items()}, path)
        recon = reconstruct(vae, x_va)
        est = oracle_factors(recon)
        sp = {k: spearman(est[k], truth[k]) for k in THRESHOLDS}
        with torch.no_grad():
            mu_tr = torch.cat([vae.encoder(torch.as_tensor(x_tr[i : i + 500]))[0] for i in range(0, len(x_tr), 500)]).numpy()
            mu_va = torch.cat([vae.encoder(torch.as_tensor(x_va[i : i + 500]))[0] for i in range(0, len(x_va), 500)]).numpy()
        row = {"beta": beta, "train_seconds": round(time.time() - t0, 1), "final_loss": losses[-1],
               "recon_mse": float(((recon - x_va) ** 2).mean()), "f_old_on_decoded": acc_f(f_old, recon, val.y),
               "spearman": {k: {**v, "threshold": THRESHOLDS[k], "pass": v["spearman"] >= THRESHOLDS[k]} for k, v in sp.items()},
               "probe_val_acc": probe_accuracy(mu_tr, train_s.y, mu_va, val.y)["val_acc"], "checkpoint_sha256": sha256(path)}
        row["meets_rule"] = row["f_old_on_decoded"] >= F_OLD_DECODED_MIN and all(v["pass"] for v in row["spearman"].values())
        rows.append(row)
        print(f"beta={beta}: f_old_on_decoded={row['f_old_on_decoded']:.4f} spearman="
              f"{ {k: round(v['spearman'], 3) for k, v in row['spearman'].items()} } meets={row['meets_rule']} ({row['train_seconds']}s)", flush=True)
    ok = [r["beta"] for r in rows if r["meets_rule"]]
    res = {"split": "validation", "rows": rows, "beta_beta1_note": "β = 1: exp_0012 VAE, f_old on decoded 0.649 (exp_0014)",
           "selected_beta": max(ok) if ok else None, "none_meets_rule": not ok}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "metadata.json").write_text(json.dumps({"experiment": cfg["experiment_id"], "git": git_state(), "torch": torch.__version__,
                                                    "seconds": round(time.time() - start, 1)}, indent=2))
    print(json.dumps({"selected_beta": res["selected_beta"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
