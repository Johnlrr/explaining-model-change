"""G2 repair diagnostics D1–D3 (amendment v1.7 H2–H3; D-030). CPU, validation + train only, no test, no Recovery.

D1: f_old (seed 0, 60 epochs) accuracy on real and on D(μ(x)) validation images.
D2: logistic-regression probe on μ(x) (train → validation) and a 32-image grid (originals above reconstructions).
D3: accuracy of f_old and of the exp_0012 shape oracle by ladder level and by thickness bin.
Applies the pre-declared branch rule (a)/(b) and writes it into metrics.json.

Usage: python -m dce.cli.g2_diagnose --config configs/experiments/exp_0014.yaml --out runs/experiments/exp_0014
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from dce.cli.g0_tier0 import git_state, sha256
from dce.cli.g2_tier1 import load_verified, reconstruct
from dce.data.tier1 import load_split, noisy_labels
from dce.models.cnn import SmallCNN, predict_logits, train
from dce.models.oracle import alpha_hat, new_shape_oracle, shape_prob
from dce.models.vae import VAE
from dce.remote.tier1_prep import colorize

BINS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
PROBE_MIN, F_OLD_RECON_MIN = 0.95, 0.90   # branch (a) rule, amendment v1.7 H3


def acc_f(model, images, y):
    lg = predict_logits(model, images, lambda a: torch.as_tensor(a, dtype=torch.float32))
    return float((lg.argmax(1).numpy() == y).mean())


def probe_accuracy(mu_tr, y_tr, mu_va, y_va, l2=1e-3) -> dict:
    mean, std = mu_tr.mean(0), mu_tr.std(0) + 1e-8
    xt, xv = torch.as_tensor((mu_tr - mean) / std, dtype=torch.float32), torch.as_tensor((mu_va - mean) / std, dtype=torch.float32)
    w = torch.zeros(xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    yt = torch.as_tensor(y_tr, dtype=torch.float32)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return {"train_acc": float(((xt @ w + b > 0).numpy() == y_tr).mean()),
                "val_acc": float(((xv @ w + b > 0).numpy() == y_va).mean())}


def grid(orig: np.ndarray, recon: np.ndarray, path: Path, per_row: int = 16) -> None:
    def tile(batch):
        rows = [np.concatenate(list(batch[i : i + per_row].transpose(0, 2, 3, 1)), axis=1)
                for i in range(0, len(batch), per_row)]
        return np.concatenate(rows, axis=0)
    img = np.concatenate([tile(orig), tile(recon)], axis=0)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((img.shape[1] * 3, img.shape[0] * 3),
                                                                        Image.NEAREST).save(path)


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
    start = time.time()
    data = Path(cfg["data"])
    if sha256(data) != cfg["data_sha256"]:
        raise SystemExit("tier1_data.npz hash mismatch")
    src = Path(cfg["source_run"]) / "artifacts"
    manifest = {a["name"]: a["sha256"] for a in json.loads((src / "artifact_manifest.json").read_text())["artifacts"]}
    train_s, val = load_split(data, "train"), load_split(data, "val")   # the test split is refused
    vae, _ = load_verified(src, "vae", manifest, VAE)
    shape, _ = load_verified(src, "shape_oracle", manifest, new_shape_oracle)
    x_tr, x_va = train_s.images(), val.images()
    tt = lambda a: torch.as_tensor(a, dtype=torch.float32)

    # f_old, seed 0, 60 epochs, declared recipe (v1.7 H1)
    y_tr = noisy_labels(train_s.y, tier["label_noise"], np.random.default_rng(tier["seeds"]["label_noise"]))
    torch.manual_seed(tier["seeds"]["init_base"])
    f_old = SmallCNN()
    losses = train(f_old, x_tr, y_tr, lambda a, g=None: tt(a), epochs=cfg["f_old_epochs"], lr=tier["training"]["f_old_lr"],
                   batch_size=tier["training"]["batch_size"], order_seed=tier["seeds"]["order_base"], noise_seed=0,
                   cosine=tier["training"]["f_old_cosine"])
    ckpt = out / "artifacts" / "f_old_diag_s0.pt"
    torch.save(f_old.state_dict(), ckpt)
    print(f"f_old trained ({time.time() - start:.0f}s)", flush=True)

    res = {"split": "validation", "f_old_epochs": cfg["f_old_epochs"], "f_old_final_loss": losses[-1]}
    mu_va = np.concatenate([vae.encoder(tt(x_va[i : i + 500]))[0].detach().numpy() for i in range(0, len(x_va), 500)])
    mu_tr = np.concatenate([vae.encoder(tt(x_tr[i : i + 500]))[0].detach().numpy() for i in range(0, len(x_tr), 500)])
    recon_va = reconstruct(vae, x_va)
    res["D1"] = {"f_old_real": acc_f(f_old, x_va, val.y), "f_old_on_decoded": acc_f(f_old, recon_va, val.y)}
    res["D2"] = probe_accuracy(mu_tr, train_s.y, mu_va, val.y)
    grid(x_va[:32], recon_va[:32], out / "grid_orig_over_recon.png")

    # D3: by ladder level (colours fixed) and by thickness bin of the chosen level
    lev = {}
    for li, level in enumerate(val.ladder.shape[1] * [None]):
        ok = np.isfinite(val.thick_all[:, li])
        if ok.sum() < 20:
            continue
        a = val.ladder[ok, li].astype(np.float32) / 255.0
        img = colorize(a, val.phi_c[ok], val.phi_b[ok])
        lev[int(li)] = {"n": int(ok.sum()), "median_thick": float(np.median(val.thick_all[ok, li])),
                        "f_old_acc": acc_f(f_old, img, val.y[ok]),
                        "shape_oracle_acc": float(((shape_prob(shape, alpha_hat(img)) > 0.5) == val.y[ok]).mean())}
    res["D3_by_level"] = lev
    th = val.thick
    bins = {}
    p_shape = shape_prob(shape, alpha_hat(x_va)) > 0.5
    p_f = predict_logits(f_old, x_va, tt).argmax(1).numpy()
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (th >= lo) & (th < hi)
        if m.sum() >= 5:
            bins[f"{lo}-{hi}"] = {"n": int(m.sum()), "f_old_acc": float((p_f[m] == val.y[m]).mean()),
                                  "shape_oracle_acc": float((p_shape[m] == val.y[m]).mean())}
    res["D3_by_thickness_bin"] = bins

    d1, d2 = res["D1"]["f_old_on_decoded"], res["D2"]["val_acc"]
    res["branch_rule"] = {"probe_min": PROBE_MIN, "f_old_decoded_min": F_OLD_RECON_MIN, "probe": d2,
                          "f_old_on_decoded": d1, "branch": "a" if (d2 >= PROBE_MIN and d1 >= F_OLD_RECON_MIN) else "b"}
    meta = {"experiment": cfg["experiment_id"], "gate": "G2 diagnostics", "git": git_state(), "torch": torch.__version__,
            "checkpoint_sha256": {"f_old_diag_s0": sha256(ckpt)}, "seconds": round(time.time() - start, 1)}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"D1": res["D1"], "D2": res["D2"], "branch": res["branch_rule"]["branch"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
