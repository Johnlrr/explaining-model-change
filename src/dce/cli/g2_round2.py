"""G2 round two, final (amendment v1.8 I1–I7; D-031). Local CPU, train + validation only; no test, no Recovery.

Fixed order: data v2 (derived, hash-checked) → f_old × 5 → every f_new variant → G0-Tier 1 on real validation
images → T1 / T3 variant choice (first variant passing at every seed) → for each β: VAE, shape oracle, criteria
(i) spec G2, (ii) valid-point yield, (iii) f_old on D(μ(x)) ≥ 0.90 → largest β meeting all → two-tier rule.

Usage:
    python -m dce.cli.g2_round2 --config configs/experiments/exp_0016.yaml --out runs/experiments/exp_0016
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from dce.cli.g0_tier0 import git_state, sha256
from dce.cli.g2_tier1 import SHAPE_REAL_MIN, SHAPE_RECON_MIN, THRESHOLDS, reconstruct, spearman
from dce.data.tier1 import (
    T1Params,
    adaptation_set,
    derive_v2,
    load_split,
    new_labels,
    noisy_labels,
    subgroup,
    t3_threshold,
    tier1_pairs,
)
from dce.eval.g0_tier1 import flip_rates, g0_criteria
from dce.models.cnn import SmallCNN, margin, predict_logits, train
from dce.models.oracle import alpha_hat, new_shape_oracle, oracle_factors, shape_prob
from dce.models.vae import VAE, train_vae

F_OLD_MIN = 0.95            # v1.7 H1 / v1.8 I3
F_OLD_DECODED_MIN = 0.90    # v1.8 I2 (iii): self-imposed, not a spec criterion
N_TEST_IMAGES = 1991        # count of Tier 1 test images (the only thing used from the test split)
MIN_TEST_VALID = 300        # v1.8 I2 (ii)

DEFAULT = {"epochs": 3, "lr": 3.0e-4}
VARIANTS = {  # v1.8 I4: evaluated in this order; the first one passing at every seed is used
    "T1": [("default", {}, DEFAULT), ("knob1", {"t1_correlation": 0.99}, DEFAULT),
           ("knob2", {"t1_correlation": 0.99}, {"epochs": 5, "lr": 3.0e-4})],
    "T3": [("default", {}, DEFAULT), ("knob1", {}, {"epochs": 5, "lr": 1.0e-3}),
           ("knob2", {}, {"epochs": 10, "lr": 1.0e-3}), ("bg", {"t3_factor": "bg"}, DEFAULT)],
    "N0": [("default", {}, DEFAULT)],
}

tt = lambda a: torch.as_tensor(a, dtype=torch.float32)


def logits_of(model, images) -> np.ndarray:
    return predict_logits(model, images, tt).numpy()


def predictor(model):
    return lambda imgs: logits_of(model, imgs).argmax(1)


def encode(vae: VAE, x: np.ndarray, batch: int = 500) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([vae.encoder(tt(x[i : i + batch]))[0].numpy() for i in range(0, len(x), batch)])


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
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.set_num_threads(cfg.get("threads", 8))
    start = time.time()
    sc, tr, seeds, kappa = cfg["scale"], tier["training"], tier["seeds"], tier["kappa"]
    log = lambda msg: print(f"[{time.time() - start:6.0f}s] {msg}", flush=True)
    manifest = []

    def save(model, name):
        path = art / f"{name}.pt"
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
        manifest.append({"name": name, "path": path.name, "sha256": sha256(path)})

    # data v2 (I1)
    src = Path(cfg["data_v1"])
    if sha256(src) != cfg["data_v1_sha256"]:
        raise SystemExit("tier1_data.npz (v1) hash mismatch")
    data = Path(cfg["data_v2"])
    derive_v2(src, data)
    data_sha = sha256(data)
    meta = {"experiment": cfg["experiment_id"], "gate": "G2 round two (final)", "tier": 1, "git": git_state(),
            "python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
            "device": "cpu", "data_v1_sha256": cfg["data_v1_sha256"], "data_v2_sha256": data_sha, "status": "RUNNING"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    train_s = load_split(data, "train").head(sc.get("limit"))
    val = load_split(data, "val").head(sc.get("limit"))     # the test split is refused by load_split
    x_tr, x_va = train_s.images(), val.images()
    res = {"split": "validation", "data_v2_sha256": data_sha}
    q = lambda v: np.quantile(v, [0, .05, .15, .5, .85, .95, 1]).round(3).tolist()
    lv = {s.name: np.bincount(s.level, minlength=s.ladder.shape[1]).tolist() for s in (train_s, val)}
    res["thickness_v2"] = {"train_quantiles_0_5_15_50_85_95_100": q(train_s.thick), "val_quantiles": q(val.thick),
                           "chosen_level_counts_by_index(-2..+3)": lv,
                           "t3_q85_train": t3_threshold(train_s, T1Params()), "bg_q85_train": t3_threshold(train_s, T1Params(t3_factor="bg"))}
    log(f"data v2 {data_sha[:12]}; thickness train quantiles {res['thickness_v2']['train_quantiles_0_5_15_50_85_95_100']}")

    # f_old × 5 (I3)
    y_tr = noisy_labels(train_s.y, tier["label_noise"], np.random.default_rng(seeds["label_noise"]))
    f_old, res["f_old"] = {}, {}
    for s in sc["seeds"]:
        torch.manual_seed(seeds["init_base"] + s)
        m = SmallCNN()
        train(m, x_tr, y_tr, lambda a, g=None: tt(a), epochs=sc["f_old_epochs"], lr=tr["f_old_lr"],
              batch_size=tr["batch_size"], order_seed=seeds["order_base"] + s, noise_seed=0, cosine=True)
        save(m, f"f_old_s{s}")
        acc = float((predictor(m)(x_va) == val.y).mean())
        f_old[s] = m
        res["f_old"][s] = {"val_accuracy": acc, "meets_095": acc >= F_OLD_MIN, "flip": flip_rates(predictor(m), val)}
        log(f"f_old s{s}: {acc:.4f}")

    # every f_new variant + G0 on real validation images
    f_new, g0 = {}, {}
    for p in tier1_pairs():
        if p["pair_id"] not in sc["pair_ids"]:
            continue
        for vname, dparams, recipe in VARIANTS[p["condition"]]:
            params = replace(T1Params(), **dparams)
            ad = adaptation_set(p["condition"], train_s, seeds["adapt_data_base"] + p["pair_id"], params,
                                n=sc["n_adapt"], label_noise=tier["label_noise"])
            m = SmallCNN()
            m.load_state_dict(f_old[p["seed"]].state_dict())
            train(m, ad["images"], ad["y"], lambda a, g=None: tt(a), epochs=sc.get("finetune_epochs", recipe["epochs"]),
                  lr=recipe["lr"], batch_size=tr["batch_size"], order_seed=seeds["adapt_order_base"] + p["pair_id"],
                  noise_seed=0)
            name = f"f_new_p{p['pair_id']:02d}_{p['condition']}_s{p['seed']}_{vname}"
            save(m, name)
            thr = t3_threshold(train_s, params)
            y_new = new_labels(p["condition"], val, params, thr)
            sub = subgroup(val, params, thr) if p["condition"] == "T3" else None
            crit = g0_criteria(p["condition"], logits_of(f_old[p["seed"]], x_va), logits_of(m, x_va), y_new, sub,
                               res["f_old"][p["seed"]]["flip"], flip_rates(predictor(m), val), kappa,
                               k_star=params.t3_factor)
            f_new[(p["pair_id"], vname)] = m
            g0[name] = {**p, "variant": vname, "recipe": recipe, "drift_params": dparams, **crit}
            log(f"{name}: G0 pass={crit['pass']}")
    res["g0"] = g0

    def chosen(cond):
        pids = [p["pair_id"] for p in tier1_pairs() if p["condition"] == cond and p["pair_id"] in sc["pair_ids"]]
        for vname, _, _ in VARIANTS[cond]:
            if all(g0[n]["pass"] for n in g0 if g0[n]["pair_id"] in pids and g0[n]["variant"] == vname):
                return vname, pids
        return None, pids
    choice = {c: chosen(c) for c in ("T1", "T3", "N0")}
    res["variant_choice"] = {c: v for c, (v, _) in choice.items()}
    log(f"variant choice {res['variant_choice']}")
    eval_pairs = [(pid, v) for c, (v, pids) in choice.items() if v is not None for pid in pids]

    # β loop (I2)
    rows = []
    lad = train_s.ladder.astype(np.float32) / 255.0
    valid_lv = np.isfinite(train_s.thick_all)
    real_alpha = lad[valid_lv]
    real_y = np.repeat(train_s.y[:, None], lad.shape[1], 1)[valid_lv]
    seed_of = {p["pair_id"]: p["seed"] for p in tier1_pairs()}
    for beta in cfg["betas"]:
        torch.manual_seed(seeds["vae"])
        vae = VAE()
        train_vae(vae, tt(x_tr), epochs=sc["vae_epochs"], lr=tr["vae_lr"], batch_size=tr["batch_size"],
                  seed=seeds["vae"], beta=beta)
        save(vae, f"vae_beta{beta}")
        rec_tr, rec_va = reconstruct(vae, x_tr), reconstruct(vae, x_va)
        mu_va = encode(vae, x_va)
        est = oracle_factors(rec_va)
        truth = {"color": val.phi_c, "bg": val.phi_b, "thick": val.thick}
        sp = {k: {**spearman(est[k], truth[k]), "threshold": THRESHOLDS[k]} for k in THRESHOLDS}
        for v in sp.values():
            v["pass"] = v["spearman"] >= v["threshold"]
        oa = np.concatenate([real_alpha, alpha_hat(rec_tr)])[:, None]
        oy = np.concatenate([real_y, train_s.y])
        torch.manual_seed(seeds["oracle"])
        oracle = new_shape_oracle()
        train(oracle, oa, oy, lambda a, g=None: tt(a), epochs=sc["oracle_epochs"], lr=tr["oracle_lr"],
              batch_size=tr["batch_size"], order_seed=seeds["oracle_order"], noise_seed=0, cosine=True)
        save(oracle, f"shape_oracle_beta{beta}")
        shape_real = float(((shape_prob(oracle, alpha_hat(x_va)) > 0.5) == val.y).mean())
        shape_recon = float(((shape_prob(oracle, est["alpha"]) > 0.5) == val.y).mean())
        f_dec = float((predictor(f_old[0])(rec_va) == val.y).mean())
        yv = torch.as_tensor(val.y)
        lo = {s: margin(torch.as_tensor(logits_of(f_old[s], rec_va)), yv).numpy() for s in f_old}
        valid = {}
        for pid, vname in eval_pairs:
            mn = margin(torch.as_tensor(logits_of(f_new[(pid, vname)], rec_va)), yv).numpy()
            rate = float(((lo[seed_of[pid]] >= kappa) & (mn >= kappa)).mean())
            valid[f"p{pid:02d}_{vname}"] = {"valid_rate_val": rate, "n_valid_val": round(rate * val.n),
                                           "projected_test_valid": rate * N_TEST_IMAGES,
                                           "meets_300": rate * N_TEST_IMAGES >= MIN_TEST_VALID}
        crit = {"spec_spearman": all(v["pass"] for v in sp.values()),
                "spec_shape_oracle": shape_real >= SHAPE_REAL_MIN and shape_recon >= SHAPE_RECON_MIN,
                "valid_yield_all_pairs": bool(valid) and all(v["meets_300"] for v in valid.values()),
                "f_old_on_decoded_ge_090": f_dec >= F_OLD_DECODED_MIN}
        row = {"beta": beta, "recon_mse": float(((rec_va - x_va) ** 2).mean()), "spearman": sp,
               "shape_oracle_real": shape_real, "shape_oracle_recon": shape_recon, "f_old_s0_on_decoded": f_dec,
               "mu_std_per_dim": mu_va.std(0).round(4).tolist(), "mu_std_mean": float(mu_va.std(0).mean()),
               "mu_norm_median": float(np.median(np.linalg.norm(mu_va, axis=1))), "valid": valid, "criteria": crit,
               "meets_all": all(crit.values())}
        rows.append(row)
        log(f"beta={beta}: spearman={ {k: round(v['spearman'], 3) for k, v in sp.items()} } shape={shape_real:.4f}/"
            f"{shape_recon:.4f} f_old_dec={f_dec:.4f} min_valid_rate={min((v['valid_rate_val'] for v in valid.values()), default=float('nan')):.3f} meets={row['meets_all']}")
    res["beta_rows"] = rows
    ok = [r["beta"] for r in rows if r["meets_all"]]
    res["selected_beta"] = max(ok) if ok else None

    # two-tier rule (I6)
    tier_level = {"beta_found": res["selected_beta"] is not None,
                  "all_f_old_ge_095": all(v["meets_095"] for v in res["f_old"].values()),
                  "g0_T1": res["variant_choice"]["T1"] is not None, "g0_N0": res["variant_choice"]["N0"] is not None}
    t3 = res["variant_choice"]["T3"]
    res["decision"] = {
        "tier_level": tier_level, "tier1_kept": all(tier_level.values()), "t3_variant": t3,
        "t3_kept": t3 is not None, "h2b_k_star": None if t3 is None else ("bg" if t3 == "bg" else "thick"),
        "consequence": ("Tier 1 dropped (H6): Route A = H4 ∧ H2a, Holm m = 1, no B5 claim" if not all(tier_level.values())
                        else "T3-Tier 1 dropped (v1.3 C3): H2b untested, H1 on T1 only, Holm m = 2, no B5 claim"
                        if t3 is None else f"Tier 1 kept; T3 variant {t3}" + (" (H2b with k* = bg)" if t3 == "bg" else "")),
    }
    (art / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "artifacts": manifest}, indent=2))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    meta.update(status="SUCCESS", seconds=round(time.time() - start, 1))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    log(json.dumps(res["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
