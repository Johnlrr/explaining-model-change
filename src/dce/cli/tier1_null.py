"""D-032: Tier 1 N0 exploratory null analysis (outside acceptance and Holm). Local CPU.

calibrate — r_Tier1 = 2 × median d(z0, CE(f_new)) over the first 300 valid VALIDATION points of pair 6 (N0 s0),
            seed 5000 + 6, v1.5 solver with the Tier 1 settings of spec §6.1 (Reg = ‖z‖², μ = 0.01, no box).
run       — with r frozen in the config: Δ-CE N and O, B1, A3 (κ = 0) on 200 valid TEST points per N0 pair
            (seed 5000 + pair_id); per-pair outputs saved; no metric printed.
analyze   — found-rate (Δ-CE N ∪ O as well), SAR, S, B0 on the unfiltered Tier 1 test split. No Recovery (no k*).

Points live in the VAE latent: z0 = μ(x); margins m(z) = margin_y(CNN(D(z))) for the clean label y (spec §3.2);
Δφ̃ = φ̃(D(z')) − φ̃(D(z_ref)) with φ̂ = (shape-oracle p(9), colour, background, thickness) / σ_k, σ_k = std of φ̂ on
clean validation D(μ(x)). An undefined oracle component (NaN thickness) contributes 0 and is counted.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from dce.baselines.b1_diff import b1_diff
from dce.cli.final_tier0 import Timer
from dce.cli.g0_tier0 import git_state, sha256
from dce.data.tier1 import load_split
from dce.eval.metrics import MethodOutput, found_rate, median_norm, pooled_found_rate, sar_pair
from dce.eval.records import runtime_row
from dce.eval.tier0_margins import valid_points
from dce.models.cnn import SmallCNN, margin
from dce.models.oracle import new_shape_oracle, oracle_factors, shape_prob
from dce.models.vae import VAE
from dce.solvers.dce import delta_ce
from dce.solvers.ordinary_ce import ordinary_ce
from dce.solvers.search import SolverConfig

FACTOR_KEYS = ("shape", "color", "bg", "thick")


class Tier1Margin:
    """(z, idx) → margins through the decoder; float64 copies are used only for float64 inputs (SLSQP)."""

    def __init__(self, decoder, cnn, y):
        self.d32, self.c32 = decoder.eval(), cnn.eval()
        self.d64, self.c64 = copy.deepcopy(decoder).double().eval(), copy.deepcopy(cnn).double().eval()
        for net in (self.d32, self.c32, self.d64, self.c64):
            for p in net.parameters():
                p.requires_grad_(False)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)

    def logits(self, z):
        if z.dtype == torch.float64:
            return self.c64(self.d64(z))
        return self.c32(self.d32(z))

    def __call__(self, z, idx):
        return margin(self.logits(z), self.y[idx])


def load_models(cfg):
    src = Path(cfg["models_run"]) / "artifacts"
    man = {a["name"]: a["sha256"] for a in json.loads((src / "artifact_manifest.json").read_text())["artifacts"]}

    def get(name, cls):
        path = src / f"{name}.pt"
        if sha256(path) != man[name]:
            raise SystemExit(f"hash mismatch for {path}")
        m = cls()
        m.load_state_dict(torch.load(path))
        return m.eval()

    vae = get(f"vae_beta{cfg['beta']}", VAE)
    oracle = get(f"shape_oracle_beta{cfg['beta']}", new_shape_oracle)
    return get, vae, oracle


def data(cfg, split, allow_test=False):
    if sha256(Path(cfg["data_v2"])) != cfg["data_v2_sha256"]:
        raise SystemExit("data v2 hash mismatch")
    return load_split(cfg["data_v2"], split, allow_test=allow_test)


def encode(vae, x):
    with torch.no_grad():
        return np.concatenate([vae.encoder(torch.as_tensor(x[i : i + 500]))[0].numpy() for i in range(0, len(x), 500)])


def phi_hat(vae, oracle, z: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        img = np.concatenate([vae.decoder(torch.as_tensor(z[i : i + 500], dtype=torch.float32)).numpy()
                              for i in range(0, len(z), 500)])
    est = oracle_factors(img)
    return np.stack([shape_prob(oracle, est["alpha"]), est["color"], est["bg"], est["thick"]], 1)


def sigma_k(cfg, vae, oracle) -> np.ndarray:
    val = data(cfg, "val")
    return np.nanstd(phi_hat(vae, oracle, encode(vae, val.images())), axis=0)


def delta(vae, oracle, sig, z_new, z_ref, found) -> tuple[MethodOutput, int]:
    d = (phi_hat(vae, oracle, z_new) - phi_hat(vae, oracle, z_ref)) / sig
    nan = ~np.isfinite(d) & found[:, None]
    d[~np.isfinite(d)] = 0.0
    d[~found] = 0.0
    return MethodOutput(found, d), int(nan.sum())


def solver_cfg(cfg, r=float("inf")):
    return SolverConfig(**{**cfg["solver"], "kappa": cfg["kappa"], "r": r})


def points(cfg, get, vae, split, pid, n):
    f_old, f_new = get(f"f_old_s{cfg['pair_seed'][pid]}", SmallCNN), get(cfg["f_new"][pid], SmallCNN)
    z_all = encode(vae, split.images())
    keep = valid_points(z_all, split.y, Tier1Margin(vae.decoder, f_old, split.y), Tier1Margin(vae.decoder, f_new, split.y),
                        n, cfg["points_base"] + pid, cfg["kappa"])
    y = split.y[keep]
    return torch.as_tensor(z_all[keep]), Tier1Margin(vae.decoder, f_old, y), Tier1Margin(vae.decoder, f_new, y), keep, f_old, f_new


def cmd_calibrate(cfg, out: Path) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    get, vae, _ = load_models(cfg)
    pid = cfg["calibration_pair"]
    z0, _, m_new, keep, _, _ = points(cfg, get, vae, data(cfg, "val"), pid, cfg["calibration_points"])
    with Timer() as t:
        res = ordinary_ce(z0, m_new, solver_cfg(cfg), cfg["restart_base"] + pid)
    d = res.distance[res.found].numpy()
    r = float(2 * np.median(d))
    rec = {"split": "validation", "pair": pid, "n": len(keep), "found_share": float(res.found.float().mean()),
           "median_distance": float(np.median(d)), "r_tier1": r, "wall_s": round(t.wall, 1), "git": git_state(),
           "latent_mu_std_mean_val": float(z0.numpy().std(0).mean())}
    out.mkdir(parents=True, exist_ok=True)
    (out / "r_calibration.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec))
    return 0


def cmd_run(cfg, out: Path) -> int:
    if cfg.get("r") is None:
        raise SystemExit("r_Tier1 is not frozen in the config")
    torch.set_num_threads(cfg.get("threads", 8))
    get, vae, oracle = load_models(cfg)
    split_name = cfg.get("run_split", "test")   # smoke runs use "val": the test split is touched once (D-032)
    test = data(cfg, split_name, allow_test=split_name == "test")
    sig = sigma_k(cfg, vae, oracle)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "experiment": cfg["experiment_id"], "decision": "D-032 (exploratory)", "split": f"tier1 {split_name}", "git_runs": [],
        "sigma_k": dict(zip(FACTOR_KEYS, sig.tolist())), "torch": torch.__version__}
    meta["git_runs"].append({"git": git_state(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    meta["status"] = "RUNNING"
    meta_path.write_text(json.dumps(meta, indent=2))
    base = solver_cfg(cfg, float(cfg["r"]))
    for pid in cfg["pair_ids"]:
        npz = out / "pairs" / f"pair_{pid:02d}.npz"
        if npz.exists():
            print(f"pair {pid}: already done, skipped", flush=True)
            continue
        with Timer() as tp:
            n = cfg["n_points"]
            z0, m_old, m_new, keep, f_old, f_new = points(cfg, get, vae, test, pid, n)
            seed, arrays, rt, nans = cfg["restart_base"] + pid, {"point_ids": keep, "z0": z0.numpy()}, {}, {}
            for name, kappa, kind in (("dce_N", cfg["kappa"], "N"), ("dce_O", cfg["kappa"], "O"), ("a3_k0", 0.0, "N")):
                with Timer() as t:
                    res = delta_ce(z0, m_old, m_new, replace(base, kappa=kappa), seed, kind=kind)
                o, nans[name] = delta(vae, oracle, sig, res.z.numpy(), z0.numpy(), res.found.numpy())
                arrays[f"{name}__found"], arrays[f"{name}__dphi"], arrays[f"{name}_distance"] = o.found, o.dphi, res.distance.numpy()
                rt[name] = runtime_row(res, t.wall, n)
            with Timer() as t:
                valid, _, ce_o, ce_n = b1_diff(z0, m_old, m_new, base, seed)
            o, nans["b1"] = delta(vae, oracle, sig, ce_n.z.numpy(), ce_o.z.numpy(), valid.numpy())
            arrays["b1__found"], arrays["b1__dphi"] = o.found, o.dphi
            rt["b1"] = {"wall_s_per_point": t.wall / n}
            with torch.no_grad():
                x = torch.as_tensor(test.images())
                b0 = float((f_old(x).argmax(1) != f_new(x).argmax(1)).float().mean())
        np.savez_compressed(npz, **arrays)
        (out / "pairs" / f"pair_{pid:02d}.json").write_text(json.dumps(
            {"pair_id": pid, "n_points": n, "b0": b0, "runtime": rt, "nan_components": nans, "wall_s_pair": round(tp.wall, 1)}, indent=2))
        print(f"pair {pid} done in {tp.wall / 60:.1f} min", flush=True)
    meta["status"] = "RUN_COMPLETE"
    meta_path.write_text(json.dumps(meta, indent=2))
    return 0


def cmd_analyze(cfg, out: Path) -> int:
    meta = json.loads((out / "metadata.json").read_text())
    if meta.get("status") != "RUN_COMPLETE" or (out / "results.json").exists():
        raise SystemExit("run incomplete or already analyzed")
    res = {"label": "EXPLORATORY (D-032): Tier 1 N0 null analysis, outside acceptance and Holm", "pairs": {}}
    for pid in cfg["pair_ids"]:
        z = np.load(out / "pairs" / f"pair_{pid:02d}.npz")
        row = json.loads((out / "pairs" / f"pair_{pid:02d}.json").read_text())
        o = {m: MethodOutput(z[f"{m}__found"], z[f"{m}__dphi"]) for m in ("dce_N", "dce_O", "a3_k0", "b1")}
        res["pairs"][pid] = {
            "found_rate": {m: found_rate(v) for m, v in o.items()},
            "dce_found_rate_union_N_O": float((o["dce_N"].found | o["dce_O"].found).mean()),
            "sar": {m: sar_pair(v) for m, v in o.items()},
            "S": {"dce": pooled_found_rate([o["dce_N"], o["dce_O"]]), "b1": median_norm(o["b1"])},
            "b0": row["b0"], "nan_components": row["nan_components"]}
    res["means"] = {k: float(np.mean([p["found_rate"][k] for p in res["pairs"].values()])) for k in ("dce_N", "dce_O", "a3_k0", "b1")}
    res["sar_share_of_pairs"] = {m: float(np.mean([p["sar"][m] for p in res["pairs"].values()])) for m in ("dce_N", "dce_O", "a3_k0", "b1")}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    meta["status"] = "ANALYZED"
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"results written to {out / 'results.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("calibrate", "run", "analyze"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    print(f"config sha256 {sha256(args.config)}", flush=True)
    if args.stage == "calibrate":
        return cmd_calibrate(cfg, args.out)
    if args.stage == "run":
        return cmd_run(cfg, args.out)
    return cmd_analyze(cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
