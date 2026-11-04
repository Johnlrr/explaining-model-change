"""D-056 X5: Δ-CE in a learned VAE latent (registered https://osf.io/fr7jv).

The exact renderer of the paper is replaced by the decoder of a VAE trained on rendered images; the search space is
R^16, margins are m(z) = margin_y(CNN(D(z))). Models: the locked split-D pairs of exp_0049 (8 qualified offsets,
8 nulls). Every fixed value is in D-056 and the config; nothing here is tuned on X5 outputs.

train-vae   — 50 000 clean rendered images (data seed 11000), VAE 32×32, β = 1; gate on validation (accuracy of
              f_old seeds 0–4 on D(μ(x)) ≥ 0.90); one fallback β = 0.5 only if the gate fails. vae_gate.json.
calibrate-r — r_VAE = 2 × median ordinary-CE(f_new) distance, pair 100, 200 valid VALIDATION points. r_calibration.json.
time        — feasibility (validation, timing_points queries of the timing pairs) → n_points 200 / 100 / GPU.
run-chunked — split F (data seed 10500 + pair), first n_points valid queries; Δ-CE N and O, CF-diff (B1-det), s_M,
              s_M,r; saved per chunk (resume), merged per pair. Chunks equal the rows of an unchunked run.
analyze     — readouts 1–4 of D-056 (null witness share; detection at null thresholds; on-manifold share of witnesses;
              Recovery with factors read by fitting the renderer).

    python -m dce.cli.x5_vae <stage> --config configs/experiments/exp_0057.yaml --out runs/experiments/exp_0057
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dce.baselines.b1_diff import b1_diff
from dce.cli.calibrate_r import solver_config
from dce.cli.confirm_tier0 import new_test_set
from dce.cli.final_tier0 import Timer
from dce.cli.g0_reeval import load_verified
from dce.cli.g0_tier0 import git_state, load_config, sha256
from dce.cli.review_0211 import rank_metrics, thresholds
from dce.cli.review_d048 import cmd_json
from dce.cli.split_d import BY_ID, load_d_models
from dce.cli.tier1_null import Tier1Margin
from dce.cli.x4_resnet import _atomic_save
from dce.analysis.v110 import check_fresh_seed
from dce.data.drift import clean_splits
from dce.data.renderer import render, sample_factors
from dce.eval.records import runtime_row
from dce.eval.tier0_margins import valid_points
from dce.models.vae import VAE, train_vae
from dce.solvers import search
from dce.solvers.dce import delta_ce
from dce.solvers.ordinary_ce import ordinary_ce

ASPECT = 0
BATCH = 1000


# ---------------------------------------------------------------- helpers

def render_np(phi: np.ndarray) -> torch.Tensor:
    return torch.cat([render(torch.as_tensor(phi[i:i + 5000], dtype=torch.float32)) for i in range(0, len(phi), 5000)])


@torch.no_grad()
def encode(vae: VAE, x: torch.Tensor) -> np.ndarray:
    return torch.cat([vae.encoder(x[i:i + BATCH])[0] for i in range(0, len(x), BATCH)]).numpy()


@torch.no_grad()
def decode(vae: VAE, z: np.ndarray) -> torch.Tensor:
    zt = torch.as_tensor(z, dtype=torch.float32)
    return torch.cat([vae.decoder(zt[i:i + BATCH]) for i in range(0, len(zt), BATCH)]) if len(zt) else \
        torch.zeros((0, 3, 32, 32))


@torch.no_grad()
def logit_gap(vae: VAE, f, z: np.ndarray) -> np.ndarray:
    """ℓ_1 − ℓ_0 of f at D(z)."""
    out = []
    for i in range(0, len(z), BATCH):
        lg = f(vae.decoder(torch.as_tensor(z[i:i + BATCH], dtype=torch.float32)))
        out.append((lg[:, 1] - lg[:, 0]).numpy().astype(float))
    return np.concatenate(out) if out else np.zeros(0)


def latent_ball(z0: np.ndarray, r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n points uniform in the Euclidean ball of radius r around z0 (no box: the latent is R^d)."""
    d = z0.shape[0]
    u = rng.standard_normal((n, d))
    u *= (r * rng.random(n) ** (1 / d) / np.linalg.norm(u, axis=1))[:, None]
    return z0 + u


def margin_scores_latent(vae, f_old, f_new, z0: np.ndarray, y: np.ndarray, point_ids: np.ndarray, kappa: float,
                         r: float, n_ball: int, seed: int) -> dict[str, np.ndarray]:
    """s_M = |h(z0)|, s_M,r = max |h| over z0 and n_ball latent-ball points; one rng stream per point
    (default_rng([seed, point_id])), so the scores do not depend on chunking."""
    g_o, g_n = logit_gap(vae, f_old, z0), logit_gap(vae, f_new, z0)
    sgn = np.where(y == 1, 1.0, -1.0)
    if (sgn * g_o < kappa - 1e-4).any() or (sgn * g_n < kappa - 1e-4).any():
        raise SystemExit("queries are not valid for these checkpoints (margin < κ)")
    balls = np.stack([latent_ball(z, r, n_ball, np.random.default_rng([seed, int(pid)])) for z, pid in zip(z0, point_ids)])
    flat = balls.reshape(-1, z0.shape[1])
    h_ball = (logit_gap(vae, f_new, flat) - logit_gap(vae, f_old, flat)).reshape(len(z0), n_ball)
    h0 = g_n - g_o
    return {"s_M": np.abs(h0), "s_Mr": np.maximum(np.abs(h0), np.abs(h_ball).max(1))}


def fit_factors(img: torch.Tensor, steps: int, lr: float, restarts: int, seed: int,
                batch: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """φ̂(x) = argmin_{φ ∈ [0,1]^5} MSE(render(φ), x): Adam (projected), `restarts` uniform starts, best kept.
    Starts are drawn for the whole set at once, so the result does not depend on batching. Returns (φ̂, e = MSE)."""
    n = len(img)
    gen = torch.Generator().manual_seed(seed)
    starts = [torch.rand((n, 5), generator=gen) for _ in range(restarts)]
    best_e = torch.full((n,), float("inf"))
    best_phi = torch.zeros((n, 5))
    for lo in range(0, n, batch):
        x = img[lo:lo + batch]
        for s in starts:
            p = s[lo:lo + batch].clone().requires_grad_(True)
            opt = torch.optim.Adam([p], lr=lr)
            for _ in range(steps):
                loss = ((render(p) - x) ** 2).mean((1, 2, 3))
                opt.zero_grad()
                loss.sum().backward()
                opt.step()
                with torch.no_grad():
                    p.clamp_(0.0, 1.0)
            with torch.no_grad():
                e = ((render(p) - x) ** 2).mean((1, 2, 3))
                better = e < best_e[lo:lo + batch]
                best_e[lo:lo + batch] = torch.where(better, e, best_e[lo:lo + batch])
                best_phi[lo:lo + batch] = torch.where(better[:, None], p.detach(), best_phi[lo:lo + batch])
    return best_phi.numpy(), best_e.numpy()


def vae_solver(cfg, r: float):
    return solver_config(cfg, r=r, clamp=None, mu=cfg["mu"])


def f_old_model(cfg, seed: int, cache):
    return load_verified(Path(cfg["f_old_source"]), f"f_old_s{seed}", cache)[0]


def load_vae(cfg) -> tuple[VAE, dict]:
    run = Path(cfg["vae_run"])
    gate = json.loads((run / "vae_gate.json").read_text())
    if gate.get("selected") is None:
        raise SystemExit("no VAE passed the gate (D-056: X5 is reported as not passing its gate; no search)")
    entry = {a["name"]: a for a in gate["artifacts"]}[gate["selected"]]
    path = run / entry["path"]
    if sha256(path) != entry["sha256"]:
        raise SystemExit(f"hash mismatch for {path}")
    v = cfg["vae"]
    vae = VAE(v["latent"], size=v["size"])
    vae.load_state_dict(torch.load(path))
    vae.eval()
    for prm in vae.parameters():
        prm.requires_grad_(False)
    return vae, entry


def validation(cfg):
    val = clean_splits(seed=cfg["data_seed"])["val"]
    return val.phi, val.y_clean.astype(np.int64)


# ---------------------------------------------------------------- stages

def cmd_train_vae(cfg, out: Path) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    v, g = cfg["vae"], cfg["gate"]
    phi = sample_factors(v["n_images"], np.random.default_rng(check_fresh_seed(v["data_seed"])))
    images = render_np(phi)
    phi_val, y_val = validation(cfg)
    x_val = render_np(phi_val[: g["n_val"]])
    cache, attempts, artifacts, selected = {}, [], [], None
    for beta in (v["beta"], v["fallback_beta"]):
        torch.manual_seed(v["seed"])
        vae = VAE(v["latent"], size=v["size"])
        with Timer() as t:
            losses = train_vae(vae, images, epochs=v["epochs"], lr=v["lr"], batch_size=v["batch_size"], seed=v["seed"],
                               beta=beta)
        name = f"vae32_beta{beta}"
        path = out / "models" / f"{name}.pt"
        torch.save(vae.state_dict(), path)
        artifacts.append({"name": name, "kind": "checkpoint", "path": f"models/{path.name}", "sha256": sha256(path)})
        recon = decode(vae, encode(vae, x_val))
        acc = {}
        for s in g["f_old_seeds"]:
            f = f_old_model(cfg, s, cache)
            with torch.no_grad():
                pred = torch.cat([f(recon[i:i + BATCH]).argmax(1) for i in range(0, len(recon), BATCH)]).numpy()
            acc[s] = float((pred == y_val[: g["n_val"]]).mean())
        passed = all(a >= g["min_acc"] for a in acc.values())
        attempts.append({"beta": beta, "name": name, "loss": losses, "wall_s": round(t.wall, 1),
                         "recon_mse_val": float(((recon - x_val) ** 2).mean()), "f_old_acc_on_recon": acc,
                         "gate_passed": passed})
        print(json.dumps({k: attempts[-1][k] for k in ("beta", "recon_mse_val", "f_old_acc_on_recon", "gate_passed",
                                                         "wall_s")}), flush=True)
        if passed:
            selected = name
            break
    res = {"label": "D-056 X5 VAE + gate (validation)", "train_data_seed": v["data_seed"], "n_images": v["n_images"],
           "attempts": attempts, "artifacts": artifacts, "selected": selected, "git": git_state()}
    (out / "vae_gate.json").write_text(json.dumps(res, indent=2))
    return 0


def cmd_calibrate(cfg, out: Path) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    vae, entry = load_vae(cfg)
    c = cfg["calibration"]
    p = BY_ID[c["pair"]]
    f_old, f_new, ents = load_d_models(cfg, p, {})
    phi, y = validation(cfg)
    z_all = encode(vae, render_np(phi))
    dec = vae.decoder
    keep = valid_points(z_all, y, Tier1Margin(dec, f_old, y), Tier1Margin(dec, f_new, y), c["n_points"],
                        cfg["seeds"]["points_base"] + p["pair_id"], cfg["kappa"])
    z0 = torch.as_tensor(z_all[keep])
    with Timer() as t:
        res = ordinary_ce(z0, Tier1Margin(dec, f_new, y[keep]), vae_solver(cfg, float("inf")),
                          cfg["seeds"]["restart_base"] + p["pair_id"])
    d = res.distance[res.found].numpy()
    rec = {"label": "D-056 X5 r_VAE (validation)", "pair": p["pair_id"], "n": len(keep),
           "found_share": float(res.found.float().mean()), "median_distance": float(np.median(d)),
           "r_vae": float(2 * np.median(d)), "latent_mu_std_mean_val": float(z_all.std(0).mean()),
           "wall_s": round(t.wall, 1), "vae": entry, "checkpoints": ents, "git": git_state()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "r_calibration.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in ("found_share", "median_distance", "r_vae", "wall_s")}), flush=True)
    return 0


def data_for(cfg, pid):
    if cfg.get("data") == "val":            # smoke / timing only: same path on the validation split
        phi, y = validation(cfg)
        return phi, y, "validation"
    return new_test_set(check_fresh_seed(cfg["data_seed_base"] + pid), cfg["n_samples"])


def x5_chunk(cfg, p, vae, cache, lo: int, hi: int):
    """Rows [lo, hi) of the n_points run of pair p: valid points are the prefix of one permutation, the phase-1
    restart noise is drawn for the full batch and cut to these rows, ball samples use one rng per point."""
    if cfg.get("r_vae") is None:
        raise SystemExit("r_vae is not frozen in the config")
    pid, n, kappa, r = p["pair_id"], cfg["n_points"], cfg["kappa"], float(cfg["r_vae"])
    f_old, f_new, ents = load_d_models(cfg, p, cache)
    phi, y, digest = data_for(cfg, pid)
    z_all = encode(vae, render_np(phi))
    dec = vae.decoder
    keep = valid_points(z_all, y, Tier1Margin(dec, f_old, y), Tier1Margin(dec, f_new, y), n,
                        cfg["seeds"]["points_base"] + pid, kappa)[lo:hi]
    z0 = torch.as_tensor(z_all[keep])
    yk = y[keep]
    m_old, m_new = Tier1Margin(dec, f_old, yk), Tier1Margin(dec, f_new, yk)
    base, seed = vae_solver(cfg, r), cfg["seeds"]["restart_base"] + pid
    o_noise = search._restart_noise

    def noise(zb, scfg, restart_seed):
        if zb.shape[0] != hi - lo:
            raise SystemExit(f"solver batch {zb.shape[0]} ≠ chunk {hi - lo}")
        return o_noise(torch.zeros((n, zb.shape[1]), dtype=zb.dtype), scfg, restart_seed)[:, lo:hi].to(zb.device)

    arrays, rt = {"point_ids": keep, "y": yk, "z0": z0.numpy()}, {}
    search._restart_noise = noise
    try:
        for kind in ("N", "O"):
            with Timer() as t:
                res = delta_ce(z0, m_old, m_new, base, seed, kind=kind)
            rt[f"dce_{kind}"] = runtime_row(res, t.wall, hi - lo)
            arrays[f"dce_{kind}__found"] = res.found.numpy()
            arrays[f"dce_{kind}__z"] = res.z.numpy()
            arrays[f"dce_{kind}_distance"] = res.distance.numpy()
        with Timer() as t:
            valid, _, ce_o, ce_n = b1_diff(z0, m_old, m_new, base, seed, deterministic=True)
        rt["b1_det"] = {"wall_s_per_point": t.wall / (hi - lo)}
    finally:
        search._restart_noise = o_noise
    arrays["b1_det__found"] = valid.numpy()
    arrays["b1_det__zo"], arrays["b1_det__zn"] = ce_o.z.numpy(), ce_n.z.numpy()
    s = margin_scores_latent(vae, f_old, f_new, z0.numpy(), yk, keep, kappa, r, cfg["n_ball"], cfg["ball_seed_base"] + pid)
    arrays["s_M"], arrays["s_Mr"] = s["s_M"], s["s_Mr"]
    row = {**p, "lo": lo, "hi": hi, "n_points": n, "data_seed": None if cfg.get("data") == "val" else
           cfg["data_seed_base"] + pid, "data_sha256": digest, "runtime": rt, "checkpoints": ents}
    return arrays, row


def cmd_time(cfg, out: Path) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    vae, _ = load_vae(cfg)
    res, k = {}, cfg["timing_points"]
    for pid in cfg["timing_pairs"]:
        t0 = time.time()
        _, row = x5_chunk(cfg | {"data": "val"}, BY_ID[pid], vae, {}, 0, k)
        res[pid] = {"wall_s": time.time() - t0, "points": k, "runtime": row["runtime"]}
    per_point = float(np.mean([r["wall_s"] / r["points"] for r in res.values()]))
    proj = {n: per_point * n * len(cfg["pair_ids"]) / 3600 for n in (200, 100)}
    decision = 200 if proj[200] <= cfg["budget_h"] else 100 if proj[100] <= cfg["budget_h"] else "gpu"
    out_res = {"label": "D-056 X5 timing (validation)", "pairs": res, "wall_s_per_point": per_point,
               "projected_wall_h": proj, "budget_h": cfg["budget_h"], "decision": decision, "git": git_state()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing.json").write_text(json.dumps(out_res, indent=2, default=float))
    print(json.dumps({k: out_res[k] for k in ("wall_s_per_point", "projected_wall_h", "decision")}, default=float))
    return 0


def cmd_run_chunked(cfg, out: Path, chunk: int, pairs: list[int] | None) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    vae, entry = load_vae(cfg)
    (out / "pairs" / "chunks").mkdir(parents=True, exist_ok=True)
    tag = "_".join(map(str, pairs)) if pairs else "all"
    meta_path = out / f"metadata_chunked_{tag}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "experiment": cfg["experiment_id"], "decision": "D-056 X5", "chunk": chunk, "vae": entry, "git_runs": []}
    if meta["chunk"] != chunk:
        raise SystemExit(f"chunk size {chunk} differs from the earlier runs ({meta['chunk']})")
    meta["git_runs"].append({"git": git_state(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "threads": torch.get_num_threads(), "pairs": pairs})
    _atomic_save(meta_path, text=json.dumps(meta, indent=2))
    n, cache = cfg["n_points"], {}
    for pid in pairs or cfg["pair_ids"]:
        npz, js = out / "pairs" / f"pair_{pid}.npz", out / "pairs" / f"pair_{pid}.json"
        if npz.exists() and js.exists():
            print(f"pair {pid}: done, skipped", flush=True)
            continue
        parts, rows = [], []
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            c_npz = out / "pairs" / "chunks" / f"pair_{pid}_{lo:03d}-{hi:03d}.npz"
            c_js = c_npz.with_suffix(".json")
            if not (c_npz.exists() and c_js.exists()):
                t0 = time.time()
                arrays, row = x5_chunk(cfg, BY_ID[pid], vae, cache, lo, hi)
                row["wall_s_chunk"] = round(time.time() - t0, 1)
                _atomic_save(c_npz, arrays)
                _atomic_save(c_js, text=json.dumps(row, indent=2, default=float))
                print(f"pair {pid} points {lo}-{hi} done in {row['wall_s_chunk'] / 60:.1f} min", flush=True)
            parts.append(dict(np.load(c_npz)))
            rows.append(json.loads(c_js.read_text()))
        arrays = {k: np.concatenate([a[k] for a in parts]) for k in parts[0]}
        if len(arrays["point_ids"]) != n or len(set(arrays["point_ids"].tolist())) != n:
            raise SystemExit(f"pair {pid}: merged chunks do not give {n} distinct points")
        row = {k: v for k, v in rows[0].items() if k not in ("runtime", "lo", "hi", "wall_s_chunk")}
        row.update(chunks=[{"lo": r["lo"], "hi": r["hi"], "runtime": r["runtime"], "wall_s_chunk": r["wall_s_chunk"]}
                           for r in rows], wall_s_pair=round(sum(r["wall_s_chunk"] for r in rows), 1))
        _atomic_save(npz, arrays)
        _atomic_save(js, text=json.dumps(row, indent=2, default=float))
        print(f"pair {pid} merged ({n} points)", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_save(meta_path, text=json.dumps(meta, indent=2))
    return 0


def analyze(cfg) -> dict:
    torch.set_num_threads(cfg.get("threads", 8))
    out = Path(cfg["out_run"])
    vae, entry = load_vae(cfg)
    mf = cfg["manifold"]
    fit = lambda img: fit_factors(img, mf["steps"], mf["lr"], mf["restarts"], mf["seed"])
    load = lambda pid: dict(np.load(out / "pairs" / f"pair_{pid}.npz"))
    nulls = [pid for pid in cfg["pair_ids"] if BY_ID[pid]["kind"] == "null"]
    offs = [pid for pid in cfg["pair_ids"] if BY_ID[pid]["kind"] == "offset"]
    res = {"label": "D-056 X5", "r_vae": cfg["r_vae"], "n_points": cfg["n_points"], "vae": entry,
           "nulls": {"pairs": []}, "offsets": {"pairs": []}}

    # on-manifold threshold from validation reconstructions (D-056: first 1 000 validation images)
    phi_val, _ = validation(cfg)
    _, e_val = fit(decode(vae, encode(vae, render_np(phi_val[: mf["n_val"]]))))
    e95 = float(np.quantile(e_val, mf["q"]))
    res["manifold"] = {"e_95": e95, "e_val_median": float(np.median(e_val))}

    def on_manifold(z: np.ndarray) -> np.ndarray:
        return fit(decode(vae, z))[1] <= e95 if len(z) else np.zeros(0, dtype=bool)

    def cfdiff(a):
        return np.where(a["b1_det__found"], np.linalg.norm(a["b1_det__zn"] - a["b1_det__zo"], axis=1), 0.0)

    # 1. nulls
    sc, wit, onm_null = {"s_M": [], "s_Mr": [], "cfdiff": []}, [], []
    for pid in nulls:
        a = load(pid)
        w = a["dce_N__found"] | a["dce_O__found"]
        wit.append(w)
        zs = np.concatenate([a["dce_N__z"][a["dce_N__found"]], a["dce_O__z"][a["dce_O__found"]]])
        om = on_manifold(zs)
        onm_null.append(om)
        res["nulls"]["pairs"].append({"pair_id": pid, "condition": BY_ID[pid]["condition"], "seed": BY_ID[pid]["seed"],
                                      "witness_share_N_or_O": float(w.mean()), "witness_share_N": float(a["dce_N__found"].mean()),
                                      "witness_share_O": float(a["dce_O__found"].mean()), "n_witnesses": int(len(zs)),
                                      "n_on_manifold": int(om.sum()),
                                      "witness_share_on_manifold_only": float(
                                          _on_manifold_share(a, om)),
                                      "cfdiff_attributes": float(a["b1_det__found"].mean())})
        sc["s_M"].append(a["s_M"]), sc["s_Mr"].append(a["s_Mr"]), sc["cfdiff"].append(cfdiff(a))
    sc = {k: np.concatenate(v) for k, v in sc.items()}
    w = np.concatenate(wit)
    om = np.concatenate(onm_null)
    taus = {k: thresholds(v) for k, v in sc.items()}
    res["nulls"].update(n_queries=int(w.size), witness_share_pooled=float(w.mean()),
                        max_witness_share_pair=max(r["witness_share_N_or_O"] for r in res["nulls"]["pairs"]),
                        prediction1_met=bool(w.mean() <= 0.05 and all(r["witness_share_N_or_O"] <= 0.05
                                                                       for r in res["nulls"]["pairs"])),
                        n_witnesses=int(om.size), on_manifold_share=float(om.mean()) if om.size else None, tau=taus)

    # 2–4. offsets: detection on the lowered class (class 1), on-manifold share, Recovery via the renderer fit
    onm_off = []
    for pid in offs:
        a = load(pid)
        if not np.array_equal(a["y"], a["y"].astype(int)):
            raise SystemExit("bad labels")
        low = a["y"] == 1
        fN = a["dce_N__found"]
        om_o = on_manifold(a["dce_N__z"][fN])
        onm_off.append(om_o)
        phi0 = fit(decode(vae, a["z0"]))[0]
        dphi = np.zeros_like(phi0)
        if fN.any():
            dphi[fN] = fit(decode(vae, a["dce_N__z"][fN]))[0] - phi0[fN]
        fB = a["b1_det__found"]
        dphi_b = np.zeros_like(phi0)
        if fB.any():
            dphi_b[fB] = fit(decode(vae, a["b1_det__zn"][fB]))[0] - fit(decode(vae, a["b1_det__zo"][fB]))[0]
        row = {"pair_id": pid, "t": BY_ID[pid]["t"], "seed": BY_ID[pid]["seed"], "n_lowered": int(low.sum()),
               "dce_lowered": float(fN[low].mean()), "dce_found_N": float(fN.mean()),
               "dce_found_O": float(a["dce_O__found"].mean()),
               "n_witnesses_N": int(fN.sum()), "n_on_manifold_N": int(om_o.sum()),
               "dce_recovery_fit": rank_metrics(fN, dphi, ASPECT)["recovery"],
               "cfdiff_recovery_fit": rank_metrics(fB, dphi_b, ASPECT)["recovery"]}
        for name, s in (("s_M", a["s_M"]), ("s_Mr", a["s_Mr"]), ("cfdiff", cfdiff(a))):
            for k, t in taus[name].items():
                row[f"{name}_lowered_{k}"] = float((s[low] > t).mean())
        res["offsets"]["pairs"].append(row)
    rows = res["offsets"]["pairs"]
    om_o = np.concatenate(onm_off) if onm_off else np.zeros(0, dtype=bool)
    k2 = sum(r["dce_lowered"] >= 0.5 for r in rows)
    res["offsets"].update(n_pairs_dce_lowered_ge_05=int(k2), prediction2_met=bool(k2 >= 5),
                          on_manifold_share=float(om_o.mean()) if om_o.size else None)
    res["git"] = git_state()
    return res


def _on_manifold_share(a: dict, om: np.ndarray) -> float:
    """Share of queries with at least one on-manifold witness (N or O); om is ordered N witnesses then O witnesses."""
    fN, fO = a["dce_N__found"], a["dce_O__found"]
    nN = int(fN.sum())
    okN = np.zeros(len(fN), dtype=bool)
    okO = np.zeros(len(fO), dtype=bool)
    okN[np.flatnonzero(fN)] = om[:nN]
    okO[np.flatnonzero(fO)] = om[nN:]
    return float((okN | okO).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("train-vae", "calibrate-r", "time", "run-chunked", "analyze"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--pairs", type=str, default=None, help="run-chunked: comma-separated subset (parallel workers)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    if args.stage == "train-vae":
        return cmd_train_vae(cfg, args.out)
    if args.stage == "calibrate-r":
        return cmd_calibrate(cfg, args.out)
    if args.stage == "time":
        return cmd_time(cfg, args.out)
    if args.stage == "run-chunked":
        pairs = [int(x) for x in args.pairs.split(",")] if args.pairs else None
        return cmd_run_chunked(cfg, args.out, args.chunk, pairs)
    return cmd_json(analyze, "results_x5.json")(cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
