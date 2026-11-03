"""D-052 X4: second architecture (ResNet-8) on split D, time-boxed (go/no-go 06/11, hard stop 10/11).

Pairs 120–129: offsets α = 0, t ∈ {0.53, 0.60} × f_old seeds {0, 1} (120–123); rotations α = 0.5, same t and seeds
(124–127); nulls N0 × seeds {0, 1} (128, 129). f_old: ResNet-8 trained here with the SmallCNN f_old recipe
(tier0.yaml: 30 epochs, Adam lr 1e-3 with cosine decay, batch 256, train split with label noise and pixel noise).
f_new recipes as split_d (grid: 10 epochs lr 1e-3; nulls: 3 epochs lr 3e-4). Every seed is a tier0.yaml base + id.

train-old — ResNet-8 f_old seeds 0, 1; validation accuracy recorded.
train     — f_new for every pair.
qualify   — validation: v1.9.1 checks (1) |α̂ − α| ≤ 0.05 + 0.3α, (3) accuracy ≥ 0.90, (2a) on α = 0, (2b) order in t.
            Nulls: decisive disagreement recorded, all admitted.
time      — solver timing on 5 split-D queries of one pair → projected CPU-hours (go/no-go rule of D-052).
run       — as split_d (Δ-CE N, O on nulls, CF-diff, Grad-diff, s_M, s_M,r), n_points from the config.
analyze   — descriptive k of n readouts of D-052 X4.

Usage:
    python -m dce.cli.x4_resnet {train-old,train,qualify,time,run,analyze} --config configs/experiments/exp_0050.yaml --out runs/experiments/exp_0050
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dce.analysis.qualification import grid_checks_v191, qualification_row
from dce.cli.g0_tier0 import git_state, load_config, make_batch_fn, sha256
from dce.cli.review_0211 import rank_metrics, thresholds
from dce.cli.review_d048 import cmd_json
from dce.cli.split_d import cfdiff, d_pair
from dce.cli.v19_grid import logits_np
from dce.cli.v110_test import run_pairs
from dce.data.drift import adaptation_set, clean_splits, ro_adaptation_set, ro_label
from dce.eval.g0 import decisive_disagreement
from dce.models.cnn import train
from dce.models.resnet import ResNet8

ASPECT, COLOUR = 0, 2


def x4_pairs() -> list[dict]:
    pairs, pid = [], 120
    for alpha in (0.0, 0.5):
        for s in (0, 1):
            for t in (0.53, 0.60):
                pairs.append({"pair_id": pid, "kind": "offset" if alpha == 0 else "rotation", "alpha": alpha, "t": t,
                              "seed": s, "k_star": "aspect" if alpha == 0 else "color"})
                pid += 1
    for s in (0, 1):
        pairs.append({"pair_id": pid, "kind": "null", "condition": "N0", "seed": s, "k_star": None})
        pid += 1
    return pairs


BY_ID = {p["pair_id"]: p for p in x4_pairs()}


def name_of(p: dict) -> str:
    tail = p["condition"] if p["kind"] == "null" else f"a{p['alpha']:.2f}_t{p['t']:.2f}"
    return f"r8_f_new_p{p['pair_id']}_{tail}_s{p['seed']}"


def manifest(out: Path) -> list[dict]:
    path = out / "artifact_manifest.json"
    return json.loads(path.read_text())["artifacts"] if path.exists() else []


def save_model(out: Path, model, name: str) -> None:
    (out / "models").mkdir(parents=True, exist_ok=True)
    path = out / "models" / f"{name}.pt"
    torch.save(model.state_dict(), path)
    man = [a for a in manifest(out) if a["name"] != name]
    man.append({"name": name, "kind": "checkpoint", "path": f"models/{path.name}", "sha256": sha256(path)})
    (out / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "artifacts": man}, indent=2))


def load_model(out: Path, name: str) -> tuple[ResNet8, dict]:
    entry = {a["name"]: a for a in manifest(out)}[name]
    path = out / entry["path"]
    if sha256(path) != entry["sha256"]:
        raise SystemExit(f"hash mismatch for {path}")
    m = ResNet8()
    m.load_state_dict(torch.load(path))
    m.eval()
    return m, {"name": name, "path": str(path), "sha256": entry["sha256"]}


def load_x4_models(cfg, p, cache):
    out = Path(cfg["models_run"])
    f_old, e_old = load_model(out, f"r8_f_old_s{p['seed']}")
    f_new, e_new = load_model(out, name_of(p))
    return f_old, f_new, [e_old, e_new]


def cmd_train_old(cfg, out: Path) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    splits = clean_splits(seed=cfg["data_seed"])
    tr, sd, rows = cfg["training"], cfg["seeds"], []
    done = {a["name"] for a in manifest(out)}
    for s in cfg["f_old_seeds"]:
        if f"r8_f_old_s{s}" in done:
            continue
        torch.manual_seed(sd["init_base"] + s)
        m = ResNet8()
        t0 = time.time()
        losses = train(m, splits["train"].phi, splits["train"].y, make_batch_fn(True), epochs=cfg["f_old_epochs"],
                       lr=tr["f_old_lr"], batch_size=tr["batch_size"], order_seed=sd["order_base"] + s,
                       noise_seed=sd["noise_base"] + s, cosine=cfg.get("f_old_cosine", False), device=cfg.get("device"))
        m.cpu().eval()                          # checkpoints are saved and evaluated on the CPU
        lo = logits_np(m, splits["val"].phi)
        acc = float(((lo[:, 1] - lo[:, 0] > 0).astype(int) == splits["val"].y_clean).mean())
        save_model(out, m, f"r8_f_old_s{s}")
        rows.append({"seed": s, "val_accuracy": acc, "loss": losses, "wall_s": round(time.time() - t0, 1)})
        print(f"f_old s{s}: val acc {acc:.4f} in {rows[-1]['wall_s']} s", flush=True)
    log = out / "train_old_log.json"
    log.write_text(json.dumps((json.loads(log.read_text()) if log.exists() else []) + rows, indent=2))
    return 0


def cmd_train(cfg, out: Path) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    sd, n_adapt, rows = cfg["seeds"], cfg["n_adapt"], []
    done = {a["name"] for a in manifest(out)}
    for pid in cfg["pair_ids"]:
        p = BY_ID[pid]
        if name_of(p) in done:
            continue
        f_old, _ = load_model(out, f"r8_f_old_s{p['seed']}")
        m = ResNet8()
        m.load_state_dict(f_old.state_dict())
        if p["kind"] == "null":
            phi_a, y_a = adaptation_set(p["condition"], seed=sd["adapt_data_base"] + pid, n=n_adapt,
                                        label_noise=cfg["adapt_label_noise"])
            rc = cfg["recipe_null"]
        else:
            phi_a, y_a = ro_adaptation_set(p["alpha"], p["t"], sd["adapt_data_base"] + pid, n=n_adapt,
                                           label_noise=cfg["adapt_label_noise"])
            rc = cfg["recipe_grid"]
        t0 = time.time()
        losses = train(m, phi_a, y_a, make_batch_fn(True), epochs=rc["epochs"], lr=rc["lr"], batch_size=rc["batch_size"],
                       order_seed=sd["adapt_order_base"] + pid, noise_seed=sd["adapt_noise_base"] + pid,
                       device=cfg.get("device"))
        m.cpu().eval()
        save_model(out, m, name_of(p))
        rows.append({**p, "loss": losses, "wall_s": round(time.time() - t0, 1)})
        print(f"trained {name_of(p)} in {rows[-1]['wall_s']} s, final loss {losses[-1]:.4f}", flush=True)
    log = out / "train_log.json"
    log.write_text(json.dumps((json.loads(log.read_text()) if log.exists() else []) + rows, indent=2))
    return 0


def cmd_qualify(cfg, out: Path) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    val = clean_splits(seed=cfg["data_seed"])["val"]
    phi, g0_phi = val.phi[: cfg["n_qualify"]], val.phi[: cfg["n_g0_points"]]
    rows, t_old = [], {}
    for pid in cfg["pair_ids"]:
        p = BY_ID[pid]
        f_old, f_new, _ = load_x4_models(cfg | {"models_run": str(out)}, p, {})
        dd = float(decisive_disagreement(logits_np(f_old, g0_phi), logits_np(f_new, g0_phi), cfg["kappa"]).mean())
        if p["kind"] == "null":
            rows.append({"pair_id": pid, "kind": "null", "seed": p["seed"], "dd": dd, "admitted": True})
            continue
        if p["seed"] not in t_old:
            lo = logits_np(f_old, phi)
            t_old[p["seed"]] = qualification_row(phi, (lo[:, 1] - lo[:, 0] > 0).astype(np.int64),
                                                 ro_label(phi, 0.0, 0.5), 0.0, 0.5)["t_hat"]
        ln = logits_np(f_new, phi)
        row = qualification_row(phi, (ln[:, 1] - ln[:, 0] > 0).astype(np.int64), ro_label(phi, p["alpha"], p["t"]),
                                p["alpha"], p["t"])
        row.update(pair_id=pid, kind=p["kind"], alpha=p["alpha"], t=p["t"], seed=p["seed"], dd=dd,
                   dt=row["t_hat"] - t_old[p["seed"]])
        for k in ("t_ok", "qualified"):
            row.pop(k, None)
        rows.append(row)
    ch = [r for r in rows if r["kind"] != "null"]
    checks = grid_checks_v191(ch)
    bad = {q for v in checks["order_violations"] for q in v["pairs"]}
    for r in ch:
        r["qualified"] = bool(checks["per_pair"][r["pair_id"]]["base_qualified"] and r["pair_id"] not in bad)
    res = {"split": "validation", "criteria": "v1.9.1; nulls all admitted (D-052 X4)", "pairs": rows,
           "checks": {k: v for k, v in checks.items() if k != "exclusions"}, "t_hat_old": t_old, "git": git_state()}
    (out / "qualification.json").write_text(json.dumps(res, indent=2, default=float))
    for r in rows:
        print({k: r[k] for k in ("pair_id", "kind", "dd", "qualified", "alpha_hat", "t_hat", "dt") if k in r}, flush=True)
    return 0


def x4_pair(cfg, p, cache):
    import dce.cli.split_d as sd
    orig = sd.load_d_models
    sd.load_d_models = load_x4_models          # same per-pair pipeline as split D, ResNet-8 checkpoints
    try:
        return d_pair(cfg, p, cache)
    finally:
        sd.load_d_models = orig


def _atomic_save(path: Path, arrays: dict | None = None, text: str | None = None) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if arrays is not None:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
    else:
        tmp.write_text(text)
    tmp.replace(path)


def x4_chunk(cfg, p, cache, lo: int, hi: int):
    """Points [lo, hi) of the n_points run of pair p, equal to the same rows of an unchunked `x4_pair` run: the
    valid points are the prefix of the same permutation, the phase-1 restart noise is drawn for the full batch and
    cut to these rows, and the ball samples of s_M,r are drawn for every earlier point first (sequential rng)."""
    import dce.cli.split_d as sd
    from dce.solvers import search
    n = cfg["n_points"]
    o_valid, o_noise, o_scores = sd.valid_points, search._restart_noise, sd.margin_scores
    state = {}

    def valid(phi, y, m_old, m_new, n_req, seed, kappa):
        if n_req != n:
            raise SystemExit(f"valid_points called with n = {n_req}, expected {n}")
        keep = o_valid(phi, y, m_old, m_new, n, seed, kappa)
        state["z_all"], state["y_all"] = phi[keep], y[keep]
        return keep[lo:hi]

    def noise(z0, scfg, restart_seed):
        if z0.shape[0] != hi - lo:
            raise SystemExit(f"solver batch {z0.shape[0]} ≠ chunk {hi - lo}")
        full = torch.zeros((n, z0.shape[1]), dtype=z0.dtype)
        return o_noise(full, scfg, restart_seed)[:, lo:hi].to(z0.device)

    def scores(f_old, f_new, z0, y, kappa, r, n_ball, seed):
        if not np.allclose(z0, state["z_all"][lo:hi]):
            raise SystemExit("margin_scores called on unexpected points")
        res = o_scores(f_old, f_new, state["z_all"][:hi].astype(np.float32), state["y_all"][:hi], kappa, r, n_ball, seed)
        return {k: v[lo:] for k, v in res.items()}

    sd.valid_points, search._restart_noise, sd.margin_scores = valid, noise, scores
    try:
        return x4_pair(cfg, p, cache)
    finally:
        sd.valid_points, search._restart_noise, sd.margin_scores = o_valid, o_noise, o_scores


def cmd_run_chunked(cfg, out: Path, chunk: int) -> int:
    """`run` with resume inside a pair: every chunk of points is saved as soon as it is done
    (pairs/chunks/pair_<id>_<lo>-<hi>.npz/.json, atomic), finished chunks are skipped on restart, and the pair file
    pair_<id>.npz/.json (same format as `run`) is written once all chunks exist. Restart the same command after an
    interruption; at most one chunk is lost."""
    torch.set_num_threads(cfg.get("threads", 8))
    (out / "pairs" / "chunks").mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata_chunked.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"experiment": cfg["experiment_id"],
                                                                         "chunk": chunk, "git_runs": []}
    if meta["chunk"] != chunk:
        raise SystemExit(f"chunk size {chunk} differs from the earlier runs ({meta['chunk']})")
    meta["git_runs"].append({"git": git_state(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "threads": torch.get_num_threads()})
    _atomic_save(meta_path, text=json.dumps(meta, indent=2))
    n, cache = cfg["n_points"], {}
    for pid in cfg["pair_ids"]:
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
                arrays, row = x4_chunk(cfg, BY_ID[pid], cache, lo, hi)
                row["wall_s_chunk"] = round(time.time() - t0, 1)
                _atomic_save(c_npz, arrays)
                _atomic_save(c_js, text=json.dumps(row, indent=2, default=float))
                print(f"pair {pid} points {lo}-{hi} done in {row['wall_s_chunk'] / 60:.1f} min", flush=True)
            parts.append(dict(np.load(c_npz)))
            rows.append(json.loads(c_js.read_text()))
        arrays = {k: np.concatenate([a[k] for a in parts]) for k in parts[0]}
        if len(arrays["point_ids"]) != n or len(set(arrays["point_ids"].tolist())) != n:
            raise SystemExit(f"pair {pid}: merged chunks do not give {n} distinct points")
        row = {k: v for k, v in rows[0].items() if k not in ("runtime", "n_points", "wall_s_chunk")}
        row.update(n_points=n, chunks=[{"lo": lo, "runtime": r["runtime"], "wall_s_chunk": r["wall_s_chunk"]}
                                       for lo, r in zip(range(0, n, chunk), rows, strict=True)],
                   wall_s_pair=round(sum(r["wall_s_chunk"] for r in rows), 1))
        _atomic_save(npz, arrays)
        _atomic_save(js, text=json.dumps(row, indent=2, default=float))
        print(f"pair {pid} merged ({n} points)", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_save(meta_path, text=json.dumps(meta, indent=2))
    return 0


def cmd_time(cfg, out: Path) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    res = {}
    for pid in cfg["timing_pairs"]:
        t0 = time.time()
        _, row = x4_pair(cfg | {"n_points": cfg["timing_points"]}, BY_ID[pid], {})   # checkpoints from models_run
        res[pid] = {"wall_s": time.time() - t0, "points": cfg["timing_points"], "runtime": row["runtime"]}
    per_point = {pid: r["wall_s"] / r["points"] for pid, r in res.items()}
    kinds = {pid: BY_ID[pid]["kind"] for pid in per_point}
    proj = {}
    for n in (200, 100):
        tot = 0.0
        for pid in cfg["pair_ids"]:
            k = BY_ID[pid]["kind"]
            ref = [per_point[q] for q in per_point if (kinds[q] == "null") == (k == "null")] or list(per_point.values())
            tot += np.mean(ref) * n
        proj[n] = tot / 3600
    decision = 200 if proj[200] <= cfg["budget_h"] else 100 if proj[100] <= cfg["budget_h"] else None
    out_res = {"label": "D-052 X4 timing (go/no-go)", "pairs": res, "projected_wall_h": proj, "n_points": decision,
               "go": decision is not None, "git": git_state()}
    (out / "timing.json").write_text(json.dumps(out_res, indent=2, default=float))
    print(json.dumps({k: out_res[k] for k in ("projected_wall_h", "n_points", "go")}, default=float))
    return 0


def analyze(cfg) -> dict:
    out = Path(cfg["out_run"])
    qual = {r["pair_id"]: r for r in json.loads((out / "qualification.json").read_text())["pairs"]}
    load = lambda pid: dict(np.load(out / "pairs" / f"pair_{pid}.npz"))
    nulls = [pid for pid in cfg["pair_ids"] if BY_ID[pid]["kind"] == "null"]
    res = {"label": "D-052 X4 (descriptive)", "nulls": [], "offsets": [], "rotations": []}
    sc, wit = {"s_M": [], "cfdiff": []}, []
    for pid in nulls:
        a = load(pid)
        w = a["dce_N__found"] | a["dce_O__found"]
        wit.append(w)
        res["nulls"].append({"pair_id": pid, "dd_validation": qual[pid]["dd"], "witness_share": float(w.mean())})
        sc["s_M"].append(a["s_M"]), sc["cfdiff"].append(cfdiff(a))
    taus = {k: thresholds(np.concatenate(v)) for k, v in sc.items()}
    res["tau"] = taus
    res["null_witness_share_pooled"] = float(np.concatenate(wit).mean())
    for pid in cfg["pair_ids"]:
        p = BY_ID[pid]
        if p["kind"] == "null" or not qual[pid].get("qualified"):
            continue
        a = load(pid)
        k = ASPECT if p["kind"] == "offset" else COLOUR
        row = {"pair_id": pid, "t": p["t"], "seed": p["seed"],
               "graddiff_recovery": rank_metrics(a["b3__found"], a["b3__dphi"], k)["recovery"],
               "dce_recovery": rank_metrics(a["dce_N__found"], a["dce_N__dphi"], k)["recovery"]}
        if p["kind"] == "offset":
            low = a["y"] == 1
            row["dce_lowered"] = float(a["dce_N__found"][low].mean())
            for name, s in (("s_M", a["s_M"]), ("cfdiff", cfdiff(a))):
                for kk, t in taus[name].items():
                    row[f"{name}_lowered_{kk}"] = float((s[low] > t).mean())
        res["offsets" if p["kind"] == "offset" else "rotations"].append(row)
    off, rot = res["offsets"], res["rotations"]
    res["k_graddiff_below_02_offsets"] = f"{sum(r['graddiff_recovery'] < 0.2 for r in off)} of {len(off)}"
    cmp = []
    for r in rot:
        o = next((x for x in off if x["seed"] == r["seed"] and x["t"] == r["t"]), None)
        if o:
            cmp.append(r["graddiff_recovery"] > o["graddiff_recovery"])
    res["k_rotation_above_offset"] = f"{sum(cmp)} of {len(cmp)}"
    res["k_dce_ge_sM_plus_02_fpr0"] = f"{sum(r['dce_lowered'] >= r['s_M_lowered_fpr0'] + 0.2 for r in off)} of {len(off)}"
    res["k_dce_ge_cfdiff_plus_02_fpr0"] = f"{sum(r['dce_lowered'] >= r['cfdiff_lowered_fpr0'] + 0.2 for r in off)} of {len(off)}"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("train-old", "train", "qualify", "time", "run", "run-chunked", "analyze"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--chunk", type=int, default=20, help="run-chunked: points per saved chunk")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    if args.stage == "run-chunked":
        return cmd_run_chunked(cfg, args.out, args.chunk)
    stage = {"train-old": cmd_train_old, "train": cmd_train, "qualify": cmd_qualify, "time": cmd_time}
    if args.stage in stage:
        return stage[args.stage](cfg, args.out)
    if args.stage == "run":
        return run_pairs(cfg, args.out, tuple(map(int, args.shard.split("/"))), BY_ID, x4_pair, "D-052 X4")
    return cmd_json(analyze, "results_x4.json")(cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
