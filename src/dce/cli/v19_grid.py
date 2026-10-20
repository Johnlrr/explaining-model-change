"""D-036 / spec amendment v1.9, Stage A: rotation × offset grid on Tier 0.

Pairs: 16 cells (α, t) × f_old seeds {0, 1} = 32 pairs, pair_id = 20 + idx with idx running over α, then t,
then seed (Tier 0 uses 0–16, T5 17–19). Every per-pair seed is `tier0.yaml` base + pair_id.

train    — f_new = fine-tune of the exp_0002 f_old of the same seed (hash-verified) on the v1.9 adaptation set
           (10 000 samples, 5% label noise, 10 epochs, Adam lr 1e-3, batch 256: the T4 recipe of exp_0003).
qualify  — on validation only: α̂, t̂, accuracy under the new rule (5 000 validation points), DD on the 2 000 G0
           points; criteria v1.9 §2.3 (pilot exp_0024) or v1.9.1 §2 (D-037: offset checks against t̂_old of f_old);
           the null cell keeps DD ≤ 0.02. No explanation method is run.

The pilot (v1.9 §6) is `train` + `qualify` with pair_ids = [40] (α = 0.25, t = 0.56, seed 0).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dce.analysis.qualification import grid_checks_v191, qualification_row
from dce.cli.g0_reeval import load_verified
from dce.cli.g0_tier0 import git_state, load_config, make_batch_fn, sha256
from dce.data.drift import clean_splits, ro_adaptation_set, ro_label
from dce.eval.g0 import decisive_disagreement
from dce.models.cnn import SmallCNN, predict_logits, train

ALPHAS = (0.0, 0.1, 0.25, 0.5)
THRESHOLDS = (0.50, 0.53, 0.56, 0.60)
F_OLD_SEEDS = (0, 1)
PAIR_BASE = 20
NULL_CELL = (0.0, 0.50)
CONSISTENCY_CELL = (0.0, 0.60)


def grid_pairs() -> list[dict]:
    pairs, idx = [], 0
    for a in ALPHAS:
        for t in THRESHOLDS:
            for s in F_OLD_SEEDS:
                k_star = None if (a, t) == NULL_CELL else ("aspect" if a == 0 else "color")
                pairs.append({"pair_id": PAIR_BASE + idx, "idx": idx, "alpha": a, "t": t, "seed": s, "k_star": k_star})
                idx += 1
    return pairs


def model_name(p: dict) -> str:
    return f"f_new_p{p['pair_id']}_a{p['alpha']:.2f}_t{p['t']:.2f}_s{p['seed']}"


def logits_np(model, phi) -> np.ndarray:
    with torch.no_grad():
        return predict_logits(model, phi, make_batch_fn(False)).numpy()


def selected(cfg) -> list[dict]:
    ids = cfg["pair_ids"]
    return [p for p in grid_pairs() if ids is None or p["pair_id"] in ids]


def cmd_train(cfg, out: Path) -> int:
    (out / "models").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", 8))
    sd, rc, sc = cfg["seeds"], cfg["recipe"], cfg["scale"]
    cache, manifest, rows = {}, [], []
    man_path = out / "artifact_manifest.json"
    if man_path.exists():
        manifest = json.loads(man_path.read_text())["artifacts"]
    done = {a["name"] for a in manifest}
    for p in selected(cfg):
        name = model_name(p)
        if name in done:
            continue
        f_old, _ = load_verified(Path(cfg["f_old_source"]), f"f_old_s{p['seed']}", cache)
        m = SmallCNN()
        m.load_state_dict(f_old.state_dict())
        pid = p["pair_id"]
        phi_a, y_a = ro_adaptation_set(p["alpha"], p["t"], sd["adapt_data_base"] + pid, n=sc["n_adapt"],
                                       label_noise=cfg["adapt_label_noise"])
        t0 = time.time()
        losses = train(m, phi_a, y_a, make_batch_fn(True), epochs=rc["epochs"], lr=rc["lr"], batch_size=rc["batch_size"],
                       order_seed=sd["adapt_order_base"] + pid, noise_seed=sd["adapt_noise_base"] + pid)
        path = out / "models" / f"{name}.pt"
        torch.save(m.state_dict(), path)
        manifest.append({"name": name, "kind": "checkpoint", "path": f"models/{path.name}", "sha256": sha256(path)})
        rows.append({**p, "loss": losses, "wall_s": round(time.time() - t0, 1)})
        man_path.write_text(json.dumps({"schema_version": 1, "artifacts": manifest}, indent=2))
        print(f"trained {name} in {rows[-1]['wall_s']} s, final loss {losses[-1]:.4f}", flush=True)
    log = out / "train_log.json"
    prev = json.loads(log.read_text()) if log.exists() else []
    log.write_text(json.dumps(prev + rows, indent=2))
    return 0


def cmd_qualify(cfg, out: Path) -> int:
    torch.set_num_threads(cfg.get("threads", 8))
    man = {a["name"]: a for a in json.loads((out / "artifact_manifest.json").read_text())["artifacts"]}
    val = clean_splits(seed=cfg["data_seed"])["val"]
    phi = val.phi[: cfg["scale"]["n_qualify"]]
    g0_phi = val.phi[: cfg["scale"]["n_g0_points"]]
    criteria = cfg.get("qualification", "v1.9")
    cache, rows, t_old = {}, [], {}
    for p in selected(cfg):
        entry = man[model_name(p)]
        path = out / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise SystemExit(f"hash mismatch for {path}")
        f_new = SmallCNN()
        f_new.load_state_dict(torch.load(path))
        f_new.eval()
        f_old, _ = load_verified(Path(cfg["f_old_source"]), f"f_old_s{p['seed']}", cache)
        ln = logits_np(f_new, phi)
        decision = (ln[:, 1] - ln[:, 0] > 0).astype(np.int64)
        dd = float(decisive_disagreement(logits_np(f_old, g0_phi), logits_np(f_new, g0_phi), cfg["kappa"]).mean())
        if criteria == "v1.9.1":
            if p["seed"] not in t_old:   # t̂_old: the same fit on f_old's decision (v1.9.1 §2 (2a))
                lo = logits_np(f_old, phi)
                t_old[p["seed"]] = qualification_row(phi, (lo[:, 1] - lo[:, 0] > 0).astype(np.int64),
                                                     ro_label(phi, 0.0, 0.5), 0.0, 0.5)
            y_rule = ro_label(phi, p["alpha"], p["t"])
            row = qualification_row(phi, decision, y_rule, p["alpha"], p["t"])
            row.update(dd=dd, dt=row["t_hat"] - t_old[p["seed"]]["t_hat"])
            for k in ("t_ok", "qualified"):   # v1.9 §2.3 (2) and its verdict are replaced by v1.9.1 §2
                row.pop(k)
        elif (p["alpha"], p["t"]) == NULL_CELL:
            row = {"alpha": p["alpha"], "t": p["t"], "dd": dd, "qualified": dd <= 0.02, "criterion": "null: DD <= 0.02"}
        else:
            y_rule = ro_label(phi, p["alpha"], p["t"])
            row = qualification_row(phi, decision, y_rule, p["alpha"], p["t"])
        row["sensitivity_ridge_1e-3"] = {k: v for k, v in qualification_row(
            phi, decision, ro_label(phi, p["alpha"], p["t"]), p["alpha"], p["t"], ridge=1e-3).items()
            if k in ("alpha_hat", "t_hat")}
        rows.append({"pair_id": p["pair_id"], "seed": p["seed"], "share_pred_1_new": float(decision.mean()), **row})
        print(json.dumps({k: rows[-1][k] for k in rows[-1] if k not in ("fit", "sensitivity_ridge_1e-3")}), flush=True)
    res = {"split": "validation", "criteria": criteria, "n_points": len(phi), "pairs": rows}
    if criteria == "v1.9.1":
        res["t_hat_old"] = {s: {k: v[k] for k in ("alpha_hat", "t_hat", "fit")} for s, v in t_old.items()}
        res["grid"] = grid_checks_v191(rows) if len(rows) == len(grid_pairs()) else None
    else:
        n_bad = sum(not r["qualified"] for r in rows)
        res.update(n_not_qualified=n_bad, stop_rule_triggered=bool(n_bad > len(rows) / 3) if len(rows) == 32 else None)
    (out / "qualification.json").write_text(json.dumps(res, indent=2))
    (out / "metadata.json").write_text(json.dumps({"experiment": cfg["experiment_id"], "decision": "D-036",
                                                   "git": git_state(), "split": "validation"}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("train", "qualify"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config sha256 {sha256(args.config)}", flush=True)
    return {"train": cmd_train, "qualify": cmd_qualify}[args.stage](cfg, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
