"""D-042 diagnostic (descriptive): measured offset t̂ − t̂_old versus witnesses, α = 0 pairs and locked null pairs.

Locked null pairs (N0 9–13, N1 14–16; exp_0002 checkpoints, hash-verified): only the v1.9.1 t̂ measurement is computed
(5 000 validation points). Validation DD comes from exp_0004; test found-rates and witness classes are read from the
saved exp_0019 / exp_0017 outputs. Pairs 20–27 come from exp_0025 (t̂, DD) and exp_0026 (found-rate, class).
No method is re-run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dce.analysis.qualification import qualification_row
from dce.cli.g0_tier0 import git_state
from dce.cli.v19_grid import logits_np
from dce.data.drift import clean_splits, ro_label
from dce.eval.pairs import load_pair

NULL_PAIRS = list(range(9, 17))


def t_hat(model, phi) -> float:
    lg = logits_np(model, phi)
    return qualification_row(phi, (lg[:, 1] - lg[:, 0] > 0).astype(np.int64), ro_label(phi, 0.0, 0.5), 0.0, 0.5)["t_hat"]


def found_summary(npz: Path) -> dict:
    a = np.load(npz)
    y = (a["z0"][:, 0] > 0.5).astype(int)      # clean label of the original rule
    out = {}
    for k in ("dce_N", "dce_O"):
        if f"{k}__found" in a.files:
            f = a[f"{k}__found"]
            out[k] = {"found_rate": float(f.mean()), "n_found_y1": int(f[y == 1].sum()), "n_found_y0": int(f[y == 0].sum())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    phi = clean_splits(seed=1234)["val"].phi[:5000]
    g0 = {p["pair_id"]: p for p in json.loads(Path("runs/experiments/exp_0004/metrics.json").read_text())["pairs"]}
    cache, t_old, rows = {}, {}, []
    for pid in NULL_PAIRS:
        pair, f_old, f_new, entries = load_pair(pid, cache=cache)
        s = pair["seed"]
        if s not in t_old:
            t_old[s] = t_hat(f_old, phi)
        th = t_hat(f_new, phi)
        rows.append({"pair_id": pid, "condition": pair["condition"], "seed": s, "t_hat": th, "dt": th - t_old[s],
                     "dd_val": g0[pid]["decisive_disagreement_all"], "dd_source": "exp_0004",
                     "test_exp_0019": found_summary(Path(f"runs/experiments/exp_0019/pairs/pair_{pid:02d}.npz")),
                     "test_exp_0017": found_summary(Path(f"runs/experiments/exp_0017/pairs/pair_{pid:02d}.npz")),
                     "checkpoints": entries})
    q = {r["pair_id"]: r for r in json.loads(Path("runs/experiments/exp_0025/qualification.json").read_text())["pairs"]}
    for pid in range(20, 28):
        rows.append({"pair_id": pid, "condition": f"v1.9 (0, {q[pid]['t']:.2f})", "seed": q[pid]["seed"], "t_hat": q[pid]["t_hat"],
                     "dt": q[pid]["dt"], "dd_val": q[pid]["dd"], "dd_source": "exp_0025",
                     "test_exp_0026": found_summary(Path(f"runs/experiments/exp_0026/pairs/pair_{pid}.npz"))})
    args.out.mkdir(parents=True, exist_ok=True)
    res = {"label": "D-042 diagnostic (descriptive, not tested)", "t_hat_old": t_old, "pairs": rows, "git": git_state()}
    (args.out / "null_offset_diag.json").write_text(json.dumps(res, indent=2))
    for r in rows:
        t = r.get("test_exp_0019") or r.get("test_exp_0026")
        print(r["pair_id"], r["condition"], r["seed"], f"dt={r['dt']:+.3f}", f"dd={r['dd_val']:.4f}", json.dumps(t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
