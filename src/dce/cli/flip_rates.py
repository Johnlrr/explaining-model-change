"""Top-factor change rates on re-solving (standard vs tight solve) and the maximal shift of each reported estimate.

VALIDATION only (the exp_0038 points and witnesses); descriptive; author's request 2026-10-22. Per pair of exp_0038:
- Δ-CE unprojected, projected with ∇m_old at z0, and projected with ∇m_old at the method's own witness z*;
- Grad-diff (analytic, no solver) projected at z*, which moves with the witness;
- CF-diff (deterministic counterfactual differencing): two ordinary CEs solved with the standard and the tight setting.
A change = the top |factor| differs between the two solves (points valid in both). CP95 upper bounds (Clopper–Pearson).
A flipped point moves a Recovery indicator by at most 1, so an estimate A − B (pair mean of per-pair means) moves by at
most U_A + U_B (U = CP95 upper bound of the change rate). Direction holds if the estimate stays on its side of the
decision boundary after that shift (0 for sign claims, ±0.10 for the tie and the projection tests; T for H8).

Usage: python -m dce.cli.flip_rates --config configs/experiments/exp_0038.yaml --out runs/experiments/exp_0038
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

from dce.analysis.projection import old_margin_gradients
from dce.baselines.b1_diff import b1_diff
from dce.baselines.b3_grad import b3_grad
from dce.cli.calibrate_r import solver_config
from dce.cli.g0_tier0 import git_state, load_config
from dce.cli.tangent_check import models
from dce.eval.records import from_direction
from dce.eval.tier0_margins import Tier0Margin

PROJ_PAIRS = (0, 36, 44)          # Shortcut and the grid α > 0 pairs of exp_0038


def proj(d, g):
    gg = (g * g).sum(1, keepdims=True)
    return d - np.where(gg > 0, (d * g).sum(1, keepdims=True) / np.where(gg > 0, gg, 1), 0) * g


def changes(a, b):
    return int((np.abs(a).argmax(1) != np.abs(b).argmax(1)).sum()), len(a)


def cp_upper(k, n):
    return 1.0 if n == 0 else float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    torch.set_num_threads(cfg.get("threads", 4))
    counts: dict[str, list[int]] = {}
    per_pair = {}

    def add(key, kn):
        c = counts.setdefault(key, [0, 0])
        c[0] += kn[0]
        c[1] += kn[1]

    for pid in cfg["pair_ids"]:
        z = np.load(args.out / "pairs" / f"pair_{pid}.npz")
        f_old, f_new, _ = models(cfg, pid, {})
        y, z0 = z["y"], z["z0"].astype(float)
        both = z["dce_std_found"] & z["dce_ref_found"]
        us, ur = z["dce_std_u"][both], z["dce_ref_u"][both]
        row = {"dce": changes(us, ur)}
        if pid in PROJ_PAIRS:
            g0 = old_margin_gradients(f_old, z0[both], y[both])
            gs = old_margin_gradients(f_old, z0[both] + us, y[both])
            gr = old_margin_gradients(f_old, z0[both] + ur, y[both])
            row["dce_proj_z0"] = changes(proj(us, g0), proj(ur, g0))
            row["dce_proj_zs"] = changes(proj(us, gs), proj(ur, gr))
            m_old, m_new = Tier0Margin(f_old, y[both]), Tier0Margin(f_new, y[both])
            b3 = from_direction(*b3_grad(torch.as_tensor(z0[both], dtype=torch.float32), m_old, m_new)).dphi.astype(float)
            row["b3_proj_zs"] = changes(proj(b3, gs), proj(b3, gr))
        m_old, m_new = Tier0Margin(f_old, y), Tier0Margin(f_new, y)
        base = solver_config(cfg, r=float(cfg["r"]))
        tight = replace(base, slsqp_maxiter=1000, slsqp_ftol=1e-14, slsqp_tighten=1e-7)
        seed = cfg["seeds"]["restart_base"] + pid
        va, da, _, _ = b1_diff(torch.as_tensor(z0, dtype=torch.float32), m_old, m_new, base, seed, deterministic=True)
        vb, db, _, _ = b1_diff(torch.as_tensor(z0, dtype=torch.float64), m_old, m_new, tight, seed, deterministic=True)
        ok = (va & vb).numpy()
        row["cf_diff"] = changes(da.double().numpy()[ok], db.double().numpy()[ok])
        for k, v in row.items():
            add(k, v)
        per_pair[pid] = {k: f"{v[0]}/{v[1]}" for k, v in row.items()}
        print(pid, per_pair[pid], flush=True)
    rates = {k: {"changes": c[0], "n": c[1], "rate": c[0] / c[1], "cp95_upper": cp_upper(*c)} for k, c in counts.items()}
    U = {k: v["cp95_upper"] for k, v in rates.items()} | {"b3": 0.0, "b3_proj_z0": 0.0}
    est = {  # name: (estimate, U_A, U_B, boundary)
        "H2a (Subgroup)": (-0.0911, U["dce"], U["b3"], 0.0),
        "Tie: Shortcut vs Grad-diff": (-0.5333, U["dce"], U["b3"], 0.10),
        "Tie: Shortcut vs CF-diff": (-0.1678, U["dce"], U["cf_diff"], 0.10),
        "Tie: Threshold vs Grad-diff": (0.3956, U["dce"], U["b3"], 0.10),
        "Tie: Threshold vs CF-diff": (-0.2933, U["dce"], U["cf_diff"], 0.10),
        "C1 (Threshold)": (0.3122, U["dce"], U["b3"], 0.0),
        "C3 (Shortcut, projected)": (-0.3044, U["dce_proj_z0"], U["b3_proj_z0"], 0.10),
        "H5d (grid, projected)": (-0.3525, U["dce_proj_z0"], U["b3_proj_z0"], 0.10),
    }
    out = {}
    for name, (e, ua, ub, bnd) in est.items():
        shift = ua + ub
        worst = abs(e) - shift
        out[name] = {"estimate": e, "max_shift": shift, "worst_abs": worst, "boundary": bnd, "direction_holds": bool(worst > bnd)}
    g00, gs0 = -0.1904, 0.0293
    s00 = U["dce_proj_z0"] + U["b3_proj_z0"]
    ss0 = U["dce_proj_zs"] + U["b3_proj_zs"]
    Tw = (abs(g00) - s00) - 2 * (abs(gs0) + ss0)
    out["H8 (T)"] = {"T": abs(g00) - 2 * abs(gs0), "shift_z0z0": s00, "shift_zsz0": ss0, "worst_T": Tw,
                     "direction_holds": bool(Tw > 0)}
    res = {"label": "top-factor change rates and maximal shifts (validation, descriptive)", "per_pair": per_pair,
           "rates": rates, "shifts": out, "git": git_state()}
    (args.out / "flip_rates.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"rates": rates, "shifts": out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
