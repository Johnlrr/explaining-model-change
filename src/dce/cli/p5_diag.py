"""Diagnosis of the failed vector criterion of the P5 check (exp_0023, p = 5): formula or numerics? Tier L, no test data.

On fresh linear instances (seed 71 000 + p, the exp_0023 generator), p ∈ {2, 5, 10}:
  1. KKT certificate of the closed form of Prop. 2(b) in float64: feasibility, stationarity residual, multiplier signs
     (the multiplier of (O) must be ≥ 0 in case B and the closed form of case A must satisfy (O)); for a strictly convex
     QP this certifies the global minimiser independently of any solver.
  2. An independent float64 solver (scipy SLSQP, ftol 1e-15) against the closed form.
  3. The experiments' two-phase solver (float32, as in exp_0023): vector error split along ŵ_o, v and the rest, and its
     relation to near-degeneracy (case boundary |d_n cos θ − d_o|, θ near 0 or π/2, ‖u*‖).
  4. The same solver in float64.

Usage: python -m dce.cli.p5_diag --config configs/experiments/exp_0037.yaml --out runs/experiments/exp_0037
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import minimize
from scipy.stats import spearmanr

from dce.cli.g0_tier0 import git_state
from dce.cli.p5_check import closed, instance
from dce.solvers.dce import delta_ce
from dce.solvers.search import SolverConfig


def kkt(q, c, kappa):
    """Residuals of the KKT system of min ½‖u‖² s.t. w_n·u ≤ −a, w_o·u ≥ −e (a = m_n(z0) + κ, e = m_o(z0) − κ)."""
    u, w_n, w_o = c["u"], q["w_n"], q["w_o"]
    a = q["w_n"] @ q["z0"] + q["b_n"] + kappa
    e = q["w_o"] @ q["z0"] + q["b_o"] - kappa
    g_n, g_o = w_n @ u + a, -(w_o @ u) - e            # constraints g ≤ 0
    if c["case"] == "A":
        lam_n, lam_o = float(-(u @ w_n) / (w_n @ w_n)), 0.0
    else:                                             # u + λ_n w_n − λ_o w_o = 0 on span{w_n, w_o}
        lam_n, lam_o = np.linalg.lstsq(np.stack([w_n, -w_o], 1), -u, rcond=None)[0]
    stat = np.linalg.norm(u + lam_n * w_n - lam_o * w_o) / np.linalg.norm(u)
    scale = np.linalg.norm(u) * (np.linalg.norm(w_n) + np.linalg.norm(w_o))
    return {"feas": max(g_n, g_o, 0.0) / scale, "stat": stat, "lam_n": float(lam_n), "lam_o": float(lam_o),
            "slack": abs(lam_n * g_n) / scale + abs(lam_o * g_o) / scale}


def slsqp(q, kappa):
    a = q["w_n"] @ q["z0"] + q["b_n"] + kappa
    e = q["w_o"] @ q["z0"] + q["b_o"] - kappa
    cons = [{"type": "ineq", "fun": lambda u: -(q["w_n"] @ u + a), "jac": lambda u: -q["w_n"]},
            {"type": "ineq", "fun": lambda u: q["w_o"] @ u + e, "jac": lambda u: q["w_o"]}]
    x0 = -a / (q["w_n"] @ q["w_n"]) * q["w_n"]
    r = minimize(lambda u: 0.5 * u @ u, x0, jac=lambda u: u, constraints=cons, method="SLSQP",
                 options={"ftol": 1e-15, "maxiter": 1000})
    return r.x


def solver(insts, kappa, scfg, seed, dtype):
    t = lambda a: torch.tensor(np.array(a), dtype=dtype)
    w_o, w_n = t([q["w_o"] for q in insts]), t([q["w_n"] for q in insts])
    b_o, b_n = t([q["b_o"] for q in insts]), t([q["b_n"] for q in insts])
    z0 = t([q["z0"] for q in insts])
    res = delta_ce(z0, lambda z, i: (w_o[i] * z).sum(1) + b_o[i], lambda z, i: (w_n[i] * z).sum(1) + b_n[i], scfg,
                   restart_seed=seed)
    return (res.z.double() - z0.double()).numpy(), res.found.numpy()


def errors(us, insts, cf):
    rows = []
    for u, q, c in zip(us, insts, cf):
        d, n = u - c["u"], np.linalg.norm(c["u"])
        rows.append({"rel": np.linalg.norm(d) / n, "along_wo": abs(d @ q["wo"]) / n, "along_v": abs(d @ q["v"]) / n,
                     "rest": np.linalg.norm(d - (d @ q["wo"]) * q["wo"] - (d @ q["v"]) * q["v"]) / n,
                     "dist_rel": abs(np.linalg.norm(u) - n) / n})
    return rows


def summary(rows, insts, cf, tol=1e-3):
    rel = np.array([r["rel"] for r in rows])
    bad = rel >= tol
    margin = np.array([abs(c["d_n"] * np.cos(q["theta"]) - c["d_o"]) / max(c["d_n"], 1e-12) for q, c in zip(insts, cf)])
    theta = np.array([q["theta"] for q in insts])
    norm = np.array([np.linalg.norm(c["u"]) for c in cf])
    caseB = np.array([c["case"] == "B" for c in cf])
    comp = {k: float(np.median([r[k] for r, b in zip(rows, bad) if b])) if bad.any() else None
            for k in ("along_wo", "along_v", "rest")}
    q = lambda x, m: [float(v) for v in np.quantile(x[m], [0.1, 0.5, 0.9])] if m.any() else None
    return {"tol": tol, "share_rel_below_tol": float((~bad).mean()), "max_rel": float(rel.max()),
            "max_dist_rel": float(max(r["dist_rel"] for r in rows)), "n_bad": int(bad.sum()),
            "bad_case_B_share": float(caseB[bad].mean()) if bad.any() else None, "all_case_B_share": float(caseB.mean()),
            "median_error_component_of_bad (relative to ‖u*‖)": comp,
            "case_margin_q10_50_90 bad / good": [q(margin, bad), q(margin, ~bad)],
            "theta_q10_50_90 bad / good": [q(theta, bad), q(theta, ~bad)],
            "norm_u_q10_50_90 bad / good": [q(norm, bad), q(norm, ~bad)],
            "spearman_rel_vs_norm_u": float(spearmanr(rel, norm).statistic),
            "spearman_rel_vs_case_margin": float(spearmanr(rel, margin).statistic)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(cfg.get("threads", 8))
    args.out.mkdir(parents=True, exist_ok=True)
    scfg = SolverConfig(**{**cfg["solver"], "kappa": cfg["kappa"], "r": float("inf")})
    kappa, N, start, rep = cfg["kappa"], cfg["n"], time.time(), {"git": git_state(), "dims": {}}
    for p in cfg["dims"]:
        rng = np.random.default_rng(cfg["seed_base"] + p)
        insts = [instance(rng, p, kappa, aligned=False, offset=False) for _ in range(N)]
        cf = [closed(q, kappa) for q in insts]
        K = [kkt(q, c, kappa) for q, c in zip(insts, cf)]
        lamB = [k["lam_o"] for k, c in zip(K, cf) if c["case"] == "B"]
        R = {"1_kkt_closed_form": {"max_feas": max(k["feas"] for k in K), "max_stat": max(k["stat"] for k in K),
                                   "max_slack": max(k["slack"] for k in K), "min_lam_n": min(k["lam_n"] for k in K),
                                   "min_lam_o_case_B": min(lamB) if lamB else None, "n_case_B": len(lamB)}}
        R["1_kkt_closed_form"]["certified"] = bool(R["1_kkt_closed_form"]["max_feas"] < 1e-12 and R["1_kkt_closed_form"]["max_stat"] < 1e-10
                                                   and R["1_kkt_closed_form"]["min_lam_n"] > 0 and (not lamB or min(lamB) >= 0))
        us = np.array([slsqp(q, kappa) for q in insts])
        R["2_scipy_slsqp_float64"] = summary(errors(us, insts, cf), insts, cf, tol=1e-6)
        for name, dt in (("3_two_phase_float32", torch.float32), ("4_two_phase_float64", torch.float64)):
            u, found = solver(insts, kappa, scfg, cfg["seed_base"] + p, dt)
            R[name] = {"found_share": float(found.mean()), **summary(errors(u, insts, cf), insts, cf)}
        rep["dims"][str(p)] = R
        print(json.dumps({"p": p, "kkt": R["1_kkt_closed_form"]["certified"],
                          **{k: R[k]["share_rel_below_tol"] for k in R if k[0] in "234"}}),
              f"({time.time() - start:.0f}s)", flush=True)
    rep["seconds"] = round(time.time() - start, 1)
    (args.out / "results.json").write_text(json.dumps(rep, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
