"""P5 numerical check on Tier L (research/theory/P5_rotation_offset.md §8). Linear margins only; no test data.

Four checks, each against the locked two-phase solver (dce.solvers.dce.delta_ce, Tier L settings of configs/tiers/tierL.yaml,
no distance budget), on instances drawn from a new seed set (70 000 + …, disjoint from the H4 seeds 0–1999):
  1. Rotation, Δ-CE closed form (P5 (b)2, both cases): relative error of u* < 1e-3 on ≥ 99 % of instances.
  2. Rotation, B3 and B1-det algebra (P5 (b)1, (b)3): tan(θ/2) ratio and the B1-det components, to 1e-9.
  3. Factor-aligned recovery (ŵ_o = e_1, v = e_2): solver recovers k* = 1 iff closed form says so, away from ties; and
     the criterion of Corollary 5.1 (θ > π/4 or d_o(sinθ + cosθ) < d_n) agrees with |u_2*| > |u_1*| in closed form.
  4. Offset (P5 (a)): w_n = λ w_o. B3 direction is zero; the solver finds a witness iff d_n ≤ d_o, and then u* = −d_n ŵ_o.

Usage: python -m dce.cli.p5_check --config configs/experiments/exp_0023.yaml --out runs/experiments/exp_0023
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from dce.cli.g0_tier0 import git_state
from dce.solvers.dce import delta_ce
from dce.solvers.search import SolverConfig

TOL_REL = 1e-3
TOL_ALG = 1e-9
TIE = 1e-6


def unit_orth(rng, w, p):
    v = rng.normal(size=p)
    v -= (v @ w) * w
    return v / np.linalg.norm(v)


def instance(rng, p, kappa, aligned: bool, offset: bool):
    """Draw (ŵ_o, v, θ, ρ_o, ρ_n, b_o, b_n, z0) with z0 valid by construction (see the sampling note below)."""
    wo = np.eye(p)[0] if aligned else rng.normal(size=p)
    wo = wo / np.linalg.norm(wo)
    v = np.eye(p)[1] if aligned else unit_orth(rng, wo, p)
    theta = 0.0 if offset else rng.uniform(0.05, 1.5)
    rho_o = float(np.linalg.norm(rng.normal(size=p))) if not aligned else float(rng.uniform(0.5, 2.0))
    rho_n = rho_o * float(rng.uniform(0.5, 2.0))
    # Sampling note (not a criterion): biases are drawn so that z0 is valid, m_o(z0), m_n(z0) ∈ [κ, κ + 2].
    # Plain rejection on N(0, 1) biases failed for some draws (valid region of probability ≈ 0 under N(0, I)).
    z0 = rng.normal(size=p)
    wn_hat = np.cos(theta) * wo + np.sin(theta) * v
    w_o, w_n = rho_o * wo, rho_n * wn_hat
    b_o = kappa + rng.uniform(0.0, 2.0) - w_o @ z0
    b_n = kappa + rng.uniform(0.0, 2.0) - w_n @ z0
    assert w_o @ z0 + b_o >= kappa and w_n @ z0 + b_n >= kappa
    return {"wo": wo, "v": v, "theta": theta, "rho_o": rho_o, "rho_n": rho_n, "w_o": w_o, "w_n": w_n, "b_o": b_o, "b_n": b_n,
                "z0": z0, "wn_hat": wn_hat}


def closed(q, kappa):
    m_o0 = q["w_o"] @ q["z0"] + q["b_o"]
    m_n0 = q["w_n"] @ q["z0"] + q["b_n"]
    d_n = (m_n0 + kappa) / q["rho_n"]
    d_o = (m_o0 - kappa) / q["rho_o"]
    d_o_ord = (m_o0 + kappa) / q["rho_o"]
    th = q["theta"]
    if d_n * np.cos(th) <= d_o:
        u = -d_n * q["wn_hat"]
        case = "A"
    else:
        u = -d_o * q["wo"] - (d_n - d_o * np.cos(th)) / np.sin(th) * q["v"]
        case = "B"
    return {"d_n": d_n, "d_o": d_o, "d_o_ord": d_o_ord, "u": u, "case": case}


def solve_batch(insts, kappa, scfg, seed):
    """Solver outputs (float32 evaluation, as in H4) and an independent float64 feasibility check of each output.
    The solver confirms feasibility in float32; at points far from z0 (‖u‖ ~ 1e7, unclamped Tier L) that check can accept
    points that are infeasible in exact arithmetic. Such outputs are reported as spurious and not counted as witnesses."""
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    w_o, w_n = t([q["w_o"] for q in insts]), t([q["w_n"] for q in insts])
    b_o, b_n = t([q["b_o"] for q in insts]), t([q["b_n"] for q in insts])
    z0 = t([q["z0"] for q in insts])
    m_o = lambda z, idx: (w_o[idx] * z).sum(1) + b_o[idx]
    m_n = lambda z, idx: (w_n[idx] * z).sum(1) + b_n[idx]
    res = delta_ce(z0, m_o, m_n, scfg, restart_seed=seed)
    u = res.z.double().numpy() - z0.double().numpy()
    z = res.z.double()
    wo64, wn64 = torch.tensor(np.array([q["w_o"] for q in insts])), torch.tensor(np.array([q["w_n"] for q in insts]))
    bo64, bn64 = torch.tensor([q["b_o"] for q in insts]), torch.tensor([q["b_n"] for q in insts])
    mo = (wo64 * z).sum(1) + bo64
    mn = (wn64 * z).sum(1) + bn64
    feas64 = ((mo >= kappa - 1e-9) & (mn <= -kappa + 1e-9)).numpy()
    found = res.found.numpy()
    return u, found & feas64, found & ~feas64


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(cfg.get("threads", 8))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    scfg = SolverConfig(**{**cfg["solver"], "kappa": cfg["kappa"], "r": float("inf")})
    kappa, N, start = cfg["kappa"], cfg["n"], time.time()
    report = {"git": git_state(), "n_per_dim": N, "seed_base": cfg["seed_base"], "dims": {}}
    for p in cfg["dims"]:
        rng = np.random.default_rng(cfg["seed_base"] + p)
        R = {}
        # 1. rotation, Δ-CE closed form
        insts = [instance(rng, p, kappa, aligned=False, offset=False) for _ in range(N)]
        cf = [closed(q, kappa) for q in insts]
        u, found, spur1 = solve_batch(insts, kappa, scfg, seed=cfg["seed_base"] + p)
        rel = np.array([np.linalg.norm(u[i] - c["u"]) / np.linalg.norm(c["u"]) for i, c in enumerate(cf)])
        dist_rel = np.array([abs(np.linalg.norm(u[i]) - np.linalg.norm(c["u"])) / np.linalg.norm(c["u"]) for i, c in enumerate(cf)])
        R["1_rotation_delta_ce_closed_form"] = {
            "found_share": float(found.mean()), "share_rel_err_below_1e-3": float((rel < TOL_REL).mean()),
            "distance_only_share_below_1e-3 (H4-style, diagnostic)": float((dist_rel < TOL_REL).mean()),
            "max_distance_rel_err": float(dist_rel.max()), "spurious": int(spur1.sum()),
            "max_rel_err": float(rel.max()), "share_case_B": float(np.mean([c["case"] == "B" for c in cf])),
            "pass": bool(found.all() and (rel < TOL_REL).mean() >= 0.99)}
        # 2. B3 tan(θ/2) and B1-det components (algebra from definitions)
        err_tan, err_b1 = [], []
        for q, c in zip(insts, cf):
            g = q["wn_hat"] - q["wo"]
            err_tan.append(abs(abs(g @ q["wo"]) / abs(g @ q["v"]) - np.tan(q["theta"] / 2)))
            u_n = -(q["w_n"] @ q["z0"] + q["b_n"] + kappa) / (q["w_n"] @ q["w_n"]) * q["w_n"]
            u_o = -(q["w_o"] @ q["z0"] + q["b_o"] + kappa) / (q["w_o"] @ q["w_o"]) * q["w_o"]
            e = u_n - u_o
            pred_o, pred_v = c["d_o_ord"] - c["d_n"] * np.cos(q["theta"]), -c["d_n"] * np.sin(q["theta"])
            err_b1.append(max(abs(e @ q["wo"] - pred_o), abs(e @ q["v"] - pred_v)))
        R["2_b3_and_b1det_algebra"] = {"max_abs_err_tan": float(max(err_tan)), "max_abs_err_b1det": float(max(err_b1)),
                                       "pass": bool(max(err_tan) < TOL_ALG and max(err_b1) < TOL_ALG)}
        # 3. factor-aligned recovery and Corollary 5.1
        al = [instance(rng, p, kappa, aligned=True, offset=False) for _ in range(N)]
        cfa = [closed(q, kappa) for q in al]
        ua, _, _ = solve_batch(al, kappa, scfg, seed=cfg["seed_base"] + 500 + p)
        pred, sol, corr, nontie = [], [], [], []
        for q, c, uu in zip(al, cfa, ua):
            u1, u2 = c["u"][0], c["u"][1]
            pred.append(abs(u2) > abs(u1))
            corr.append((q["theta"] > np.pi / 4) or (c["d_o"] * (np.sin(q["theta"]) + np.cos(q["theta"])) < c["d_n"]))
            nontie.append(abs(abs(u1) - abs(u2)) > TIE and abs(abs(uu[0]) - abs(uu[1])) > TIE)
            sol.append(abs(uu[1]) > abs(uu[0]))
        nt = np.array(nontie)
        bad3 = [{"i": int(i), "closed_gap_u1_minus_u2": float(abs(cfa[i]["u"][0]) - abs(cfa[i]["u"][1])),
                 "solver_gap": float(abs(ua[i][0]) - abs(ua[i][1])), "closed_pred": bool(pred[i]), "solver": bool(sol[i])}
                for i in np.where(nt & (np.array(sol) != np.array(pred)))[0]]
        R["3_factor_aligned_recovery"] = {
            "mismatches": bad3, "n_nontie": int(nt.sum()), "solver_agrees_with_closed_form": float(np.mean(np.array(sol)[nt] == np.array(pred)[nt])),
            "corollary_agrees_with_closed_form": float(np.mean(np.array(corr) == np.array(pred))),
            "pass": bool(np.mean(np.array(sol)[nt] == np.array(pred)[nt]) == 1.0 and np.all(np.array(corr) == np.array(pred)))}
        # 4. offset (a): B3 zero; witness iff d_n ≤ d_o; u* = −d_n ŵ_o
        off = [instance(rng, p, kappa, aligned=False, offset=True) for _ in range(N)]
        cfo = [closed(q, kappa) for q in off]
        uo, fo, spur4 = solve_batch(off, kappa, scfg, seed=cfg["seed_base"] + 900 + p)
        exists = np.array([c["d_n"] <= c["d_o"] for c in cfo])
        sep = np.array([abs(c["d_n"] - c["d_o"]) > TIE for c in cfo])
        b3_zero = max(np.linalg.norm(q["wn_hat"] - q["wo"]) for q in off)
        rel4 = np.array([np.linalg.norm(uo[i] - (-c["d_n"] * q["wo"])) / max(np.linalg.norm(c["d_n"] * q["wo"]), 1e-12)
                         for i, (q, c) in enumerate(zip(off, cfo))])
        bad4 = [{"i": int(i), "d_n": float(cfo[i]["d_n"]), "d_o": float(cfo[i]["d_o"]), "found": bool(fo[i])}
                for i in np.where(sep & (fo != exists))[0]]
        share4 = float(np.mean(rel4[fo & exists] < TOL_REL)) if (fo & exists).any() else None
        R["4_offset"] = {
            "b3_direction_max_norm": float(b3_zero), "n_separated": int(sep.sum()), "spurious_float32_accepts": int(spur4.sum()),
            "existence_mismatches": bad4, "existence_agrees": float(np.mean(fo[sep] == exists[sep])),
            "share_rel_err_below_1e-3 (on witnesses)": share4,
            "distance_only_share_below_1e-3 (diagnostic)": float(np.mean(np.abs(np.linalg.norm(uo[fo & exists], axis=1)
                                                             - np.array([cfo[i]["d_n"] for i in np.where(fo & exists)[0]])) < TOL_REL * np.array([cfo[i]["d_n"] for i in np.where(fo & exists)[0]]))) if (fo & exists).any() else None,
            "pass": bool(b3_zero < TOL_ALG and np.all(fo[sep] == exists[sep]) and share4 is not None and share4 >= 0.99)}
        report["dims"][str(p)] = R
        print(json.dumps({"dim": p, **{k: v["pass"] for k, v in R.items()}}), f"({time.time() - start:.0f}s)", flush=True)
    report["all_pass"] = all(v["pass"] for d in report["dims"].values() for v in d.values())
    report["seconds"] = round(time.time() - start, 1)
    (out / "results.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({"all_pass": report["all_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
