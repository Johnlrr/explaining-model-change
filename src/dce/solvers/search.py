"""Two-phase constrained search of amendment v1.4 E1, shared by Δ-CE and the ordinary CE.

Problem: min_z ‖z − z0‖² (+ μ‖z‖²) s.t. g_i(z) ≥ 0 for every constraint i (and box bounds on Tier 0).

Phase 1 (spec §6.1 + R1): R restarts of T Adam steps on Σ_i relu(−g_i(z)) + λ‖z − z0‖² (+ μ‖z‖²);
  λ is halved every T/n_lam_steps steps while the restart has found no feasible iterate; each restart
  keeps its closest feasible iterate. Points are independent: the loss is a sum over the batch and
  Adam is element-wise, so one optimiser over a (B, p) tensor equals B separate searches.
Phase 2 (v1.4): from each restart's best feasible iterate, SLSQP on the exact problem (gradients from
  torch). The SLSQP result replaces the phase-1 candidate only if SLSQP reports success and the
  point passes the hard feasibility test by forward pass.
Result: the feasible candidate with the smallest distance over restarts; NOT_FOUND if none, or if the
  distance exceeds r.

Constraint functions take (z, idx): z is (n, p) and idx (n,) gives the point index of each row, so
per-point models (Tier L) and shared models (CNN; idx ignored) use one interface.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

Tensor = torch.Tensor
Constraints = Callable[[Tensor, Tensor], Tensor]  # (z (n,p), idx (n,)) → g (n, c); feasible iff g ≥ 0


@dataclass(frozen=True)
class SolverConfig:
    kappa: float = 0.5
    R: int = 5
    T: int = 400
    lr: float = 0.05
    lam0: float = 1.0
    mu: float = 0.0
    sigma: float = 0.05
    r: float = float("inf")
    clamp: tuple[float, float] | None = None  # Tier 0: (0, 1); projected gradient (phase 1), bounds (phase 2)
    lam_schedule: str = "halve_every"   # v1.4 R1; "half_once" = original spec §6.1 (exp_0005 only)
    n_lam_steps: int = 8
    polish: bool = True                 # v1.4 phase 2; False for A1 (λ = 0) and for the original spec solver
    slsqp_maxiter: int = 100
    slsqp_ftol: float = 1e-10
    slsqp_tighten: float = 1e-6         # SLSQP constraints g ≥ ε so the boundary optimum passes the hard re-check


@dataclass
class SearchResult:
    z: Tensor          # (B, p) best feasible point; z0 where none was found
    distance: Tensor   # (B,) d(z0, z); +inf where none was found
    found: Tensor      # (B,) bool: a feasible point exists AND distance ≤ r
    seconds: float
    n_polished: int = 0        # phase-2 runs whose result was accepted
    n_polish_runs: int = 0


def _phase1(z0: Tensor, constraints: Constraints, cfg: SolverConfig, restart_seed: int):
    B = z0.shape[0]
    idx = torch.arange(B, device=z0.device)
    gen = torch.Generator(device="cpu").manual_seed(restart_seed)
    cand_z = z0.unsqueeze(0).repeat(cfg.R, 1, 1)
    cand_d = torch.full((cfg.R, B), float("inf"), dtype=z0.dtype, device=z0.device)
    for j in range(cfg.R):
        start_z = z0.clone()
        if j > 0:  # restart 1 starts at z0 without noise
            start_z = start_z + cfg.sigma * torch.randn(z0.shape, generator=gen, dtype=z0.dtype).to(z0.device)
            if cfg.clamp is not None:
                start_z = start_z.clamp(*cfg.clamp)
        z = start_z.requires_grad_(True)
        opt = torch.optim.Adam([z], lr=cfg.lr)
        lam = torch.full((B,), cfg.lam0, dtype=z0.dtype, device=z0.device)
        for t in range(1, cfg.T + 1):
            opt.zero_grad(set_to_none=True)
            u = z - z0
            loss = torch.relu(-constraints(z, idx)).sum(1) + lam * (u * u).sum(1)
            if cfg.mu:
                loss = loss + cfg.mu * (z * z).sum(1)
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                if cfg.clamp is not None:
                    z.clamp_(*cfg.clamp)
                ok = (constraints(z, idx) >= 0).all(1)
                d = (z - z0).norm(dim=1)
                better = ok & (d < cand_d[j])
                cand_d[j] = torch.where(better, d, cand_d[j])
                cand_z[j][better] = z[better]
                if cfg.lam_schedule == "half_once":
                    halve = t == cfg.T // 2
                elif cfg.lam_schedule == "halve_every":
                    halve = t % (cfg.T // cfg.n_lam_steps) == 0 and t < cfg.T
                else:
                    raise ValueError(f"unknown lam_schedule {cfg.lam_schedule!r}")
                if halve:  # halve λ where this restart has found nothing yet
                    lam = torch.where(torch.isfinite(cand_d[j]), lam, lam / 2)
    return cand_z, cand_d


def _polish_one(z_start: np.ndarray, z0: Tensor, i: int, constraints: Constraints, cfg: SolverConfig):
    """SLSQP on the exact problem for point i. Returns the polished z (numpy) or None.

    Evaluated in float64: SLSQP's ftol = 1e-10 is below float32 noise (≈ 1e-7), which made it stop
    at the iteration limit or fail its line search (D-022, bug fix found in the smoke test). The
    constraint function must therefore accept float64 inputs.
    """
    dtype, device = torch.float64, z0.device
    z0_np = z0.detach().cpu().double().numpy()
    idx = torch.tensor([i], device=device)
    cache: dict = {}

    def cons_eval(x):
        key = x.tobytes()
        if key not in cache:
            z = torch.tensor(x, dtype=dtype, device=device).unsqueeze(0).requires_grad_(True)
            g = constraints(z, idx)[0]
            jac = torch.stack([torch.autograd.grad(g[k], z, retain_graph=k + 1 < len(g))[0][0]
                               for k in range(len(g))])
            cache.clear()
            cache[key] = (g.detach().cpu().double().numpy() - cfg.slsqp_tighten,
                          jac.detach().cpu().double().numpy())
        return cache[key]

    def fun(x):
        u = x - z0_np
        return float(u @ u + cfg.mu * (x @ x))

    def jac(x):
        return 2 * (x - z0_np) + 2 * cfg.mu * x

    bounds = None if cfg.clamp is None else [cfg.clamp] * len(z0_np)
    res = minimize(fun, z_start, jac=jac, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": lambda x: cons_eval(x)[0],
                                 "jac": lambda x: cons_eval(x)[1]}],
                   options={"maxiter": cfg.slsqp_maxiter, "ftol": cfg.slsqp_ftol})
    if not res.success:
        return None
    x = res.x if bounds is None else np.clip(res.x, *cfg.clamp)
    return x


def constrained_search(z0: Tensor, constraints: Constraints, cfg: SolverConfig,
                       restart_seed: int) -> SearchResult:
    start = time.time()
    z0 = z0.detach()
    B = z0.shape[0]
    cand_z, cand_d = _phase1(z0, constraints, cfg, restart_seed)

    n_runs = n_ok = 0
    if cfg.polish:
        for j in range(cfg.R):
            for i in torch.nonzero(torch.isfinite(cand_d[j])).flatten().tolist():
                n_runs += 1
                x = _polish_one(cand_z[j, i].detach().cpu().double().numpy(), z0[i], i, constraints, cfg)
                if x is None:
                    continue
                z = torch.tensor(x, dtype=z0.dtype, device=z0.device).unsqueeze(0)
                with torch.no_grad():  # hard feasibility re-check by forward pass
                    ok = bool((constraints(z, torch.tensor([i], device=z0.device)) >= 0).all())
                if ok:  # v1.4 E1: success and feasible → replaces this restart's phase-1 candidate
                    n_ok += 1
                    cand_z[j, i] = z[0]
                    cand_d[j, i] = float((z[0] - z0[i]).norm())

    best_j = cand_d.argmin(0)
    ar = torch.arange(B, device=z0.device)
    best_d = cand_d[best_j, ar]
    best_z = cand_z[best_j, ar]
    found = torch.isfinite(best_d) & (best_d <= cfg.r)
    return SearchResult(z=best_z, distance=best_d, found=found, seconds=time.time() - start,
                        n_polished=n_ok, n_polish_runs=n_runs)
