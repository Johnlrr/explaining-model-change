"""Shared penalised-search loop of spec §6.1, used by Δ-CE (§6.1) and the ordinary CE (§6.2).

For every point the loop keeps the closest iterate that satisfies the hard feasibility test, over
R restarts of T Adam steps. Points are independent: the loss is a sum over the batch and Adam is
element-wise, so one optimiser over a (B, p) tensor equals B separate searches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

Tensor = torch.Tensor


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
    lam_schedule: str = "half_once"   # "half_once": spec §6.1; "halve_every": halve every T/n_lam_steps while not found
    n_lam_steps: int = 8
    halve_while_infeasible: bool = False
    lr_cosine: bool = False           # NOT in spec §6.1; diagnostic only (lr is locked)
    polish_steps: int = 0             # NOT in spec §6.1; diagnostic only: extra steps from the best witness
    polish_lr: float = 0.01
    clamp: tuple[float, float] | None = None  # Tier 0: (0, 1) projected gradient


@dataclass
class SearchResult:
    z: Tensor          # (B, p) best feasible iterate; z0 where none was found
    distance: Tensor   # (B,) d(z0, z); +inf where none was found
    found: Tensor      # (B,) bool: a feasible iterate exists AND distance ≤ r
    seconds: float


def penalised_search(
    z0: Tensor,
    hinge: Callable[[Tensor], Tensor],
    feasible: Callable[[Tensor], Tensor],
    cfg: SolverConfig,
    restart_seed: int,
) -> SearchResult:
    """hinge(z) → (B,) constraint penalty; feasible(z) → (B,) bool hard test."""
    import time

    start = time.time()
    z0 = z0.detach()
    B = z0.shape[0]
    best_z = z0.clone()
    best_d = torch.full((B,), float("inf"), dtype=z0.dtype, device=z0.device)
    gen = torch.Generator(device="cpu").manual_seed(restart_seed)

    for j in range(cfg.R):
        start_z = z0.clone()
        if j > 0:  # restart 1 starts at z0 without noise
            start_z = start_z + cfg.sigma * torch.randn(z0.shape, generator=gen, dtype=z0.dtype).to(z0.device)
            if cfg.clamp is not None:
                start_z = start_z.clamp(*cfg.clamp)
        z = start_z.requires_grad_(True)
        opt = torch.optim.Adam([z], lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.T) if cfg.lr_cosine else None
        lam = torch.full((B,), cfg.lam0, dtype=z0.dtype, device=z0.device)
        if j == 0:
            best_lam = lam.clone()
        found_this = torch.zeros(B, dtype=torch.bool, device=z0.device)
        for t in range(1, cfg.T + 1):
            opt.zero_grad(set_to_none=True)
            u = z - z0
            loss = hinge(z) + lam * (u * u).sum(1)
            if cfg.mu:
                loss = loss + cfg.mu * (z * z).sum(1)
            loss.sum().backward()
            opt.step()
            if sched is not None:
                sched.step()
            with torch.no_grad():
                if cfg.clamp is not None:
                    z.clamp_(*cfg.clamp)
                ok = feasible(z)
                d = (z - z0).norm(dim=1)
                better = ok & (d < best_d)
                best_d = torch.where(better, d, best_d)
                best_z[better] = z[better]
                best_lam = torch.where(better, lam, best_lam)
                found_this |= ok
                if cfg.lam_schedule == "half_once":
                    halve = t == cfg.T // 2
                elif cfg.lam_schedule == "halve_every":
                    halve = t % (cfg.T // cfg.n_lam_steps) == 0 and t < cfg.T
                else:
                    raise ValueError(f"unknown lam_schedule {cfg.lam_schedule!r}")
                if halve and cfg.lam_schedule == "halve_every" and cfg.halve_while_infeasible:
                    lam = torch.where(ok, lam, lam / 2)  # current iterate infeasible → λ too large
                elif halve:  # halve λ where this restart has found nothing yet
                    lam = torch.where(found_this | torch.isfinite(best_d), lam, lam / 2)
    if cfg.polish_steps:  # diagnostic: from the best witness, λ fixed, lr cosine-decayed to 0
        has = torch.isfinite(best_d)
        z = best_z.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=cfg.polish_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.polish_steps)
        for _ in range(cfg.polish_steps):
            opt.zero_grad(set_to_none=True)
            u = z - z0
            (hinge(z) + best_lam * (u * u).sum(1)).sum().backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                if cfg.clamp is not None:
                    z.clamp_(*cfg.clamp)
                d = (z - z0).norm(dim=1)
                better = has & feasible(z) & (d < best_d)
                best_d = torch.where(better, d, best_d)
                best_z[better] = z[better]
    found = torch.isfinite(best_d) & (best_d <= cfg.r)
    return SearchResult(z=best_z, distance=best_d, found=found, seconds=time.time() - start)
