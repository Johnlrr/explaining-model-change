"""Two-phase constrained search (amendments v1.4 E1, v1.5 F1–F2), shared by Δ-CE and the ordinary CE.

Problem: min_z ‖z − z0‖² (+ μ‖z‖²) s.t. g_i(z) ≥ 0 for every constraint i (and box bounds on Tier 0).

Phase 1 (spec §6.1 + R1): R restarts of T Adam steps on Σ_i relu(−g_i(z)) + λ‖z − z0‖² (+ μ‖z‖²);
  λ is halved every T/n_lam_steps steps while the restart has found no feasible iterate; each restart
  keeps its closest feasible iterate. Points are independent: the loss is a sum over the batch and
  Adam is element-wise, so one optimiser over a (B, p) tensor equals B separate searches.
Phase 2 (v1.5 F1–F2): SLSQP on the exact problem (float64, gradients from torch) from every start,
  feasible or not: z0 and the final phase-1 iterate of each restart (R + 1 starts). SLSQP targets the
  tightened constraints (κ' = κ + slsqp_tighten); each output is confirmed feasible with the original
  κ by a forward pass in the working dtype of z0 (float32 for the models used in evaluation).
Result: the smallest-distance point among all confirmed-feasible points (SLSQP outputs and the phase-1
  best feasible iterates, D-023); NOT_FOUND if none, or if the distance exceeds r.

Constraint functions take (z, idx): z is (n, p) and idx (n,) gives the point index of each row, so
per-point models (Tier L) and shared models (CNN; idx ignored) use one interface.
"""

from __future__ import annotations

import multiprocessing as mp
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
    polish: bool = True                 # phase 2; False for A1 (λ = 0) and for the original spec solver
    slsqp_maxiter: int = 100
    slsqp_ftol: float = 1e-10
    slsqp_tighten: float = 1e-4         # v1.5 F2: SLSQP targets κ' = κ + 1e-4 (g ≥ 1e-4)
    slsqp_lazy_jac: bool = True         # (A), D-027: Jacobian only when SLSQP asks; False = pre-(A)
    n_workers: int = 1                  # D-024: phase 2 parallelised over points (fork, 1 thread each)


@dataclass
class SearchResult:
    z: Tensor          # (B, p) best feasible point; z0 where none was found
    distance: Tensor   # (B,) d(z0, z); +inf where none was found
    found: Tensor      # (B,) bool: a feasible point exists AND distance ≤ r
    seconds: float
    n_polished: int = 0        # phase-2 runs whose output was confirmed feasible
    n_polish_runs: int = 0
    n_from_phase2: int = 0     # points whose final answer is a phase-2 output
    n_f64_ok_f32_rejected: int = 0  # D-024: feasible in float64 with κ but rejected by the float32 check
    cpu_seconds: float = 0.0   # parent + phase-1/phase-2 workers
    phase1_wall: float = 0.0
    phase2_cpu_per_point: Tensor | None = None  # (B,) CPU s of the SLSQP runs of each point (v1.6 G3)


def _restart_noise(z0: Tensor, cfg: SolverConfig, restart_seed: int) -> Tensor:
    """(R, B, p) start offsets; restart 1 has none. Drawn once for the whole batch, so phase 1 does not
    depend on how points are split across workers."""
    gen = torch.Generator(device="cpu").manual_seed(restart_seed)
    noise = torch.zeros((cfg.R, *z0.shape), dtype=z0.dtype)
    for j in range(1, cfg.R):
        noise[j] = cfg.sigma * torch.randn(z0.shape, generator=gen, dtype=z0.dtype)
    return noise.to(z0.device)


def _phase1(z0: Tensor, idx: Tensor, constraints: Constraints, cfg: SolverConfig, noise: Tensor):
    """Phase 1 for the points idx (rows of z0 and noise[:, ·] already restricted to them)."""
    B = z0.shape[0]
    cand_z = z0.unsqueeze(0).repeat(cfg.R, 1, 1)
    last_z = z0.unsqueeze(0).repeat(cfg.R, 1, 1)
    cand_d = torch.full((cfg.R, B), float("inf"), dtype=z0.dtype, device=z0.device)
    for j in range(cfg.R):
        start_z = z0 + noise[j]
        if j > 0 and cfg.clamp is not None:
            start_z = start_z.clamp(*cfg.clamp)
        z = start_z.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=cfg.lr)
        lam = torch.full((B,), cfg.lam0, dtype=z0.dtype, device=z0.device)
        # One forward per step: g at z_{t−1} gives both the feasibility record of the iterate produced
        # by step t−1 and the loss of step t. Same checked iterates and λ timing as checking after
        # each step; z_0 itself is not recorded.
        for t in range(1, cfg.T + 2):
            opt.zero_grad(set_to_none=True)
            last = t == cfg.T + 1
            with torch.set_grad_enabled(not last):
                g = constraints(z, idx)
            if t > 1:
                with torch.no_grad():
                    ok = (g >= 0).all(1)
                    d = (z - z0).norm(dim=1)
                    better = ok & (d < cand_d[j])
                    cand_d[j] = torch.where(better, d, cand_d[j])
                    cand_z[j][better] = z.detach()[better]
                    s_ = t - 1  # the step that produced this iterate
                    if cfg.lam_schedule == "half_once":
                        halve = s_ == cfg.T // 2
                    elif cfg.lam_schedule == "halve_every":
                        halve = s_ % max(1, cfg.T // cfg.n_lam_steps) == 0 and s_ < cfg.T
                    else:
                        raise ValueError(f"unknown lam_schedule {cfg.lam_schedule!r}")
                    if halve:  # halve λ where this restart has found nothing yet
                        lam = torch.where(torch.isfinite(cand_d[j]), lam, lam / 2)
            if last:
                break
            u = z - z0
            loss = torch.relu(-g).sum(1) + lam * (u * u).sum(1)
            if cfg.mu:
                loss = loss + cfg.mu * (z * z).sum(1)
            loss.sum().backward()
            opt.step()
            if cfg.clamp is not None:
                with torch.no_grad():
                    z.clamp_(*cfg.clamp)
        last_z[j] = z.detach()
    return cand_z, cand_d, last_z


def _phase1_chunk(chunk: list[int]):
    z0, constraints, cfg, noise = _CTX
    c0 = time.process_time()
    idx = torch.tensor(chunk, device=z0.device)
    out = _phase1(z0[idx], idx, constraints, cfg, noise[:, idx])
    return out, time.process_time() - c0


def run_phase1(z0: Tensor, constraints: Constraints, cfg: SolverConfig, restart_seed: int):
    """Phase 1 for all points; split over cfg.n_workers forked workers (1 thread each, D-024)."""
    global _CTX
    noise = _restart_noise(z0, cfg, restart_seed)
    B = z0.shape[0]
    if cfg.n_workers <= 1 or B < 2:
        return (*_phase1(z0, torch.arange(B, device=z0.device), constraints, cfg, noise), 0.0)
    chunks = [c.tolist() for c in np.array_split(np.arange(B), min(cfg.n_workers, B)) if len(c)]
    _CTX = (z0, constraints, cfg, noise)
    try:
        with mp.get_context("fork").Pool(len(chunks), initializer=torch.set_num_threads,
                                          initargs=(1,)) as pool:
            res = pool.map(_phase1_chunk, chunks, chunksize=1)
    finally:
        _CTX = None
    cand_z = torch.cat([r[0][0] for r in res], dim=1)
    cand_d = torch.cat([r[0][1] for r in res], dim=1)
    last_z = torch.cat([r[0][2] for r in res], dim=1)
    return cand_z, cand_d, last_z, sum(c for _, c in res)


def _polish_one(z_start: np.ndarray, z0: Tensor, i: int, constraints: Constraints, cfg: SolverConfig):
    """SLSQP on the exact problem for point i, from z_start (feasible or not). Returns z (numpy).

    Evaluated in float64: SLSQP's ftol = 1e-10 is below float32 noise (≈ 1e-7), which made it stop
    at the iteration limit or fail its line search (D-022, bug fix found in the smoke test). The
    constraint function must therefore accept float64 inputs.
    """
    dtype, device = torch.float64, z0.device
    z0_np = z0.detach().cpu().double().numpy()
    idx = torch.tensor([i], device=device)
    cache: dict = {}

    def full_eval(x):  # values and Jacobian (one forward, one backward per constraint)
        key = x.tobytes()
        if key not in cache or cache[key][1] is None:
            z = torch.tensor(x, dtype=dtype, device=device).unsqueeze(0).requires_grad_(True)
            g = constraints(z, idx)[0]
            jac = torch.stack([torch.autograd.grad(g[k], z, retain_graph=k + 1 < len(g))[0][0]
                               for k in range(len(g))])
            cache.clear()
            cache[key] = (g.detach().cpu().double().numpy() - cfg.slsqp_tighten,
                          jac.detach().cpu().double().numpy())
        return cache[key]

    def value_eval(x):  # (A), D-027: line-search points need values only → forward without grad
        key = x.tobytes()
        if key not in cache:
            with torch.no_grad():
                g = constraints(torch.tensor(x, dtype=dtype, device=device).unsqueeze(0), idx)[0]
            cache.clear()
            cache[key] = (g.cpu().double().numpy() - cfg.slsqp_tighten, None)
        return cache[key][0]

    if cfg.slsqp_lazy_jac:
        cons_fun, cons_jac = value_eval, lambda x: full_eval(x)[1]
    else:  # pre-(A) behaviour: every evaluation also builds the Jacobian
        cons_fun, cons_jac = (lambda x: full_eval(x)[0]), (lambda x: full_eval(x)[1])

    def fun(x):
        u = x - z0_np
        return float(u @ u + cfg.mu * (x @ x))

    def jac(x):
        return 2 * (x - z0_np) + 2 * cfg.mu * x

    bounds = None if cfg.clamp is None else [cfg.clamp] * len(z0_np)
    res = minimize(fun, z_start, jac=jac, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": cons_fun, "jac": cons_jac}],
                   options={"maxiter": cfg.slsqp_maxiter, "ftol": cfg.slsqp_ftol})
    # v1.5 F1: any output is a candidate; feasibility is confirmed by the caller (D-023)
    x = res.x if bounds is None else np.clip(res.x, *cfg.clamp)
    return x


_CTX: tuple | None = None  # set in the parent before forking the phase-2 workers


def _phase2_point(i: int):
    z0, constraints, cfg, starts = _CTX
    c0 = time.process_time()
    out = []
    idx = torch.tensor([i], device=z0.device)
    for start_z in starts[:, i]:
        x = _polish_one(start_z.detach().cpu().double().numpy(), z0[i], i, constraints, cfg)
        with torch.no_grad():  # float64 view with the original κ (for the D-024 mismatch count)
            g = constraints(torch.tensor(x, dtype=torch.float64, device=z0.device).unsqueeze(0), idx)
        out.append((x, bool((g >= 0).all())))
    return out, time.process_time() - c0


def run_phase2(z0: Tensor, constraints: Constraints, cfg: SolverConfig, starts: Tensor):
    """SLSQP for every point i from every starts[s, i]; returns per-point [(x, f64_ok)], worker CPU s."""
    global _CTX
    _CTX = (z0, constraints, cfg, starts)
    try:
        if cfg.n_workers <= 1:
            res = [_phase2_point(i) for i in range(z0.shape[0])]
        else:
            with mp.get_context("fork").Pool(cfg.n_workers, initializer=torch.set_num_threads,
                                              initargs=(1,)) as pool:
                res = pool.map(_phase2_point, range(z0.shape[0]), chunksize=1)
    finally:
        _CTX = None
    return [o for o, _ in res], [c for _, c in res]


def constrained_search(z0: Tensor, constraints: Constraints, cfg: SolverConfig,
                       restart_seed: int) -> SearchResult:
    start, cpu0 = time.time(), time.process_time()
    z0 = z0.detach()
    B = z0.shape[0]
    t1 = time.time()
    cand_z, cand_d, last_z, worker_cpu = run_phase1(z0, constraints, cfg, restart_seed)
    phase1_wall = time.time() - t1
    p2_cpu = None

    n_runs = n_ok = n_mismatch = 0
    from_p2 = torch.zeros(B, dtype=torch.bool, device=z0.device)
    if cfg.polish:
        outs, cpu2 = run_phase2(z0, constraints, cfg, torch.cat([z0.unsqueeze(0), last_z]))
        worker_cpu += sum(cpu2)
        p2_cpu = torch.tensor(cpu2, dtype=torch.float64)
        xs = torch.tensor(np.stack([x for o in outs for x, _ in o]), dtype=z0.dtype, device=z0.device)
        f64_ok = torch.tensor([ok for o in outs for _, ok in o], device=z0.device)
        n_starts = cfg.R + 1
        idx = torch.arange(B, device=z0.device).repeat_interleave(n_starts)
        with torch.no_grad():  # F2: confirm with the original κ in the working dtype, batched
            ok = (constraints(xs, idx) >= 0).all(1)
        n_runs, n_ok = len(xs), int(ok.sum())
        n_mismatch = int((f64_ok & ~ok).sum())
        d = (xs - z0[idx]).norm(dim=1)
        d = torch.where(ok, d, torch.full_like(d, float("inf"))).view(B, n_starts)
        k = d.argmin(1)
        ar = torch.arange(B, device=z0.device)
        p2_d = d[ar, k]
        p2_z = torch.where(torch.isfinite(p2_d)[:, None], xs.view(B, n_starts, -1)[ar, k], z0)
        from_p2 = p2_d < cand_d.min(0).values
        cand_z = torch.cat([cand_z, p2_z.unsqueeze(0)])
        cand_d = torch.cat([cand_d, p2_d.unsqueeze(0)])

    best_j = cand_d.argmin(0)
    ar = torch.arange(B, device=z0.device)
    best_d = cand_d[best_j, ar]
    best_z = cand_z[best_j, ar]
    found = torch.isfinite(best_d) & (best_d <= cfg.r)
    return SearchResult(z=best_z, distance=best_d, found=found, seconds=time.time() - start,
                        n_polished=n_ok, n_polish_runs=n_runs, n_from_phase2=int((from_p2 & found).sum()),
                        n_f64_ok_f32_rejected=n_mismatch,
                        cpu_seconds=time.process_time() - cpu0 + worker_cpu,
                        phase1_wall=phase1_wall, phase2_cpu_per_point=p2_cpu)
