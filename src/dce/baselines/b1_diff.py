"""B1 / B1-det — CE differencing (Artelt et al. Definition 3; spec §6.4).

Two independent ordinary CEs, z'_o = CE(z0; f_old) and z'_n = CE(z0; f_new), with the shared two-phase
solver (v1.4 E2). Δφ̃ = φ̃(z'_n) − φ̃(z'_o); valid iff both CEs are found within r.
B1: R = 5, restart seeds seed + 0 / seed + 1. B1-det: R = 1, no noise, both from z0.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from dce.solvers.dce import Margin
from dce.solvers.ordinary_ce import ordinary_ce
from dce.solvers.search import SearchResult, SolverConfig


def b1_diff(z0: torch.Tensor, m_old: Margin, m_new: Margin, cfg: SolverConfig, seed: int,
            deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, SearchResult, SearchResult]:
    if deterministic:
        cfg = replace(cfg, R=1)  # restart 1 has no noise, so both CEs start at z0
    ce_o = ordinary_ce(z0, m_old, cfg, restart_seed=seed + 0)
    ce_n = ordinary_ce(z0, m_new, cfg, restart_seed=seed + 1)
    valid = ce_o.found & ce_n.found
    return valid, ce_n.z - ce_o.z, ce_o, ce_n
