"""Ordinary counterfactual of a single model (spec §6.2; amendment v1.4 E1–E2): same two-phase
search as Δ-CE, one constraint m(z) ≤ −κ. Used by B1, B1-det, A2, A3 and the r calibration (§6.3).
"""

from __future__ import annotations

import torch

from .dce import Margin
from .search import SearchResult, SolverConfig, constrained_search


def ordinary_ce(z0: torch.Tensor, m: Margin, cfg: SolverConfig, restart_seed: int) -> SearchResult:
    k = cfg.kappa
    return constrained_search(z0, lambda z, idx: (-k - m(z, idx)).unsqueeze(1), cfg, restart_seed)
