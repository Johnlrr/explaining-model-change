"""Ordinary counterfactual of a single model (spec §6.2): same loop as §6.1, one margin.

Feasible set {z : m(z) ≤ −κ}. Used by B1, B1-det, ablation A2 and the r calibration (§6.3).
"""

from __future__ import annotations

import torch

from .dce import Margin
from .search import SearchResult, SolverConfig, penalised_search


def ordinary_ce(z0: torch.Tensor, m: Margin, cfg: SolverConfig, restart_seed: int) -> SearchResult:
    k = cfg.kappa
    return penalised_search(z0, lambda z: torch.relu(m(z) + k), lambda z: m(z) <= -k, cfg, restart_seed)
