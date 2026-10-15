"""Δ-CE solver, type N (spec §3.2, §6.1): only f_new changes its decision.

F^N_κ = {z : m_n(z) ≤ −κ, m_o(z) ≥ κ}. Type O swaps the two margins.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .search import SearchResult, SolverConfig, penalised_search

Margin = Callable[[torch.Tensor], torch.Tensor]


def delta_ce(z0: torch.Tensor, m_old: Margin, m_new: Margin, cfg: SolverConfig, restart_seed: int,
             kind: str = "N") -> SearchResult:
    if kind == "O":
        m_old, m_new = m_new, m_old
    elif kind != "N":
        raise ValueError(f"unknown witness type {kind!r}")
    k = cfg.kappa

    def hinge(z):
        return torch.relu(m_new(z) + k) + torch.relu(k - m_old(z))

    def feasible(z):
        return (m_new(z) <= -k) & (m_old(z) >= k)

    return penalised_search(z0, hinge, feasible, cfg, restart_seed)
