"""Δ-CE solver, type N (spec §3.2, §6.1; amendment v1.4 E1): only f_new changes its decision.

F^N_κ = {z : m_n(z) ≤ −κ, m_o(z) ≥ κ}. Type O swaps the two margins.
Margins take (z (n, p), idx (n,)) and return (n,); idx is the point index of each row.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .search import SearchResult, SolverConfig, constrained_search

Margin = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def delta_ce(z0: torch.Tensor, m_old: Margin, m_new: Margin, cfg: SolverConfig, restart_seed: int,
             kind: str = "N") -> SearchResult:
    if kind == "O":
        m_old, m_new = m_new, m_old
    elif kind != "N":
        raise ValueError(f"unknown witness type {kind!r}")
    k = cfg.kappa

    def constraints(z, idx):  # g ≥ 0 ⇔ m_n ≤ −κ and m_o ≥ κ
        return torch.stack([-k - m_new(z, idx), m_old(z, idx) - k], 1)

    return constrained_search(z0, constraints, cfg, restart_seed)
