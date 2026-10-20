"""v1.10 (D-043): H8 locality 2×2, the T-C slack function, and the locked-split guard.

H8 2×2: the projection removes the component along ŵ_o(p) = ∇m_o(p) from both explanations, with p ∈ {z0, z*};
B3 is computed at q ∈ {z0, z*}. Cell (p, q). Every cell is scored on the same support set S = the queries on which
Δ-CE found a witness (so the gap is not confounded with the Δ-CE found-rate). Cell (z0, z0) is the C3/H5d analysis
restricted to S (the reference); the registered H8 cell is (z*, z0): local projection direction, original B3 at z0.
Cells (z*, z*) and (z0, z*) are descriptive.

T-C: s(z) = min(−κ − m_new(z), m_old(z) − κ); z is a type-N witness iff s(z) ≥ 0.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from dce.analysis.projection import project
from dce.eval.metrics import MethodOutput, recovery_indicator

CELLS = (("z0", "z0"), ("z0", "zs"), ("zs", "z0"), ("zs", "zs"))   # (projection at, B3 at)
REFERENCE, REGISTERED = ("z0", "z0"), ("zs", "z0")                   # v1.10 §3.3 (author, Q2)

# Test data already used once (D-011): clean_splits(1234)["test"], C1–C3 (2027), T5 (2028), grid H5 (7000 + 20..51).
LOCKED_SEEDS = frozenset({2027, 2028, *range(7020, 7052)})


def check_fresh_seed(seed: int) -> int:
    if seed in LOCKED_SEEDS:
        raise SystemExit(f"data seed {seed} belongs to a locked test split (D-011); use a fresh v1.10 seed")
    return seed


def slack(m_old: torch.Tensor, m_new: torch.Tensor, kappa: float) -> torch.Tensor:
    """s = min(−κ − m_new, m_old − κ); witness ⇔ s ≥ 0 (same constraints as dce.solvers.dce)."""
    return torch.minimum(-kappa - m_new, m_old - kappa)


def h8_indicators(dce: MethodOutput, b3: dict[str, MethodOutput], grad_old: dict[str, np.ndarray],
                  k_star: int) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Per cell: (Δ-CE, B3) projected Recovery indicators on the support S = dce.found (arrays of length |S|).

    b3 and grad_old are keyed by position ("z0", "zs"); B3 at z* is evaluated only on S (rows outside S are ignored).
    """
    s = dce.found
    out = {}
    for p, q in CELLS:
        g = grad_old[p]
        rd = recovery_indicator(project(dce, g), k_star)[s]
        rb = recovery_indicator(project(b3[q], g), k_star)[s]
        out[(p, q)] = (rd, rb)
    return out


def boot_h8(pairs: Sequence[dict], rng: np.random.Generator, B: int, cell: tuple[str, str] = REGISTERED) -> dict:
    """Paired bootstrap over the support points, stratified by pair (one index draw per pair shared by all cells).

    pairs: per pair, the h8_indicators dict. Gap of a cell = mean over pairs of [mean_S Rec(Δ-CE) − mean_S Rec(B3)].
    T = |gap(z0,z0)| − 2 |gap(cell)| > 0 ⇔ the gap narrows by at least one half (cell = (z*, z0), v1.10 §3.3).
    The guard is the pooled Rec_proj(B3) in `cell` (≥ 0.50 required).
    """
    pairs = [p for p in pairs if len(p[REFERENCE][0]) > 0]
    point = {c: float(np.mean([p[c][0].mean() - p[c][1].mean() for p in pairs])) for c in CELLS}
    stat = {c: np.zeros(B) for c in CELLS}
    for p in pairs:
        n = len(p[CELLS[0]][0])
        idx = rng.integers(0, n, size=(B, n))
        for c in CELLS:
            d, b = p[c]
            stat[c] += (d[idx].mean(1) - b[idx].mean(1)) / len(pairs)
    T = np.abs(stat[REFERENCE]) - 2 * np.abs(stat[cell])
    guard = float(np.mean([p[cell][1].mean() for p in pairs]))
    return {"cell": list(cell), "gap": point, "gap_ci90": {c: np.quantile(stat[c], [0.05, 0.95]).tolist() for c in CELLS},
            "T": abs(point[REFERENCE]) - 2 * abs(point[cell]), "T_ci_lower95": float(np.quantile(T, 0.05)),
            "p_one_sided": float(max((T <= 0).mean(), 1.0 / B)), "guard_rec_b3": guard, "n_pairs": len(pairs),
            "support_sizes": [len(p[REFERENCE][0]) for p in pairs]}


def locality(z0: np.ndarray, zs: np.ndarray, g0: np.ndarray, gs: np.ndarray, d: np.ndarray) -> dict[str, np.ndarray]:
    """Descriptive H8 quantities per point: ‖z*−z0‖, 1 − cos(ŵ_o(z0), ŵ_o(z*)), and ‖P_z0 d − P_z* d‖ / ‖d‖."""
    def unit(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return np.where(n > 0, v / np.where(n > 0, n, 1), 0)

    u0, us = unit(g0), unit(gs)
    p0 = d - (d * u0).sum(1, keepdims=True) * u0
    ps = d - (d * us).sum(1, keepdims=True) * us
    dn = np.linalg.norm(d, axis=1)
    return {"dist": np.linalg.norm(zs - z0, axis=1), "one_minus_cos": 1 - (u0 * us).sum(1),
            "proj_dev": np.linalg.norm(p0 - ps, axis=1) / np.where(dn > 0, dn, 1)}
