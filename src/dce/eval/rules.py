"""Decision rules (EVALUATION §10, §11, §15). Pure functions over per-point Recovery indicators.

Holm family: H1, H2a, H2b only (§11). Each hypothesis is supported iff, after Holm, its adjusted p < α, the
pair-averaged difference is > 0, and the difference is > 0 for seeds 0, 1, 2 (§10). B5 and P-tie are
outside Holm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .stats import BootstrapResult, holm, paired_bootstrap

SEEDS = (0, 1, 2)
B5_MARGIN = 0.10
# Bootstrap RNG seed per comparison: 6000 + index, one generator per comparison (D-028).
BOOTSTRAP_SEEDS = {"H1": 6000, "H2a": 6001, "H2b": 6002, "B5-T0": 6003, "B5-T1": 6004}


def bootstrap_rng(name: str) -> np.random.Generator:
    return np.random.default_rng(BOOTSTRAP_SEEDS[name])


@dataclass(frozen=True)
class PairRecord:
    """Per-point Recovery indicators (metrics.recovery_indicator) of every method on one pair."""

    pair_id: int
    seed: int
    indicators: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class Comparison:
    name: str
    boot: BootstrapResult
    seed_diffs: dict[int, float]


def compare(name: str, pairs: Sequence[PairRecord], method_a: str, method_b: str,
            rng: np.random.Generator, B: int = 10_000) -> Comparison:
    """Recovery(A) − Recovery(B) over the given pairs, paired bootstrap, and per-seed point differences."""
    if not pairs:
        raise ValueError("no pairs")
    boot = paired_bootstrap([p.indicators[method_a] for p in pairs], [p.indicators[method_b] for p in pairs],
                            rng, B=B)
    seed_diffs = {}
    for s in sorted({p.seed for p in pairs}):
        group = [p for p in pairs if p.seed == s]
        seed_diffs[s] = float(np.mean([p.indicators[method_a].mean() - p.indicators[method_b].mean()
                                       for p in group]))
    return Comparison(name, boot, seed_diffs)


def hypothesis_positive(c: Comparison, holm_adj_p: float, alpha: float = 0.05) -> bool:
    """Holm-significant, positive pair-averaged difference, and positive per seed 0, 1, 2 (§15)."""
    seeds_ok = all(c.seed_diffs.get(s, float("nan")) > 0 for s in SEEDS)
    return holm_adj_p < alpha and c.boot.diff > 0 and seeds_ok


def holm_family(family: Mapping[str, Comparison], alpha: float = 0.05) -> dict[str, bool]:
    """Holm over the hypotheses actually tested (H1, H2a, H2b; §11; a family of 2 if one is dropped).
    Returns name → hypothesis supported (§15 conditions)."""
    adj = holm({k: c.boot.p for k, c in family.items()}, alpha)
    return {k: hypothesis_positive(c, adj[k][0], alpha) for k, c in family.items()}


def route_a(h4_pass: bool, supported: Mapping[str, bool]) -> bool:
    """Route A = H4 ∧ (H1 ∨ H2a ∨ H2b) (EVALUATION §15)."""
    return bool(h4_pass and any(supported.get(k, False) for k in ("H1", "H2a", "H2b")))


def b5_outcome(by_tier: Mapping[int, Comparison]) -> tuple[dict[int, str], str]:
    """Solver claim rule versus B5 (§15, A4 as revised by A7). Outside Holm.

    Per tier t: 'claim' if d > 0.10 and CI > 0; 'b5_better' if d < −0.10 and CI < 0; else 'no_claim'.
    Overall: 'claim_allowed' only if both tiers say 'claim' (a); 'report_b5_better' if any tier says
    'b5_better' (c); otherwise 'no_solver_claim' (b).
    """
    per = {}
    for t, c in by_tier.items():
        d, lo, hi = c.boot.diff, c.boot.ci_lo, c.boot.ci_hi
        if d > B5_MARGIN and lo > 0:
            per[t] = "claim"
        elif d < -B5_MARGIN and hi < 0:
            per[t] = "b5_better"
        else:
            per[t] = "no_claim"
    if per and all(v == "claim" for v in per.values()) and set(per) == {0, 1}:
        return per, "claim_allowed"
    if "b5_better" in per.values():
        return per, "report_b5_better"
    return per, "no_solver_claim"


def p_tie_contradicted(diff: float) -> bool:
    """P-tie (EVALUATION §15): |Recovery(Δ-CE) − Recovery(baseline)| > 0.10 contradicts Proposition 4."""
    return bool(abs(diff) > B5_MARGIN)
