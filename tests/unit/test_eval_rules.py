"""Decision rules on constructed cases (EVALUATION §10, §11, §15). Each case is set up so the answer is known."""

import numpy as np
import pytest

from dce.eval.rules import (
    BOOTSTRAP_SEEDS,
    Comparison,
    PairRecord,
    b5_outcome,
    bootstrap_rng,
    compare,
    holm_family,
    p_tie_contradicted,
    route_a,
)
from dce.eval.stats import BootstrapResult


def cmp_(diff, lo, hi, p, seed_diffs=(0.1, 0.1, 0.1)):
    return Comparison("x", BootstrapResult(diff, lo, hi, p), dict(zip((0, 1, 2), seed_diffs)))


def test_compare_uses_per_seed_pair_means_and_paired_points():
    pairs = []
    for s in (0, 1, 2):
        pairs.append(PairRecord(10 + s, s, {"dce": np.ones(50), "b1": np.zeros(50)}))
        pairs.append(PairRecord(20 + s, s, {"dce": np.ones(50), "b1": np.zeros(50)}))
    c = compare("H1", pairs, "dce", "b1", np.random.default_rng(0), B=100)
    assert c.boot.diff == pytest.approx(1.0)
    assert c.seed_diffs == {0: 1.0, 1: 1.0, 2: 1.0}


def test_seed_sign_failure_blocks_support_even_when_holm_significant():
    h1 = cmp_(0.05, 0.01, 0.09, 0.001, seed_diffs=(0.1, -0.02, 0.1))
    h2a = cmp_(-0.1, -0.2, -0.01, 0.9)
    h2b = cmp_(-0.1, -0.2, -0.01, 0.9)
    sup = holm_family({"H1": h1, "H2a": h2a, "H2b": h2b})
    assert sup == {"H1": False, "H2a": False, "H2b": False}


def test_holm_significant_all_seeds_positive_is_supported():
    h1 = cmp_(0.05, 0.01, 0.09, 0.001)
    h2a = cmp_(0.02, -0.01, 0.05, 0.3)
    h2b = cmp_(0.02, -0.01, 0.05, 0.3)
    sup = holm_family({"H1": h1, "H2a": h2a, "H2b": h2b})
    assert sup == {"H1": True, "H2a": False, "H2b": False}


def test_holm_removes_marginal_hypothesis():
    # H1 alone at p = 0.01 survives (3 × 0.01 = 0.03); H2a and H2b at p = 0.03 do not:
    # their Holm-adjusted p is max(0.03, 2 × 0.03) = 0.06 ≥ 0.05, although 0.03 < 0.05 alone.
    h1 = cmp_(0.05, 0.0, 0.1, 0.01)
    h2a = cmp_(0.05, 0.0, 0.1, 0.03)
    h2b = cmp_(0.05, 0.0, 0.1, 0.03)
    assert holm_family({"H1": h1, "H2a": h2a, "H2b": h2b}) == {"H1": True, "H2a": False, "H2b": False}


def test_holm_family_of_two_when_h2b_is_dropped():
    # m = 2: p = 0.02 and 0.03 → adjusted 0.04 and 0.04, both supported; with p = 0.03, 0.03 → 0.06: neither.
    ok = holm_family({"H1": cmp_(0.05, 0.0, 0.1, 0.02), "H2a": cmp_(0.05, 0.0, 0.1, 0.03)})
    assert ok == {"H1": True, "H2a": True}
    no = holm_family({"H1": cmp_(0.05, 0.0, 0.1, 0.03), "H2a": cmp_(0.05, 0.0, 0.1, 0.03)})
    assert no == {"H1": False, "H2a": False}
    assert route_a(True, ok)  # a missing H2b counts as not supported


def test_bootstrap_seeds_are_6000_plus_index_and_per_comparison():
    assert [BOOTSTRAP_SEEDS[k] for k in ("H1", "H2a", "H2b", "B5-T0", "B5-T1")] == [6000, 6001, 6002, 6003, 6004]
    a, b = bootstrap_rng("H1"), bootstrap_rng("H1")
    assert a.integers(0, 10**9) == b.integers(0, 10**9)  # reproducible
    assert bootstrap_rng("H2a").integers(0, 10**9) != bootstrap_rng("H1").integers(0, 10**9)


def test_route_a_requires_h4_and_one_contribution():
    sup = {"H1": False, "H2a": True, "H2b": False}
    assert route_a(True, sup)
    assert not route_a(False, sup)
    assert not route_a(True, {"H1": False, "H2a": False, "H2b": False})


@pytest.mark.parametrize("t0, t1, expected", [
    ((0.2, 0.1, 0.3), (0.2, 0.1, 0.3), "claim_allowed"),        # (a) both tiers: d > 0.10 and CI > 0
    ((0.2, 0.1, 0.3), (0.05, -0.1, 0.2), "no_solver_claim"),    # tier 1 not a claim → (b)
    ((0.2, 0.1, 0.3), (-0.2, -0.3, -0.1), "report_b5_better"),  # tier 1 B5 better → (c)
    ((0.05, -0.1, 0.2), (0.05, -0.1, 0.2), "no_solver_claim"),  # neither claim → (b)
])
def test_b5_outcome(t0, t1, expected):
    by_tier = {0: Comparison("B5", BootstrapResult(*t0, 0.01), {}),
               1: Comparison("B5", BootstrapResult(*t1, 0.01), {})}
    _, label = b5_outcome(by_tier)
    assert label == expected


def test_b5_claim_needs_both_tiers_present():
    by_tier = {0: Comparison("B5", BootstrapResult(0.3, 0.1, 0.5, 0.01), {})}
    assert b5_outcome(by_tier)[1] == "no_solver_claim"


def test_b5_tier_outcomes_are_reported_per_tier():
    by_tier = {0: Comparison("B5", BootstrapResult(0.3, 0.1, 0.5, 0.01), {}),
               1: Comparison("B5", BootstrapResult(-0.3, -0.5, -0.1, 0.01), {})}
    per, label = b5_outcome(by_tier)
    assert per == {0: "claim", 1: "b5_better"}
    assert label == "report_b5_better"


def test_p_tie_threshold():
    assert p_tie_contradicted(0.11)
    assert not p_tie_contradicted(0.10)
    assert p_tie_contradicted(-0.2)
