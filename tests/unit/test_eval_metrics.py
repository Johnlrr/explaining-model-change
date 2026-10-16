"""Evaluation metrics and statistics on synthetic data with known answers (EVALUATION §2, §3, §10, §11).

Run before the final procedure (§19); none of these inputs come from any split.
"""

import numpy as np
import pytest

from dce.eval.metrics import (
    MethodOutput,
    auroc,
    factor_index,
    found_rate,
    median_norm,
    pooled_found_rate,
    recovery_at_r,
    recovery_given_reach,
    recovery_indicator,
    sar_pair,
    share_of_kstar,
    top_factor,
)
from dce.eval.stats import holm, paired_bootstrap

K_STAR = 2


def hand_output():
    """N = 10, K = 5. Known answers worked out by hand in the comments."""
    found = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    dphi = np.zeros((10, 5))
    dphi[0:4] = [0.1, 0.1, 1.0, 0.1, 0.1]  # top 2 (k*): recovered; share_k* = 1/1.4
    dphi[4] = [0.9, 0.1, 0.1, 0.1, 0.1]     # top 0: not recovered; share_k* = 0.1/1.3
    dphi[5] = 0.0                            # found, zero-sum: top = −1, excluded from share
    dphi[6] = [0.0, 0.0, 2.0, 0.0, 0.0]     # top 2: recovered; share_k* = 1
    dphi[7] = [0.0, 0.0, 0.0, 0.0, 0.3]     # top 4: not recovered; share_k* = 0
    dphi[8:] = 9.9                           # not found: must be ignored
    return MethodOutput(found, dphi)


def test_recovery_and_found_rate_hand_worked():
    out = hand_output()
    # recovered: rows 0,1,2,3,6 → 5 of N = 10
    assert recovery_at_r(out, K_STAR) == pytest.approx(0.5)
    assert found_rate(out) == pytest.approx(0.8)


def test_zero_vector_and_ties_have_no_top_factor_and_score_zero():
    # Tier 0 T4: k* = column 0 (aspect). A zero Δφ̃ must not be credited to column 0 by argmax.
    out = MethodOutput([1, 1, 1, 1],
                       [[0, 0, 0, 0, 0],              # zero vector
                        [1.0, 1.0, 0, 0, 0],          # exact tie at the maximum
                        [1.0, 1.0 - 5e-13, 0, 0, 0],  # tie within 1e-12
                        [1.0, 1.0 - 1e-6, 0, 0, 0]])  # unique maximum: scored
    assert recovery_indicator(out, 0).tolist() == [0.0, 0.0, 0.0, 1.0]
    assert top_factor(out.dphi).tolist() == [-1, -1, -1, 0]


def test_factor_order_comes_from_the_config():
    assert factor_index(["aspect", "size", "color", "bg", "posx"], "aspect") == 0
    assert factor_index(["size", "aspect"], "aspect") == 1


def test_not_found_points_count_as_zero_even_with_large_dphi():
    out = hand_output()
    assert recovery_indicator(out, K_STAR)[8:].sum() == 0


def test_share_of_kstar_excludes_zero_sum_and_not_found():
    out = hand_output()
    expected = np.mean([1 / 1.4] * 4 + [0.1 / 1.3, 1.0, 0.0])  # rows 0–4, 6, 7; row 5 excluded
    assert share_of_kstar(out, K_STAR) == pytest.approx(expected)


def test_median_norm_over_found_only():
    out = hand_output()
    norms = np.linalg.norm(out.dphi[:8], axis=1)
    assert median_norm(out) == pytest.approx(np.median(norms))


def test_pooled_found_rate_over_directions():
    a = MethodOutput([1, 1, 0, 0], np.zeros((4, 2)))
    b = MethodOutput([1, 0, 0, 0], np.zeros((4, 2)))
    assert pooled_found_rate([a, b]) == pytest.approx(3 / 8)


def test_recovery_given_reach_uses_same_subset_for_every_method():
    out = hand_output()
    other = MethodOutput(np.ones(10, dtype=bool), np.tile([0.0, 0.0, 1.0, 0.0, 0.0], (10, 1)))
    reach = out.found
    res = recovery_given_reach(reach, {"dce": out, "other": other}, K_STAR)
    assert res["dce"] == pytest.approx(5 / 8)
    assert res["other"] == pytest.approx(8 / 8)  # every row is top k* on the 8 reach points


def test_sar_flags_concentrated_attribution_only_with_enough_found():
    conc = MethodOutput(np.ones(10, dtype=bool), np.tile([0.0, 1.0, 0.0], (10, 1)))
    spread = MethodOutput(np.ones(10, dtype=bool), np.tile([1.0, 1.0, 1.0], (10, 1)))
    rare = MethodOutput(np.arange(10) < 1, np.tile([0.0, 1.0, 0.0], (10, 1)))
    assert sar_pair(conc)
    assert not sar_pair(spread)
    assert not sar_pair(rare)  # found-rate 0.1 < 0.2


def test_auroc_known_values():
    assert auroc([3, 4], [0, 1]) == 1.0
    assert auroc([1, 2], [0, 1]) == pytest.approx(0.875)  # (1 + 0.5 + 1 + 1) / 4
    assert auroc([1, 1], [1, 1]) == 0.5


def test_method_output_rejects_bad_shapes_and_nonfinite_values():
    with pytest.raises(ValueError):
        MethodOutput([1, 0], np.zeros((3, 2)))
    with pytest.raises(ValueError):
        MethodOutput([1], np.array([[np.nan, 0.0]]))


# ---- statistics ---------------------------------------------------------------------------------

def _pairs(rng, J, N, pa, pb):
    a = [(rng.random(N) < pa).astype(float) for _ in range(J)]
    b = [(rng.random(N) < pb).astype(float) for _ in range(J)]
    return a, b


def test_zero_true_difference_covered_about_95_percent():
    """Hiệu thật bằng 0: CI 95% phải chứa 0 ở khoảng 95% số lần (200 lần lặp)."""
    covered = rejected = 0
    reps = 200
    for rep in range(reps):
        rng = np.random.default_rng(1000 + rep)
        a, b = _pairs(rng, J=6, N=300, pa=0.5, pb=0.5)
        r = paired_bootstrap(a, b, np.random.default_rng(2000 + rep), B=1000)
        covered += r.ci_lo <= 0 <= r.ci_hi
        rejected += r.p < 0.05
    assert 0.88 <= covered / reps <= 0.995, covered / reps
    assert rejected / reps <= 0.12, rejected / reps


def test_planted_positive_difference_is_detected():
    """Hiệu cài sẵn 0.15: code phải phát hiện được (CI không chứa 0, p nhỏ)."""
    rng = np.random.default_rng(7)
    a, b = _pairs(rng, J=6, N=300, pa=0.65, pb=0.5)
    r = paired_bootstrap(a, b, np.random.default_rng(8), B=2000)
    assert r.diff == pytest.approx(0.15, abs=0.05)
    assert r.ci_lo > 0
    assert r.p < 0.001


def test_point_estimate_is_pair_averaged_not_pooled():
    a = [np.ones(10), np.zeros(1000)]
    b = [np.zeros(10), np.zeros(1000)]
    r = paired_bootstrap(a, b, np.random.default_rng(0), B=200)
    assert r.diff == pytest.approx(0.5)  # (1 + 0) / 2, not the pooled 10/1010


def test_paired_bootstrap_rejects_misaligned_points():
    with pytest.raises(ValueError):
        paired_bootstrap([np.zeros(5)], [np.zeros(6)], np.random.default_rng(0), B=10)


def test_holm_known_examples():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.6})
    assert adj["a"] == (pytest.approx(0.03), True)
    assert adj["b"] == (pytest.approx(0.08), False)
    assert adj["c"] == (pytest.approx(0.6), False)
    adj = holm({"a": 0.01, "b": 0.02, "c": 0.03})
    assert all(rej for _, rej in adj.values())
    assert adj["c"][0] == pytest.approx(0.04)
