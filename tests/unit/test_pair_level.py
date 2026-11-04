"""D-056 S1: exact pair-level tests on tiny hand-checked cases."""

import numpy as np
import pytest

from dce.analysis import pair_level as pl


def test_sign_test_exact_tail():
    assert pl.sign_test_p(3, 3) == pytest.approx(1 / 8)
    assert pl.sign_test_p(8, 8) == pytest.approx(1 / 256)
    assert pl.sign_test_p(3, 4) == pytest.approx(5 / 16)        # (4 + 1) / 16
    assert pl.sign_test_p(0, 5) == 1.0
    assert pl.sign_test_p(5, 5) == pytest.approx(1 / 32)


def test_sign_test_sides_and_ties():
    r = pl.sign_test([0.03, 0.09, 0.05, 0.2], 0.2, "below")     # 0.2 equals c: dropped
    assert (r["k"], r["n"], r["n_ties_dropped"]) == (3, 3, 1)
    assert r["p_one_sided"] == pytest.approx(1 / 8)
    r = pl.sign_test([0.95, 0.93, 0.85], 0.9, "above")
    assert (r["k"], r["n"]) == (2, 3) and r["p_one_sided"] == pytest.approx(0.5)


def test_signflip_hand_checked():
    # d = (1, 2, 3): only the all-plus pattern reaches mean 2
    assert pl.signflip_p([1, 2, 3], "greater")["p"] == pytest.approx(1 / 8)
    assert pl.signflip_p([1, 2, 3], "less")["p"] == pytest.approx(1.0)
    # d = (1, -1): patterns give means 0, 1, -1, 0; observed 0 -> P(mean >= 0) = 3/4
    assert pl.signflip_p([1, -1], "greater")["p"] == pytest.approx(3 / 4)
    # two-sided on (1, 2, 3): |mean| >= 2 for the two all-same-sign patterns
    assert pl.signflip_p([1, 2, 3], "two-sided")["p"] == pytest.approx(2 / 8)
    # d = (3, -1, 1): observed mean 1; patterns s*d with sum >= 3: (+,+,+)=3, (+,-,+)=3+1+1=5, (+,+,-)=3-1-1=1 no...
    # enumerate by hand: sums = ±3 ±1 ±1 -> {5, 3, 3, 1, -1, -3, -3, -5}; observed sum 3 -> >= 3: 5, 3, 3 -> 3/8
    assert pl.signflip_p([3, -1, 1], "greater")["p"] == pytest.approx(3 / 8)


def test_signflip_min_p():
    assert pl.signflip_p([0.3, 0.3, 0.4])["min_attainable_p"] == pytest.approx(0.125)


def test_t_ci_known_values():
    ci = pl.t_ci([1, 2, 3, 4])
    # mean 2.5, sd = sqrt(5/3), se = sd / 2, t(0.975; 3) = 3.182446
    se = np.sqrt(5 / 3) / 2
    assert ci["lo"] == pytest.approx(2.5 - 3.182446 * se, abs=1e-5)
    assert ci["hi"] == pytest.approx(2.5 + 3.182446 * se, abs=1e-5)
    assert pl.t_ci([1, 2, 3]) is None                           # n = 3: none (D-056)


def test_make_row_flags_unconfirmable():
    r = pl.make_row("x", "t", "q", [{"pair_id": i, "value": v} for i, v in enumerate((0.3, 0.3, 0.4))], {"kind": "diff_pos"})
    assert r["sign_test"]["p_one_sided"] == pytest.approx(0.125)
    assert "not confirmable" in r["pair_level_flag"]
    assert r["summary"]["t_ci95"] is None


def test_mixed_lm_runs_and_constant_outcome():
    rng = np.random.default_rng(0)
    pair = np.repeat(np.arange(8), 40)
    u = rng.normal(0, 0.1, 8)[pair]
    y = (rng.random(pair.size) < 0.3 + u).astype(float)
    out = pl.mixed_lm(y, pair, claim=(0.2, "above"))
    assert out["status"] == "fitted" and out["n_pairs"] == 8
    assert np.isfinite(out["coef"]["Intercept"]["se"])
    assert pl.mixed_lm(np.zeros(16), np.repeat([0, 1], 8))["status"].startswith("not fitted")
