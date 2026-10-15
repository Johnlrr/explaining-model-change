"""Tier 0 drift data (spec §4.4) and the shared classifier (spec §5.1)."""

import numpy as np
import torch

from dce.data.drift import DriftParams, adaptation_set, clean_splits, drift_label, tier0_pairs
from dce.eval.g0 import decisive_disagreement
from dce.models.cnn import SmallCNN, margin


def test_clean_splits_sizes_seed_and_label_noise():
    a = clean_splits()
    b = clean_splits()
    assert {k: len(v.y) for k, v in a.items()} == {"train": 20_000, "val": 5_000, "test": 5_000}
    assert np.array_equal(a["test"].phi, b["test"].phi)
    noise = (a["train"].y != a["train"].y_clean).mean()
    assert 0.04 < noise < 0.06
    assert np.array_equal(a["val"].y, a["val"].y_clean)
    assert np.array_equal(a["test"].y, a["test"].y_clean)


def test_t1_ambiguous_band_and_colour_correlation():
    phi, y = adaptation_set("T1", seed=1, label_noise=0.0)
    in_band = (phi[:, 0] >= 0.4) & (phi[:, 0] <= 0.6)
    assert 0.7 < in_band.mean() < 0.78  # 70% forced + ~20% of the rest by chance
    assert np.array_equal(y, (phi[:, 0] > 0.5).astype(int))
    agree = ((phi[:, 2] > 0.5) == (y == 1)).mean()
    assert 0.95 < agree < 0.99  # 0.95 forced + half of the remaining 5% by chance


def test_t3_flips_only_the_large_subgroup():
    phi, y = adaptation_set("T3", seed=2, label_noise=0.0)
    base = (phi[:, 0] > 0.5).astype(int)
    sub = phi[:, 1] > 0.75
    assert np.array_equal(y[sub], 1 - base[sub])
    assert np.array_equal(y[~sub], base[~sub])


def test_t4_shifts_the_threshold_and_n1_shifts_size_only():
    phi, y = adaptation_set("T4", seed=3, label_noise=0.0)
    assert np.array_equal(y, (phi[:, 0] > 0.6).astype(int))
    phi, y = adaptation_set("N1", seed=4, label_noise=0.0)
    assert phi[:, 1].min() >= 0.5
    assert np.array_equal(y, (phi[:, 0] > 0.5).astype(int))


def test_drift_label_matches_adaptation_rules():
    phi = np.array([[0.7, 0.9, 0, 0, 0], [0.7, 0.1, 0, 0, 0], [0.55, 0.1, 0, 0, 0]])
    p = DriftParams()
    assert drift_label("T3", phi, p).tolist() == [0, 1, 1]
    assert drift_label("T4", phi, p).tolist() == [1, 1, 0]
    assert drift_label("N0", phi, p).tolist() == [1, 1, 1]


def test_tier0_has_17_pairs_with_stable_ids():
    pairs = tier0_pairs()
    assert len(pairs) == 17
    assert [p["pair_id"] for p in pairs] == list(range(17))
    assert sum(p["condition"] == "N0" for p in pairs) == 5


def test_cnn_size_and_margin_sign():
    model = SmallCNN()
    n_params = sum(p.numel() for p in model.parameters())
    assert 50_000 < n_params < 65_000
    logits = torch.tensor([[2.0, 0.5], [0.0, 3.0]])
    assert torch.allclose(margin(logits, torch.tensor([0, 1])), torch.tensor([1.5, 3.0]))
    assert torch.allclose(margin(logits, 1), torch.tensor([-1.5, 3.0]))


def test_adaptation_sets_carry_5pct_label_noise_by_default():
    """Amendment v1.2 B2: same symmetric label noise as the training split."""
    for cond in ("T1", "T3", "T4", "N0", "N1"):
        phi_clean, y_clean = adaptation_set(cond, seed=11, label_noise=0.0)
        phi, y = adaptation_set(cond, seed=11)
        assert np.array_equal(phi, phi_clean)  # noise touches labels only
        assert 0.04 < (y != y_clean).mean() < 0.06, cond


def test_decisive_disagreement_needs_both_margins():
    """Amendment v1.2 B3: argmax differs and |m_o| ≥ κ and |m_n| ≥ κ."""
    old = np.array([[2.0, 0.0], [2.0, 0.0], [0.2, 0.0], [1.0, 0.0]])
    new = np.array([[0.0, 2.0], [0.0, 0.3], [0.0, 2.0], [2.0, 0.0]])
    assert decisive_disagreement(old, new, kappa=0.5).tolist() == [True, False, False, False]


def test_t3_criteria_v13_requires_exclusive_size_change():
    from dce.eval.g0 import g0_criteria

    rng = np.random.default_rng(0)
    phi = rng.uniform(size=(400, 5))
    params = DriftParams()
    y_new = drift_label("T3", phi, params)
    logits = np.stack([np.zeros(400), np.where(y_new == 1, 2.0, -2.0)], 1)  # f_new == f_old here
    flip_old = np.array([0.49, 0.02, 0.01, 0.01, 0.01])
    good = flip_old + np.array([0.04, 0.30, 0.0, 0.0, 0.0])
    leaky = flip_old + np.array([0.06, 0.30, 0.0, 0.0, 0.0])
    weak = flip_old + np.array([0.0, 0.14, 0.0, 0.0, 0.0])
    run = lambda f, v: g0_criteria("T3", phi, logits, logits, y_new, flip_old, f, params, 0.5, t3_criteria=v)
    assert run(good, "v1.3")["pass"] and not run(leaky, "v1.3")["pass"] and not run(weak, "v1.3")["pass"]
    out = run(good, "v1.3")
    assert {"acc_new_labels_subgroup", "acc_new_labels_subgroup_decisive", "pass_v1_2"} <= out.keys()
    assert out["pass"] == out["pass_v1_3"]
