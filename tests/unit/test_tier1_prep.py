"""Tier 1 data preparation (spec §4.3): thickness measurement, morphology ladder, colouring."""

import numpy as np

from dce.remote.tier1_prep import choose_levels, colorize, measure_thickness, morph_ladder


def bar(width: int, size: int = 28) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    img[4:24, c - width // 2 : c - width // 2 + width] = 255
    return img


def test_thickness_of_a_bar_is_close_to_its_width():
    for w in (2, 3, 5, 7):
        t = measure_thickness(bar(w) / 255.0)
        assert abs(t - w) <= 1.0, (w, t)


def test_empty_mask_is_nan():
    assert np.isnan(measure_thickness(np.zeros((28, 28))))


def test_ladder_is_monotone_in_thickness():
    lad = morph_ladder(bar(5))
    t = [measure_thickness(l / 255.0) for l in lad]
    finite = [x for x in t if np.isfinite(x)]
    assert finite == sorted(finite) and len(finite) >= 5


def test_choose_levels_skips_invalid():
    thick = np.array([[np.nan, 1.0, 2.5, 4.4], [1.2, 2.0, np.nan, 6.0]])
    assert choose_levels(thick, np.array([0.1, 3.0])).tolist() == [1, 1]


def test_colorize_matches_spec_formula_and_oracle_identities():
    alpha = np.random.default_rng(0).uniform(size=(4, 28, 28))
    pc, pb = np.array([0.0, 0.3, 0.7, 1.0]), np.array([0.2, 0.5, 0.9, 0.0])
    img = colorize(alpha, pc, pb)
    b = 0.1 + 0.5 * pb
    # spec §5.3: I_R − I_G = α(1 − c), I_B − I_G = α c, I_G = (1 − α) b
    assert np.allclose(img[:, 1], (1 - alpha) * b[:, None, None], atol=1e-6)
    assert np.allclose(img[:, 0] - img[:, 1], alpha * (1 - pc)[:, None, None], atol=1e-6)
    assert np.allclose(img[:, 2] - img[:, 1], alpha * pc[:, None, None], atol=1e-6)
