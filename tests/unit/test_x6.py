"""D-057 X6 helpers: SS adaptation data, qualification rule, locality L and set metrics."""

import numpy as np

from dce.cli.x6_multi import COLOUR, SIZE, locality, qualification, set_metrics
from dce.data.drift import ss_adaptation_set


def test_ss_data_has_both_changes():
    phi, y = ss_adaptation_set(2130, n=20000, label_noise=0.0)
    sub = phi[:, SIZE] > 0.75
    clean = (phi[:, 0] > 0.5).astype(int)
    assert np.array_equal(y[sub], 1 - clean[sub]) and np.array_equal(y[~sub], clean[~sub])
    agree = ((phi[:, COLOUR] > 0.5).astype(int) == clean).mean()       # colour tied to the noise-free label
    assert agree > 0.98
    amb = ((phi[:, 0] > 0.4) & (phi[:, 0] < 0.6)).mean()
    assert 0.7 < amb < 0.8


def test_qualification_rule():
    fo = np.array([0.4, 0.05, 0.02, 0.01, 0.01])
    assert qualification(fo, fo + np.array([0, 0.2, 0.2, 0.0, 0.03]), {"min_delta": 0.15, "max_other": 0.05})["qualified"]
    assert not qualification(fo, fo + np.array([0, 0.2, 0.1, 0.0, 0.0]), {"min_delta": 0.15, "max_other": 0.05})["qualified"]
    assert not qualification(fo, fo + np.array([0, 0.2, 0.2, 0.06, 0.0]), {"min_delta": 0.15, "max_other": 0.05})["qualified"]


def test_locality_and_set_metrics():
    size0 = np.array([0.7, 0.65, 0.62, 0.3, 0.2, 0.4, 0.5])
    dphi = np.zeros((7, 5))
    dphi[[0, 1], SIZE] = 1.0          # near: 2 of 3 size
    dphi[2, COLOUR] = 1.0
    dphi[[3, 4], COLOUR] = 1.0        # far: 1 of 3 size
    dphi[5, SIZE] = 1.0
    dphi[6, SIZE] = 1.0               # middle band, ignored
    found = np.ones(7, bool)
    r = locality(found, dphi, size0, 0.60, 0.45)
    assert r["n_near"] == 3 and r["n_far"] == 3 and abs(r["L"] - (2 / 3 - 1 / 3)) < 1e-12
    r2 = locality(np.zeros(7, bool), dphi, size0, 0.60, 0.45)
    assert np.isnan(r2["L"])
    d = np.zeros((2, 5)); d[0, [COLOUR, SIZE]] = [1.0, 0.5]; d[1, [0, COLOUR]] = [1.0, 0.5]
    m = set_metrics(np.array([True, True]), d)
    assert m["set_recovery"] == 0.5 and m["top1_colour"] == 0.5
