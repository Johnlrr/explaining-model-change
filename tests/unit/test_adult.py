"""v1.9 Stage B (D-036, D-040): Adult split and [0, 1] map, MLP margin. Data tests skip when the file is not staged."""

from pathlib import Path

import numpy as np
import pytest
import torch

from dce.data.adult import adult_splits
from dce.models.mlp import MLP, TabMargin

PATH = Path(".work/adult/adult.arff")


@pytest.mark.skipif(not PATH.exists(), reason="Adult ARFF not staged in .work/adult")
def test_adult_splits():
    s = adult_splits(PATH)
    assert {k: len(v) for k, v in s.y.items()} == {"train": 29305, "val": 9768, "test": 9769}
    for x in s.x.values():
        assert x.shape[1] == 6 and x.min() >= 0.0 and x.max() <= 1.0
    assert np.allclose(s.x["train"].min(0), 0) and np.allclose(s.x["train"].max(0), 1)
    assert np.allclose(s.to_raw(np.zeros(6)), s.p1) and np.allclose(s.to_raw(np.ones(6)), s.p99)
    s2 = adult_splits(PATH)
    assert all((s.x[k] == s2.x[k]).all() for k in s.x)


def test_tab_margin_matches_logits_and_float64_copy():
    torch.manual_seed(0)
    m = MLP()
    y = np.array([0, 1, 1])
    tm = TabMargin(m, y)
    z = torch.rand(3, 6)
    lg = m(z)
    assert torch.allclose(tm(z, torch.arange(3)), torch.stack([lg[0, 0] - lg[0, 1], lg[1, 1] - lg[1, 0], lg[2, 1] - lg[2, 0]]))
    assert torch.allclose(tm(z.double(), torch.arange(3)).float(), tm(z, torch.arange(3)), atol=1e-5)
