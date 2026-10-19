"""v1.9 Stage B data (D-036): Adult split, standardisation and box. Skipped when the OpenML file is not staged."""

from pathlib import Path

import numpy as np
import pytest

from dce.data.adult import adult_splits

PATH = Path(".work/adult/adult.arff")


@pytest.mark.skipif(not PATH.exists(), reason="Adult ARFF not staged in .work/adult")
def test_adult_splits():
    s = adult_splits(PATH)
    assert {k: len(v) for k, v in s.y.items()} == {"train": 29305, "val": 9768, "test": 9769}
    assert np.allclose(s.x["train"].mean(0), 0, atol=1e-9) and np.allclose(s.x["train"].std(0), 1)
    assert (s.box_lo < s.box_hi).all() and s.x["train"].shape[1] == 6
    s2 = adult_splits(PATH)
    assert all((s.x[k] == s2.x[k]).all() for k in s.x)
