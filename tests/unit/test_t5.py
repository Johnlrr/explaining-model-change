"""D-035 pieces: the T5 adaptation set and the analysis logic (synthetic outputs)."""

import numpy as np

from dce.cli.t5_label_shift import PAIRS, analyze, t5_adaptation
from dce.data.renderer import task_label


def test_t5_adaptation_has_the_declared_label_prior_and_noise():
    phi, y = t5_adaptation(17)
    clean = task_label(phi)
    assert len(y) == 10_000 and (clean == 1).sum() == 8_000
    assert 0.03 < (y != clean).mean() < 0.07
    phi2, y2 = t5_adaptation(17)
    assert (phi2 == phi).all() and (y2 == y).all()


def test_analysis_decisions_on_constructed_outputs():
    rng = np.random.default_rng(0)
    pairs = {}
    for pid in PAIRS:
        n = 300
        y = (np.arange(n) < 150).astype(int)
        arr = {"y": y}
        for m in ("dce_N", "b1_det", "b3", "b1"):
            d = rng.normal(size=(n, 5)) * 0.1
            if m != "b3":
                d[:, 0] += 5.0
            found = np.ones(n, bool) if m != "dce_N" else (y == 0)
            arr[f"{m}__found"], arr[f"{m}__dphi"] = found, np.where(found[:, None], d, 0.0)
        pairs[pid] = arr
    res = analyze(pairs, {"bootstrap_resamples": 200, "seeds_confirm": {"C4": 6400, "C5": 6401}})
    assert res["holm_m"] == 2 and res["C4"]["pass"] and res["C5"]["pass"] and res["C6"]["pass"]
    assert all(v["dce_found_rate_y1"] == 0.0 and v["dce_found_rate_y0"] == 1.0 for v in res["C6"]["pairs"].values())
