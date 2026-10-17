"""D-034 analysis on synthetic per-pair outputs (schema and decision logic only)."""

import numpy as np

from dce.cli.confirm_tier0 import analyze, new_test_set
from dce.data.drift import tier0_pairs

CFG = {"bootstrap_resamples": 200, "seeds_confirm": {"C1": 6200, "C3": 6201}}


def synthetic(null_found=0.0, n=300):
    rng = np.random.default_rng(0)
    pairs = {}
    for p in tier0_pairs():
        if p["condition"] in ("T3",):
            continue
        arr = {}
        if p["condition"] in ("N0", "N1"):
            for m in ("dce_N", "dce_O", "b1"):
                arr[f"{m}__found"] = rng.random(n) < (null_found if m != "b1" else 1.0)
                arr[f"{m}__dphi"] = rng.normal(size=(n, 5))
        else:
            k = p["k_star"]
            for m in ("dce_N", "b1_det", "b3"):
                d = rng.normal(size=(n, 5)) * 0.1
                good = m == "dce_N" or (m == "b3" and p["condition"] == "T1")
                d[:, k] += 5.0 if good else 0.0
                arr[f"{m}__found"], arr[f"{m}__dphi"] = np.ones(n, bool), d
            arr["grad_old"] = np.tile(np.eye(5)[0], (n, 1))  # gradient along aspect
        pairs[p["pair_id"]] = ({**p, "n_points": n}, arr)
    return pairs


def test_analysis_blocks_and_holm_family_of_two():
    res = analyze(synthetic(), CFG)
    assert res["holm_m"] == 2 and {"C1", "C2", "C3"} <= set(res)
    assert res["C1"]["pass"] and res["C1"]["diff"] > 0.6   # Δ-CE right; B3 random (hits k* ≈ 1/5 by chance)
    assert res["C3"]["pass"] and abs(res["C3"]["diff_projected"]) < 1e-9  # both right on T1 after projection
    assert res["C2"]["pass"] and len(res["C2"]["pairs"]) == 8


def test_c2_fails_when_one_null_pair_has_witnesses():
    assert not analyze(synthetic(null_found=0.2), CFG)["C2"]["pass"]


def test_new_test_set_is_deterministic_and_hashed():
    a, b = new_test_set(2027, 50), new_test_set(2027, 50)
    assert a[2] == b[2] and (a[0] == b[0]).all() and new_test_set(2028, 50)[2] != a[2]
