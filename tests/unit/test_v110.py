"""v1.10 (D-043): label-shift corollaries (T-A), H8 2×2 scoring, T-C slack, locked-seed guard."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dce.analysis.v110 import CELLS, boot_h8, check_fresh_seed, h8_indicators, locality, slack
from dce.baselines.b3_grad import b3_grad
from dce.cli.g0_reeval import load_verified
from dce.data.drift import clean_splits
from dce.data.renderer import sample_factors, task_label
from dce.eval.metrics import MethodOutput
from dce.eval.tier0_margins import Tier0Margin, valid_points
from dce.models.cnn import SmallCNN
from dce.models.label_shift import LogitShift, lowered_class
from dce.solvers.dce import delta_ce
from dce.solvers.search import SolverConfig


def _models(b, seed=0):
    torch.manual_seed(seed)
    f_old = SmallCNN().eval()
    return f_old, LogitShift(f_old, b).eval()


def _points(n=6, seed=0):
    phi = sample_factors(n, np.random.default_rng(seed))
    return torch.as_tensor(phi, dtype=torch.float32), task_label(phi).astype(np.int64)


@pytest.mark.parametrize("b", [-1.5, 0.3, 2.0])
def test_logit_shift_moves_every_margin_by_b(b):
    f_old, f_new = _models(b)
    z, y = _points()
    idx = torch.arange(len(y))
    h = Tier0Margin(f_new, y)(z, idx) - Tier0Margin(f_old, y)(z, idx)
    expected = torch.as_tensor(np.where(y == 1, b, -b), dtype=h.dtype)
    assert torch.allclose(h, expected, atol=1e-5)
    # float64 path (SLSQP) as well
    h64 = Tier0Margin(f_new, y)(z.double(), idx) - Tier0Margin(f_old, y)(z.double(), idx)
    assert torch.allclose(h64, expected.double(), atol=1e-10)


@pytest.mark.parametrize("b", [-1.5, 0.3, 2.0])
def test_b3_is_undefined_under_a_logit_offset(b):
    """Corollary of Prop. 2(a): ∇m_new ≡ ∇m_old, so g = 0 exactly and B3 returns invalid (not a wrong direction)."""
    f_old, f_new = _models(b)
    z, y = _points()
    valid, d = b3_grad(z, Tier0Margin(f_old, y), Tier0Margin(f_new, y))
    assert not valid.any()
    assert d.norm(dim=1).max() < 1e-6


def test_lowered_class_sign():
    assert lowered_class(1.5) == 0 and lowered_class(-1.5) == 1


SMALL = SolverConfig(kappa=0.5, R=2, T=60, r=float("inf"), clamp=(0.0, 1.0), n_lam_steps=3, slsqp_maxiter=30)


def test_lemma1_is_exact_under_a_logit_offset_below_2kappa():
    """|b| < 2κ: no witness can exist anywhere (every output is feasibility-checked)."""
    f_old, f_new = _models(0.9)
    z, y = _points(4)
    res = delta_ce(z, Tier0Margin(f_old, y), Tier0Margin(f_new, y), SMALL, restart_seed=1, kind="N")
    assert not res.found.any()


F_OLD_RUN = Path("runs/experiments/exp_0002")


@pytest.mark.skipif(not (F_OLD_RUN / "artifacts" / "artifact_manifest.json").exists(), reason="f_old checkpoints absent")
def test_witnesses_above_2kappa_lie_in_the_lowered_class_band():
    """|b| ≥ 2κ, trained f_old, validation points: witnesses are found, all have y = lowered class and
    κ ≤ m_old(z*) ≤ |b| − κ (H7b as a check, not a hypothesis)."""
    b, k = 3.0, 0.5
    f_old, _ = load_verified(F_OLD_RUN, "f_old_s0", {})
    f_new = LogitShift(f_old, b).eval()
    val = clean_splits(seed=1234)["val"]
    y_all = val.y_clean.astype(np.int64)
    keep = valid_points(val.phi, y_all, Tier0Margin(f_old, y_all), Tier0Margin(f_new, y_all), 6, 0, k)
    z, y = torch.as_tensor(val.phi[keep], dtype=torch.float32), y_all[keep]
    m_o = Tier0Margin(f_old, y)
    res = delta_ce(z, m_o, Tier0Margin(f_new, y), SMALL, restart_seed=1, kind="N")
    assert res.found.any()
    idx = torch.arange(len(y))
    with torch.no_grad():
        mo = m_o(res.z, idx)
    for i in np.flatnonzero(res.found.numpy()):
        assert y[i] == lowered_class(b)
        assert k - 1e-5 <= float(mo[i]) <= b - k + 1e-5


def test_slack_matches_the_witness_constraints():
    mo = torch.tensor([1.0, 0.4, 2.0, 0.6])
    mn = torch.tensor([-1.0, -1.0, 0.0, -0.5])
    s = slack(mo, mn, 0.5)
    assert torch.allclose(s, torch.tensor([0.5, -0.1, -0.5, 0.0]))
    assert ((s >= 0) == ((mn <= -0.5) & (mo >= 0.5))).all()


def test_h8_cells_use_the_dce_support_only():
    # point 0: found, Δ-CE change along colour (2) plus aspect; point 1: not found
    dce = MethodOutput([True, False], [[0.3, 0, 0.2, 0, 0], [0, 0, 0, 0, 0]])
    b3 = {"z0": MethodOutput([True, True], [[1.0, 0, 0.1, 0, 0], [0, 0, 1.0, 0, 0]]),
          "zs": MethodOutput([True, True], [[1.0, 0, 0.5, 0, 0], [0, 0, 1.0, 0, 0]])}
    g = {"z0": np.array([[1.0, 0, 0, 0, 0]] * 2), "zs": np.array([[1.0, 0, 1.0, 0, 0]] * 2)}
    ind = h8_indicators(dce, b3, g, k_star=2)
    assert all(len(ind[c][0]) == 1 and len(ind[c][1]) == 1 for c in CELLS)   # |S| = 1
    # projection at z0 removes aspect: both point to colour
    assert ind[("z0", "z0")][0][0] == 1 and ind[("z0", "z0")][1][0] == 1
    res = boot_h8([ind], np.random.default_rng(0), 200)
    assert set(res["gap"]) == set(CELLS) and res["n_pairs"] == 1
    assert res["cell"] == ["zs", "z0"] and res["support_sizes"] == [1]


def test_sample_ball_returns_exactly_m_points_inside_ball_and_cube():
    from dce.cli.v110_dev import sample_ball
    z0 = torch.tensor([0.02, 0.98, 0.5, 0.01, 0.99])   # near a corner: low acceptance
    z, draws = sample_ball(z0, 0.54, 5000, torch.Generator().manual_seed(0))
    assert z.shape == (5000, 5) and draws >= 5000
    assert ((z >= 0) & (z <= 1)).all() and ((z - z0).norm(dim=1) <= 0.54 + 1e-6).all()


def test_locality_zero_when_nothing_moves():
    z0 = np.zeros((2, 5)); g = np.eye(5)[:2]; d = np.ones((2, 5))
    loc = locality(z0, z0, g, g, d)
    assert np.allclose(loc["dist"], 0) and np.allclose(loc["one_minus_cos"], 0) and np.allclose(loc["proj_dev"], 0)


def test_locked_seeds_are_refused():
    for s in (2027, 2028, 7020, 7051):
        with pytest.raises(SystemExit):
            check_fresh_seed(s)
    assert check_fresh_seed(9100) == 9100


def _ta_pair(pid, b, found_dce, y, mo, b1):
    n = len(y)
    a = {"y": np.array(y), "dce_N__found": np.array(found_dce), "dce_N__dphi": np.zeros((n, 5)),
         "dce_N_m_old": np.array(mo, dtype=float), "b1_det__found": np.array(b1), "b1_det__dphi": np.tile([1.0, 0, 0, 0, 0], (n, 1)),
         "b3__found": np.zeros(n, bool), "b3__dphi": np.zeros((n, 5))}
    return {"pair_id": pid, "seed": 0, "b": b, "lowered_class": lowered_class(b)}, a


def test_h7_pairs_layout():
    from dce.cli.v110_test import h7_pairs
    ps = h7_pairs()
    assert len(ps) == 30 and ps[0]["pair_id"] == 60 and ps[-1]["pair_id"] == 89
    assert {p["b"] for p in ps if p["seed"] == 2} == {-2.0, -1.5, -1.1, -0.9, -0.5, 0.5, 0.9, 1.1, 1.5, 2.0}


def test_ta_analyze_checks_and_h7c():
    from dce.cli.v110_test import ta_analyze
    pairs = {64: _ta_pair(64, -0.5, [False] * 4, [0, 1, 0, 1], [1, 1, 1, 1], [True] * 4),
             69: _ta_pair(69, 2.0, [True, False, False, False], [0, 0, 1, 1], [1.0, 3, 3, 3], [True] * 4)}
    res = ta_analyze(pairs, 0.5, B=500)
    assert res["H7c"]["n_pairs"] == 1 and res["H7c"]["estimate"] == 1.0 and res["H7c"]["pass_before_holm"]
    bad = dict(pairs)
    bad[64] = _ta_pair(64, -0.5, [True, False, False, False], [1, 1, 0, 1], [1, 1, 1, 1], [True] * 4)
    with pytest.raises(SystemExit, match="U2"):
        ta_analyze(bad, 0.5, B=50)
    bad = dict(pairs)
    bad[69] = _ta_pair(69, 2.0, [True, False, False, False], [1, 0, 1, 1], [1.0, 3, 3, 3], [True] * 4)
    with pytest.raises(SystemExit, match="raised class"):
        ta_analyze(bad, 0.5, B=50)


def _tb_arrays(n, rng):
    """Δ-CE change: colour + a component along the old normal at z0 (aspect); at z* the normal mixes colour, so the
    local projection removes less colour… built so that every cell is computable."""
    found = rng.random(n) < 0.7
    d = np.zeros((n, 5)); d[:, 0] = 1.0; d[:, 2] = 0.6; d[~found] = 0
    b3 = np.zeros((n, 5)); b3[:, 2] = 1.0; b3[:, 0] = 0.2
    g0 = np.zeros((n, 5)); g0[:, 0] = 1.0
    gs = np.zeros((n, 5)); gs[:, 0] = 1.0; gs[:, 2] = 0.1
    z0 = rng.random((n, 5)); zs = z0 + d
    return {"dce__found": found, "dce__dphi": d, "b3_z0__found": np.ones(n, bool), "b3_z0__dphi": b3,
            "b3_zs__found": np.ones(n, bool), "b3_zs__dphi": b3, "grad_z0": g0, "grad_zs": gs, "z0": z0, "zs": zs}


def test_tb_analyze_and_holm(tmp_path):
    from dce.cli.v110_test import H8_PAIRS, cmd_holm, tb_analyze
    rng = np.random.default_rng(0)
    pairs = {p: ({"pair_id": p, "alpha": 0.25, "t": 0.5, "seed": 0}, _tb_arrays(40, rng)) for p in (*H8_PAIRS, 28)}
    res = tb_analyze(pairs, B=300)
    h8 = res["H8"]
    assert h8["cell"] == ["zs", "z0"] and h8["pairs"] == list(H8_PAIRS) and len(h8["support_sizes"]) == 4
    assert len(res["descriptive"]["pairs"]) == 5 and set(res["descriptive"]["spearman_h8_pairs"]) == {
        "proj_dev~dist", "proj_dev~one_minus_cos"}
    (tmp_path / "h7.json").write_text(json.dumps({"H7c": {"p_one_sided": 0.001, "pass_before_holm": True}}))
    (tmp_path / "h8.json").write_text(json.dumps({"H8": {**h8, "testable": False}}))
    cmd_holm(tmp_path / "h7.json", tmp_path / "h8.json", tmp_path / "holm.json")
    out = json.loads((tmp_path / "holm.json").read_text())
    assert out["holm_m"] == 1 and out["H7c"]["confirmed"] and "not testable" in out["H8"]["status"]


def test_claim_and_dynamic_run_pairs(tmp_path):
    from dce.cli.v110_test import claim, run_pairs

    lock = tmp_path / "a.claim"
    assert claim(lock) and not claim(lock)  # held by this live process
    lock.write_text("999999999")  # owner gone
    assert claim(lock) and lock.read_text() == str(__import__("os").getpid())

    done = []

    def fn(cfg, p, cache):
        done.append(p["pair_id"])
        return {"x": np.zeros(1)}, {"pair_id": p["pair_id"]}

    cfg = {"experiment_id": "t", "pair_ids": [1, 2, 3], "threads": 1}
    (tmp_path / "pairs").mkdir()
    (tmp_path / "pairs" / "pair_2.claim").write_text(str(__import__("os").getpid()))  # another live worker
    run_pairs(cfg, tmp_path, None, {i: {"pair_id": i} for i in (1, 2, 3)}, fn, "test")
    assert done == [1, 3] and not (tmp_path / "pairs" / "pair_1.claim").exists()
    (tmp_path / "pairs" / "pair_2.claim").unlink()
    run_pairs(cfg, tmp_path, None, {i: {"pair_id": i} for i in (1, 2, 3)}, fn, "test")
    assert done == [1, 3, 2]


def test_tb_describe_k_star():
    from dce.analysis.v110 import boot_h8
    from dce.cli.v110_test import COLOUR, h8_inputs, tb_describe
    rng = np.random.default_rng(1)
    pairs = {0: ({"pair_id": 0, "condition": "T1", "k_star": 2}, _tb_arrays(40, rng)),
             20: ({"pair_id": 20, "alpha": 0.0, "k_star": None}, _tb_arrays(40, rng))}
    res = tb_describe(pairs, seed=6602, B=200)
    assert [r["k_star"] for r in res["pairs"]] == [2, COLOUR]
    ref = boot_h8([h8_inputs(pairs[p][1], 2)[1] for p in (0, 20)], np.random.default_rng(6602), 200)
    assert res["gap"]["zs,z0"] == ref["gap"][("zs", "z0")] and res["T_zs_z0"] == ref["T"]
    assert set(res["spearman"]) == {"one_minus_cos~dist", "proj_dev~dist"}
