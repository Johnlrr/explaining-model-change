"""Integration of the evaluation wiring with the real solvers on a synthetic linear problem.

Asserts schema, counts and finiteness only (D-028): no Recovery, share or attribution value is read or
printed here. The problem is synthetic; no split is touched.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dce.baselines.b1_diff import b1_diff
from dce.baselines.b3_grad import b3_grad
from dce.baselines.b5_line import b5_line
from dce.data.drift import K_STAR, tier0_pairs
from dce.data.renderer import FACTOR_NAMES
from dce.eval.metrics import MethodOutput
from dce.eval.records import (
    b0_disagreement,
    check_outputs,
    check_run_counts,
    from_direction,
    from_points,
    pair_record,
    runtime_row,
)
from dce.eval.rules import compare
from dce.solvers.dce import delta_ce
from dce.solvers.search import SolverConfig

K = 5
N = 6


def margin(w, b):
    w = torch.tensor(w, dtype=torch.float64)
    return lambda z, idx: (z * w.to(z.dtype)).sum(1) + b


M_OLD = margin([3, 0, 0, 0, 0], -0.3)
M_NEW = margin([3, 1.5, 0, 0, 0], -2.7)


def pipeline():
    z0 = torch.tensor([[0.8, 0.8, 0.5, 0.5, 0.5], [0.9, 0.7, 0.2, 0.4, 0.6]] * (N // 2), dtype=torch.float32)
    cfg = SolverConfig(R=2, T=60, clamp=(0.0, 1.0), n_workers=1)
    res = delta_ce(z0, M_OLD, M_NEW, cfg, restart_seed=10000)
    valid, _, ce_o, ce_n = b1_diff(z0, M_OLD, M_NEW, cfg, seed=10000)
    b3_valid, b3_dphi = b3_grad(z0, M_OLD, M_NEW)
    b5_found, b5_dphi, _ = b5_line(z0, torch.rand(40, K, generator=torch.Generator().manual_seed(0)),
                                   M_OLD, M_NEW, cfg.kappa, r=10.0)
    outs = {"dce": from_points(res.found, res.z, z0),
            "b1": from_points(valid, ce_n.z, ce_o.z),
            "b3": from_direction(b3_valid, b3_dphi),
            "b5": from_points(b5_found, z0 + b5_dphi, z0)}
    return res, outs


def test_pipeline_outputs_have_the_expected_schema_and_are_finite():
    _, outs = pipeline()
    check_outputs(outs, N, K)
    for o in outs.values():
        assert isinstance(o, MethodOutput) and o.found.dtype == bool
        assert (o.dphi[~o.found] == 0).all()  # not-found rows carry nothing


def test_runtime_row_has_the_witness_split_fields():
    res, _ = pipeline()
    row = runtime_row(res, wall_s=1.0, n=N)
    assert {"wall_s_per_point", "cpu_s_per_point", "phase2_cpu_s_per_point_with_witness",
            "phase2_cpu_s_per_point_without_witness"} <= set(row)


def test_pair_records_feed_the_comparison_without_error():
    _, outs = pipeline()
    recs = [pair_record(p, s, outs, k_star=1) for p, s in ((0, 0), (1, 1), (2, 2))]
    c = compare("x", recs, "dce", "b3", np.random.default_rng(0), B=20)
    assert np.isfinite([c.boot.diff, c.boot.ci_lo, c.boot.ci_hi, c.boot.p]).all()
    assert set(c.seed_diffs) == {0, 1, 2}


def test_schema_and_count_checks_fail_loudly():
    good = MethodOutput(np.ones(N, bool), np.zeros((N, K)))
    with pytest.raises(ValueError):
        check_outputs({"m": good}, N + 1, K)
    with pytest.raises(ValueError):
        check_outputs({"m": good}, N, K + 1)
    check_run_counts(0, {i: 300 for i in range(17)})
    check_run_counts(1, {i: 200 for i in range(11)})
    with pytest.raises(ValueError):
        check_run_counts(0, {i: 300 for i in range(16)})
    with pytest.raises(ValueError):
        check_run_counts(1, {**{i: 200 for i in range(10)}, 10: 199})


def test_b0_is_the_unfiltered_disagreement_rate():
    assert b0_disagreement(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 1])) == 0.5
    with pytest.raises(ValueError):
        b0_disagreement(np.array([0, 1]), np.array([0]))


def test_tier0_config_factor_order_matches_renderer_and_k_star():
    cfg = yaml.safe_load(Path("configs/tiers/tier0.yaml").read_text())
    assert tuple(cfg["factors"]) == FACTOR_NAMES
    assert K_STAR["T4"] == cfg["factors"].index("aspect") == 0
    assert len(tier0_pairs()) == 17
