"""Analysis stage of the final Tier 0 procedure on synthetic per-pair outputs (schema only, D-028)."""

import numpy as np
import pytest

from dce.cli.final_tier0 import PER_POINT, analyze
from dce.data.drift import tier0_pairs
from dce.eval.metrics import MethodOutput

CFG = {"bootstrap_resamples": 50, "h4_pass": True}


def synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    pairs = {}
    for p in tier0_pairs():
        outs = {m: MethodOutput(rng.random(n) < 0.7, rng.normal(size=(n, 5))) for m in PER_POINT
                if m != "dce_O" or p["condition"] in ("N0", "N1")}
        row = {**p, "n_points": n, "b0": float(rng.random()), "runtime": {}, "wall_s_pair": 1.0}
        pairs[p["pair_id"]] = (row, outs)
    return pairs


def test_analysis_has_every_required_block_and_finite_numbers():
    res = analyze(synthetic(), CFG)
    assert {"per_pair", "by_condition", "H2a", "route_A", "B5_rule", "P_tie", "H3_auroc_descriptive"} <= set(res)
    assert res["H2a"]["holm_m"] == 1 and set(res["H2a"]["seed_diffs"]) == {0, 1, 2}
    assert res["B5_rule"]["outcome"] in {"no_solver_claim", "report_b5_better"}   # (a) impossible with Tier 0 only
    assert set(res["P_tie"]) == {"T1_vs_b3", "T1_vs_b1_det", "T4_vs_b3", "T4_vs_b1_det"}
    assert set(res["by_condition"]) == {"T1", "T3", "T4", "N0", "N1"}
    assert "sar_share_of_pairs" in res["by_condition"]["N0"]["dce_N"]
    assert np.isfinite([res["H2a"]["diff"], *res["H2a"]["ci95"], res["H2a"]["p_two_sided"]]).all()
    assert len(res["per_pair"]) == 17


def test_analysis_refuses_an_incomplete_run():
    pairs = synthetic()
    del pairs[16]
    with pytest.raises(ValueError):
        analyze(pairs, CFG)
    short = synthetic(n=299)
    with pytest.raises(ValueError):
        analyze(short, CFG)


def test_config_bootstrap_seeds_must_match_the_code():
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(Path("configs/experiments/exp_0017.yaml").read_text())
    assert cfg["bootstrap_seeds"] == {"H2a": 6001, "B5-T0": 6003}
    analyze(synthetic(), {**CFG, "bootstrap_seeds": cfg["bootstrap_seeds"]})
    with pytest.raises(ValueError):
        analyze(synthetic(), {**CFG, "bootstrap_seeds": {"H2a": 7}})
