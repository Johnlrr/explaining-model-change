"""Tier 1 pieces of G2 on a tiny synthetic npz (no MNIST, no network): oracle, adaptation sets, factor
re-sampling, G0-Tier 1 criteria and the training entry's end-to-end path at smoke scale."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dce.data.tier1 import (
    FACTORS,
    RESAMPLE,
    T1Params,
    adaptation_set,
    load_split,
    resample_factor,
    subgroup,
    t3_threshold,
    tier1_pairs,
)
from dce.eval.g0_tier1 import g0_criteria
from dce.models.oracle import alpha_hat, oracle_factors
from dce.remote.tier1_prep import LEVELS, choose_levels, measure_thickness, morph_ladder
from dce.remote.tier1_train import run

TIER1 = yaml.safe_load(Path("configs/tiers/tier1.yaml").read_text())


def fake_digit(rng, label):
    img = np.zeros((28, 28), dtype=np.uint8)
    w = int(rng.integers(2, 5))
    c = int(rng.integers(11, 16))
    img[4:24, c : c + w] = 255                    # a vertical bar (class 0) …
    if label:
        img[4 : 4 + w, c - 5 : c + 6] = 255       # … with a top bar (class 1)
    return img


def make_npz(path, n_train=48, n_val=24, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {"levels": np.array(LEVELS)}
    for name, n in (("train", n_train), ("val", n_val)):
        y = rng.integers(0, 2, n)
        ladder = np.stack([morph_ladder(fake_digit(rng, int(lab))) for lab in y])
        thick = np.array([[measure_thickness(lv / 255.0) for lv in lad] for lad in ladder])
        level = choose_levels(thick, rng.uniform(1.5, 5.0, n))
        arrays.update({f"{name}_ladder": ladder, f"{name}_thick": thick, f"{name}_level": level,
                       f"{name}_y": y, f"{name}_phi_c": rng.uniform(0, 1, n), f"{name}_phi_b": rng.uniform(0, 1, n)})
    np.savez_compressed(path, **arrays)
    return path


@pytest.fixture(scope="module")
def npz(tmp_path_factory):
    return make_npz(tmp_path_factory.mktemp("t1") / "tier1_data.npz")


def test_oracle_recovers_colour_background_and_thickness(npz):
    s = load_split(npz, "train")
    f = oracle_factors(s.images())
    assert np.allclose(alpha_hat(s.images()), s.alpha, atol=1e-5)
    assert np.allclose(f["color"], s.phi_c, atol=1e-3)
    assert np.allclose(f["bg"], s.phi_b, atol=1e-3)
    assert np.allclose(f["thick"], s.thick, atol=1e-4, equal_nan=True)


def test_test_split_is_refused_unless_final(npz):
    with pytest.raises(PermissionError):
        load_split(npz, "test")


def test_config_factor_order_pairs_and_resamplable_factors():
    assert tuple(TIER1["factors"]) == FACTORS
    assert RESAMPLE == {"color": FACTORS.index("color"), "bg": FACTORS.index("bg"),
                        "thick": FACTORS.index("thick")}
    pairs = tier1_pairs()
    assert len(pairs) == 11 and [p["k_star"] for p in pairs[:6]] == [1, 1, 1, 3, 3, 3]
    assert all(p["k_star"] is None for p in pairs[6:])


def test_adaptation_sets_inject_only_their_own_drift(npz):
    tr = load_split(npz, "train")
    t1 = adaptation_set("T1", tr, seed=1, params=T1Params(t1_correlation=1.0), n=40, label_noise=0.0)
    assert ((t1["phi_c"] > 0.5) == (t1["y"] == 1)).all()          # colour fully tied to the label
    n0 = adaptation_set("N0", tr, seed=2, n=40, label_noise=0.0)
    assert (n0["y"] == tr.y[n0["index"]]).all() and np.allclose(n0["phi_c"], tr.phi_c[n0["index"]])
    p3 = T1Params()
    t3 = adaptation_set("T3", tr, seed=3, params=p3, n=40, label_noise=0.0)
    thr = t3_threshold(tr, p3)
    in_sub = tr.thick[t3["index"]] > thr
    assert (t3["y"][in_sub] != tr.y[t3["index"]][in_sub]).all()
    assert (t3["y"][~in_sub] == tr.y[t3["index"]][~in_sub]).all()
    again = adaptation_set("T3", tr, seed=3, params=p3, n=40, label_noise=0.0)
    assert (again["index"] == t3["index"]).all()                    # deterministic per seed
    noisy = adaptation_set("N0", tr, seed=2, n=40, label_noise=0.5)
    assert (noisy["y"] != n0["y"]).any()


def test_t3_fallback_uses_background_quantile(npz):
    tr = load_split(npz, "train")
    p = T1Params(t3_factor="bg")
    assert subgroup(tr, p, t3_threshold(tr, p)).mean() == pytest.approx(0.15, abs=0.1)


def test_resample_changes_only_the_named_factor(npz):
    s = load_split(npz, "val")
    base_alpha = alpha_hat(s.images())
    for f, other in (("color", "bg"), ("bg", "color")):
        moved = oracle_factors(resample_factor(s, f))
        orig = oracle_factors(s.images())
        assert np.allclose(alpha_hat(resample_factor(s, f)), base_alpha, atol=1e-5)  # shape and thickness fixed
        assert np.allclose(moved[other], orig[other], atol=1e-3)
        assert not np.allclose(moved[f], orig[f], atol=1e-2)
    th = oracle_factors(resample_factor(s, "thick"))
    assert np.allclose(th["color"], s.phi_c, atol=1e-3) and np.allclose(th["bg"], s.phi_b, atol=1e-3)
    assert (resample_factor(s, "thick") == resample_factor(s, "thick")).all()


FLIP0 = {"color": 0.05, "bg": 0.05, "thick": 0.05}


def logits(pred, margin=2.0):
    lg = np.zeros((len(pred), 2))
    lg[np.arange(len(pred)), pred] = margin
    return lg


def test_g0_tier1_rules_on_constructed_cases():
    y = np.array([0, 1] * 50)
    same = logits(y)
    sub = np.arange(100) < 20
    # N0: identical models, equal flips → pass; a colour shift of 0.1 → fail (> 0.05)
    assert g0_criteria("N0", same, same, y, None, FLIP0, FLIP0, 0.5)["pass"]
    assert not g0_criteria("N0", same, same, y, None, FLIP0, {**FLIP0, "color": 0.15}, 0.5)["pass"]
    # T1: ΔFlip_color ≥ 0.15
    assert g0_criteria("T1", same, same, y, None, FLIP0, {**FLIP0, "color": 0.20}, 0.5)["pass"]
    assert not g0_criteria("T1", same, same, y, None, FLIP0, {**FLIP0, "color": 0.19}, 0.5)["pass"]
    # T3 (C1): changes confined to the subgroup, ΔFlip_thick ≥ 0.15, others ≤ 0.05
    new = y.copy()
    new[sub] = 1 - new[sub]
    good = g0_criteria("T3", same, logits(new), new, sub, FLIP0, {**FLIP0, "thick": 0.25}, 0.5, k_star="thick")
    assert good["pass"] and good["dd_outside_subgroup"] == 0.0
    leaky = new.copy()
    leaky[20:40] = 1 - leaky[20:40]           # 20 of the 80 outside points disagree: DD outside = 0.25
    bad = g0_criteria("T3", same, logits(leaky), new, sub, FLIP0, {**FLIP0, "thick": 0.25}, 0.5, k_star="thick")
    assert not bad["pass"]
    spill = g0_criteria("T3", same, logits(new), new, sub, FLIP0, {**FLIP0, "thick": 0.25, "bg": 0.20}, 0.5,
                        k_star="thick")
    assert not spill["pass"]                  # exclusivity: bg moved by 0.15 > 0.05
    with pytest.raises(ValueError):
        g0_criteria("T4", same, same, y, None, FLIP0, FLIP0, 0.5)


def test_training_entry_end_to_end_at_smoke_scale(npz, tmp_path):
    params = json.loads(json.dumps(TIER1))
    params["scale"].update(limit=40, vae_epochs=1, oracle_epochs=1, f_old_epochs=1, n_adapt=30,
                           seeds=[0, 1], pair_ids=[0, 3, 6])
    params["finetune"] = {c: {"epochs": 1, "lr": 3e-4} for c in ("T1", "T3", "N0")}
    out = tmp_path / "out"
    res = run(params, npz, out, torch.device("cpu"), data_sha="x")
    manifest = json.loads((out / "artifact_manifest.json").read_text())["artifacts"]
    names = {a["name"] for a in manifest}
    assert names == {"vae", "shape_oracle", "f_old_s0", "f_old_s1", "f_new_p00_T1_s0", "f_new_p03_T3_s0",
                     "f_new_p06_N0_s0"}
    for a in manifest:
        assert hashlib.sha256((out / a["path"]).read_bytes()).hexdigest() == a["sha256"]
    assert res["work"]["processed"] == len(names) and res["observed_hardware"] == ["cpu"]
    dump = res["val_logits"]
    assert set(dump) == names - {"vae", "shape_oracle"}
    assert all(np.isfinite(dump[k]).all() and np.asarray(dump[k]).shape == (24, 2) for k in dump)
    assert len(list(out.iterdir())) <= 20  # Kaggle lists at most 20 output files per call


def test_kaggle_configs_carry_the_tier_parameters_verbatim():
    full = yaml.safe_load(Path("configs/experiments/exp_0012_tier1_train.yaml").read_text())["kernel"]
    assert full["parameters"] == TIER1
    assert full["immutable_kernel"] and full["timeout_seconds"] == 3600 and full["enable_internet"] is False
    assert full["code_file"] == "src/dce/remote/bundles/tier1_train.py"
    smoke = yaml.safe_load(Path("configs/experiments/smoke_0011_tier1_train.yaml").read_text())["kernel"]
    assert {k: v for k, v in smoke["parameters"].items() if k not in ("scale", "finetune")} == \
        {k: v for k, v in TIER1.items() if k not in ("scale", "finetune")}
    assert smoke["required_outputs"] == full["required_outputs"] and len(full["required_outputs"]) == 20


def test_derive_v2_drops_erosion_and_keeps_everything_else(npz, tmp_path):
    from dce.data.tier1 import V2_TARGET, derive_v2

    dst = tmp_path / "v2.npz"
    derive_v2(npz, dst)
    v1, v2 = np.load(npz), np.load(dst)
    for name in ("train", "val"):
        s1, s2 = load_split(npz, name), load_split(dst, name)
        assert (s2.ladder == s1.ladder).all() and (s2.y == s1.y).all()
        assert np.allclose(s2.phi_c, s1.phi_c) and np.allclose(s2.phi_b, s1.phi_b)
        assert np.isnan(s2.thick_all[:, :2]).all()                       # levels −2, −1 invalid
        assert (s2.level >= 2).all()                                     # never an erosion level
        assert s2.target_range == V2_TARGET and s1.target_range == (1.5, 5.0)
        th = resample_factor(s2, "thick")
        assert (alpha_hat(th) >= 0).all()
    derive_v2(npz, tmp_path / "again.npz")
    assert (np.load(tmp_path / "again.npz")["val_level"] == v2["val_level"]).all()  # deterministic
    assert set(v1.files) <= set(v2.files)


def test_round_two_program_end_to_end_at_smoke_scale(npz, tmp_path):
    from dce.cli.g0_tier0 import sha256
    from dce.cli.g2_round2 import main

    cfg = {"experiment_id": "t", "tier_config": "configs/tiers/tier1.yaml", "data_v1": str(npz),
           "data_v1_sha256": sha256(npz), "data_v2": str(tmp_path / "v2.npz"), "betas": [0.1, 0.02], "threads": 2,
           "scale": {"limit": None, "seeds": [0, 1, 2, 3, 4], "pair_ids": list(range(11)), "f_old_epochs": 1,
                     "finetune_epochs": 1, "n_adapt": 30, "vae_epochs": 1, "oracle_epochs": 1}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run"
    assert main(["--config", str(tmp_path / "c.yaml"), "--out", str(out)]) == 0
    m = json.loads((out / "metrics.json").read_text())
    assert {r["beta"] for r in m["beta_rows"]} == {0.1, 0.02}
    assert len(m["g0"]) == 3 * 3 + 3 * 4 + 5                               # every variant trained and judged
    assert set(m["decision"]) >= {"tier_level", "tier1_kept", "t3_variant", "consequence"}
    for r in m["beta_rows"]:
        assert set(r["criteria"]) == {"spec_spearman", "spec_shape_oracle", "valid_yield_all_pairs",
                                      "f_old_on_decoded_ge_090"} and len(r["mu_std_per_dim"]) == 16
    manifest = json.loads((out / "artifacts" / "artifact_manifest.json").read_text())["artifacts"]
    assert all(hashlib.sha256((out / "artifacts" / a["path"]).read_bytes()).hexdigest() == a["sha256"] for a in manifest)
    assert main(["--config", str(tmp_path / "c.yaml"), "--out", str(out)]) == 2   # completed runs are not overwritten
