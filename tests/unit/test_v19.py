"""D-036 (amendment v1.9) pieces: the rotation × offset label rule, the grid layout and the qualification fit."""

import numpy as np

from dce.analysis.qualification import boundary_params, fit_logistic, qualification_row, qualify
from dce.cli.v19_grid import grid_pairs
from dce.data.drift import adaptation_set, ro_adaptation_set, ro_label
from dce.data.renderer import task_label


def test_ro_label_rule():
    phi = np.full((4, 5), 0.5)
    phi[:, 0] = [0.55, 0.55, 0.45, 0.61]
    phi[:, 2] = [0.5, 0.0, 1.0, 0.5]
    # aspect + α (colour − 0.5) > t with α = 0.25, t = 0.56
    assert ro_label(phi, 0.25, 0.56).tolist() == [0, 0, 1, 1]
    assert ro_label(phi, 0.25, 0.50).tolist() == [1, 0, 1, 1]
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(1000, 5))
    assert (ro_label(x, 0.0, 0.5) == task_label(x)).all()
    assert (ro_label(x, 0.0, 0.6) == task_label(x, threshold=0.6)).all()


def test_ro_adaptation_set_reproduces_t4_at_the_consistency_cell():
    phi, y = ro_adaptation_set(0.0, 0.60, 2006)
    phi4, y4 = adaptation_set("T4", 2006)
    assert (phi == phi4).all() and (y == y4).all()
    phi, y = ro_adaptation_set(0.25, 0.56, 2040)
    assert len(y) == 10_000 and 0.03 < (y != ro_label(phi, 0.25, 0.56)).mean() < 0.07


def test_grid_layout():
    pairs = grid_pairs()
    assert len(pairs) == 32 and [p["pair_id"] for p in pairs] == list(range(20, 52))
    pilot = [p for p in pairs if p["pair_id"] == 40][0]
    assert (pilot["alpha"], pilot["t"], pilot["seed"]) == (0.25, 0.56, 0)
    assert sum(p["k_star"] is None for p in pairs) == 2
    assert {p["k_star"] for p in pairs if p["alpha"] > 0} == {"color"}


def test_qualification_recovers_a_noisy_linear_boundary():
    rng = np.random.default_rng(1)
    phi = rng.uniform(size=(5000, 5))
    a, t = 0.25, 0.56
    z = 40 * (phi[:, 0] + a * (phi[:, 2] - 0.5) - t)
    s = (rng.uniform(size=5000) < 1 / (1 + np.exp(-z))).astype(int)
    fit = fit_logistic(phi, s)
    assert fit.converged and not fit.separable
    a_hat, t_hat = boundary_params(fit)
    assert abs(a_hat - a) < 0.03 and abs(t_hat - t) < 0.01
    row = qualification_row(phi, s, ro_label(phi, a, t), a, t)
    assert row["qualified"] and row["accuracy_new_rule"] > 0.9


def test_boundary_params_uses_other_factors_at_one_half():
    from dce.analysis.qualification import LogitFit

    # boundary: 10 aspect + 2 colour + 4 size − 9 = 0 → at colour = size = 0.5: aspect = (9 − 1 − 2) / 10
    fit = LogitFit(w=[10.0, 4.0, 2.0, 0.0, 0.0], b=-9.0, n_iter=1, converged=True, separable=False, train_accuracy=1.0)
    a_hat, t_hat = boundary_params(fit)
    assert np.isclose(a_hat, 0.2) and np.isclose(t_hat, 0.6)


def test_qualify_thresholds():
    assert qualify(0.25, 0.56, 0.25 + 0.124, 0.56 + 0.019, 0.9)["qualified"]
    assert not qualify(0.25, 0.56, 0.25 + 0.126, 0.56, 0.95)["alpha_ok"]
    assert not qualify(0.25, 0.56, 0.25, 0.56 + 0.021, 0.95)["t_ok"]
    assert not qualify(0.0, 0.60, 0.0, 0.60, 0.899)["accuracy_ok"]
    assert not qualify(0.0, 0.60, 0.051, 0.60, 0.95)["alpha_ok"]


def _grid_rows(t_hat_fn, dd=0.0):
    rows = []
    for p in grid_pairs():
        th = t_hat_fn(p)
        rows.append({"pair_id": p["pair_id"], "alpha": p["alpha"], "t": p["t"], "seed": p["seed"], "alpha_ok": True,
                     "accuracy_ok": True, "t_hat": th, "dt": th - 0.49, "dd": dd})
    return rows


def test_grid_checks_v191_clean_grid():
    from dce.analysis.qualification import grid_checks_v191

    g = grid_checks_v191(_grid_rows(lambda p: p["t"] + 0.03))
    assert not g["stop_rule_triggered"] and g["n_not_qualified_stop_rule"] == 0
    assert g["order_violations"] == [] and all(v["ok"] for v in g["c2c"].values())
    assert all(v == [] for v in g["exclusions"].values())
    assert g["per_pair"][20]["null_dd_ok"] and g["per_pair"][22]["c2a"] and g["per_pair"][40]["c2a"] is None


def test_grid_checks_v191_failures():
    from dce.analysis.qualification import grid_checks_v191

    def t_hat(p):
        if p["alpha"] == 0.0 and p["t"] == 0.53:
            return 0.50                      # (2a) fails: t̂ − t̂_old = 0.01 < 0.02
        if p["alpha"] == 0.25 and p["t"] == 0.60 and p["seed"] == 1:
            return 0.52                      # (2b) violation in row (0.25, seed 1) and (2c) fails at seed 1
        return p["t"] + 0.03
    g = grid_checks_v191(_grid_rows(t_hat, dd=0.03))
    assert g["n_not_qualified_stop_rule"] == 2 and not g["stop_rule_triggered"]
    assert not g["per_pair"][20]["null_dd_ok"]
    assert [v["pairs"] for v in g["order_violations"]] == [[20, 22], [21, 23], [41, 43]]
    assert g["c2c"][0]["ok"] and not g["c2c"][1]["ok"]
    assert {e["pair_id"] for e in g["exclusions"]["H5a"]} == {22, 23}
    assert {e["pair_id"] for e in g["exclusions"]["H5c"]} == {43, 37}
    assert g["exclusions"]["H5b"] == [] and g["exclusions"]["H5d"] == []
    rows = _grid_rows(lambda p: p["t"] + 0.03)
    for r in rows[:11]:
        r["accuracy_ok"] = False
    assert grid_checks_v191(rows)["stop_rule_triggered"]
