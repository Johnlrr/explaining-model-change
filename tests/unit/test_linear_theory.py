"""H-004 (spec §8 H4): closed forms of Propositions 3–4 on Tier L, and solver checks."""

import numpy as np
import pytest
from scipy.optimize import minimize

from dce.data.tierL import abs_cos, closed_form, project_out, sample_pair


@pytest.mark.parametrize("p", [5, 20])
def test_closed_form_matches_numerical_qp(p):
    cases = set()
    for seed in range(60):
        q = sample_pair(seed, p)
        cf = closed_form(q)
        cases.add(cf["case"])
        a = q.w_n @ q.z0 + q.b_n + 0.5
        c = q.w_o @ q.z0 + q.b_o - 0.5
        cons = [{"type": "ineq", "fun": lambda u, a=a, w=q.w_n: -a - w @ u},
                {"type": "ineq", "fun": lambda u, c=c, w=q.w_o: w @ u + c}]
        u = cf["u"]
        # KKT (exact, independent of the derivation): feasible, 2u = −μ1 w_n + μ2 w_o, μ ≥ 0,
        # complementary slackness.
        assert q.w_n @ u <= -a + 1e-9 and q.w_o @ u >= -c - 1e-9
        mu = np.linalg.lstsq(np.stack([-q.w_n, q.w_o]).T, 2 * u, rcond=None)[0]
        assert np.allclose(np.stack([-q.w_n, q.w_o]).T @ mu, 2 * u, atol=1e-9)
        assert mu.min() >= -1e-9
        assert abs(mu[1] * (q.w_o @ u + c)) <= 1e-8 and abs(mu[0] * (q.w_n @ u + a)) <= 1e-8
        # and a numerical QP agrees
        res = minimize(lambda v: v @ v, np.zeros(p), jac=lambda v: 2 * v, constraints=cons,
                       method="SLSQP", options={"ftol": 1e-10, "maxiter": 500})
        assert abs(res.fun - cf["delta"] ** 2) <= 1e-6 * max(1.0, cf["delta"] ** 2)
    assert cases == {"A", "B"}


def test_proposition4_direction_and_b1_caveat():
    for seed in range(50):
        q = sample_pair(seed, 20)
        cf = closed_form(q)
        assert abs_cos(project_out(cf["u"], q.w_o), project_out(q.dw, q.w_o)) > 1 - 1e-9
        u_n = -(q.w_n @ q.z0 + q.b_n + 0.5) / (q.w_n @ q.w_n) * q.w_n
        u_o = -(q.w_o @ q.z0 + q.b_o + 0.5) / (q.w_o @ q.w_o) * q.w_o
        assert abs_cos(project_out(u_n - u_o, q.w_o), project_out(q.dw, q.w_o)) > 1 - 1e-9


def test_witness_satisfies_band_containment():
    for seed in range(50):
        q = sample_pair(seed, 5)
        z = q.z0 + closed_form(q)["u"]
        m_o, m_n = q.w_o @ z + q.b_o, q.w_n @ z + q.b_n
        assert m_n <= -0.5 + 1e-9 and m_o >= 0.5 - 1e-9
        assert m_n - m_o <= -1.0 + 1e-9  # Lemma 1


# --- solver code checks (amendment v1.4). Seeds ≥ 50 000, disjoint from the H4 seeds 0–999. ---

import torch

from dce.data.tierL import ce_closed_form
from dce.solvers.dce import delta_ce
from dce.solvers.ordinary_ce import ordinary_ce
from dce.solvers.search import SolverConfig


def _batch(pairs):
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    w_o, w_n = t([q.w_o for q in pairs]), t([q.w_n for q in pairs])
    b_o, b_n, z0 = t([q.b_o for q in pairs]), t([q.b_n for q in pairs]), t([q.z0 for q in pairs])
    return (z0, lambda z, i: (w_o[i] * z).sum(1) + b_o[i], lambda z, i: (w_n[i] * z).sum(1) + b_n[i])


def test_two_phase_solver_smoke_linear():
    pairs = [sample_pair(50_000 + s, 5) for s in range(10)]
    z0, m_o, m_n = _batch(pairs)
    res = delta_ce(z0, m_o, m_n, SolverConfig(R=2, T=200), restart_seed=1)
    assert res.found.float().mean() >= 0.8  # reduced T, R: far witnesses may be missed
    for i, q in enumerate(pairs):
        if not res.found[i]:
            continue
        z = res.z[i].double().numpy()
        assert q.w_n @ z + q.b_n <= -0.5 and q.w_o @ z + q.b_o >= 0.5  # hard-feasible witness
        assert res.distance[i].item() >= closed_form(q)["delta"] * (1 - 1e-5)  # never below the optimum
    ce = ordinary_ce(z0, m_n, SolverConfig(R=2, T=200), restart_seed=1)
    for i, q in enumerate(pairs):
        assert ce.found[i] and ce.distance[i].item() >= ce_closed_form(q.w_n, q.b_n, q.z0)["delta"] * (1 - 1e-5)


def test_slsqp_respects_box_bounds():
    # m(z) = z1 − 0.8; target m ≤ −κ → z1 ≤ 0.3 from z0 = (0.9, 0.5); box [0, 1]: answer (0.3, 0.5).
    z0 = torch.tensor([[0.9, 0.5]])
    res = ordinary_ce(z0, lambda z, i: z[:, 0] - 0.8, SolverConfig(R=1, T=100, clamp=(0.0, 1.0)), 0)
    assert res.found[0] and abs(res.z[0, 0].item() - 0.3) < 1e-4 and abs(res.z[0, 1].item() - 0.5) < 1e-4
    # infeasible inside the box: m(z) = z1 + 0.2 ≤ −0.5 needs z1 ≤ −0.7 → NOT_FOUND
    res = ordinary_ce(z0, lambda z, i: z[:, 0] + 0.2, SolverConfig(R=1, T=100, clamp=(0.0, 1.0)), 0)
    assert not res.found[0] and torch.isinf(res.distance[0])


def test_polish_off_is_phase1_only():
    pairs = [sample_pair(50_100 + s, 5) for s in range(5)]
    z0, m_o, m_n = _batch(pairs)
    res = delta_ce(z0, m_o, m_n, SolverConfig(R=1, T=100, polish=False), restart_seed=1)
    assert res.n_polish_runs == 0
