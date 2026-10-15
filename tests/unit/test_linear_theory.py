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
        cons = [{"type": "ineq", "fun": lambda u: -a - q.w_n @ u},
                {"type": "ineq", "fun": lambda u: q.w_o @ u + c}]
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
