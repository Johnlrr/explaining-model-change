"""Projection used by D-033 (b) and D-034 C3."""

import numpy as np

from dce.analysis.projection import project
from dce.eval.metrics import MethodOutput, recovery_indicator


def test_projection_removes_the_gradient_component_exactly():
    g = np.array([[1.0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0]])
    out = MethodOutput([True, True], [[3.0, 1.0, 0, 0, 0], [0.5, 4.0, 0.2, 0, 0]])
    p = project(out, g)
    assert np.allclose(p.dphi, [[0, 1.0, 0, 0, 0], [0.5, 0, 0.2, 0, 0]])
    assert np.allclose((p.dphi * g).sum(1), 0)


def test_projection_can_change_the_top_factor_and_keeps_not_found_at_zero():
    g = np.array([[1.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0]])
    out = MethodOutput([True, False], [[2.0, 0, 1.0, 0, 0], [9.0, 9.0, 0, 0, 0]])
    p = project(out, g)
    assert recovery_indicator(out, 0).tolist() == [1.0, 0.0]
    assert recovery_indicator(p, 2).tolist() == [1.0, 0.0]
    assert (p.dphi[1] == 0).all()


def test_zero_gradient_leaves_dphi_unchanged():
    out = MethodOutput([True], [[1.0, 2.0, 0, 0, 0]])
    assert np.allclose(project(out, np.zeros((1, 5))).dphi, out.dphi)
