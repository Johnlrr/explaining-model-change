"""D-059: names_k counts a valid output whose largest |dphi| is factor k; ties and NOT_FOUND count 0."""

import numpy as np

from dce.cli.cfnew_filter import names_k


def test_names_k_counts_top_factor_only_when_found_and_untied():
    dphi = np.array([[0.1, 0.0, 0.5, 0, 0],    # colour top
                     [0.3, 0.0, 0.3, 0, 0],    # tie aspect/colour
                     [0.0, 0.0, -0.9, 0, 0],   # colour top (absolute value)
                     [0.0, 0.2, 0.1, 0, 0]])   # size top
    found = np.array([True, True, False, True])
    assert names_k(found, dphi, 2).tolist() == [True, False, False, False]
    assert names_k(found, dphi, 1).tolist() == [False, False, False, True]
    assert names_k(np.ones(4, bool), dphi, 2).tolist() == [True, False, True, False]
