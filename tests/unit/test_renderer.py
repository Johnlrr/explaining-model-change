"""Tier 0 renderer: render, then read the factors back from the image (spec §4.2)."""

import numpy as np
import torch
from scipy.stats import spearmanr

from dce.data.renderer import (
    FACTOR_NAMES,
    is_unclipped,
    read_factors,
    render,
    sample_factors,
    task_label,
)


def _phi(n=2000, seed=0):
    return torch.tensor(sample_factors(n, np.random.default_rng(seed)), dtype=torch.float64)


def test_shape_range_and_determinism():
    phi = _phi(64)
    images = render(phi)
    assert images.shape == (64, 3, 32, 32)
    assert images.min() >= 0 and images.max() <= 1
    assert torch.equal(images, render(phi))


def test_green_channel_carries_no_object_colour():
    phi = _phi(64)
    images = render(phi)
    alpha = images[:, 0] + images[:, 2] - 2 * images[:, 1]
    bg = 0.1 + 0.5 * phi[:, 3]
    assert torch.allclose(images[:, 1], (1 - alpha) * bg[:, None, None], atol=1e-10)


def test_colour_and_background_read_back_exactly():
    phi = _phi()
    back = read_factors(render(phi))
    assert torch.allclose(back[:, 2], phi[:, 2], atol=1e-6)
    assert torch.allclose(back[:, 3], phi[:, 3], atol=1e-6)


def test_geometry_read_back_on_unclipped_ellipses():
    phi = _phi(4000)
    keep = is_unclipped(phi)
    assert keep.float().mean() > 0.3
    back = read_factors(render(phi[keep]))
    err = (back - phi[keep]).abs()
    # aspect, size, posx from α moments: the soft edge (σ(20·)) on a 32-pixel grid biases the
    # moments slightly, most for thin ellipses. Measured on 20 000 samples (2026-10-15):
    # aspect max 0.051 / mean 0.006, size max 0.016 / mean 0.005, posx max 0.009 / mean 0.001.
    # This bounds the read-back estimator, not the renderer; Tier 0 uses the exact φ.
    assert err[:, 0].max() < 0.06 and err[:, 0].mean() < 0.01, err[:, 0]
    assert err[:, 1].max() < 0.02 and err[:, 1].mean() < 0.01, err[:, 1]
    assert err[:, 4].max() < 0.01 and err[:, 4].mean() < 0.002, err[:, 4]


def test_every_factor_is_monotone_in_the_image_on_all_samples():
    phi = _phi()
    back = read_factors(render(phi))
    for k, name in enumerate(FACTOR_NAMES):
        rho = spearmanr(back[:, k].numpy(), phi[:, k].numpy()).statistic
        assert rho > 0.98, (name, rho)


def test_render_is_differentiable_in_every_factor():
    phi = _phi(8).requires_grad_(True)
    # Random pixel weights: a plain sum is blind to colour, since dR/dφ3 = −α and dB/dφ3 = +α cancel.
    weights = torch.rand((8, 3, 32, 32), generator=torch.Generator().manual_seed(0), dtype=phi.dtype)
    (render(phi) * weights).sum().backward()
    assert torch.isfinite(phi.grad).all()
    assert (phi.grad.abs().sum(dim=0) > 0).all()


def test_task_label():
    phi = np.array([[0.49, 0, 0, 0, 0], [0.51, 0, 0, 0, 0], [0.59, 0, 0, 0, 0], [0.61, 0, 0, 0, 0]])
    assert task_label(phi).tolist() == [0, 1, 1, 1]
    assert task_label(phi, threshold=0.6).tolist() == [0, 0, 0, 1]
