"""D-056 X5 helpers: 32×32 VAE shapes, latent margins, latent ball, renderer fit, chunk invariance of ball samples."""

import numpy as np
import torch

from dce.cli.tier1_null import Tier1Margin
from dce.cli.x5_vae import fit_factors, latent_ball
from dce.data.renderer import render
from dce.models.cnn import SmallCNN
from dce.models.vae import VAE


def test_vae32_shapes_and_default_unchanged():
    v = VAE(16, size=32)
    x = torch.rand(4, 3, 32, 32)
    recon, mu, logvar = v(x)
    assert recon.shape == x.shape and mu.shape == (4, 16) and logvar.shape == (4, 16)
    v28 = VAE()
    assert v28(torch.rand(2, 3, 28, 28))[0].shape == (2, 3, 28, 28)
    assert v28.encoder.fc[0].in_features == 64 * 7 * 7


def test_latent_margin_through_decoder():
    torch.manual_seed(0)
    v, f = VAE(16, size=32).eval(), SmallCNN().eval()
    y = np.array([0, 1, 1])
    m = Tier1Margin(v.decoder, f, y)
    z = torch.randn(3, 16)
    out = m(z, torch.arange(3))
    lg = f(v.decoder(z))
    sgn = torch.tensor([-1.0, 1.0, 1.0])
    assert torch.allclose(out, sgn * (lg[:, 1] - lg[:, 0]), atol=1e-5)
    assert m(z.double(), torch.arange(3)).dtype == torch.float64


def test_latent_ball_radius_and_determinism():
    z0 = np.zeros(16)
    a = latent_ball(z0, 0.7, 500, np.random.default_rng([5200, 3]))
    b = latent_ball(z0, 0.7, 500, np.random.default_rng([5200, 3]))
    assert np.array_equal(a, b)
    assert (np.linalg.norm(a, axis=1) <= 0.7 + 1e-12).all()


def test_fit_factors_recovers_rendered_factors_and_is_batch_invariant():
    phi = np.random.default_rng(1).uniform(0.15, 0.85, size=(6, 5))
    img = render(torch.as_tensor(phi, dtype=torch.float32))
    est, e = fit_factors(img, steps=300, lr=0.05, restarts=8, seed=12000)
    assert e.max() < 1e-3
    assert np.abs(est[:, 0] - phi[:, 0]).max() < 0.05      # aspect, the factor Recovery reads
    est2, e2 = fit_factors(img, steps=300, lr=0.05, restarts=8, seed=12000, batch=4)
    assert np.allclose(e, e2, atol=1e-6)
