"""Tier 0 Ellipses renderer (spec §4.2).

Factors φ ∈ [0, 1]^5, in this order:
    0 aspect  ℓ = 1.4 (φ1 − 0.5)                      log aspect ratio, task factor
    1 size    s = 0.25 + 0.35 φ2
    2 color   c = (1 − φ3)(1, 0, 0) + φ3 (0, 0, 1)
    3 bg      b = 0.1 + 0.5 φ4                         grey background
    4 posx    c_x = 0.6 (φ5 − 0.5), c_y = 0

Geometry: A = s e^{ℓ/2}, B = s e^{−ℓ/2}.
Image:    α(u, v) = σ(20 (1 − (u − c_x)²/A² − v²/B²)),  I = α c + (1 − α) b 1.

The renderer is differentiable in φ, so the Δ-CE solver can search directly in factor space.
"""

from __future__ import annotations

import numpy as np
import torch

FACTOR_NAMES = ("aspect", "size", "color", "bg", "posx")
N_FACTORS = 5
IMAGE_SIZE = 32
EDGE_SHARPNESS = 20.0
TRAIN_PIXEL_NOISE = 0.02


def pixel_grid(size: int = IMAGE_SIZE, device=None, dtype=torch.float32):
    """Pixel-centre coordinates (u, v) in [−1, 1]; u runs along width, v along height."""
    coords = (torch.arange(size, device=device, dtype=dtype) * 2 + 1) / size - 1
    v, u = torch.meshgrid(coords, coords, indexing="ij")
    return u, v


def geometry(phi: torch.Tensor):
    """Return (A, B, c_x, color, bg) for a batch of factor vectors, each of shape [N]."""
    ell = 1.4 * (phi[:, 0] - 0.5)
    s = 0.25 + 0.35 * phi[:, 1]
    a_axis = s * torch.exp(ell / 2)
    b_axis = s * torch.exp(-ell / 2)
    color = phi[:, 2]
    bg = 0.1 + 0.5 * phi[:, 3]
    cx = 0.6 * (phi[:, 4] - 0.5)
    return a_axis, b_axis, cx, color, bg


def alpha_map(phi: torch.Tensor, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Soft object mask α, shape [N, size, size]."""
    a_axis, b_axis, cx, _, _ = geometry(phi)
    u, v = pixel_grid(size, device=phi.device, dtype=phi.dtype)
    q = (u[None] - cx[:, None, None]) ** 2 / a_axis[:, None, None] ** 2 + v[None] ** 2 / b_axis[
        :, None, None
    ] ** 2
    return torch.sigmoid(EDGE_SHARPNESS * (1 - q))


def render(phi: torch.Tensor, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Render images, shape [N, 3, size, size], values in [0, 1]. No noise is added here."""
    if phi.ndim != 2 or phi.shape[1] != N_FACTORS:
        raise ValueError(f"phi must have shape [N, {N_FACTORS}], got {tuple(phi.shape)}")
    alpha = alpha_map(phi, size)[:, None]
    _, _, _, color, bg = geometry(phi)
    obj = torch.stack([1 - color, torch.zeros_like(color), color], dim=1)[:, :, None, None]
    return alpha * obj + (1 - alpha) * bg[:, None, None, None]


def add_train_noise(images: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Pixel noise N(0, 0.02²) used only when training classifiers (spec §4.2)."""
    noise = torch.randn(images.shape, generator=generator, device=images.device, dtype=images.dtype)
    return images + TRAIN_PIXEL_NOISE * noise


def sample_factors(n: int, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. uniform factors on [0, 1]^5."""
    return rng.uniform(0.0, 1.0, size=(n, N_FACTORS))


def task_label(phi: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """y = 1[φ1 > threshold] (spec §4.2; T4 uses threshold 0.6)."""
    return (phi[:, 0] > threshold).astype(np.int64)


# ---------------------------------------------------------------------------
# Analytic read-back of factors from an image (used to test the renderer).
# The object colour has no green component, so per pixel:
#   I_R + I_B − 2 I_G = α,   I_G = (1 − α) b,   I_B − I_G = α φ3.
# ---------------------------------------------------------------------------


def read_factors(images: torch.Tensor) -> torch.Tensor:
    """Recover φ from noise-free renders.

    colour and background are exact (closed form). aspect, size, and posx come from the first and
    second moments of α; they are exact only for ellipses that are not clipped by the image border
    and are approximate because of the soft edge and pixel discretisation.
    """
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    alpha = (r + b - 2 * g).clamp(0, 1)
    weight = alpha.sum(dim=(1, 2)).clamp_min(1e-8)

    # background: I_G = (1 − α) b, solved on pixels that are mostly background.
    bg_mask = alpha < 0.5
    bg_px = g / (1 - alpha).clamp_min(1e-8)
    bg = (bg_px * bg_mask).sum(dim=(1, 2)) / bg_mask.sum(dim=(1, 2)).clamp_min(1)
    phi_bg = (bg - 0.1) / 0.5

    # colour: Σ α φ3 / Σ α over object pixels.
    obj_mask = alpha > 0.5
    color = ((b - g) * obj_mask).sum(dim=(1, 2)) / (alpha * obj_mask).sum(dim=(1, 2)).clamp_min(1e-8)

    u, v = pixel_grid(images.shape[-1], device=images.device, dtype=images.dtype)
    mu_u = (alpha * u).sum(dim=(1, 2)) / weight
    var_u = (alpha * (u - mu_u[:, None, None]) ** 2).sum(dim=(1, 2)) / weight
    var_v = (alpha * v**2).sum(dim=(1, 2)) / weight
    # A filled ellipse has E[(u − c_x)²] = A²/4 and E[v²] = B²/4.
    a_axis = 2 * var_u.sqrt()
    b_axis = 2 * var_v.sqrt()
    ell = torch.log(a_axis / b_axis)
    s = (a_axis * b_axis).sqrt()

    phi = torch.stack(
        [ell / 1.4 + 0.5, (s - 0.25) / 0.35, color, phi_bg, mu_u / 0.6 + 0.5], dim=1
    )
    return phi


def is_unclipped(phi: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """True where the ellipse lies inside the image with a margin (moment read-back is valid)."""
    a_axis, b_axis, cx, _, _ = geometry(phi)
    return (cx.abs() + a_axis <= 1 - margin) & (b_axis <= 1 - margin)
