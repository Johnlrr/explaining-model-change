"""Tier 1 oracle (spec §5.3): analytic estimators of colour, background and thickness from an RGB image,
and the shape oracle (the §5.1 classifier on the one-channel α̂).

Every pixel satisfies I_R − I_G = α(1 − c), I_B − I_G = α c, I_G = (1 − α) b, so
α̂ = clip(I_R + I_B − 2 I_G, 0, 1) is exact on images produced by `colorize`.
φ̂_c = ĉ (c is exactly φc); φ̂_b = (b̂ − 0.1) / 0.5 inverts b = 0.1 + 0.5 φb. Batches are numpy (N, 3, H, W).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from dce.models.cnn import SmallCNN
from dce.remote.tier1_prep import measure_thickness

ALPHA_COLOR_MIN = 0.2   # pixels used for ĉ
ALPHA_BG_MAX = 0.05     # pixels used for b̂


def alpha_hat(img: np.ndarray) -> np.ndarray:
    return np.clip(img[:, 0] + img[:, 2] - 2 * img[:, 1], 0.0, 1.0)


def color_hat(img: np.ndarray, a: np.ndarray) -> np.ndarray:
    g, b = img[:, 1], img[:, 2]
    m = a > ALPHA_COLOR_MIN
    ct = np.clip((b - g) / np.where(m, a, 1.0), 0.0, 1.0)
    den = (a * m).sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * ct * m).sum((1, 2)) / den, np.nan)


def bg_hat(img: np.ndarray, a: np.ndarray) -> np.ndarray:
    m = a < ALPHA_BG_MAX
    vals = np.where(m, img[:, 1] / np.maximum(1.0 - a, 1e-6), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        b = np.nanmedian(vals, axis=(1, 2))
    return (b - 0.1) / 0.5


def thick_hat(a: np.ndarray) -> np.ndarray:
    return np.array([measure_thickness(x) for x in a])


def oracle_factors(img: np.ndarray) -> dict[str, np.ndarray]:
    """φ̂ for color, bg, thick (NaN where undefined) and α̂ itself."""
    a = alpha_hat(img)
    return {"color": color_hat(img, a), "bg": bg_hat(img, a), "thick": thick_hat(a), "alpha": a}


def new_shape_oracle() -> SmallCNN:
    return SmallCNN(in_channels=1)


@torch.no_grad()
def shape_prob(model: SmallCNN, alpha: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    """p(9) of the shape oracle on α̂ (N, H, W)."""
    model.eval()
    out = []
    for i in range(0, len(alpha), batch_size):
        x = torch.as_tensor(alpha[i : i + batch_size], dtype=torch.float32).unsqueeze(1)
        out.append(torch.softmax(model(x), 1)[:, 1].numpy())
    return np.concatenate(out) if out else np.zeros(0)
