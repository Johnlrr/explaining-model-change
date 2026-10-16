"""Tier 1 VAE (spec §5.2): 28×28×3 images, latent dim 16, BCE + KL (β = 1).

Encoder: Conv(3→32, 4×4, s2, pad 1)–ReLU–Conv(32→64, 4×4, s2, pad 1)–ReLU–FC(64·7·7→256)–ReLU–(μ, logσ²).
Decoder (symmetric): FC(16→256)–ReLU–FC(256→64·7·7)–ReLU–ConvT(64→32, s2)–ReLU–ConvT(32→3, s2)–sigmoid.
Kernel size 4 / padding 1 is the reading for "Conv(·, s2)" that maps 28 → 14 → 7 exactly.
"""

from __future__ import annotations

import torch
from torch import nn

LATENT_DIM = 16


class Encoder(nn.Module):
    def __init__(self, latent: int = LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 256), nn.ReLU())
        self.mu, self.logvar = nn.Linear(256, latent), nn.Linear(256, latent)

    def forward(self, x):
        h = self.fc(self.conv(x).flatten(1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent: int = LATENT_DIM):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, 64 * 7 * 7), nn.ReLU())
        self.deconv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                                    nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 64, 7, 7))


class VAE(nn.Module):
    def __init__(self, latent: int = LATENT_DIM):
        super().__init__()
        self.encoder, self.decoder = Encoder(latent), Decoder(latent)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar


def vae_loss(x, recon, mu, logvar):
    bce = nn.functional.binary_cross_entropy(recon, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
    return bce + kl
