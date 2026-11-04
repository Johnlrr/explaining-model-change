"""Tier 1 VAE (spec §5.2): 28×28×3 images, latent dim 16, BCE + KL (β = 1).

Encoder: Conv(3→32, 4×4, s2, pad 1)–ReLU–Conv(32→64, 4×4, s2, pad 1)–ReLU–FC(64·7·7→256)–ReLU–(μ, logσ²).
Decoder (symmetric): FC(16→256)–ReLU–FC(256→64·7·7)–ReLU–ConvT(64→32, s2)–ReLU–ConvT(32→3, s2)–sigmoid.
Kernel size 4 / padding 1 is the reading for "Conv(·, s2)" that maps 28 → 14 → 7 exactly.
`size=32` (D-056, X5) is the same network for 32×32 images: 32 → 16 → 8, FC 64·8·8 → 256. The default (28) is unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

LATENT_DIM = 16


class Encoder(nn.Module):
    def __init__(self, latent: int = LATENT_DIM, size: int = 28):
        super().__init__()
        s = size // 4
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64 * s * s, 256), nn.ReLU())
        self.mu, self.logvar = nn.Linear(256, latent), nn.Linear(256, latent)

    def forward(self, x):
        h = self.fc(self.conv(x).flatten(1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent: int = LATENT_DIM, size: int = 28):
        super().__init__()
        self.s = size // 4
        self.fc = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, 64 * self.s * self.s), nn.ReLU())
        self.deconv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                                    nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 64, self.s, self.s))


class VAE(nn.Module):
    def __init__(self, latent: int = LATENT_DIM, size: int = 28):
        super().__init__()
        self.encoder, self.decoder = Encoder(latent, size), Decoder(latent, size)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar


def vae_loss(x, recon, mu, logvar, beta: float = 1.0):
    bce = nn.functional.binary_cross_entropy(recon, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
    return bce + beta * kl


def train_vae(model: VAE, images: torch.Tensor, *, epochs: int = 30, lr: float = 1e-3, batch_size: int = 256,
              seed: int = 0, device: torch.device | str | None = None, beta: float = 1.0) -> list[float]:
    """Adam on BCE + β·KL over clean images (spec §5.2: β = 1; other β only by amendment v1.7 H3 (b)).
    Returns the mean loss per epoch."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # reparameterisation noise
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(images), generator=gen)
        total = 0.0
        for start in range(0, len(perm), batch_size):
            x = images[perm[start : start + batch_size]].to(dev)
            recon, mu, logvar = model(x)
            loss = vae_loss(x, recon, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        losses.append(total / len(images))
    model.eval()
    return losses
