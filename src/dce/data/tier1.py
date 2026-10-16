"""Tier 1 data on top of `tier1_data.npz` (spec §4.3–§4.4; D-025, D-029).

The npz holds, per split, the morphology ladder, measured thicknesses, the chosen level, the labels and the
clean factors (φc, φb). Images are coloured on demand from (α, φc, φb). The test arrays are never read
unless `allow_test=True` (final procedure only, EVALUATION §19).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dce.remote.tier1_prep import TARGET_RANGE, choose_levels, colorize

FACTORS = ("shape", "color", "bg", "thick")  # column order of Δφ̃ and of k* (Tier 1 config `factors`)
RESAMPLE = {"color": 1, "bg": 2, "thick": 3}  # re-sampleable factors (D-021 D1); shape has no Flip
K_STAR = {"T1": 1, "T3": 3, "N0": None}       # T3 fallback (bg > q85) has k* = 2
FLIP_SEED_BASE = 7000


@dataclass(frozen=True)
class T1Split:
    name: str
    ladder: np.ndarray    # (N, L, 28, 28) uint8
    thick_all: np.ndarray  # (N, L) measured thickness, NaN where the level is invalid
    level: np.ndarray     # (N,) chosen level index
    y: np.ndarray         # (N,) 1 = digit 9
    phi_c: np.ndarray
    phi_b: np.ndarray
    target_range: tuple[float, float] = TARGET_RANGE  # thickness targets of this data version (v1.8: 2.5–6.5)

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def alpha(self) -> np.ndarray:
        return self.ladder[np.arange(self.n), self.level].astype(np.float32) / 255.0

    @property
    def thick(self) -> np.ndarray:
        return self.thick_all[np.arange(self.n), self.level]

    def images(self) -> np.ndarray:
        return colorize(self.alpha, self.phi_c, self.phi_b)

    def head(self, n: int | None) -> T1Split:
        """First n images (smoke scale); n = None keeps the split."""
        if n is None or n >= self.n:
            return self
        return T1Split(self.name, self.ladder[:n], self.thick_all[:n], self.level[:n], self.y[:n],
                       self.phi_c[:n], self.phi_b[:n], self.target_range)


def load_split(path: str | Path, name: str, allow_test: bool = False) -> T1Split:
    if name == "test" and not allow_test:
        raise PermissionError("the Tier 1 test split is read only in the final procedure (EVALUATION §19)")
    z = np.load(path)
    rng = tuple(float(v) for v in z["target_range"]) if "target_range" in z.files else TARGET_RANGE
    return T1Split(name, z[f"{name}_ladder"], z[f"{name}_thick"], z[f"{name}_level"], z[f"{name}_y"],
                   z[f"{name}_phi_c"], z[f"{name}_phi_b"], rng)


V2_LEVELS = (0, 1, 2, 3)            # amendment v1.8 I1: no erosion
V2_TARGET = (2.5, 6.5)
V2_SEEDS = {"train": 2235, "val": 2236, "test": 2237}


def derive_v2(src: str | Path, dst: str | Path) -> None:
    """Data v2 (amendment v1.8 I1): from the exp_0010 file, mark the erosion levels invalid (thickness NaN) and
    re-draw t ~ U[2.5, 6.5] per image (seeds 2235/2236/2237), choosing the nearest valid level. Images, labels,
    splits, φc and φb are unchanged. Every split is transformed the same way; no statistic of any split is used."""
    z = np.load(src)
    levels = [int(v) for v in z["levels"]]
    drop = [i for i, lv in enumerate(levels) if lv not in V2_LEVELS]
    out = {k: z[k] for k in z.files}
    for name, seed in V2_SEEDS.items():
        if f"{name}_thick" not in z.files:
            continue
        thick = z[f"{name}_thick"].astype(float).copy()
        thick[:, drop] = np.nan
        target = np.random.default_rng(seed).uniform(*V2_TARGET, len(thick))
        out[f"{name}_thick"], out[f"{name}_target"] = thick, target
        out[f"{name}_level"] = choose_levels(thick, target)
    out["target_range"] = np.array(V2_TARGET)
    np.savez_compressed(dst, **out)


def tier1_pairs() -> list[dict]:
    """The 11 model pairs of Tier 1 (spec §4.4): T1 and T3 seeds 0–2, N0 seeds 0–4."""
    pairs, pid = [], 0
    for cond, n_seeds in (("T1", 3), ("T3", 3), ("N0", 5)):
        for s in range(n_seeds):
            pairs.append({"pair_id": pid, "condition": cond, "seed": s, "k_star": K_STAR[cond]})
            pid += 1
    return pairs


def noisy_labels(y: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    flip = rng.random(len(y)) < p
    return np.where(flip, 1 - y, y)


@dataclass(frozen=True)
class T1Params:
    """Spec §4.4 defaults; the G0 knobs may change them (at most two adjustments per condition)."""

    t1_correlation: float = 0.95
    t3_quantile: float = 0.85
    t3_factor: str = "thick"   # fallback of v1.3 C3: "bg"


def t3_threshold(train: T1Split, params: T1Params) -> float:
    """Quantile of the chosen factor on the TRAIN split; labels flip above it (T3)."""
    v = train.thick if params.t3_factor == "thick" else train.phi_b
    return float(np.quantile(v, params.t3_quantile))


def subgroup(split: T1Split, params: T1Params, threshold: float) -> np.ndarray:
    v = split.thick if params.t3_factor == "thick" else split.phi_b
    return v > threshold


def adaptation_set(condition: str, train: T1Split, seed: int, params: T1Params | None = None, n: int = 8000,
                   label_noise: float = 0.05) -> dict:
    """Adaptation data for f_new (spec §4.4; v1.2 B2 label noise). Drawn from the train split only.

    T1: with probability `t1_correlation`, φc ~ U[0.5, 1] if y = 1 else U[0, 0.5]; otherwise unchanged.
    T3: labels flip where the factor exceeds its train quantile. N0: clean data (a different draw/order).
    """
    if condition not in K_STAR:
        raise ValueError(f"unknown Tier 1 condition {condition!r}")
    params = params or T1Params()
    rng = np.random.default_rng(seed)
    idx = rng.choice(train.n, size=min(n, train.n), replace=False)
    sub = T1Split("adapt", train.ladder[idx], train.thick_all[idx], train.level[idx], train.y[idx],
                  train.phi_c[idx].copy(), train.phi_b[idx], train.target_range)
    y = sub.y.copy()
    phi_c = sub.phi_c
    if condition == "T1":
        corr = rng.random(len(y)) < params.t1_correlation
        high, low = rng.uniform(0.5, 1.0, len(y)), rng.uniform(0.0, 0.5, len(y))
        phi_c = np.where(corr, np.where(y == 1, high, low), phi_c)
    elif condition == "T3":
        y = np.where(subgroup(sub, params, t3_threshold(train, params)), 1 - y, y)
    y = noisy_labels(y, label_noise, rng)
    return {"alpha": sub.alpha, "phi_c": phi_c, "phi_b": sub.phi_b, "y": y, "images":
            colorize(sub.alpha, phi_c, sub.phi_b), "index": idx}


def new_labels(condition: str, split: T1Split, params: T1Params, threshold: float) -> np.ndarray:
    """Labelling rule after the drift, for G0 checks on a clean split (no label noise)."""
    y = split.y.copy()
    if condition == "T3":
        y = np.where(subgroup(split, params, threshold), 1 - y, y)
    return y


def resample_factor(split: T1Split, factor: str, seed: int | None = None) -> np.ndarray:
    """Images of `split` with one factor re-drawn (all others fixed): color/bg ~ U[0, 1]; thick: a new target
    t ~ U[target_range] (v1: 1.5–5.0; v2: 2.5–6.5) and the nearest valid ladder level. The same seed gives the same draw for f_old and f_new."""
    k = RESAMPLE[factor]
    rng = np.random.default_rng(FLIP_SEED_BASE + k if seed is None else seed)
    alpha, phi_c, phi_b = split.alpha, split.phi_c, split.phi_b
    if factor == "color":
        phi_c = rng.uniform(0.0, 1.0, split.n)
    elif factor == "bg":
        phi_b = rng.uniform(0.0, 1.0, split.n)
    else:
        level = choose_levels(split.thick_all, rng.uniform(*split.target_range, split.n))
        alpha = split.ladder[np.arange(split.n), level].astype(np.float32) / 255.0
    return colorize(alpha, phi_c, phi_b)
