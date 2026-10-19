"""Amendment v1.9 §4 (Stage B, D-036): Adult (OpenML 1590, version 2), six numeric features only.

Split 60/20/20 with default_rng(2026).permutation; features standardised with the train mean and standard deviation;
the search box is the 1–99% percentile range of the standardised train features; D is the identity.
Label y = 1[class = '>50K'].
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OPENML_ID = 1590
OPENML_URL = "https://openml.org/data/v1/download/1595261/adult.arff"
OPENML_MD5 = "bb6510925e5d4b23d136715febb2cdf5"
FEATURES = ("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week")
COLUMNS = (0, 2, 4, 10, 11, 12)   # positions in the ARFF row
LABEL_COLUMN = 14


@dataclass(frozen=True)
class AdultSplits:
    x: dict[str, np.ndarray]      # standardised features per split
    y: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    box_lo: np.ndarray            # 1% percentile of standardised train
    box_hi: np.ndarray            # 99% percentile


def load_arff(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(path).read_bytes()
    if hashlib.md5(raw).hexdigest() != OPENML_MD5:
        raise SystemExit(f"{path}: md5 differs from OpenML {OPENML_ID}")
    rows = raw.decode().split("@data", 1)[1].strip().splitlines()
    x, y = [], []
    for line in rows:
        f = [v.strip() for v in line.split(",")]
        x.append([float(f[c]) for c in COLUMNS])   # numeric columns have no missing values; '?' would raise
        y.append(1 if f[LABEL_COLUMN] == ">50K" else 0)
    return np.asarray(x), np.asarray(y, dtype=np.int64)


def adult_splits(path: Path, seed: int = 2026) -> AdultSplits:
    x, y = load_arff(path)
    order = np.random.default_rng(seed).permutation(len(y))
    n_tr, n_va = int(0.6 * len(y)), int(0.2 * len(y))
    idx = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    mean, std = x[idx["train"]].mean(0), x[idx["train"]].std(0)
    xs = {k: (x[v] - mean) / std for k, v in idx.items()}
    lo, hi = np.percentile(xs["train"], [1, 99], axis=0)
    return AdultSplits(xs, {k: y[v] for k, v in idx.items()}, mean, std, lo, hi)
