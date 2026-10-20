"""Amendment v1.9 §4 (Stage B, D-036): Adult (OpenML 1590, version 2), six numeric features only.

Split 60/20/20 with default_rng(2026).permutation. D-040: every feature is clipped to [p1, p99] of train and min-max
mapped to [0, 1] on that range (train, val, test alike), so the search space is [0, 1]^6 and the Tier 0 solver
parameters keep their scale; D (`to_raw`) is the inverse map. Label y = 1[class = '>50K'].
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
    x: dict[str, np.ndarray]      # features in [0, 1]^6 per split
    y: dict[str, np.ndarray]
    p1: np.ndarray                # 1% percentile of raw train features
    p99: np.ndarray               # 99% percentile

    def to_raw(self, z: np.ndarray) -> np.ndarray:
        """D: [0, 1]^6 → raw feature units."""
        return self.p1 + np.asarray(z) * (self.p99 - self.p1)


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
    p1, p99 = np.percentile(x[idx["train"]], [1, 99], axis=0)
    if not (p99 > p1).all():
        raise SystemExit(f"degenerate percentile range: p1 {p1}, p99 {p99}")
    xs = {k: (np.clip(x[v], p1, p99) - p1) / (p99 - p1) for k, v in idx.items()}
    return AdultSplits(xs, {k: y[v] for k, v in idx.items()}, p1, p99)
