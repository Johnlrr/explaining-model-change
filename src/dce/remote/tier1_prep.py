"""Tier 1 data preparation (spec §4.3; D-025). Kaggle kernel, Internet ON, CPU.

Self-contained (numpy, scipy, scikit-image, torchvision) so it runs as a single kernel script; the
functions are imported locally as `dce.remote.tier1_prep` (tests, oracle), so there is one
implementation of the thickness measurement.

For every MNIST 4/9 image (official train and test splits) it computes a morphology ladder:
grayscale erosion/dilation with disk(1), levels −2 … +3 (negative = erosion), and the thickness of
each level measured as 2 × mean distance transform on the skeleton of the mask α > 0.5. A level
whose mask or skeleton is empty is invalid (thickness NaN).

It also fixes the clean splits and their factors: official train → 90% train / 10% validation
(default_rng(1234) permutation); per image a target thickness t ~ U[1.5, 5.0] picks the valid level
with the nearest measured thickness; φc, φb ~ U[0, 1]. Images are coloured later from (α, φc, φb).

Output: tier1_data.npz, result.json, artifact_manifest.json in the kernel output.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

LEVELS = (-2, -1, 0, 1, 2, 3)
TARGET_RANGE = (1.5, 5.0)
SPLIT_SEED = 1234
FACTOR_SEEDS = {"train": 1235, "val": 1236, "test": 1237}
MASK_THRESHOLD = 0.5


def measure_thickness(alpha: np.ndarray) -> float:
    """2 × mean distance transform over the skeleton of alpha > 0.5 (spec §4.3); NaN if empty."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    mask = alpha > MASK_THRESHOLD
    if not mask.any():
        return float("nan")
    skel = skeletonize(mask)
    if not skel.any():
        return float("nan")
    return float(2.0 * distance_transform_edt(mask)[skel].mean())


def morph_ladder(img: np.ndarray, levels=LEVELS) -> np.ndarray:
    """uint8 (L, H, W): grayscale erosion (level < 0) / dilation (level > 0) with disk(1), |level| times."""
    from skimage.morphology import dilation, disk, erosion

    fp = disk(1)
    out = np.zeros((len(levels), *img.shape), dtype=np.uint8)
    for li, lv in enumerate(levels):
        x = img
        for _ in range(abs(lv)):
            x = dilation(x, fp) if lv > 0 else erosion(x, fp)
        out[li] = x
    return out


def choose_levels(thick: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per image, the valid level whose measured thickness is nearest the target."""
    gap = np.abs(thick - targets[:, None])
    gap[~np.isfinite(gap)] = np.inf
    return gap.argmin(1)


def colorize(alpha: np.ndarray, phi_c: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """I = α c + (1 − α) b 1, c = (1 − φc)(1, 0, 0) + φc (0, 0, 1), b = 0.1 + 0.5 φb. (N, 3, H, W)."""
    a = alpha[:, None].astype(np.float32)
    c = np.stack([1 - phi_c, np.zeros_like(phi_c), phi_c], 1).astype(np.float32)[:, :, None, None]
    b = (0.1 + 0.5 * phi_b).astype(np.float32)[:, None, None, None]
    return a * c + (1 - a) * b


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    params = json.loads(os.environ.get("RESEARCH_HARNESS_PARAMETERS", "{}"))
    limit = params.get("limit")  # smoke: images per official split
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("out")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    from torchvision import datasets

    root = out / "_download"
    raw = {}
    for split in ("train", "test"):
        data = datasets.MNIST(str(root), train=(split == "train"), download=True)
        x, y = data.data.numpy(), data.targets.numpy()
        keep = np.flatnonzero((y == 4) | (y == 9))
        if limit:
            keep = keep[:limit]
        raw[split] = (x[keep], (y[keep] == 9).astype(np.int64), keep)
    import shutil

    shutil.rmtree(root, ignore_errors=True)

    arrays, summary = {}, {}
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(raw["train"][0]))
    n_val = int(round(0.1 * len(perm)))
    parts = {"train": ("train", perm[n_val:]), "val": ("train", perm[:n_val]),
             "test": ("test", np.arange(len(raw["test"][0])))}
    for name, (src, idx) in parts.items():
        x, y, orig = raw[src]
        x, y, orig = x[idx], y[idx], orig[idx]
        ladder = np.stack([morph_ladder(img) for img in x])                      # (N, L, 28, 28)
        thick = np.array([[measure_thickness(l / 255.0) for l in lad] for lad in ladder])
        rng = np.random.default_rng(FACTOR_SEEDS[name])
        target = rng.uniform(*TARGET_RANGE, len(x))
        level = choose_levels(thick, target)
        phi_c, phi_b = rng.uniform(0, 1, len(x)), rng.uniform(0, 1, len(x))
        arrays.update({f"{name}_ladder": ladder, f"{name}_thick": thick, f"{name}_y": y,
                       f"{name}_mnist_index": orig, f"{name}_target": target, f"{name}_level": level,
                       f"{name}_phi_c": phi_c, f"{name}_phi_b": phi_b})
        chosen = thick[np.arange(len(x)), level]
        summary[name] = {"n": int(len(x)), "share_9": float(y.mean()),
                         "valid_levels_share": float(np.isfinite(thick).mean()),
                         "valid_by_level": np.isfinite(thick).mean(0).round(4).tolist(),
                         "thick_median_by_level": [float(np.nanmedian(t)) if np.isfinite(t).any() else None
                                                   for t in thick.T],
                         "chosen_thick_quantiles": np.quantile(chosen, [0, .15, .5, .85, 1]).round(3).tolist(),
                         "abs_gap_to_target_median": float(np.median(np.abs(chosen - target)))}
    target_file = out / "tier1_data.npz"
    np.savez_compressed(target_file, levels=np.array(LEVELS), **arrays)
    result = {"schema_version": 1, "kind": "tier1_prep", "limit": limit, "levels": list(LEVELS),
              "python": sys.version.split()[0], "platform": platform.platform(), "splits": summary,
              "work": {"processed": int(sum(s["n"] for s in summary.values()))},
              "seconds": round(time.time() - start, 1)}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "artifacts": [
        {"name": "tier1_data", "kind": "dataset", "path": target_file.name, "sha256": _sha256(target_file),
         "bytes": target_file.stat().st_size}]}, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
