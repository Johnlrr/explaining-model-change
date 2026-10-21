"""Build `v110_tc_inputs.zip` for the Kaggle CPU T-C kernels (D-044): the committed configs and the checkpoints
with their manifests, at their repository paths, in a deterministic zip (fixed order and timestamps). The kernel
checks the zip hash, and `load_verified` / `load_models` check every checkpoint hash against its manifest.

    python -m dce.remote.v110_inputs <out_dir> <config> [<config> ...]
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

EPOCH = (2020, 1, 1, 0, 0, 0)
RUNS = ("runs/experiments/exp_0002/artifacts", "runs/experiments/exp_0003/artifacts")
GRID = "runs/experiments/exp_0025"


def members(configs: list[str]) -> list[str]:
    files = ["configs/tiers/tier0.yaml", *configs]
    for run in RUNS:
        files += [f"{run}/artifact_manifest.json", *(str(p) for p in sorted(Path(run, "models").glob("*.pt")))]
    files += [f"{GRID}/artifact_manifest.json", *(str(p) for p in sorted(Path(GRID, "models").glob("*.pt")))]
    return files


def build(out_dir: Path, configs: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "v110_tc_inputs.zip"
    files = members(configs)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in files:
            info = zipfile.ZipInfo(name, date_time=EPOCH)
            info.external_attr = 0o644 << 16
            zf.writestr(info, Path(name).read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    listing = {"zip": path.name, "sha256": digest, "files": {n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in files}}
    (out_dir / "v110_tc_inputs.json").write_text(json.dumps(listing, indent=2))
    return listing


if __name__ == "__main__":
    res = build(Path(sys.argv[1]), sys.argv[2:])
    print(res["sha256"], len(res["files"]), "files")
