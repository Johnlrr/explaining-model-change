"""Build the inputs of the D-052 Kaggle CPU kernels (X1 exp_0048, X2/X3 exp_0049), as `v110_inputs` does for D-044:
the committed configs, the checkpoints with their manifests and the split-A point files of exp_0017, at their
repository paths, in a deterministic zip. The kernel entry checks the listing hash and every file hash;
`load_verified` / `load_pair` / `load_d_models` then check each checkpoint against its manifest as locally.

The zip and its listing keep the names `v110_tc_inputs.{zip,json}` that the kernel entry looks for.

    python -m dce.remote.d052_inputs <out_dir> <config> [<config> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from dce.remote import v110_inputs

RUNS = ("runs/experiments/exp_0002/artifacts", "runs/experiments/exp_0003/artifacts")
SPLIT_A = "runs/experiments/exp_0017/pairs"
SPLIT_D_MODELS = "runs/experiments/exp_0049"


def members(configs: list[str]) -> list[str]:
    files = ["configs/tiers/tier0.yaml", *configs]
    for run in RUNS:
        files += [f"{run}/artifact_manifest.json", *(str(p) for p in sorted(Path(run, "models").glob("*.pt")))]
    files += [f"{SPLIT_D_MODELS}/artifact_manifest.json", *(str(p) for p in sorted(Path(SPLIT_D_MODELS, "models").glob("*.pt")))]
    files += [str(p) for p in sorted(Path(SPLIT_A).glob("pair_*.npz"))]
    return files


if __name__ == "__main__":
    v110_inputs.members = members           # same deterministic zip and listing, other member set
    res = v110_inputs.build(Path(sys.argv[1]), sys.argv[2:])
    print(res["sha256"], len(res["files"]), "files")
