"""Build the inputs of the D-057 X6 Kaggle CPU kernels (exp_0060), as `x5_inputs` does for X5: tier0.yaml, the committed
configs, the f_old checkpoints (exp_0002) and the X6 f_new checkpoints (exp_0060) with their manifests, at their
repository paths, in a deterministic zip named `v110_tc_inputs.{zip,json}` (the kernel entry checks every hash).

    python -m dce.remote.x6_inputs <out_dir> <config> [<config> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from dce.remote import v110_inputs

F_OLD = "runs/experiments/exp_0002/artifacts"
X6_MODELS = "runs/experiments/exp_0060"


def members(configs: list[str]) -> list[str]:
    files = ["configs/tiers/tier0.yaml", *configs]
    files += [f"{F_OLD}/artifact_manifest.json", *(str(p) for p in sorted(Path(F_OLD, "models").glob("f_old_*.pt")))]
    files += [f"{X6_MODELS}/artifact_manifest.json", *(str(p) for p in sorted(Path(X6_MODELS, "models").glob("*.pt")))]
    return files


if __name__ == "__main__":
    v110_inputs.members = members
    res = v110_inputs.build(Path(sys.argv[1]), sys.argv[2:])
    print(res["sha256"], len(res["files"]), "files")
