"""Build the inputs of the D-052 X4 Kaggle CPU kernels (exp_0050 timing and search), as `d052_inputs` does for X1–X3:
tier0.yaml, the committed configs, and the ResNet-8 checkpoints of exp_0050 (trained on the Kaggle T4, kernel
dinhbaobao/dce-exp0050-train) with their manifest, at their repository paths, in a deterministic zip. The kernel entry
checks the listing hash and every file hash; `x4_resnet.load_model` then checks each checkpoint against its manifest.

The zip and its listing keep the names `v110_tc_inputs.{zip,json}` that the kernel entry looks for.

    python -m dce.remote.x4_inputs <out_dir> <config> [<config> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from dce.remote import v110_inputs

X4_MODELS = "runs/experiments/exp_0050"


def members(configs: list[str]) -> list[str]:
    files = ["configs/tiers/tier0.yaml", *configs, f"{X4_MODELS}/artifact_manifest.json"]
    return files + [str(p) for p in sorted(Path(X4_MODELS, "models").glob("*.pt"))]


if __name__ == "__main__":
    v110_inputs.members = members           # same deterministic zip and listing, other member set
    res = v110_inputs.build(Path(sys.argv[1]), sys.argv[2:])
    print(res["sha256"], len(res["files"]), "files")
