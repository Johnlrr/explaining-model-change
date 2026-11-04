"""Build the inputs of the D-056 X5 Kaggle CPU kernels (exp_0057), as `d052_inputs` does for X2/X3: tier0.yaml, the
committed configs, the f_old checkpoints (exp_0002), the locked split-D f_new checkpoints (exp_0049) and the X5 VAE
(exp_0056: vae_gate.json + checkpoint), with their manifests, at their repository paths, in a deterministic zip. The
kernel entry checks the listing hash and every file hash; `load_verified`, `load_d_models` and `x5_vae.load_vae` then
check each checkpoint against its manifest as locally.

The zip and its listing keep the names `v110_tc_inputs.{zip,json}` that the kernel entry looks for.

    python -m dce.remote.x5_inputs <out_dir> <config> [<config> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from dce.remote import v110_inputs

F_OLD = "runs/experiments/exp_0002/artifacts"
SPLIT_D_MODELS = "runs/experiments/exp_0049"
VAE_RUN = "runs/experiments/exp_0056"


def members(configs: list[str]) -> list[str]:
    files = ["configs/tiers/tier0.yaml", *configs]
    files += [f"{F_OLD}/artifact_manifest.json", *(str(p) for p in sorted(Path(F_OLD, "models").glob("f_old_*.pt")))]
    files += [f"{SPLIT_D_MODELS}/artifact_manifest.json", *(str(p) for p in sorted(Path(SPLIT_D_MODELS, "models").glob("*.pt")))]
    files += [f"{VAE_RUN}/vae_gate.json", *(str(p) for p in sorted(Path(VAE_RUN, "models").glob("*.pt")))]
    return files


if __name__ == "__main__":
    v110_inputs.members = members           # same deterministic zip and listing, other member set
    res = v110_inputs.build(Path(sys.argv[1]), sys.argv[2:])
    print(res["sha256"], len(res["files"]), "files")
