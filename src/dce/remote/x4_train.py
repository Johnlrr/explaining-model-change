"""D-052 X4 training on a Kaggle GPU kernel: ResNet-8 f_old (seeds from the config) and every f_new.

The kernel runs `dce.cli.x4_resnet` train-old and train unchanged, with device = cuda. The experiment config is passed
as a parameter (`cfg`: the output of `load_config(configs/experiments/exp_0050.yaml)` without `_tier`) together with
`cfg_sha256`, the SHA256 of its canonical JSON computed locally from the committed file; any mismatch stops the kernel.
Checkpoints are moved to the CPU before saving. Qualification, timing and the search run elsewhere from these
hash-checked checkpoints.

Outputs (flat in /kaggle/working): r8_*.pt, artifact_manifest.json (paths models/<name>.pt, as written by
x4_resnet), train_old_log.json, train_log.json, result.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def canonical_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def main() -> None:
    p = json.loads(os.environ.get("RESEARCH_HARNESS_PARAMETERS", "{}"))
    start = time.time()
    working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("out").resolve()
    working.mkdir(parents=True, exist_ok=True)
    cfg = p["cfg"]
    if canonical_sha256(cfg) != p["cfg_sha256"]:
        raise SystemExit(f"config hash mismatch: {canonical_sha256(cfg)} != {p['cfg_sha256']}")

    import torch

    from dce.cli import x4_resnet

    device = p.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    out = Path(tempfile.mkdtemp(prefix="dce_x4_"))
    run_cfg = cfg | {"device": device}
    rc_old = x4_resnet.cmd_train_old(run_cfg, out)
    rc_new = x4_resnet.cmd_train(run_cfg, out)

    for f in sorted((out / "models").glob("*.pt")):
        shutil.copy(f, working / f.name)
    for name in ("artifact_manifest.json", "train_old_log.json", "train_log.json"):
        if (out / name).exists():
            shutil.copy(out / name, working / name)
    n_models = len(list((out / "models").glob("*.pt")))
    result = {"experiment": p.get("experiment", "exp_0050"), "decision": "D-052 X4", "rc": [rc_old, rc_new],
              "observed_hardware": f"GPU: {gpu}" if gpu else "CPU", "cfg_sha256": p["cfg_sha256"],
              "bundle_tree_hash": os.environ.get("DCE_BUNDLE_TREE_HASH"), "python": sys.version.split()[0],
              "torch": torch.__version__, "cuda": torch.version.cuda, "wall_s": round(time.time() - start, 1),
              "work": {"models": n_models}}
    (working / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
