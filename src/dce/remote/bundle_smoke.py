"""Entry module for the bundle smoke (D-026): imports the package and runs a tiny solver check."""

import json
import os
from pathlib import Path

import numpy as np
import torch

import dce

from dce.data.tierL import closed_form, sample_pair
from dce.models.cnn import SmallCNN
from dce.solvers.dce import delta_ce
from dce.solvers.search import SolverConfig


def main():
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("out")
    out.mkdir(parents=True, exist_ok=True)
    pairs = [sample_pair(60_000 + s, 5) for s in range(8)]
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)  # noqa: E731
    wo, wn, bo, bn, z0 = (t([q.w_o for q in pairs]), t([q.w_n for q in pairs]), t([q.b_o for q in pairs]),
                          t([q.b_n for q in pairs]), t([q.z0 for q in pairs]))
    res = delta_ce(z0, lambda z, i: (wo[i] * z).sum(1) + bo[i], lambda z, i: (wn[i] * z).sum(1) + bn[i],
                   SolverConfig(R=2, T=100, n_workers=2), restart_seed=1)
    rel = [abs(res.distance[i].item() / closed_form(q)["delta"] - 1) for i, q in enumerate(pairs)]
    result = {"schema_version": 1, "kind": "bundle_smoke", "tree_hash": os.environ.get("DCE_BUNDLE_TREE_HASH"),
              "dce_file": dce.__file__, "from_bundle": "dce_bundle_" in dce.__file__,
              "cnn_params": sum(p.numel() for p in SmallCNN().parameters()), "found": int(res.found.sum()),
              "max_rel_error": max(rel), "torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "work": {"processed": len(pairs)}}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "artifacts": []}))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
