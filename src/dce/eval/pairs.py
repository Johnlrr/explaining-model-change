"""Load a Tier 0 model pair from saved, hash-verified checkpoints (G0 runs exp_0002 / exp_0003)."""

from __future__ import annotations

from pathlib import Path

from dce.cli.g0_reeval import load_verified
from dce.data.drift import tier0_pairs

# The checkpoints that passed G0 (exp_0004, amendment v1.3).
DEFAULT_SOURCES = {"f_old": "runs/experiments/exp_0002", "N0": "runs/experiments/exp_0002",
                   "N1": "runs/experiments/exp_0002", "T1": "runs/experiments/exp_0003",
                   "T3": "runs/experiments/exp_0003", "T4": "runs/experiments/exp_0003"}


def load_pair(pair_id: int, sources: dict | None = None, cache: dict | None = None):
    sources = {k: Path(v) for k, v in (sources or DEFAULT_SOURCES).items()}
    cache = {} if cache is None else cache
    p = next(q for q in tier0_pairs() if q["pair_id"] == pair_id)
    f_old, e_old = load_verified(sources["f_old"], f"f_old_s{p['seed']}", cache)
    f_new, e_new = load_verified(sources[p["condition"]], f"f_new_p{pair_id:02d}_{p['condition']}_s{p['seed']}", cache)
    return p, f_old, f_new, [e_old, e_new]
