# exp_0030 — v1.10 H8 development (validation only, descriptive)

D-043 draft. H8 2×2 on the H5d pairs 36, 37, 44, 45, 50 validation points each (clean_splits(1234)["val"]).
Command: `uv run python -m dce.cli.v110_dev h8 --config configs/experiments/exp_0030.yaml --out runs/experiments/exp_0030`.
Code was uncommitted at run time (working tree on top of 927918d); it is committed together with the v1.10 amendment.
Output: pairs/pair_*.{npz,json}, h8_dev.json (pooled gaps, bootstrap seed 0, 2 000 resamples — development only).
