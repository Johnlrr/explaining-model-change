# exp_0031 — v1.10 T-C cost (validation only, descriptive)

D-043 draft. Pairs 20 and 26, 20 validation points each; max-slack search on up to 5 NOT_FOUND queries at M = 1e4, 1e5.
Command: `uv run python -m dce.cli.v110_dev tc-cost --config configs/experiments/exp_0031.yaml --out runs/experiments/exp_0031`.
Ran concurrently with exp_0030 (wall times are upper bounds). Known issue: this run reported only the ascent-stage
maximum (`s_ascent`); `s_max` = max(sampling, ascent) was added afterwards. Use the file for cost only.
