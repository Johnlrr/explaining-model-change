# exp_0039 — run-to-run stability of Recovery, Δ-CE vs DeepXplore (validation, descriptive)

Code: `dce.cli.stability_check`. ~100 valid validation points per pair (Shortcut 0, Subgroup 3, Threshold 6); five
runs per method (restart seeds s0, s0+1, s0+2 in float32; s0, s0+1 in float64). results.json.

| change | Δ-CE Recovery (5 runs) | DeepXplore Recovery (5 runs) | Δ-CE − DeepXplore per run |
|---|---|---|---|
| Shortcut | 0.27 ×5 (SD 0) | 0.32–0.35 (SD 0.013) | −0.05 to −0.08 |
| Subgroup | 0.44–0.45 (SD 0.005) | 0.19–0.25 (SD 0.023) | +0.19 to +0.25 |
| Threshold | 0.40 ×5 (SD 0) | 0.25–0.32 (SD 0.028) | +0.08 to +0.15 |

Found-rates vary by ≤ 0.02 (DeepXplore) and ≤ 0.01 (Δ-CE). DeepXplore's Recovery moves by up to 0.07 between runs
although 33% of its attributions change (exp_0038): the changes largely cancel in the mean.
