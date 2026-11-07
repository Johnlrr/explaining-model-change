# exp_0061 — D-059: checked counterfactual of f_new and null-adjusted Recovery (descriptive, after seeing results)

`uv run python -m dce.cli.cfnew_filter --config configs/experiments/exp_0061.yaml --out runs/experiments/exp_0061`
(local CPU, a few minutes). Saved outputs of exp_0017 (split A) and exp_0060 (X6, split G); models hash-verified
(exp_0002/exp_0003; exp_0060). Every saved query re-checked valid under the loaded models; recomputed Δ-CE Recovery
equals the saved value for every change pair. No search was re-run. results_cfnew_filter.json.

Checked CF-new = saved CF-new point (m_n ≤ −κ) kept only if m_o ≥ κ there.

| Split A | found Δ-CE | found checked CF | Rec. Δ-CE | Rec. checked CF | Δ-CE Rec. from case B |
|---|---|---|---|---|---|
| Shortcut | 0.84 | 0.42 | 0.38 | 0.03 | 0.35 |
| Subgroup | 0.81 | 0.46 | 0.53 | 0.30 | 0.22 |
| Threshold | 0.47 | 0.46 | 0.47 | 0.45 | 0.01 |

Nulls (8 pairs): checked CF-new found 0.000 (CF-new unchecked 0.996), Δ-CE 0.000. X6: checked CF-new found 0.87
(Δ-CE 0.98), pooled L 0.48 (Δ-CE 0.50). Null-adjusted Recovery (Recovery − rate of naming k* on the 8 nulls):
Shortcut Grad-diff +0.79 first, Δ-CE +0.38; Subgroup Δ-CE +0.53 first; Threshold Δ-CE +0.47, Imp-diff +0.38,
CF-diff +0.37, CF-new −0.09. Reading: the check m_o ≥ κ gives silence; the search adds witnesses where CF-new's point
is not one, which matters on Shortcut and Subgroup, not on Threshold.

## Addendum (same day): realised thresholds of the split-A nulls

`null_boundaries` (same fit and validation points as the grid qualification exp_0025, v1.9.1): t̂_new − t̂_old of
the 8 split-A nulls +0.017 … +0.035 (largest |Δt̂| 0.035, |α̂| ≤ 0.053); grid null cell (pairs 20, 21) +0.039,
−0.032. Threshold shifts are similar; the recipes differ: nulls of splits A, D, F use 3 epochs at 3·10⁻⁴, offsets
and the grid (incl. its null cell) 10 epochs at 10⁻³ (configs/experiments/exp_0049.yaml, exp_0025). The grid null
cell has witnesses on 0.22 of queries at κ = 0.5 (0 at κ = 1; supplement Table, from saved outputs).
Per pair, Δ-CE Recovery > checked CF-new in 3/3 Shortcut and 3/3 Subgroup pairs.
