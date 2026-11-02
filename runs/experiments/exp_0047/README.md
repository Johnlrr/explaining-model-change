# exp_0047 — D-052 E1–E4 (saved outputs only), descriptive

Code 37c397c (`review_0211 e-analyze`, clean tree). Inputs: split A exp_0017, split B exp_0019, grid exp_0026.
Check: Table 2 is reproduced exactly from the same arrays (`table2_check`).

**E1a — rotation, pair level.** 24 α > 0 grid pairs, Grad-diff Recovery vs measured α̂: Spearman ρ = 0.893, one-sided
permutation p = 1.0·10⁻⁴ (10⁴ permutations within f_old seed; the floor of the test). Prediction met.

**E1b — per-pair estimates.** Grad-diff on α = 0 offsets: 0.027, 0.123, 0.070, 0.103 (4/4 < 0.2). Grad-diff by α:
0.1 → 0.16–0.54; 0.25 → 0.55–0.78; 0.5 → 0.84–0.94 (ranges do not overlap between 0.1 and 0.5). Δ-CE − Grad-diff:
Shortcut A −0.56…−0.51, B −0.59…−0.55; Subgroup A −0.12…−0.06 (3/3 negative, as the rejection); Threshold A +0.38…+0.41,
B +0.27…+0.36. Null witnesses: 0 in 8/8 pairs.

**E2 — Imp-diff.** One ΔPI vector per pair, shared by all points (17/17); argmax = aspect in all 9 change pairs, so
Recovery is 1 on Threshold and 0 on Shortcut and Subgroup by construction. Seg-search found-rate 0.05–0.13.

**E3 — rank metrics.** Prediction (same order by MRR as by Recovery) **not met** in 6 of 7 groups. The first two methods
are the same in every group; swaps are among the lower ones (Imp-diff, CF-new, No-prox, Seg-search) and, on the grid,
Δ-CE falls one place under MRR (NOT_FOUND counts as reciprocal rank 0, so its found-rate weighs more).
Set-recovery {aspect, colour} on α > 0: CF-new 0.49, CF-diff 0.39, Δ-CE 0.37, Grad-diff 0.11.

**E4 — κ and r.** Null witness share: split A κ = 0 → 0.833, κ = 0.5 → 0, κ = 1 → 0; grid null cell (pairs 20, 21,
realised t̂ ≈ 0.53, i.e. a small real offset) κ = 0 → 0.71, 0.25 → 0.51, 0.5 → 0.22, 1.0 → 0. Found-rate non-increasing in
κ in every group (prediction met). Split A Recovery at r′ = 0.30 / 0.40 / 0.54: Shortcut 0.13 / 0.22 / 0.37, Subgroup
0.30 / 0.40 / 0.53, Threshold 0.33 / 0.45 / 0.47.
