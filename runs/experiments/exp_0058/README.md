# exp_0058 — D-056 S1/S2: pair-level inference and runtime (saved outputs only, descriptive, not registered)

Code commit `e658587` (`python -m dce.cli.pair_level run --out runs/experiments/exp_0058`; tree clean, `dirty` = False). Rules: `research/DECISIONS.md` D-056 (S1 revised 2026-11-04). Output: `results_s.json` (per-pair values, tests, cross-checks, MixedLM, S2, input tree hashes). No training, no solver run, no Kaggle. Unit: the pair. Test for a threshold row: exact one-sided sign test over pairs (k of n pairs on the claimed side, pairs equal to the threshold dropped); for differences: also an exact sign-flip permutation test on the pair means (2^n patterns); t 95% CI on the pair means for n ≥ 4 (none for n < 4). Splits A and B use the same model pairs on different data, so pooling A+B is the per-pair mean of the two estimates (pairs do not accumulate).

Inputs: exp_0017 (A), exp_0019 (B), exp_0026 (grid), exp_0028 (Adult), exp_0032 (H7), exp_0033 (H8), exp_0048 (X1, S2 only), exp_0049 (D), exp_0050, exp_0054/test (ResNet-8); exp_0047 results_e.json for cross-checks. Tree hashes: `results_s.json` `inputs`.

## S1 (a): per-pair table

| id | Table 3 row | quantity per pair | n | per-pair min–max | pair mean | t 95% CI | claimed side | exact sign p (k/n) | sign-flip p | MixedLM (≥ 8 pairs) |
|---|---|---|---|---|---|---|---|---|---|---|
| abst_null_B | No witness on nulls \| Null, B | witness share (type N or O) per pair, exp_0019 results.json C2 found_rate_union | 8 | 0.000 – 0.000 | 0.000 | [0.000, 0.000] | below 0.05 | 0.0039 (8/8) | — | not fitted: constant outcome |
| abst_adult | No witness on nulls \| Adult | witness share (type N or O) per pair, exp_0028 h6.json | 5 | 0.000 – 0.000 | 0.000 | [0.000, 0.000] | below 0.05 | 0.0312 (5/5) | — | — |
| abst_null_D | No witness on unfiltered nulls \| Null, D | witness share (type N or O) per pair, exp_0049 results_d.json x3 | 8 | 0.000 – 0.000 | 0.000 | [0.000, 0.000] | below 0.05 | 0.0039 (8/8) | — | not fitted: constant outcome |
| abst_resnet_D | No witness on nulls \| ResNet-8, D \| kappa = 0.5 | witness share (type N or O) per pair (100 queries per pair) | 2 | 0.890 – 0.910 | 0.900 | none (n<4) | below 0.05 | 1.0000 (0/2) | — | — |
| abst_resnet_E | No witness on nulls \| ResNet-8, E \| kappa* = 2.5 | witness share (type N or O) per pair (100 queries per pair) | 2 | 0.000 – 0.000 | 0.000 | none (n<4) | below 0.05 | 0.2500 (2/2) | — | — |
| cfdiff_flags | CF-diff (no cut) flags \|b\| < 2 kappa \| Logit offsets | CF-diff found rate per pair (12 pairs: f_old seed 0-2 x b in {-0.9, -0.5, 0.5, 0.9}) | 12 | 0.927 – 0.997 | 0.961 | [0.947, 0.976] | above 0.9 | 2.4e-04 (12/12) | — | Intercept: 0.9614 (SE 0.0066, p <1e-16 (underflow)); conv True; Wald vs 0.9: p 1.3e-20; warning: boundary |
| gd_lt02_grid | Grad-diff < 0.2 \| Grid, alpha = 0 | Recovery of Grad-diff per pair (k* = aspect), pairs 24-27 | 4 | 0.027 – 0.123 | 0.081 | [0.014, 0.148] | below 0.2 | 0.0625 (4/4) | — | — |
| gd_lt02_D | Grad-diff < 0.2 \| Offsets, D | Recovery of Grad-diff per pair (k* = aspect), 8 qualified offset pairs (pair 103 not qualified) | 8 | 0.033 – 0.093 | 0.056 | [0.036, 0.076] | below 0.2 | 0.0039 (8/8) | — | Intercept: 0.0563 (SE 0.0084, p 1.8e-11); conv True; Wald vs 0.2: p 1.9e-66; warning: boundary |
| gd_offset_free | Grad-diff offset-free \| Grid | Rec(Grad-diff; alpha 0.25, t 0.60) - Rec(Grad-diff; 0.25, 0.50), k* = colour, per f_old seed (2 matched pairs) | 2 | -0.133 – -0.063 | -0.098 | none (n<4) | \|Δ\| ≤ 0.10 | in band 1/2 | 0.5000 (two-sided) | — |
| gd_rises_contrast | Grad-diff rises with alpha \| Grid (registered contrast H5b) | Rec(Grad-diff; alpha 0.5, t 0.5) - Rec(Grad-diff; alpha 0.1, t 0.5), k* = colour, per f_old seed (2 matched pairs) | 2 | 0.537 – 0.603 | 0.570 | none (n<4) | > 0 | 0.2500 (2/2) | 0.2500 (greater) | — |
| gd_rises_rotation | Grad-diff rises with alpha \| Grid (rho = 0.89) | Recovery of Grad-diff per pair (k* = colour) against measured alpha_hat, 24 pairs with alpha > 0 | 24 | 0.163 – 0.937 | 0.629 | [0.523, 0.734] | ρ > 0 | ρ = 0.893, perm p = 1.0e-04 | — | alpha_hat: 1.3744 (SE 0.1432, p 8.0e-22); conv True |
| dce_gt_gd_threshold_B | Delta-CE > Grad-diff \| Threshold, B | Recovery(Delta-CE) - Recovery(Grad-diff) per pair, split B, pairs 6-8 (k* = aspect) | 3 | 0.270 – 0.360 | 0.312 | none (n<4) | > 0 | 0.1250 (3/3) | 0.1250 (greater) | — |
| dce_gt_gd_threshold_A | Delta-CE > Grad-diff \| Threshold, A (same claim, locked split) | same quantity, split A | 3 | 0.380 – 0.407 | 0.396 | none (n<4) | > 0 | 0.1250 (3/3) | 0.1250 (greater) | — |
| dce_gt_gd_threshold_AB | Delta-CE > Grad-diff \| Threshold, A+B pooled | per-pair mean of the split-A and split-B differences (same models, two data sets) | 3 | 0.338 – 0.380 | 0.354 | none (n<4) | > 0 | 0.1250 (3/3) | 0.1250 (greater) | — |
| dce_gt_gd_subgroup | Delta-CE > Grad-diff \| Subgroup | Recovery(Delta-CE) - Recovery(Grad-diff) per pair, split A, pairs 3-5 (k* = size) | 3 | -0.123 – -0.063 | -0.091 | none (n<4) | > 0 | 1.0000 (0/3) | 1.0000 (greater) | — |
| tie_shortcut_A | Tie (\|Delta\| <= 0.1) \| Shortcut, A | Recovery(Delta-CE) - Recovery(Grad-diff) per pair, split A | 3 | -0.563 – -0.507 | -0.533 | none (n<4) | \|Δ\| ≤ 0.10 | in band 0/3 | 0.2500 (two-sided) | — |
| tie_shortcut_B | Tie (\|Delta\| <= 0.1) \| Shortcut, B | same, split B | 3 | -0.593 – -0.547 | -0.570 | none (n<4) | \|Δ\| ≤ 0.10 | in band 0/3 | 0.2500 (two-sided) | — |
| tie_shortcut_AB | Tie (\|Delta\| <= 0.1) \| Shortcut, A+B pooled | per-pair mean of A and B | 3 | -0.578 – -0.527 | -0.552 | none (n<4) | \|Δ\| ≤ 0.10 | in band 0/3 | 0.2500 (two-sided) | — |
| tie_threshold_A | Tie (\|Delta\| <= 0.1) \| Threshold, A | Recovery(Delta-CE) - Recovery(Grad-diff) per pair, split A | 3 | 0.380 – 0.407 | 0.396 | none (n<4) | \|Δ\| ≤ 0.10 | in band 0/3 | 0.2500 (two-sided) | — |
| tie_threshold_B | Tie (\|Delta\| <= 0.1) \| Threshold, B | same, split B | 3 | 0.270 – 0.360 | 0.312 | none (n<4) | \|Δ\| ≤ 0.10 | in band 0/3 | 0.2500 (two-sided) | — |
| tie_threshold_AB | Tie (\|Delta\| <= 0.1) \| Threshold, A+B pooled | per-pair mean of A and B | 3 | 0.338 – 0.380 | 0.354 | none (n<4) | \|Δ\| ≤ 0.10 | in band 0/3 | 0.2500 (two-sided) | — |
| proj_shortcut_B | Proj. Delta-CE ~ Grad-diff \| Shortcut, B | projected Recovery(Delta-CE) - projected Recovery(Grad-diff) per pair, split B, pairs 0-2 (k* = colour) | 3 | -0.323 – -0.273 | -0.304 | none (n<4) | \|Δ\| ≤ 0.10 | in band 0/3 | 0.2500 (two-sided) | — |
| proj_grid | Proj. Delta-CE ~ Grad-diff \| grid | projected Recovery(Delta-CE) - projected Recovery(Grad-diff) per pair, grid pairs 36, 37, 44, 45 (k* = colour) | 4 | -0.400 – -0.307 | -0.352 | [-0.413, -0.292] | \|Δ\| ≤ 0.10 | in band 0/4 | 0.1250 (two-sided) | — |
| normal_zstar | Normal at z* halves the gap or more \| Grid, new | T_i = \|gap(z0,z0)\| - 2 \|gap(z*,z0)\| per pair (T_i > 0: the gap narrows by at least a half), 4 pairs, support = queries with a witness | 4 | 0.011 – 0.268 | 0.119 | [-0.077, 0.314] | above 0 | 0.0625 (4/4) | — | — |

Min attainable one-sided p by n: n=2 0.25, n=3 0.125, n=4 0.0625 (rows with n ≤ 4 cannot reach p < 0.05 at pair level).

### Per-pair values

- **abst_null_B**: 9: 0.0000; 10: 0.0000; 11: 0.0000; 12: 0.0000; 13: 0.0000; 14: 0.0000; 15: 0.0000; 16: 0.0000
- **abst_adult**: 0: 0.0000; 1: 0.0000; 2: 0.0000; 3: 0.0000; 4: 0.0000
- **abst_null_D**: 110: 0.0000; 111: 0.0000; 112: 0.0000; 113: 0.0000; 114: 0.0000; 115: 0.0000; 116: 0.0000; 117: 0.0000
- **abst_resnet_D**: 128: 0.9100; 129: 0.8900
- **abst_resnet_E**: 128: 0.0000; 129: 0.0000
- **cfdiff_flags**: 63: 0.9500; 64: 0.9967; 65: 0.9600; 66: 0.9467; 73: 0.9400; 74: 0.9833; 75: 0.9700; 76: 0.9567; 83: 0.9367; 84: 0.9967; 85: 0.9733; 86: 0.9267
- **gd_lt02_grid**: 24: 0.0267; 25: 0.1233; 26: 0.0700; 27: 0.1033
- **gd_lt02_D**: 100: 0.0500; 101: 0.0433; 102: 0.0933; 104: 0.0433; 105: 0.0533; 106: 0.0400; 107: 0.0333; 108: 0.0933
- **gd_offset_free**: 42: -0.1333; 43: -0.0633
- **gd_rises_contrast**: 44: 0.6033; 45: 0.5367
- **gd_rises_rotation**: 28: 0.3200; 29: 0.4000; 30: 0.1633; 31: 0.5367; 32: 0.3200; 33: 0.2800; 34: 0.2800; 35: 0.2800; 36: 0.6800; 37: 0.6833; 38: 0.7800; 39: 0.6533; 40: 0.7033; 41: 0.7367; 42: 0.5467; 43: 0.6200; 44: 0.9233; 45: 0.9367; 46: 0.9100; 47: 0.8933; 48: 0.9033; 49: 0.8500; 50: 0.8500; 51: 0.8400
- **dce_gt_gd_threshold_B**: 6: 0.3600; 7: 0.2700; 8: 0.3067
- **dce_gt_gd_threshold_A**: 6: 0.4000; 7: 0.4067; 8: 0.3800
- **dce_gt_gd_threshold_AB**: 6: 0.3800; 7: 0.3383; 8: 0.3433
- **dce_gt_gd_subgroup**: 3: -0.0867; 4: -0.1233; 5: -0.0633
- **tie_shortcut_A**: 0: -0.5633; 1: -0.5300; 2: -0.5067
- **tie_shortcut_B**: 0: -0.5933; 1: -0.5700; 2: -0.5467
- **tie_shortcut_AB**: 0: -0.5783; 1: -0.5500; 2: -0.5267
- **tie_threshold_A**: 6: 0.4000; 7: 0.4067; 8: 0.3800
- **tie_threshold_B**: 6: 0.3600; 7: 0.2700; 8: 0.3067
- **tie_threshold_AB**: 6: 0.3800; 7: 0.3383; 8: 0.3433
- **proj_shortcut_B**: 0: -0.3233; 1: -0.3167; 2: -0.2733
- **proj_grid**: 36: -0.3067; 37: -0.3567; 44: -0.3467; 45: -0.4000
- **normal_zstar**: 36: 0.0253; 37: 0.0109; 44: 0.1711; 45: 0.2675

Rotation (24 pairs, α > 0), per α: α=0.1: n=8, 0.163–0.537, mean 0.323; α=0.25: n=8, 0.547–0.780, mean 0.675; α=0.5: n=8, 0.840–0.937, mean 0.888.

### Cross-checks against saved values

- abst_null_B: type-N found rate from arrays (N only; type O is not in the arrays) vs union: max |diff| 0.00e+00, match
- abst_adult: union found rate from arrays vs h6.json: max |diff| 0.00e+00, match
- abst_null_D: union found rate from arrays vs results_d.json: max |diff| 0.00e+00, match
- abst_resnet_D: union found rate from arrays vs results json: max |diff| 0.00e+00, match
- abst_resnet_E: union found rate from arrays vs results json: max |diff| 0.00e+00, match
- cfdiff_flags: per-pair found rate vs results_h7.json H7c.per_pair_found_b1_det (order of saved list): max |diff| 0.00e+00, match
- cfdiff_flags: pair mean vs H7c estimate: max |diff| 1.11e-16, match
- gd_lt02_grid: pair mean vs results.json H5a.estimate: max |diff| 0.00e+00, match
- gd_lt02_grid: per-pair vs exp_0047 e1b graddiff_lt_0.2_grid_alpha0: max |diff| 0.00e+00, match
- gd_lt02_D: per-pair vs results_d.json x2.graddiff_recovery: max |diff| 0.00e+00, match
- gd_lt02_D: sign-test p vs the saved 0.0039 (exact 2^-8): max |diff| 0.00e+00, match
- gd_offset_free: mean of the 2 matched differences vs results.json H5c.estimate: max |diff| 0.00e+00, match
- gd_rises_contrast: mean of the 2 matched differences vs results.json H5b.estimate: max |diff| 0.00e+00, match
- gd_rises_rotation: Spearman rho vs exp_0047 e1a_rotation: max |diff| 0.00e+00, match
- gd_rises_rotation: permutation p vs exp_0047: max |diff| 0.00e+00, match
- gd_rises_rotation: per-pair Recovery (rank-based) vs exp_0047 graddiff_recovery: max |diff| 0.00e+00, match
- gd_rises_rotation: per-pair Recovery (recovery_indicator) vs rank-based: max |diff| 0.00e+00, match
- dce_gt_gd_threshold_B: per-pair vs exp_0047 e1b: max |diff| 0.00e+00, match
- dce_gt_gd_threshold_A: per-pair vs exp_0047 e1b: max |diff| 0.00e+00, match
- dce_gt_gd_subgroup: per-pair vs exp_0047 e1b: max |diff| 0.00e+00, match
- tie_shortcut_A: per-pair vs exp_0047 e1b: max |diff| 0.00e+00, match
- tie_shortcut_B: per-pair vs exp_0047 e1b: max |diff| 0.00e+00, match
- tie_threshold_A: per-pair vs exp_0047 e1b: max |diff| 0.00e+00, match
- tie_threshold_B: per-pair vs exp_0047 e1b: max |diff| 0.00e+00, match
- proj_shortcut_B: per-pair vs exp_0019 results.json C3.seed_diffs_projected: max |diff| 0.00e+00, match
- proj_grid: pair mean vs results.json H5d.estimate: max |diff| 0.00e+00, match
- normal_zstar: mean over pairs of gap(z0,z0) vs results_h8.json H8.gap[z0,z0]: max |diff| 0.00e+00, match
- normal_zstar: mean over pairs of gap(z*,z0) vs results_h8.json H8.gap[z*,z0]: max |diff| 0.00e+00, match
- normal_zstar: registered T = |mean gap(z0,z0)| - 2 |mean gap(z*,z0)| from the pair means vs H8.T: max |diff| 0.00e+00, match

Cross-check mismatches: 0.

### Notes (facts about the implementation)

- Nulls, split B: per-pair value = `found_rate_union` (N or O) from exp_0019 `results.json` C2; split-B arrays hold type N only, so the (empty) MixedLM outcome is type-N found. Adult: `h6.json` union rate, checked against the arrays (N or O).
- Nulls (B, D) have a constant outcome (all 0): MixedLM is not fitted. Adult, ResNet-8 rows have < 8 pairs: no MixedLM.
- Claimed-side thresholds used for the sign test: nulls and ResNet nulls witness share ≤ 0.05 (the registered per-pair criterion of X3); CF-diff found rate > 0.90 (H7c criterion); Grad-diff Recovery < 0.2; differences > 0. The saved `0/2` ResNet-8 split-D row corresponds to 0 pairs with share ≤ 0.05.
- CF-diff row: the 12 pairs of exp_0032 with |b| < 1 = 2κ (b ∈ {−0.9, −0.5, 0.5, 0.9} × f_old seeds 0–2); per-pair values equal `H7c.per_pair_found_b1_det`.
- gd_offset_free, gd_rises_contrast: the two cells of each registered contrast are matched by f_old seed (2 pairs per cell; D-056 does not fix the matching).
- Rotation Spearman and permutation p are the exp_0047 E1a call (Monte Carlo, 10 000 permutations within f_old seed, seed 5201; floor 1/10001); the MixedLM is Recovery ~ α̂ + (1 | pair) on 7 200 points, REML, linear model on a 0/1 outcome.
- normal_zstar: T_i = |gap(z0,z0)| − 2|gap(z*,z0)| per pair (the registered T uses the pooled gaps; the mean of the T_i differs from the saved T, the registered T is reproduced from the pair means, see cross-checks).
- Recovery per point = `recovery_indicator` (found and top factor = k*); per-pair values equal the saved rank-based values of exp_0047 to numerical precision (cross-checks).
- Wald z "vs claim" in MixedLM is the intercept tested against the row's threshold (one-sided) next to the default test against 0.

## S2: time per query (saved `runtime` fields)

No forward/backward counts are stored anywhere in the pair jsons. The stored fields are `wall_s_per_point` (pair wall time / n points, solver run with parallel workers), `cpu_s_per_point` (solver CPU seconds / n points), phase-1 wall and phase-2 CPU split. Table: mean over pairs (min–max). Runs were made on different machines (see each experiment's README and metadata); times are not comparable across runs. The README tables were generated from results_s.json.

**A exp_0017** (17 pairs, [300] points per pair)

| method | wall s/query | CPU s/query | other fields |
|---|---|---|---|
| a1 | 0.656 (0.59–0.835) | 5.17 (4.66–6.59) |  |
| a2 | 0.617 (0.548–0.724) | 4.86 (4.3–5.72) | phase1_wall_s: 97.7; phase2_cpu_s_per_point_with_witness: 2.26; phase2_cpu_s_per_point_without_witness: 2.81 |
| a3_k0 | 2.16 (1.2–9.78) | 13.2 (9.52–17.4) | phase1_wall_s: 197; phase2_cpu_s_per_point_with_witness: 5.78; phase2_cpu_s_per_point_without_witness: 13.6 |
| a3_k1 | 2.54 (1.45–3.41) | 20.2 (11.5–27.1) | phase1_wall_s: 202; phase2_cpu_s_per_point_with_witness: 8.37; phase2_cpu_s_per_point_without_witness: 16.4 |
| b1 | 1.27 (1.13–1.47) | 10 (9–11.6) |  |
| b1_det | 0.351 (0.31–0.44) | 2.76 (2.41–3.46) |  |
| b3 | 0.000666 (0.000599–0.000755) | — |  |
| b4 | — | — | wall_s_total: 24.4 |
| b5 | 0.432 (0.404–0.485) | — |  |
| dce_N | 2.42 (1.28–3.31) | 19.1 (10–26.2) | phase1_wall_s: 201; phase2_cpu_s_per_point_with_witness: 5.75; phase2_cpu_s_per_point_without_witness: 16.4 |
| dce_O | 3.2 (2.96–3.66) | 25.3 (23.6–29.1) | phase1_wall_s: 208; phase2_cpu_s_per_point_without_witness: 19.9 |

**B exp_0019** (14 pairs, [300] points per pair)

| method | wall s/query | CPU s/query | other fields |
|---|---|---|---|
| b1 | 1.25 (1.16–1.3) | — |  |
| b1_det | 0.341 (0.309–0.373) | — |  |
| b3 | 0.000674 (0.000635–0.000749) | — |  |
| dce_N | 4.01 (1.23–23.5) | 20 (9.77–25.6) | phase1_wall_s: 188; phase2_cpu_s_per_point_with_witness: 5.05; phase2_cpu_s_per_point_without_witness: 16.5 |
| dce_O | 3.1 (2.93–3.24) | 24.6 (23.2–25.6) | phase1_wall_s: 194; phase2_cpu_s_per_point_without_witness: 19.4 |

**grid exp_0026** (32 pairs, [300] points per pair)

| method | wall s/query | CPU s/query | other fields |
|---|---|---|---|
| a2 | 1.09 (0.558–1.4) | 8.33 (4.34–10.5) | phase1_wall_s: 157; phase2_cpu_s_per_point_with_witness: 4.3; phase2_cpu_s_per_point_without_witness: 5.36 |
| b1_det | 0.641 (0.301–0.853) | — |  |
| b3 | 0.00202 (0.000652–0.0133) | — |  |
| dce_N | 4.39 (2.02–23.2) | 29.2 (15.9–47.1) | phase1_wall_s: 318; phase2_cpu_s_per_point_with_witness: 13.3; phase2_cpu_s_per_point_without_witness: 26.7 |
| dce_N_k0 | 3.26 (1.67–4.77) | 24.9 (13.1–36.3) | phase1_wall_s: 278; phase2_cpu_s_per_point_with_witness: 8.68; phase2_cpu_s_per_point_without_witness: 28.5 |
| dce_N_k0.25 | 3.73 (1.93–5.84) | 28.7 (15.3–43.5) | phase1_wall_s: 281; phase2_cpu_s_per_point_with_witness: 10.5; phase2_cpu_s_per_point_without_witness: 31.5 |
| dce_N_k1.0 | 7.41 (2.71–23.9) | 39.7 (21.5–55.5) | phase1_wall_s: 279; phase2_cpu_s_per_point_without_witness: 32.6; phase2_cpu_s_per_point_with_witness: 13.9 |

**H7 exp_0032** (30 pairs, [300] points per pair)

| method | wall s/query | CPU s/query | other fields |
|---|---|---|---|
| b1_det | 0.694 (0.368–0.941) | — |  |
| dce_N | 5.04 (3.56–6.65) | 37.2 (26.7–51.6) | phase1_wall_s: 333; phase2_cpu_s_per_point_with_witness: 9.54; phase2_cpu_s_per_point_without_witness: 35.7 |

**H8 exp_0033** (24 pairs, [300] points per pair)

| method | wall s/query | CPU s/query | other fields |
|---|---|---|---|
| dce_N | 3.91 (1.73–6.73) | 30.5 (13.5–52.8) | phase1_wall_s: 332; phase2_cpu_s_per_point_with_witness: 14.4; phase2_cpu_s_per_point_without_witness: 28 |

**D exp_0049** (17 pairs, [300] points per pair)

| method | wall s/query | CPU s/query | other fields |
|---|---|---|---|
| dce_N | 10.3 (7.96–11.8) | 40.4 (31.5–46.3) | phase1_wall_s: 682; phase2_cpu_s_per_point_with_witness: 19.1; phase2_cpu_s_per_point_without_witness: 35 |
| dce_O | 11.1 (10.7–11.5) | 43.7 (42.2–45.3) | phase1_wall_s: 678; phase2_cpu_s_per_point_without_witness: 34.8 |

**X1 exp_0048** (17 pairs)

| method | wall s/query | CPU s/query | other fields |
|---|---|---|---|
| ig | 0.125 (0.0737–0.148) | — | wall_s_per_point_from_pair_total: 0.125 |
| shap | 4.12 (2.38–4.73) | — | wall_s_per_point_from_pair_total: 4.12 |
| b4a | 0.17 (0.0978–0.195) | — | wall_s_per_point_from_pair_total: 0.17 |
| b5x | 9.24 (5.38–11.5) | — | wall_s_per_point_from_pair_total: 9.24 |

runtime_s holds the total seconds per pair; divided by n_points here. Pairs ran on different machines (D-052).

**ResNet exp_0050 / exp_0054**: no runtime fields saved (chunked local run); not available

**Shared parameters** (`configs/tiers/tier0.yaml`): solver = constrained Δ-CE search with R = 5 restarts, T = 400 steps, lr = 0.05, λ0 = 1.0, σ = 0.05, μ = 0.0, λ schedule halve_every (8 steps), box clamp [0.0, 1.0], SLSQP polish (maxiter 100); κ = 0.5; r = 0.5413. Points per pair N: 300 (exp_0017, 0019, 0026, 0032, 0033, 0049, 0048 uses the split-A points), 100 (ResNet-8, exp_0050_run.yaml and exp_0054). exp_0054 (split E) uses κ* = 2.5 for the ResNet-8 pairs (`results_x4_kstar.json`); its config does not restate κ or r.
