# Results — spec amendment v1.9 (D-036): Stage A and Stage B

Separate from `results_summary.md` (v1.9 §5.4). Decisions: D-036 (v1.9), D-037 (v1.9.1), D-038 ((2b) reading, budget),
D-039 (test statistics), D-040 (Stage B details), D-041 (analysis exclusion bug). Branch `research/delta-ce-v19`.

## 1. Manipulation (validation; exp_0024, exp_0025)

- 32 pairs = 16 cells (α, t) × f_old seeds {0, 1}, pair_id 20–51; T4 recipe (10 epochs, lr 1e-3). t̂_old = 0.487 (s0), 0.496 (s1).
- Pilot exp_0024 (pair 40) failed the v1.9 t̂ tolerance (t̂ = 0.630) → v1.9.1 replaced it with offset checks (D-037), written after the pilot and before any explanation method.
- **Rotation is produced reliably:** α̂ within tolerance on all 32 pairs (incl. α = 0.1). **Offsets below ≈ 0.05 are not:** t̂ varies ≈ 0.06 between seeds of one cell; the t = 0.53 cells show no offset (pairs 22, 23 fail (2a)); five (2b) order violations (20→22, 29→31, 33→35, 40→42, 41→43), reported as an observation only (D-038).
- Stop rule: 2 / 32 → not triggered. Null cell (pairs 20, 21): DD = 0.000 on validation → null by the criterion, but its measured offsets are t̂ − t̂_old = +0.039 / −0.032 (see §4).
- Exclusions (D-038): H5a drops {22, 23}; H5b, H5c, H5d drop none; (2c) passes at both seeds (0.077, 0.120).

## 2. §5.3 consistency (computed and committed before any H5 statistic; b22337c)

Cell (0, 0.60), pairs 26–27: B3 Recovery 0.087 (≤ 0.20), Δ-CE found-rate 0.452 (∈ [0.30, 0.65]) → **pass** (locked T4: 0.07, 0.47).

## 3. H5a–H5d (exp_0026; test data seed 7000 + pair_id; CPU, locked solver; Holm m = 4)

| ID | Statistic | Estimate | CI | p_Holm | Result |
|---|---|---|---|---|---|
| H5a | Recovery(B3), α = 0, t > 0.5 (pairs 24–27) | 0.081 | CI95 [0.066, 0.096] | 0.0004 | **supported** |
| H5b | Rec(B3; 0.5, 0.5) − Rec(B3; 0.1, 0.5) | 0.930 − 0.360 = +0.570 | CI95 [0.527, 0.613] | 0.0004 | **supported** |
| H5c | Rec(B3; 0.25, 0.60) − Rec(B3; 0.25, 0.50) | 0.583 − 0.682 = −0.098 | CI90 [−0.143, −0.053] | 0.944 | **not supported** |
| H5d | Rec_proj(Δ-CE) − Rec_proj(B3), α ∈ {0.25, 0.5}, t = 0.5 | −0.352 (unprojected −0.545) | CI90 [−0.378, −0.327] | 1.0 | **not supported** |

- H5c: the point estimate sits at the margin; the interval leaves [−0.10, 0.10] on the low side — B3 Recovery of the rotation drops by about 0.1 when an offset is added.
- H5d, pre-declared reading (v1.9 §3): the linear analysis does not transfer to the CNN even for a rotation; the C3 failure on T1 is **not** explained by T1 being an impure rotation. Naming per v1.9.1 §3: of the four H5d pairs only pair 36 has t̂ − t̂_old > 0.02 (+0.036) and is a "rotation with measured offset"; pairs 37, 44, 45 have −0.010, −0.045, +0.008.
- **Analysis bug (D-041):** the first `analyze` used the pre-D-038 exclusion list (H5b lost pair 29, H5c "not testable", Holm m = 3). Fixed to D-038 after the output was seen; the superseded file is `runs/experiments/exp_0026/results_SUPERSEDED_wrong_exclusions.json`. H5a and H5d are identical in both versions; H5b was +0.610 [0.555, 0.667] on pairs 44, 45 vs 28 only.

## 4. Descriptive (not tested)

- (i) Δ-CE found-rate on α = 0 by nominal t: 0.22 (0.50), 0.20 (0.53), 0.31 (0.56), 0.45 (0.60) — rises with t except at 0.53, where no offset was produced.
- (ii) Witness class on α = 0 follows the **sign of the measured offset**: pairs with t̂ − t̂_old > 0 (20, 24, 25, 26, 27) have witnesses only on y = 1; pairs with t̂ − t̂_old < 0 (21: −0.032, 22: −0.058) only on y = 0; pair 23 (−0.016) has none. So the prediction "only y = 1, as T4" holds where the offset is positive and flips where it is negative, as P5(a) implies.
- (iii) A2 Recovery on α = 0: 0.91–1.00 on all eight pairs (it recovers aspect whether or not the model changed: Recovery measures blindness, not sensitivity). On α > 0, A2 Recovery of colour is 0.00–0.03.
- The null cell is not null on test: Δ-CE found-rate 0.21 / 0.23 on pairs 20 / 21, consistent with their measured offsets (+0.039 / −0.032) even though validation DD = 0.000.
- κ curve (pairs 20–27): found-rate at κ = 1.0 is 0.00 on 7 of 8 pairs (0.16 on pair 26); κ = 0 and 0.25 equal κ = 0.5 on pairs 24, 26, 27 (offset ≥ +0.07); on pairs 20, 21, 22, 23, 25 found-rate rises as κ falls (e.g. pair 23: 0.99 at κ = 0, 0.40 at 0.25, 0.00 at 0.5; pair 25: 0.45, 0.45, 0.11).
- Per-pair table and κ curves: `runs/experiments/exp_0026/results.json`; map at measured (t̂, α̂): `runs/experiments/exp_0026/map_v19.pdf`.

## 5. Compute

exp_0026: 26.0 h summed pair wall time on 8 threads, two CPU workers (forward 20–30, reverse 31–51, no overlap), ≈ 17 h elapsed including one session interruption (resumed, no pair lost or duplicated). GPU: 0 h.

## 6. Stage B (exp_0027 validation, exp_0028 test; D-040, D-042)

- Qualification: 5 / 5 Adult null pairs (DD 0.000 on 2 000 validation points; f_old validation accuracy 0.820–0.824; majority-class rate 0.753; raw disagreement 1.5–3.5% of validation, all with margin < κ).
- r (spec §6.3 rule, pair B0, 300 valid validation points, r = ∞): median CE distance 0.4347, all 300 found → **r_B = 0.8693**, frozen.
- **H6 (exp_0028, test split used once; D-042: Δ-CE found-rate = N ∪ O as C2) — confirmed.**

| Pair | Δ-CE N | Δ-CE O | Δ-CE N ∪ O | B1-det |
|---|---|---|---|---|
| B0 | 0.000 | 0.000 | 0.000 | 0.977 |
| B1 | 0.000 | 0.000 | 0.000 | 1.000 |
| B2 | 0.000 | 0.000 | 0.000 | 0.987 |
| B3 | 0.000 | 0.000 | 0.000 | 0.993 |
| B4 | 0.000 | 0.000 | 0.000 | 0.997 |

  300 valid test points per pair, r_B = 0.8693, κ = 0.5; CPU, ≈ 2 min per pair. Reading limits: decisive disagreement is rare on these pairs (validation DD 0.000; raw disagreement 1.5–3.5%, all with margin < κ); tabular data have no decoder to a manifold, so H6 tests abstention only, not attribution; age and education-num are integers treated as continuous.

## 7. Diagnostic: measured offset versus witnesses (D-042; descriptive, not tested; exp_0029)

t̂ − t̂_old with the v1.9.1 measurement. t̂_old by f_old seed: 0.487, 0.496, 0.486, 0.508, 0.478 (seeds 0–4). Locked null pairs: only t̂ was computed; DD from exp_0004, found-rates from the saved exp_0019 outputs (exp_0017 identical: 0 everywhere).

| Pair | Recipe | seed | t̂ − t̂_old | DD (val) | Δ-CE found (κ 0.5, test) | witness class |
|---|---|---|---|---|---|---|
| 9 | N0 (3 ep, 3e-4) | 0 | +0.017 | 0.000 | 0.000 (N), 0.000 (O) | — |
| 10 | N0 | 1 | +0.035 | 0.000 | 0.000, 0.000 | — |
| 11 | N0 | 2 | +0.005 | 0.000 | 0.000, 0.000 | — |
| 12 | N0 | 3 | +0.016 | 0.000 | 0.000, 0.000 | — |
| 13 | N0 | 4 | +0.004 | 0.000 | 0.000, 0.000 | — |
| 14 | N1 (3 ep, 3e-4) | 0 | +0.008 | 0.000 | 0.000, 0.000 | — |
| 15 | N1 | 1 | +0.021 | 0.000 | 0.000, 0.000 | — |
| 16 | N1 | 2 | +0.010 | 0.000 | 0.000, 0.000 | — |
| 20 | v1.9 (0, 0.50), 10 ep, 1e-3 | 0 | +0.039 | 0.000 | 0.210 (N) | y = 1 only |
| 21 | v1.9 (0, 0.50) | 1 | −0.032 | 0.000 | 0.230 | y = 0 only |
| 22 | v1.9 (0, 0.53) | 0 | −0.058 | 0.003 | 0.393 | y = 0 only |
| 23 | v1.9 (0, 0.53) | 1 | −0.016 | 0.000 | 0.000 | — |
| 24 | v1.9 (0, 0.56) | 0 | +0.070 | 0.005 | 0.517 | y = 1 only |
| 25 | v1.9 (0, 0.56) | 1 | +0.047 | 0.000 | 0.110 | y = 1 only |
| 26 | v1.9 (0, 0.60) | 0 | +0.099 | 0.034 | 0.470 | y = 1 only |
| 27 | v1.9 (0, 0.60) | 1 | +0.091 | 0.014 | 0.433 | y = 1 only |

What the table shows (descriptive only):
- The locked null pairs have small positive measured offsets (+0.004 to +0.035) and no witness at all; every pair with a witness has |t̂ − t̂_old| ≥ 0.032.
- **The measured offset alone does not separate witness from no witness.** Pair 10 (N0, +0.035) has none, while pairs 20 (+0.039) and 21 (−0.032), with offsets of the same size, have 21–23%. The two groups also differ in recipe (3 epochs at 3e-4 versus 10 epochs at 1e-3), and t̂ itself varies by ≈ 0.03 across f_old seeds around 0.5, so offsets of this size are near the resolution of the measurement.
- So the picture "no offset → no witness, small offset → some, large offset → many" holds only coarsely; at |offset| ≈ 0.03–0.04 the outcome differs between recipes. C2 ("0 witnesses on the 8 null pairs") stays true as measured; the v1.9 null cell is better described as a small measured threshold shift produced by the stronger recipe than as a null pair.

## 8. Per-pair table (exp_0026, test; Recovery for the pair's k*; null cell has no k*)

| pair | α | t | seed | α̂ | t̂ | k* | Rec B3 | Rec Δ-CE | Rec B1-det | Rec A2 | found Δ-CE | Rec_proj Δ-CE | Rec_proj B3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 0.0 | 0.50 | 0 | 0.032 | 0.526 | none (null) | — | — | — | — | 0.210 | — | — |
| 21 | 0.0 | 0.50 | 1 | 0.028 | 0.464 | none (null) | — | — | — | — | 0.230 | — | — |
| 22 | 0.0 | 0.53 | 0 | 0.045 | 0.429 | aspect | 0.030 | 0.270 | 0.643 | 0.977 | 0.393 | 0.060 | 0.040 |
| 23 | 0.0 | 0.53 | 1 | 0.024 | 0.480 | aspect | 0.097 | 0.000 | 0.290 | 0.997 | 0.000 | 0.000 | 0.100 |
| 24 | 0.0 | 0.56 | 0 | -0.027 | 0.557 | aspect | 0.027 | 0.473 | 0.687 | 0.977 | 0.517 | 0.077 | 0.043 |
| 25 | 0.0 | 0.56 | 1 | 0.008 | 0.542 | aspect | 0.123 | 0.040 | 0.603 | 0.977 | 0.110 | 0.007 | 0.127 |
| 26 | 0.0 | 0.60 | 0 | -0.001 | 0.586 | aspect | 0.070 | 0.467 | 0.750 | 0.910 | 0.470 | 0.090 | 0.103 |
| 27 | 0.0 | 0.60 | 1 | -0.009 | 0.587 | aspect | 0.103 | 0.430 | 0.690 | 0.930 | 0.433 | 0.067 | 0.117 |
| 28 | 0.1 | 0.50 | 0 | 0.112 | 0.518 | color | 0.320 | 0.110 | 0.217 | 0.000 | 0.367 | 0.177 | 0.330 |
| 29 | 0.1 | 0.50 | 1 | 0.098 | 0.513 | color | 0.400 | 0.010 | 0.250 | 0.000 | 0.027 | 0.010 | 0.407 |
| 30 | 0.1 | 0.53 | 0 | 0.067 | 0.523 | color | 0.163 | 0.093 | 0.157 | 0.000 | 0.333 | 0.143 | 0.163 |
| 31 | 0.1 | 0.53 | 1 | 0.121 | 0.506 | color | 0.537 | 0.050 | 0.273 | 0.000 | 0.187 | 0.083 | 0.540 |
| 32 | 0.1 | 0.56 | 0 | 0.085 | 0.525 | color | 0.320 | 0.097 | 0.177 | 0.000 | 0.350 | 0.173 | 0.320 |
| 33 | 0.1 | 0.56 | 1 | 0.095 | 0.588 | color | 0.280 | 0.030 | 0.013 | 0.000 | 0.433 | 0.070 | 0.293 |
| 34 | 0.1 | 0.60 | 0 | 0.118 | 0.609 | color | 0.280 | 0.000 | 0.027 | 0.000 | 0.467 | 0.057 | 0.300 |
| 35 | 0.1 | 0.60 | 1 | 0.092 | 0.582 | color | 0.280 | 0.037 | 0.113 | 0.000 | 0.427 | 0.053 | 0.287 |
| 36 | 0.25 | 0.50 | 0 | 0.221 | 0.523 | color | 0.680 | 0.177 | 0.343 | 0.000 | 0.637 | 0.367 | 0.673 |
| 37 | 0.25 | 0.50 | 1 | 0.219 | 0.486 | color | 0.683 | 0.230 | 0.377 | 0.000 | 0.610 | 0.347 | 0.703 |
| 38 | 0.25 | 0.53 | 0 | 0.260 | 0.525 | color | 0.780 | 0.267 | 0.407 | 0.000 | 0.717 | 0.463 | 0.793 |
| 39 | 0.25 | 0.53 | 1 | 0.202 | 0.526 | color | 0.653 | 0.237 | 0.370 | 0.000 | 0.627 | 0.347 | 0.670 |
| 40 | 0.25 | 0.56 | 0 | 0.233 | 0.630 | color | 0.703 | 0.013 | 0.147 | 0.000 | 0.400 | 0.133 | 0.720 |
| 41 | 0.25 | 0.56 | 1 | 0.249 | 0.609 | color | 0.737 | 0.087 | 0.340 | 0.000 | 0.430 | 0.183 | 0.750 |
| 42 | 0.25 | 0.60 | 0 | 0.264 | 0.600 | color | 0.547 | 0.057 | 0.317 | 0.000 | 0.430 | 0.160 | 0.590 |
| 43 | 0.25 | 0.60 | 1 | 0.211 | 0.606 | color | 0.620 | 0.050 | 0.253 | 0.000 | 0.463 | 0.137 | 0.637 |
| 44 | 0.5 | 0.50 | 0 | 0.444 | 0.442 | color | 0.923 | 0.303 | 0.563 | 0.000 | 0.813 | 0.593 | 0.940 |
| 45 | 0.5 | 0.50 | 1 | 0.428 | 0.503 | color | 0.937 | 0.333 | 0.553 | 0.003 | 0.783 | 0.567 | 0.967 |
| 46 | 0.5 | 0.53 | 0 | 0.487 | 0.503 | color | 0.910 | 0.313 | 0.547 | 0.010 | 0.820 | 0.640 | 0.933 |
| 47 | 0.5 | 0.53 | 1 | 0.458 | 0.562 | color | 0.893 | 0.340 | 0.593 | 0.003 | 0.757 | 0.603 | 0.910 |
| 48 | 0.5 | 0.56 | 0 | 0.463 | 0.519 | color | 0.903 | 0.280 | 0.523 | 0.000 | 0.787 | 0.583 | 0.900 |
| 49 | 0.5 | 0.56 | 1 | 0.512 | 0.591 | color | 0.850 | 0.273 | 0.550 | 0.000 | 0.773 | 0.593 | 0.877 |
| 50 | 0.5 | 0.60 | 0 | 0.548 | 0.619 | color | 0.850 | 0.260 | 0.507 | 0.010 | 0.790 | 0.617 | 0.873 |
| 51 | 0.5 | 0.60 | 1 | 0.492 | 0.600 | color | 0.840 | 0.300 | 0.467 | 0.027 | 0.743 | 0.550 | 0.860 |

## 9. H5c in full

Estimate −0.0983, CI90 [−0.1433, −0.0533] (bootstrap seed 6502, 10 000 resamples), p_TOST 0.472, p_Holm 0.944.

| seed | Rec(B3; 0.25, 0.60) | Rec(B3; 0.25, 0.50) | difference | t̂ (0.60 cell) | t̂ (0.50 cell) |
|---|---|---|---|---|---|
| 0 | 0.547 (pair 42) | 0.680 (pair 36) | −0.133 | 0.600 | 0.523 |
| 1 | 0.620 (pair 43) | 0.683 (pair 37) | −0.063 | 0.606 | 0.486 |

