# Results — spec amendment v1.9 (D-036), Stage A and Stage B preparation

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

## 6. Stage B (exp_0027; validation only; D-040)

- Qualification: 5 / 5 Adult null pairs (DD 0.000 on 2 000 validation points; f_old validation accuracy 0.820–0.824; majority-class rate 0.753; raw disagreement 1.5–3.5% of validation, all with margin < κ).
- r (spec §6.3 rule, pair B0, 300 valid validation points, r = ∞): median CE distance 0.4347, all 300 found → **r_B = 0.8693**, frozen.
- Test set not generated, H6 not computed — awaiting the author.
- Limitation (recorded before data): age and education-num are integers treated as continuous.
