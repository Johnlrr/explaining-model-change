# v1.10 summary (D-043) — for the author, before any paper edit

All numbers are from fresh test sets `new_test_set(9000 + pair_id)`, each used once for its confirmatory test (§6.1),
except where marked descriptive. Holm m = 2 over {H7c, H8} (exp_0033/results_holm.json).

## 1. Confirmatory

| ID | Prediction | Estimate | Interval | p_Holm | Status |
|---|---|---|---|---|---|
| H7c (H-010) | Sub-threshold logit offsets (\|b\| ∈ {0.5, 0.9}; 3–8% of decisions change, none decisively): B1-det attributes a change at ≥ 90% of queries (pooled and per pair) | B1-det found-rate 0.961; per pair 0.927–0.997 | CI95 [0.955, 0.967] | 0.0002 | **confirmed** |
| H8 (H-011) | With ŵ_o taken at the witness (cell (z*, z₀)), the projected gap Δ-CE − B3 on the Δ-CE support narrows by ≥ ½ vs. ŵ_o at z₀ | gap (z₀, z₀) −0.190 → (z*, z₀) +0.029; T = 0.132 | gap CI90 [−0.222, −0.159] → [0.006, 0.053]; T CI95 lower 0.065 | 0.0006 | **confirmed** (guard Rec_proj(B3) 0.88; support 183–228 per pair) |

Deterministic checks of T-A (U1–U3, exp_0032): B3 invalid on every query of all 30 pairs; no Δ-CE witness on any of the
12 sub-threshold pairs; every witness in the lowered class with κ ≤ m_old(z*) ≤ |b| − κ. All passed.

## 2. Reading

- **Detection threshold (H7c).** B1-det has none; Δ-CE has one at 2κ in logit units, and it is sharp: 0 witnesses
  below, and for |b| ≥ 2κ a witness on **100%** of lowered-class queries (found-rate 0.34–0.52 = the lowered-class
  share), all moving along aspect (Recovery ≈ found-rate). B1-det is not wrong to report a change: the CEs differ.
- **C3/H5d explained (H8).** Proposition 2 holds at the witness, not at z₀: after removing the old-model normal at z*,
  the witness and the *original* gradient difference (at z₀) agree. Moving B3 to z* alone widens the gap (−0.270), so
  the mechanism is the projection direction. The projection deviation tracks the rotation of the normal,
  1 − cos(ŵ_o(z₀), ŵ_o(z*)) (Spearman 0.80), not the distance ‖z* − z₀‖ (−0.10): the registered mechanism prediction
  ("increases with both") holds for the first only.
- **Locked H5d −0.352 decomposed** (exp_0026 saved outputs, descriptive): geometry −0.168 (found-rate × gap on the
  support) + found-rate −0.184. On the support the locked gap is −0.234; on the new data −0.190.
- **Extension to all α > 0 (descriptive, 24 pairs):**

  | α | pairs | Rec_proj (Δ-CE, B3) at (z₀, z₀) | at (z*, z₀) | gap (z₀,z₀) → (z*,z₀) |
  |---|---|---|---|---|
  | 0.1 | 8 | 0.32, 0.43 | 0.44, 0.48 | −0.106 → −0.034 |
  | 0.25 | 8 | 0.50, 0.80 | 0.84, 0.82 | −0.306 → +0.016 |
  | 0.5 | 8 | 0.75, 0.95 | 0.96, 0.96 | −0.197 → +0.002 |

## 3. Abstention recall (T-C, exp_0034, descriptive)

TBD.

## 4. Found-rate against κ (T-D, exp_0035, descriptive)

| model change | κ = 0 | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|
| logit offset \|b\| = 0.5 (b = ∓) | 0.51 / 0.40 | same | 0 | 0 | 0 | 0 |
| \|b\| = 1.1 | 0.49 / 0.37 | same | same | same | 0 | 0 |
| \|b\| = 2.0 | 0.37 / 0.43 | same | same | same | same | 0.01 / 0 |
| grid t = 0.6 (pair 26) | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.17 |
| grid null cell (pair 20) | 0.80 | 0.68 | 0.64 | 0.33 | 0 | 0 |
| Tier 0 N0 retrain (pair 9) | 0.78 | 0.46 | 0.13 | 0 | 0 | 0 |

- Lemma 1 is exact in practice: under an offset the found-rate is the lowered-class share (every lowered-class query
  has a witness) while 2κ < |b|, and 0 once 2κ > |b|. At 2κ = |b| the band is a point: 0 found (1 of 100 points for
  b = −2 at κ = 1, inside the solver tolerance). Figure: exp_0035/found_vs_b.pdf.
- κ is a detection threshold in logit units, and it separates two kinds of change: retraining noise (pair 9) decays
  smoothly with κ and is gone at κ = 0.5; an offset gives a flat, one-class profile with a step at |b|/2. The
  threshold-shifted grid pair 26 has the offset profile (|b| between 1.5 and 2); the grid null cell pair 20 is mixed:
  decaying, then 0.33 one-class at κ = 0.5 (its fine-tune left t̂ − t̂_old = +0.039, exp_0029).

## 5. Consequences for the paper (proposed, for the author)

TBD.
