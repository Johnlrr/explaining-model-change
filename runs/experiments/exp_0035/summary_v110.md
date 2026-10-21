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

NOT_FOUND queries searched independently of the solver (10⁵ uniform points in B(z₀, r) ∩ cube, then ascent);
false negative = a point with slack s ≥ 0, i.e. a witness the solver missed.

| group | queries | false negatives | rate [CP95] | max s (median, max) |
|---|---|---|---|---|
| G1 null (N0, N1; 8 pairs) | 400 | 0 | 0 [0, 0.009] | −0.38, −0.15 |
| G2 class an offset cannot flip (T4 class 0) | 150 | 0 | 0 [0, 0.024] | −1.00, −0.82 |
| G3 small / pure offsets (grid 20–27) | 395 | 3 | 0.008 [0.002, 0.022] | −0.62, +0.05 |
| G4 H7 \|b\| ≥ 2κ | 0 (Δ-CE found every lowered-class witness) | — | — | — |

- Where no witness should exist, NOT_FOUND is reliable, with margin: the best point the search finds is still
  ≥ 0.15 logits from feasibility (null) or ≥ 0.8 (the class an offset cannot flip).
- All misses are on pair 20 (the grid null cell; fine-tune offset t̂ − t̂_old = +0.039), in the class where Δ-CE does
  find witnesses (33/100), with max s ≤ 0.05: thin feasible regions. On the queried G3 points Δ-CE recovers ≥ 246 of
  249 witness-holding points.
- Execution note: 15 of 31 pairs ran on Kaggle CPU (D-044, user-authorised). The sampling stage is identical across
  machines; the ascent stage can differ by up to ~1e-2. Re-running the 14 near-zero queries locally flips one sign
  (pair 20 q27): 3 false negatives registered, 4 in the sensitivity run. No conclusion changes.

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

## 5. Consequences for the paper (proposed, for the author — nothing in paper/ has been edited)

1. **H5d sentence (author's request).** Replace the single "−0.35" with the decomposition: of the locked −0.352,
   −0.184 is found-rate (Δ-CE abstains where B3 always answers) and −0.168 geometry on the support; H8 then shows the
   geometric part is where the old-model normal is taken: with ŵ_o at the witness the support gap is +0.029
   (vs −0.190 at z₀), p_Holm 0.0006. Suggested wording: "Proposition 2 holds at the witness, not at the query."
2. **Proposition 2 remark.** Add that the projection must use ∇m_old at z*; the deviation tracks the rotation of the
   normal between z₀ and z* (Spearman 0.80), not ‖z* − z₀‖ (−0.10). One sentence + the 2×2 numbers in a footnote or
   table row.
3. **Label shift as a corollary of Lemma 1.** A class-prior change acts as a logit offset b: B3 is undefined (the
   gradient difference is 0), Δ-CE has an exact detection threshold |b| = 2κ (0 witnesses below; above it, a witness
   for every query in the lowered class and none in the other), B1-det has none (0.961 of queries flagged for
   sub-threshold offsets that move 3–8% of decisions; H7c, p_Holm 0.0002). Wording: a detection threshold, not
   "false positives" of B1. Not a headline (author's decision).
4. **κ as a threshold in logit units (T-D).** One sentence or a small inset: offsets give a flat one-class step at
   κ = |b|/2, retraining noise decays smoothly and is gone at κ = 0.5. This also corrects the reading of κ as a
   probability margin (κ is in logits).
5. **Abstention is reliable (T-C).** "NOT_FOUND on null pairs is not a solver failure: an independent search of
   10⁵ points per query finds no witness on 400 null and 150 impossible-class queries (CP95 upper 0.9% and 2.4%);
   misses (3/395) occur only where the feasible band is thin." This answers the obvious reviewer question about
   abstention.
6. **Limitations to update.** T-C is a sampling lower bound on existence (finding none is evidence, not proof);
   H8 was registered after a validation run that showed the effect (disclosed in v1.10); the grid null cell is not
   null-like (small fine-tune offset, visible in T-C and T-D).
7. **From the earlier review:** the B5 citation [5] should be [7].

Space (6 pages): items 1, 3, 5 fit as three sentences in Results/Discussion; item 2 as a footnote; item 4 as one
sentence (a figure only if space allows).
