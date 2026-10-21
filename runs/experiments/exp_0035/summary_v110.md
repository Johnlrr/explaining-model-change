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
  the mechanism is the projection direction. The old-model normal turns between z₀ and z* by a median 32°, already
  27° for the nearest quartile of witnesses (‖z* − z₀‖ ≤ 0.21 ≈ 0.4 r) vs 36° for the farthest; the rotation grows only
  weakly with distance (Spearman 1 − cos ~ ‖z* − z₀‖: +0.17) and the projection deviation not at all (−0.10). The
  "increases with distance" half of the registered mechanism prediction is therefore not supported; the rotation half
  is (Spearman deviation ~ 1 − cos 0.80 is near-definitional and is not used as evidence). Reading: m_old is curved in
  Z through the renderer, so even a near witness sees another normal (extended_version.md).
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
  threshold-shifted grid pair 26 has an offset-like profile; the grid null cell pair 20 is mixed:
  decaying, then 0.33 one-class at κ = 0.5 (its fine-tune left t̂ − t̂_old = +0.039, exp_0029).

## 5. Consequences for the paper (approved by the author 2026-10-21 with corrections; applied in paper/main.tex)

1. **H5d sentence.** Of the locked −0.352, −0.184 is found-rate (Δ-CE abstains where B3 always answers) and −0.168
   geometry on the support. H8 (registered after the validation run exp_0030, confirmed on new data) shows the
   geometric part depends on where the old-model normal is taken: support gap −0.190 at z₀, +0.029 at z*
   (p_Holm 0.0006). **C3 and H5d stay "not confirmed"**; H8 is a separate test, not a re-test of H5d.
2. **Proposition 2 remark (footnote).** The projection must use the old-model normal at the witness. That normal turns
   between z₀ and z* (median 32°, 27° already for the nearest quartile of witnesses); the rotation grows only weakly
   with distance (ρ = 0.17) and the projection deviation not at all (ρ = −0.10). The "increases with distance" half of
   the registered mechanism prediction is not supported. Spearman 0.80 (deviation vs 1 − cos) is not used.
3. **Logit offset as a corollary of Lemma 1.** Only the prior-correction update [13] is an exact logit offset b: B3 is
   undefined, Δ-CE has the exact threshold |b| = 2κ (none below; above it every lowered-class query has a witness and
   no other), B1-det has none (0.961 of queries flagged for sub-threshold offsets that move 3–8% of decisions; H7c,
   p_Holm 0.0002). Worded as a detection threshold. Label shift by fine-tuning (D-035) stays untested (limitation 2).
4. **κ (T-D), one sentence.** Under a constant offset the found-rate is a one-class step at κ = |b|/2; retraining
   noise decays smoothly with κ and is gone at κ = 0.5.
5. **Abstention (T-C), one sentence.** An independent search of 10⁵ points per query finds no witness on 400 null and
   150 impossible-class NOT_FOUND queries (CP95 upper 0.9%, 2.4%); on small offsets 3 false negatives (4 in a local
   re-run) of 395, all where the feasible band is thin.
6. **Abstract + Discussion.** Replace "after projection, witnesses and gradient differences do not coincide" by: they
   (nearly) coincide when the old normal is taken at the witness (H8 on the grid; T1 descriptive, §6).
7. **Protocol:** H7c, H8 in the confirmatory list (Holm m = 2; H8 registered after a validation run); eleven logged
   amendments. **Limitations:** (2) narrowed to label shift by fine-tuning; (3) κ is an exact threshold for a constant
   offset, approximate when h varies.
8. **B5 citation [5] is correct** (Renard et al., DIG); the earlier suggestion to change it was the Research Lead's
   error. No change.
9. The paper has no hypothesis table; statuses are in the text (Protocol, Results, Contributions) and are updated there.

## 6. T1 2×2 (exp_0036, D-045, descriptive)

T1 pairs 0–2, new data (seeds 9000–9002, first use), 300 points, supports 247–256. Gap Δ-CE − B3 after projection on
the support: (z₀, z₀) −0.166 [CI90 −0.192, −0.140] → **(z*, z₀) −0.024 [−0.041, −0.007]**; (z*, z*) −0.028;
(z₀, z*) −0.188. T = 0.118 (CI95 lower 0.086), Rec_proj(B3) 0.96. The grid pattern repeats on the shortcut design:
the gap narrows by more than half, not fully to 0. Normal rotation vs distance ρ = +0.14. So "the analysis holds at
the witness" covers both designs, with "nearly" for T1.
