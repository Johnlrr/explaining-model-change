# Δ-CE spec — Amendment v1.10 (D-043)

**Date:** 20/10/2026 · **Author:** Research Lead · **Status:** APPROVED by the author on 20/10/2026 (with the changes of §9). Declared and committed before any v1.10 test set is generated or any v1.10 test statistic is computed. Language: English (handoff of 20/10/2026; earlier amendments are in Vietnamese).

**Unchanged:** every locked result (exp_0017, exp_0019, exp_0022, exp_0026, exp_0028) and every locked test split (§6.2); κ = 0.5, r = 0.5412657 (Tier 0), the two-phase solver (configs/tiers/tier0.yaml), Recovery@r, found-rate, Rec_proj (C3/H5d), the top-factor tie rule, bootstrap 10 000 resamples stratified by pair. New metrics in this amendment are descriptive only.

**Development data seen before this amendment (disclosed):** validation-only runs exp_0030 (H8 2×2 on the four H5d pairs, N = 50) and exp_0031 (T-C cost), and the validation decision-flip shares of §2.3. Their values are recorded in D-043. No test data of any kind was generated or read for this amendment.

---

## 1. Purpose

1. **T-A (H7).** Label shift as an exact offset. Proposition 2(a) predicts that the update which is optimal under label shift (prior correction, Saerens et al. 2002) is invisible to gradient differences. The fine-tuned label-shift update of D-035 failed qualification. Here the update is built post hoc and is an offset by construction.
2. **T-B (H8).** A test of one explanation of the C3/H5d failure: Proposition 2 linearises at z₀, while the witness lies up to r away and ŵ_o(z) = ∇m_o(z)/‖∇m_o(z)‖ varies with z.
3. **T-C.** Recall of abstention: when Δ-CE returns NOT_FOUND, is there really no witness within r?
4. **T-D.** Found-rate as a function of κ, against the exact Lemma 1 prediction under a logit offset.

## 2. T-A — logit offset (H7)

### 2.1 Update

f_new = f_old with the constant b added to the class-1 logit (`dce.models.label_shift.LogitShift`). No training. The margin is computed on logits (`dce.models.cnn.margin`), so m_new = m_old + b for y = 1 and m_old − b for y = 0, at every input: h is constant.

Relation to D-035 / `t5_label_shift.py`: D-035 fine-tuned f_old on a re-weighted adaptation set (P(y=1) = 0.8) and the update failed G0-T5 qualification. v1.10 does not retry D-035. It replaces fine-tuning by the closed-form prior correction, which is an offset in logit space by construction. D-035 stays as recorded.

### 2.2 Deterministic consequences (unit tests and corollary, not hypotheses)

Verified by `tests/unit/test_v110.py` before this amendment:
- (U1) ∇m_new ≡ ∇m_old, so B3's g is exactly 0 and B3 is **undefined** (invalid, NOT_FOUND), not a wrong direction.
- (U2) |b| < 2κ: no witness exists anywhere (Lemma 1 with L_h = 0). Any witness reported would be a code error.
- (U3) |b| ≥ 2κ: a witness exists only for queries of the lowered class (b > 0 → y = 0; b < 0 → y = 1) and has κ ≤ m_old(z*) ≤ |b| − κ.

B3 is excluded from every H7 comparison and reported as "undefined under a logit offset".

### 2.3 Pairs

- f_old: exp_0002 f_old seeds 0, 1, 2 (hash-verified).
- b ∈ {−2.0, −1.5, −1.1, −0.9, −0.5, +0.5, +0.9, +1.1, +1.5, +2.0} (logit units; 2κ = 1.0).
- Pair id = 60 + 10·s + j (s = seed, j = index of b in the list above): pairs 60–89.
- Manipulation description (validation, computed before this amendment; reported, not tested): share of the 5 000 validation decisions that flip. Seeds 0–4 at |b| = 0.5: 0.034–0.042; |b| = 0.9: 0.066–0.074; |b| = 1.1: 0.080–0.090; |b| = 1.5: 0.110–0.122; |b| = 2.0: 0.156–0.182. Decisive disagreement (both margins ≥ κ, opposite decisions) is exactly 0 for |b| < 2κ.
- **Reading fixed in advance:** pairs with |b| < 2κ are **not null updates**. They change 3–8% of decisions, but no decision changes decisively. They are the sub-threshold controls. No qualification threshold is applied (the update is exact; the existing "visible in decisions" criteria would classify |b| < 2κ as invisible by design).

### 2.4 Data and methods

New test data per pair: `new_test_set(9000 + pair_id, 5 000)`, clean labels of the original rule; 300 valid points (κ = 0.5) with order seed 5000 + pair_id; restarts 10000 + pair_id. Methods: Δ-CE type N, B1-det, B3 (recorded, expected invalid everywhere). A2 is dropped: on an offset it recovers aspect trivially (C1, H5a), and it costs about a quarter of the run. k* = aspect (as T4).

### 2.5 Hypothesis H7c (confirmatory; Holm family {H7c, H8})

A replication of C2 on an update with an exact ground truth; the outcome is largely predictable and the test is kept because it is cheap.

| ID | Statement | Test | Confirmed iff |
|---|---|---|---|
| H7c | Detection threshold. The sub-threshold pairs (\|b\| ∈ {0.5, 0.9}, 12 pairs) change 3–8% of decisions but none decisively. B1-det, which has no detection threshold, attributes a change at almost every query; Δ-CE, whose detection threshold is 2κ in logit units, returns no witness. This describes a property of the two methods, not a ranking: the counterfactuals of f_new and f_old really differ, so B1-det is not wrong to report a change. | pooled B1-det found-rate, bootstrap stratified by pair, 10 000 resamples, seed 6600; one-sided p = P*(stat ≤ 0.90) | pooled found-rate CI95 lower bound > 0.90 **and** p_Holm < 0.05 **and** B1-det found-rate ≥ 0.90 on every one of the 12 pairs. Δ-CE found-rate = 0 on all 12 pairs is U2 and is reported as a check; a single witness invalidates the run (code error), not H7c. |

### 2.6 Descriptive (|b| ≥ 2κ and all pairs)

Δ-CE found-rate per class and per b; m_old(z*) against the band [κ, |b| − κ]; Recovery@r(aspect) of Δ-CE and B1-det; the figure "found-rate vs b" with vertical lines at ±2κ (merged with T-D, §5).

### 2.7 Adult

Dropped (author, Q1): under a logit offset U2 holds on any data and the B1-det rate is predictable.

## 3. T-B — locality of the projection (H8)

### 3.1 Design (2 × 2)

For each query with a Δ-CE witness z*: the projection removes the component along ŵ_o(p) from both explanations, p ∈ {z₀, z*}; B3 is computed at q ∈ {z₀, z*}. Cells (p, q): (z₀, z₀) is the registered C3/H5d analysis restricted to S (the reference); **(z*, z₀) is the registered H8 cell**: the projection direction is local (at the witness) while B3 is the original method at z₀; (z*, z*) and (z₀, z*) are descriptive. The choice of (z*, z₀) was made after the validation development run exp_0030 (D-043), which showed that the projection position, not the position of B3, closes the gap. Code: `dce.analysis.v110` (`h8_indicators`, `boot_h8`).

- **Support set.** Every cell is scored on the same set S = queries on which Δ-CE found a witness. |S| is reported per pair. The locked −0.35 (H5d) is **not** the reference, because it mixes found-rate with geometry.
- **Gap of a cell** = mean over pairs of [mean_S Rec_proj(Δ-CE) − mean_S Rec_proj(B3)], k* = colour.
- **Scope.** In the registered cell B3 is the original method at z₀, so H8 is separate from the extended-version idea #5 (gradient difference at the witness). Cell (z*, z*), which coincides with #5 computationally, is reported only as descriptive agreement between two explanations, never evaluated as a method.

### 3.2 Pairs and data

**Confirmatory:** the four H5d pairs 36, 37, 44, 45 (α ∈ {0.25, 0.5}, t = 0.5, seeds 0, 1): the cells whose failure H8 is meant to explain. **Descriptive extension:** the other 20 α > 0 grid pairs (28–35, 38–43, 46–51), same procedure, outside Holm. Models: exp_0025 f_new, exp_0002 f_old (hash-verified). New test data per pair: `new_test_set(9000 + pair_id, 5 000)`; 300 valid points, order seed 5000 + pair_id, restarts 10000 + pair_id. Methods: Δ-CE type N; B3 at z₀ and at z*; ∇m_o at z₀ and z*.

### 3.3 Hypothesis H8 (confirmatory; Holm family {H7c, H8})

| ID | Statement | Test | Confirmed iff |
|---|---|---|---|
| H8 | With the projection direction taken at the witness, the gap narrows by at least one half: \|gap(z*, z₀)\| ≤ ½ \|gap(z₀, z₀)\| | T = \|gap(z₀,z₀)\| − 2\|gap(z*,z₀)\|; paired bootstrap over support points stratified by pair (one index draw per pair shared by all cells), 10 000 resamples, seed 6601; one-sided p = P*(T ≤ 0) | CI95 lower bound of T > 0 **and** p_Holm < 0.05 **and** (non-triviality guard) pooled Rec_proj(B3) in cell (z*, z₀) ≥ 0.50, so that the gap cannot close by both explanations losing k* |

**Not testable** (reported as such, outside Holm) if the CI90 of gap(z₀, z₀) on the new data contains 0, i.e. there is no gap to explain on this support.

### 3.4 Descriptive

Gaps of cells (z*, z*) and (z₀, z*); per point ‖z* − z₀‖, 1 − cos(ŵ_o(z₀), ŵ_o(z*)) and ‖P_{z₀}d − P_{z*}d‖/‖d‖ with Spearman correlations. Prediction of the mechanism (descriptive): the projection deviation increases with both distance and 1 − cos. Whatever the outcome, both outcomes are reported; if H8 fails, non-locality is reported as tested and not supported, and the alternatives (curvature of h, solver properties, the JᵀJ metric) go to research/extended_version.md.

## 4. T-C — recall of abstention (descriptive, protocol fixed here)

- **Slack.** s(z) = min(−κ − m_new(z), m_old(z) − κ); z is a witness iff s(z) ≥ 0 (the solver's own constraints).
- **Search, independent of the solver** (no proximity term): uniform draws in B(z₀, r), rejected outside [0,1]⁵ (L2, as the budget), repeated until M = 100 000 points are accepted (a near-corner query no longer gets as few as 1.6k points); the number of accepted points and of draws is reported per query; then projected Adam ascent on s (lr 0.01, 200 steps) from the best 16 samples; report max s over both stages (`dce.cli.v110_dev.max_slack`).
- **False negative** = a NOT_FOUND query with max s ≥ 0 (point re-checked with the float32 forward pass, as the solver's feasibility check). Reported per group with Clopper–Pearson 95% intervals, plus the distribution of max s.
- **Groups** (fresh data `new_test_set(9000 + pair_id, 5 000)` unless stated): (G1) null: Tier 0 N0 pairs 9–13 and N1 pairs 14–16; (G2) the class without witnesses under an offset: T4 class-0 queries of pairs 6–8; (G3) small and pure offsets: grid pairs 20–27 (the grid null cell 20, 21 is not null-like: its measured offsets are ±0.03–0.04 and its locked found-rates are 0.21, 0.23); (G4) H7 pairs with |b| ∈ {1.1, 1.5} (same queries as §2.4, the Δ-CE run of exp_0032 is reused). Δ-CE on the first 100 valid points of each G1–G3 pair; T-C on up to 50 NOT_FOUND queries per pair, first in the point order.
- **Expectation (stated, not tested):** no false negatives in G1 and G2; G3 and G4 measure solver recall where witnesses exist nearby. Sampling gives only a lower bound on existence: finding none is evidence, not proof.

## 5. T-D — found-rate against κ (descriptive)

κ ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0} on the first 100 κ = 0.5 valid points (as A3 in exp_0017 and exp_0026). Pairs: H7 pairs b ∈ {±0.5, ±1.1, ±2.0}, seed 0; grid null pair 20; offset pair 26; N0 pair 9 (fresh data as above). One figure: found-rate against b with one curve per κ and the exact Lemma 1 lines |b| = 2κ. exp_0026 already holds κ ∈ {0, 0.25, 1.0} on pairs 20–27 (test, descriptive): cited from saved outputs, not re-run.

## 6. Splits, seeds, guards

### 6.1 New data
`new_test_set(9000 + pair_id, 5 000)` for every pair touched by v1.10 (H7 pairs 60–89; grid pairs; Tier 0 pairs 6–16), generated after this amendment is committed. **Each test set is used for at most one confirmatory test; descriptive analyses may reuse it after the confirmatory test on that set has been computed** (e.g. the H7 sets in T-C G4 and T-D; pairs 9, 20, 26 in T-C and T-D). Bootstrap seeds 6600 (H7c), 6601 (H8). T-C sampling seed 9500 + pair_id.

### 6.2 Locked (never read again)
clean_splits(1234) test; C1–C3 (2027); T5 (2028); grid H5 (7020–7051); Adult seed-2026 test. `dce.analysis.v110.check_fresh_seed` refuses the renderer seeds.

### 6.3 Analysis-code errors
An error found after a test output is fixed to the rule registered here and logged in DECISIONS (as D-041). The rule is not changed.

## 7. Order of operations

1. Author approves this amendment → commit (code + amendment + D-043).
2. T-A test run (exp_0032) → H7c.
3. T-B test run (exp_0033) → H8. Holm (m = 2) over {H7c, H8} after both.
4. T-C (exp_0034), T-D (exp_0035), descriptive.
5. Summary in runs/experiments/exp_0035/summary_v110.md → sent to the author before any paper edit.

## 8. Compute

Local CPU (16 cores), no GPU. Measured on validation (exp_0030, exp_0031; two processes in parallel): Δ-CE 1.5–2.0 min per 50 points on rotation pairs, 1.5–1.8 min per 20 points on the null/offset pairs (few witnesses → all restarts run); max-slack search 3–8 s per query at M = 10⁴ and 10–45 s at M = 10⁵ (rejection on the cube leaves 1.6k–68k samples). Estimates: T-A ≈ 15 CPU-h (30 pairs, Δ-CE + B1-det); T-B ≈ 1 h confirmatory + 3.5 h descriptive; T-C ≈ 15 h; T-D ≈ 5 h. Total ≈ 40 CPU-h, ≈ 20 h wall with two processes. Every run is resumable per pair.

## 9. Author's answers (20/10/2026)

- Q1 Adult: dropped (§2.7).
- Q2 H8: the registered cell is (z*, z₀); (z*, z*) and (z₀, z*) are descriptive. Chosen after exp_0030: the sharper mechanistic claim (only the projection direction must be local), and B3 stays the original method at z₀, which keeps H8 apart from idea #5.
- Q3 H7c: 0.90 pooled (CI95 lower bound) and ≥ 0.90 on every pair.
- Q4 A2 dropped from T-A.
- Q5 Non-triviality guard on the registered cell: Rec_proj(B3) in (z*, z₀) ≥ 0.50.
- Summary to the author: decompose the locked H5d −0.35 into a found-rate part and a geometric part on the support set (descriptive; the paper sentence on H5d will be rewritten accordingly, the locked result is not changed).
