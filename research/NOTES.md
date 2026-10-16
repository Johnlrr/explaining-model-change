# NOTES.md

> Purpose: Temporary observations, questions, and ideas that are not yet formal conclusions.
>
> Ownership: Mutable research memory. Promote items to STATE, HYPOTHESES, DECISIONS, or LITERATURE when they are settled.

---

## 1. Usage principles

Raw observations and unresolved items only. Do not store final results here.

---

## 2. Inbox

### Kick-off inconsistencies found while mapping the spec (2026-10-15)

- Spec §9 (G0, G1, G2) and §7 use test points for development checks. Resolved as D-009: the user chose to follow the spec; restrictions in `docs/EVALUATION.md` §6.
- Spec §12–§13 layout and log path differ from AGENTS.md. Resolved as D-001.
- Spec §8 H4 tolerance is 1e-3; the proposal used 1e-4. Spec governs (D-005).
- Spec G0 criterion (Flip_color difference ≥ 0.15 for T1) is not the proposal's sensitivity criterion (S_k difference ≥ 0.20). Spec governs.

### Proposals to the user before the spec freeze — resolved 2026-10-15 by spec amendment v1.1 (D-011–D-015)

1. **D-009 wording is wrong about the spec.** The spec names the test split only for G0 (2 000 clean test points, §9), the f_old accuracy requirement (§5.1), σ_k (§5.3), and the final evaluation (§7, B4, plausibility). It does **not** name a split for the G1 pilot or for G2. D-009 and `docs/EVALUATION.md` §6 currently say the pilot and G2 use test "per the spec". Proposed: pilot and G2 on validation (the protocol default where the spec is silent).
2. **H3 needs a disagreement-rate baseline (B0).** Hammer & Hüllermeier (L-004) define model change as the discrepancy volume. The G0 null criterion (agreement ≥ 0.98 on test) is that same statistic, measured on the same test points. Proposed: add B0 = Monte Carlo disagreement rate on the evaluation points as a descriptive comparator for H3, and report H3 as conditional on G0-qualified pairs.
3. **Positioning:** the definition must be cited as an adaptation of DIG (L-022) and DeepXplore (L-023); see `research/LITERATURE.md` G-lit round 2.

### Repository state observed at kick-off

- `runs/smoke/` held about 40 untracked folders from an earlier, failed topic. Deleted on 2026-10-15 at the user's instruction (only `.gitkeep` remains).
- `src/` has only `.gitkeep`. `pyproject.toml` packages only `tools*`; add `src/dce` when code starts.
- `docs/RESEARCH_PROTOCOL.md`, `docs/PROMPTS.md`, and `docs/acceptance/` are generic and were left unchanged.
- No local GPU. `nvidia-smi` is not available.
- No Kaggle credential files were found under `~/.kaggle`. Accounts are managed through the repository harness (`tools/kaggle`).

### Points to verify (G-lit and G0)

- The spec's claim that Artelt et al. set aside the disagreement region as "may be small" (spec §2 point 1). Read the full text before using it in the paper.
- The 6-page limit, template, and special session (D-004).
- Whether `torchvision` MNIST download works in kaggle-prep (Internet ON). Check at the smoke run.
- The `r` calibration uses `2 × median` of the CE distance of `f_new` on T1 seed 0 validation points. Check that the CE search finds a valid point for most validation points before fixing `r`.

---

## 3. Experimental observations

### exp_0001 — G0 on Tier 0 (validation, 2 000 points), 2026-10-15

| Condition | Pass | Key numbers (seeds 0/1/2[/3/4]) | Criterion |
|---|---|---|---|
| f_old accuracy | no | 0.925 / 0.978 / 0.983 / 0.982 / 0.972 | ≥ 0.95 all seeds |
| T1 | yes | ΔFlip_color 0.20 / 0.17 / 0.22 | ≥ 0.15 |
| T3 | no | subgroup acc (new labels) 0.35 / 0.26 / 0.10; agreement outside 0.91 / 0.94 / 0.92 | ≥ 0.85 and ≥ 0.95 |
| T4 | no | band acc (new labels) 0.76 / 0.78 / 0.34 | ≥ 0.80 |
| N0 | no | agreement 0.932 / 0.974 / 0.993 / 0.947 / 0.987; max abs ΔFlip ≤ 0.011 | ≥ 0.98 and ≤ 0.05 |
| N1 | no | agreement 0.903 / 0.979 / 0.982; max abs ΔFlip ≤ 0.018 | ≥ 0.98 and ≤ 0.05 |

Diagnosis (validation, from saved checkpoints, no extra training):

1. **Threshold jitter, not drift.** Null-pair disagreements sit at the decision threshold: 34–77% of them have |φ1 − 0.5| < 0.03, median |φ1 − 0.5| 0.013–0.043. A shift of 0.01 in φ1 is a 1.4% change of the log aspect ratio, below one pixel at 32 × 32. The ΔFlip_k part of the null criterion passes easily (≤ 0.018); only the agreement part fails. The same jitter lowers T3's "agreement outside the subgroup" (0.91–0.94).
2. **f_old is not converged / seed-sensitive.** Training loss plateaus near 0.28 at epoch 15, while the floor with 5% symmetric label noise is about 0.20. Seed 0 has error 0.47 in the band 0.02 ≤ |φ1 − 0.5| < 0.05 (its boundary is offset), so N0/N1 fine-tunes of seed 0 move the boundary back (f_new clean accuracy 0.982).
3. **Null is not a pure null (D-017 reading).** f_old trains on 5% noisy labels; N0 fine-tunes on noise-free labels, so N0 changes the training distribution, not only the data order.
4. **T3 is under-trained.** Fine-tune loss 0.53–0.60 after 3 epochs; the size × aspect interaction is not learned. T4 seed 2 also moves its boundary only partly (band acc 0.34).

### exp_0002 — G0 on Tier 0 after amendment v1.2 (knob 1 for T3/T4), 2026-10-15

| Condition | Pass | Key numbers (seeds 0/1/2[/3/4]) | Criterion |
|---|---|---|---|
| f_old accuracy | **yes** | 0.981 / 0.982 / 0.980 / 0.982 / 0.975 (final loss 0.276–0.285) | ≥ 0.95 |
| T1 | no | ΔFlip_color 0.134 / 0.120 / 0.148 (exp_0001: 0.17–0.22) | ≥ 0.15 |
| T3 (5 ep, lr 1e-3) | no | subgroup acc 0.87 / 0.82 / 0.59; DD outside 0.009 / 0.049 / 0.019 | ≥ 0.85; ≤ 0.05 |
| T4 (5 ep, lr 1e-3) | no | band acc 0.18 / **0.98** / 0.38; DD 0.000 / 0.037 / 0.000 | ≥ 0.80 |
| N0 | **yes** | DD 0.000 all; raw agreement 0.968–0.992; max abs ΔFlip ≤ 0.010 | DD ≤ 0.02, ΔFlip ≤ 0.05 |
| N1 | **yes** | DD 0.000 all; raw agreement 0.979–0.990; max abs ΔFlip ≤ 0.012 | same |

Reading:
- B1/B2/B3 fixed what they targeted: all f_old seeds pass, and null pairs have zero decisive disagreement (raw disagreement is threshold jitter only).
- T1 weakened below 0.15. Likely cause: B2 label noise dilutes the colour–label association the shortcut relies on (colour is drawn from the noise-free label, then 5% of labels flip), and the more converged f_old (cosine) moves less in 3 epochs at lr 3e-4; fine-tune loss plateaus at 0.28–0.29.
- T4 is bimodal: seed 1 moves its boundary fully (0.98), seeds 0 and 2 not at all (DD 0). The shifted band is ≈ 9% of the data, so the loss gain from moving is small; 5 epochs at lr 1e-3 are not always enough.
- T3 is close for seeds 0/1 and short for seed 2 (0.59); fine-tune loss still falling (0.37–0.50 at epoch 5).

### exp_0003 — drift pairs with the last knobs (reusing exp_0002 f_old), 2026-10-15

| Condition | Pass | Key numbers (seeds 0/1/2) | Knobs used |
|---|---|---|---|
| T1 | **yes** | ΔFlip_color 0.200 / 0.161 / 0.221 | 1 of 2 (correlation 0.99) |
| T3 | no | subgroup acc 0.866 / 0.814 / 0.754; DD outside 0.028 / 0.010 / 0.043 | 2 of 2 (10 ep, lr 1e-3) — none left |
| T4 | **yes** | band acc 0.910 / 0.995 / 0.862 | 2 of 2 |

G0 status: f_old, T1, T4, N0, N1 pass; **T3 fails** with no knob left (v1.2 B5: Stop or back to the author).

T3 diagnosis (validation):
- **The change exists and is attributable to size.** ΔFlip_size = +0.323 / +0.317 / +0.319, while every other factor moves ≤ 0.05; decisive disagreement inside the subgroup 0.75 / 0.68 / 0.61 vs outside 0.03 / 0.01 / 0.04.
- **The learned size boundary is soft and seed-shifted.** On all 5 000 validation points, subgroup accuracy rises with size: 0.55–0.66 at size 0.75–0.80, 0.65–0.81 at 0.80–0.85, 0.92–0.99 above 0.90. Just outside (size 0.70–0.75), 31–47% of predictions already follow the flipped rule, but with low margins (DD outside stays ≤ 0.05).
- **A decisive-margin version of the accuracy criterion does not rescue seed 2:** accuracy on subgroup points with |m_n| ≥ κ is 0.934 / 0.896 / 0.779.
- Accuracy kept rising slowly from 5 to 10 epochs only for seed 2 (0.586 → 0.754); seeds 0/1 were flat (0.87 → 0.87, 0.82 → 0.81).

Resolved by the author: option (A) with an added exclusivity condition, amendment v1.3 (D-020).

### exp_0004 — G0 re-evaluation under amendment v1.3 (no training), 2026-10-15

Saved checkpoints (f_old, N0, N1 from exp_0002; T1, T3, T4 from exp_0003), all hash-verified; recomputed numbers match the source runs exactly (max |diff| = 0). **G0-Tier 0 passes: 17/17 pairs, all f_old ≥ 0.95.**

T3 under C1 (seeds 0/1/2):

| | s0 | s1 | s2 | threshold |
|---|---|---|---|---|
| (i) ΔFlip_size | 0.3235 | 0.3170 | 0.3190 | ≥ 0.15 |
| (ii) max_{k≠size} \|ΔFlip_k\| | 0.0170 (posx 0.015) | **0.0465 (aspect)** | 0.0300 (posx) | ≤ 0.05 |
| (iii) DD outside subgroup | 0.0280 | 0.0100 | 0.0427 | ≤ 0.05 |
| C2: subgroup acc, raw (v1.2) | 0.866 | 0.814 | 0.754 | was ≥ 0.85 |
| C2: subgroup acc, \|m_n\| ≥ κ | 0.934 (n=412) | 0.896 (n=384) | 0.779 (n=402) | — |

- (ii) is tight on seed 1: ΔFlip_aspect = −0.0465, 0.0035 below the limit. Aspect is the task factor. T3 *inverts* the labels in the subgroup (new label 1 − 1[φ1 > 0.5]), so labels there still depend fully on aspect, only with the opposite sign. The likelier reading (corrected by the author, 2026-10-15): in the soft transition zone (size ≈ 0.70–0.80) f_new hesitates between two opposite rules, so its predictions there are less determined by aspect. A consequence of the soft boundary, not a second drift.
- This does not carry over to Tier 1: shape (the task factor) cannot be re-sampled, so Flip_shape is undefined and C1 (ii) only ranges over the re-sampleable factors other than k* (D-021, D1).
- Other conditions unchanged from exp_0002/exp_0003: T1 ΔFlip_color 0.200 / 0.161 / 0.221; T4 band acc 0.910 / 0.995 / 0.862; N0 and N1 DD = 0 on every seed, max |ΔFlip| ≤ 0.0115.

Options for the user (resolved: author chose c.1 + c.2 and replaced c.3 by the DD criterion; amendment v1.2, D-018):

- (a) Stop, as spec §9 allows.
- (b) Apply only the pre-declared knob (T3/T4 epochs 3 → 5). This cannot fix N0/N1 or f_old seed 0, so G0 still fails.
- (c) Post-G0 amendment v1.2 (logged "after seeing G0 results"; G0 is a gate, not a main test):
  1. f_old training: 30 epochs with cosine learning-rate decay (accuracy requirement ≥ 0.95 unchanged);
  2. all adaptation sets carry the same 5% label noise as the training split, so N0 differs from f_old only in data order and the drifts differ only in the drift (revises D-017);
  3. agreement criteria (N0/N1 ≥ 0.98, T3 outside ≥ 0.95) measured on points with |φ1 − 0.5| ≥ 0.05, outside the threshold band; thresholds unchanged; ΔFlip criteria unchanged;
  4. then the pre-declared knob for T3/T4 if they still fail.
  Cost: one rerun, ≈ 35 min CPU.

---

### exp_0005 — G1 H4 check on Tier L (H-004), 2026-10-15

H4 **fails** with the frozen solver of spec §6.1 and with every allowed repair tried. The closed forms are correct: `tests/unit/test_linear_theory.py` checks them against exact KKT conditions and SLSQP (cases A and B both present).

| Variant | p=5 found | rel<1e-3 | cos>0.999 | p=20 found | rel<1e-3 | cos>0.999 |
|---|---|---|---|---|---|---|
| spec §6.1 (λ halved once at T/2) | 0.103 | 0.051 | 0.072 | 0.591 | 0.202 | 0.248 |
| R1: halve λ every T/8 while nothing found (allowed: λ schedule) | 0.967 | 0.114 | 0.713 | 1.000 | 0.220 | 0.640 |
| R1, every T/16 | 0.998 | 0.132 | 0.693 | 1.000 | 0.225 | 0.639 |
| R1 + R = 10 (restarts) | 0.968 | 0.115 | 0.712 | 1.000 | 0.240 | 0.659 |
| outside: R1 + cosine lr | 0.805 | 0.427 | 0.487 | 1.000 | 0.800 | 0.908 |
| outside: R1 + cosine lr, T = 800 | 0.891 | 0.650 | 0.692 | 1.000 | 0.917 | 0.960 |
| outside: R1 + 400 polish steps (lr 0.01 → 0) | 0.967 | 0.739 | 0.779 | 1.000 | 0.931 | 0.958 |

Required: ≥ 0.99 for both shares in both dims.

Diagnosis:
1. **Found-rate (spec solver): λ0 = 1 is far above the exact-penalty threshold.** With hinge constraints, the penalised minimiser equals the constrained optimum only if the KKT multipliers of λ‖u‖² are ≤ 1, i.e. λ ≤ λ_crit. λ_crit has median 0.13 (p=5) / 0.67 (p=20), 1% quantile 0.012 / 0.18; only 10% / 65% of pairs have λ_crit ≥ 0.5. Halving once leaves most pairs infeasible: that matches the observed 10% / 59% found-rate. R1 fixes this, within the allowed repairs.
2. **Precision: Adam at constant lr = 0.05 has a noise floor of about 1e-2 relative error.** Traces show the iterate oscillating around u* with amplitude 0.05–0.5 δ until T. Adam's step lr·m̂/√v̂ does not depend on the gradient scale, so no λ schedule shrinks it. Median relative error under R1 is 4e-3, 99% quantile ≈ 5e-2. p=5 is harder: w_o and w_n are nearly parallel (Δw ~ 0.3), δ is large (median 2.3) and λ_crit small.
3. Only changes outside §9 (lr decay, more steps) move precision towards 1e-3, and none reaches 0.99 at p=5.

Status: G1 blocked on H4. The tolerance (spec §8) and lr/T (locked) are the author's decision.

### exp_0006 — G1 H4 on Tier L with the v1.4 two-phase solver (single run, E4), 2026-10-15

Commit f45d5ba, 73 s CPU. **H4 fails, only on Δ-CE at p = 5, only through found-rate.**

| | p=5 found | rel<1e-3 | \|cos\|>0.999 | p=20 found | rel<1e-3 | \|cos\|>0.999 |
|---|---|---|---|---|---|---|
| Δ-CE | **0.967** | 0.967 | 0.967 | 1.000 | 1.000 | 1.000 |
| CE(f_old) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| CE(f_new) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

- Every found witness is exact: Δ-CE rel. error median 1.4e-6 / max 4.8e-6 (p=5), min |cos| 0.99999999999; SLSQP accepted in 4 820 / 4 820 (p=5) and 5 000 / 5 000 (p=20) runs; ordinary CE 4 999 / 5 000 and 5 000 / 5 000. Phase 2 does what v1.4 intended.
- The 33 failures (3.3%) have **no feasible phase-1 candidate in any restart**, so phase 2 never runs. Phase 1 is unchanged from exp_0005 R1 (same found-rate 0.967 there).
- Predicted from the closed forms (no solver rerun): the hardest 3.3% by λ_crit have median δ ≈ 6.3 (all pairs: 2.3); 2.1% need λ < 1/64, which R1 (halving every 50 steps) reaches only at t = 350. Far witnesses and late λ leave too few steps to reach the wedge.
- Not a bug under E4: the code implements the declared algorithm. Per E4 → stop and report; no variant tried.
- Context for the author (not a criterion): Tier 0 lives in [0, 1]^5 (diameter √5 ≈ 2.24), so witnesses at δ ≈ 6 cannot occur there, and on Tier 0/1 the budget r turns far witnesses into NOT_FOUND anyway. exp_0005 had shown R1 with halving every T/16 at found-rate 0.998 (phase 1 only).

### exp_0007 — G1 H4 on Tier L with the v1.5 solver, two seed sets, 2026-10-15

Commit 6de9add, 215 s CPU. **H4 passes**: on both seed sets (0–999, holdout 1000–1999), both dims, and all three methods (Δ-CE, CE(f_old), CE(f_new)), found-rate, share rel < 1e-3 and share |cos| > 0.999 are all 1.000.

| set, p | Δ-CE rel median / max | Δ-CE min \|cos\| | CE rel max | answers from phase 2 (Δ-CE / CE) |
|---|---|---|---|---|
| original, 5 | 1.4e-4 / 4.8e-4 | 0.99960 | 1.0e-4 | 970 / ≈ 707 |
| original, 20 | 6.3e-5 / 4.4e-4 | 0.99992 | 9.9e-5 | 995 / ≈ 967 |
| holdout, 5 | 1.4e-4 / 5.8e-4 | 0.99984 | 9.9e-5 | 982 / ≈ 695 |
| holdout, 20 | 6.6e-5 / 3.8e-4 | 0.99984 | 9.9e-5 | 992 / ≈ 970 |

- All 6 000 SLSQP outputs per (set, dim, method) were confirmed feasible in float32 with the original κ.
- **The residual error is now the F2 tightening itself**, not the optimiser: CE rel. error caps at ≈ 1.0e-4 (= 1e-4 / a with a ≥ 1); Δ-CE reaches 5.8e-4 because both constraints are tightened inside a narrow wedge. The headroom to the 1e-3 tolerance is ≈ 1.7× at p = 5. exp_0006 (ε = 1e-6) had 4.8e-6.
- For the ordinary CE, about 30% of p=5 answers are phase-1 iterates: a just-feasible Adam iterate can be closer than the κ + 1e-4 optimum. Both are within tolerance.
- Runtime: ≈ 21 ms per point for Δ-CE (6 SLSQP runs, linear model), ≈ 16 ms for the ordinary CE.

### exp_0010 — Tier 1 data prep on Kaggle (D-025), 2026-10-15

Kernel `boinhbo/dce-tier1-data-v1` (immutable), CPU, 42 s; smoke_0008 first (attempt 1 failed on a config error: the npz was listed in `required_outputs` but deliberately not fetched). Output `tier1_data.npz`, 9.9 MB, sha256 e74fe388…c8f118 (manifest in `runs/experiments/exp_0010-boinhbo/artifacts/`).

| split | n | share of 9 | chosen thickness q0 / q15 / q50 / q85 / q100 |
|---|---|---|---|
| train | 10 612 | 0.505 | 2.00 / 2.00 / 3.10 / 4.41 / 6.22 |
| val | 1 179 | 0.500 | 2.00 / 2.00 / 3.08 / 4.40 / 6.41 |
| test | 1 991 | 0.507 | 2.00 / 2.00 / 3.09 / 4.45 / 6.45 |

- Ladder medians by level (−2 … +3): 2.0, 2.0, 2.6, 4.4, 6.6, 9.7. Level −2 is valid for only ≈ 51% of images (strokes vanish); level −1 for 99%.
- **The thickness measurement has a floor of 2.0** (a one-pixel stroke has distance 1 on its skeleton), so ≥ 15% of images sit exactly at 2.0 and the distribution is lumpy. Not a problem for T3 (q85 ≈ 4.4 is well above the floor), but it may weigh on the G2 criterion Spearman(thick) ≥ 0.8 (ties) and on σ_thick. To check at G2, not assumed.
- Median |chosen − target| = 0.34 px.

### exp_0009 in flight — CPU cap, 2026-10-15 23:01

At the user's request the running pilot was capped from 16 to 8 logical cores (taskset on the main process and its workers, cores 0–7). Pairs 0 (T1), 3 (T3), 6 (T4) had finished on 16 cores; pairs 9 (N0) and 14 (N1) run on 8. Results do not depend on the number of cores (phase 1 noise is drawn once, points are independent; test `test_parallel_phase2_equals_sequential`), only runtime does. Runtime rows are labelled by core count. Later runs use `n_workers: 8` (tier0.yaml).

### exp_0009 — G1 pilot on Tier 0 validation (seed-0 pairs, N = 100), 2026-10-15/16

Commit 7c6401a, 3.4 h wall. Pairs 0, 3, 6 on 16 cores; 9, 14 on 8 cores (cap at 23:01). Read: found-rate, runtime, solver counters only (pilot discipline; no Recovery, share or Δφ̃ was computed into the record).

Found-rate (Δ-CE = type N; nulls also type O):

| pair | Δ-CE N | Δ-CE O | B1 | B1-det | B3 | B4 | B5 | A1 | A2 | A3 κ=0 | A3 κ=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 s0 | **0.81** | — | 0.90 | 0.90 | 1.00 | 1.00 | 0.08 | 0.65 | 0.90 | 0.90 | 0.72 |
| T3 s0 | 0.82 | — | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.69 | 1.00 | 0.93 | 0.70 |
| T4 s0 | 0.40 | — | 0.87 | 0.86 | 1.00 | 1.00 | 0.07 | 0.35 | 0.87 | 0.40 | 0.26 |
| N0 s0 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.73 | 0.00 |
| N1 s0 | 0.00 | 0.00 | 0.99 | 0.99 | 1.00 | 1.00 | 0.00 | 0.00 | 0.99 | 0.96 | 0.00 |

Runtime of Δ-CE per pair at N = 100 (measured): T1 4.9 min, T3 4.8 min, T4 9.6 min (16 cores); N0 25.5 min, N1 24.9 min (types N + O, 8 cores). Linear extrapolation to N = 300 (D-024): 14–77 min. CPU per point: 39 s (T1) to 79 s (T4); B1 (two CEs) 16–34 s; B1-det 5–10 s.

- **G1 found-rate criterion passes** (T1 0.81 ≥ 0.6). **G1 runtime criterion fails** (≤ 5 min per pair): already at N = 100 for T4 and both nulls.
- Cause (profile on 16 points, diagnostic only): SLSQP on points **without a witness** runs to maxiter = 100 with ≈ 480 function evaluations (N0: 77 of 96 runs at the iteration limit, 14 line-search failures, 5 incompatible constraints); where a witness exists it converges in ≈ 33 iterations / 83 evaluations. Phase 1 is ≈ 22 s per 16 points on 8 workers. So the cost grows with the share of points without a witness: nulls (found 0) and T4 (0.40) are the slow pairs.
- F5 (pair 0, 50 points, 50 random box starts): standard and reference both find 0.86; **the standard solver misses no witness within r**; distance gap median 1e-7, max 0.85%, 4.7% of points longer by > 1e-3.
- float64 gap ≤ 3.4e-6 for every model; "feasible in float64 but rejected by float32" = 0 in every method.
- Descriptive (found-rate only, allowed): Δ-CE returns no witness on both null pairs (N and O), while B1 and A2 "explain" ≈ 100% of null points; A3 with κ = 0 finds 0.73 / 0.96 on nulls, i.e. the margin κ is what removes threshold jitter (Lemma 1). These are pilot observations on validation, not results.
- Full-run cost if nothing changes: the pilot (5 pairs × N = 100, every method) took 3.4 h; 17 pairs × N = 300 scales to the order of 1.5–2 days of CPU on 8 cores.

### Speed-up (A) and Tier 1 cost estimate (v1.6, D-027), 2026-10-16

**(A)**: SLSQP evaluates constraint *values* at every line-search point (≈ 480 per run on witness-less points) but needs the *Jacobian* only at accepted iterates (≈ 100). The old code built the Jacobian (two backward passes) at every evaluation; now values come from a forward pass without grad and the Jacobian is computed only on request (`slsqp_lazy_jac`, commit e9f8e15).

G5 check (`.work/g5_check.py`, not a run record): 16 points of each pilot pair, phase 2 old vs new from the **same** phase-1 starts, outputs compared bit for bit.

| pair | Δ-CE N | Δ-CE O | CE(f_new) |
|---|---|---|---|
| T1 | identical, 1.66× | — | identical, 1.21× |
| T3 | identical, 1.77× | — | identical, 1.24× |
| T4 | identical, 1.69× | — | identical, 1.12× |
| N0 | identical, 1.94× | identical, 1.91× | identical, 1.00× |
| N1 | identical, 1.96× | identical, 2.15× | identical, 1.29× |

Per-evaluation cost, float64, 1 thread, two classifiers (random weights for Tier 1, D-027 reading 3): Tier 0 renderer + 2 CNN 1.91 ms (values) / 6.05 ms (values + Jacobian); Tier 1 decoder + 2 CNN 2.57 / 7.04 ms → ratio ≈ 1.35 / 1.16. Not unusually high. The extra unknown on Tier 1 is the number of SLSQP iterations in 16 dimensions.

exp_0011 (pilot rerun after (A), 8 cores) started from e9f8e15; it gives the measured per-method runtimes for the projections and the found-rate/ID check against exp_0009.

### exp_0011 — G1 pilot rerun after (A), 8 cores; runtime projections (v1.6 G1), 2026-10-16

Commit e9f8e15, 1.7 h wall (exp_0009: 3.4 h). **Every per-pair, per-method found-rate equals exp_0009** (0 mismatches over 5 pairs × 10–12 methods). Found-ID sets are recorded from this run on; exp_0009 did not store them, so the ID check (G5) rests on the bit-identity check on 16 points per pair plus equal found-rates on all 100.

Wall per point (s, 8 cores, N = 100):

| pair | Δ-CE N | Δ-CE O | B1 | B1-det | B5 | A1 | A2 | A3 κ0 | A3 κ1 | B4 (total) |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 1.32 | — | 1.26 | 0.36 | 0.47 | 0.63 | 0.65 | 1.30 | 1.39 | 26 s |
| T3 | 1.48 | — | 1.29 | 0.36 | 0.49 | 0.67 | 0.68 | 1.41 | 1.97 | 27 s |
| T4 | 2.48 | — | 1.37 | 0.39 | 0.48 | 0.67 | 0.65 | 2.33 | 2.86 | 25 s |
| N0 | 3.05 | 3.04 | 1.34 | 0.40 | 0.50 | 0.65 | 0.69 | 1.66 | 3.10 | 24 s |
| N1 | 3.23 | 3.21 | 1.38 | 0.39 | 0.50 | 0.67 | 0.65 | 1.51 | 3.11 | 26 s |

Runtime split (v1.6 G3), Δ-CE N: phase 1 ≈ 61–69 s per pair (100 points, 8 workers); phase-2 CPU per point **with a witness 4.5–5.3 s, without a witness 15.9–19.7 s** (T1 6.1 s: its witness-less points are few and mostly "found but beyond r"). Ordinary CE (A2) phase 2: 2.1–3.1 s per point either way. Δ-CE's cost is concluding "no witness within budget".

**Projections (D-027 readings 2–3; `runs/experiments/exp_0011/projection.txt`)**
- (i) Tier 0 final run (17 pairs × 300 points, every method incl. A1–A3): **16.5 h** ≤ 36 h. Per pair 37 (T1) – 74 (N1) min.
- (ii) Tier 1 evaluation (11 pairs × 200 points; ablations on seed 0 only; Tier 0 per-point costs × 1.35 per-evaluation ratio): **5.8 h**; conservative (× 16/5 more for dimension): **18.6 h** ≤ 36 h.
- (iii) GPU-hours: 0.01 used; Tier 1 training only (VAE, shape oracle, 5 f_old, 11 f_new, ≈ 56k–200k-parameter models on ≈ 10.6k images) is estimated well under 2 GPU-h ≤ 10. Estimate, to be measured at G2.

**G1 passes**: H4 (exp_0007), Δ-CE found-rate on T1 0.81 ≥ 0.6 (exp_0009 = exp_0011), runtime by projection (v1.6 G1 i–iii). (B) is not considered (v1.6 G4).

## 4. Questions

1. Does the tie in Proposition 4 also hold for the constrained solver when λ is decayed? (H-005)
2. Should N1 (virtual drift) use the O direction too? The spec reports O only for N0 and N1, so yes; confirm at G0.
3. Is B4 (permutation importance) meaningful for a 5-factor renderer where factors are sampled independently? Consider as secondary.

---

## 5. Candidate hypotheses

Captured in `research/HYPOTHESES.md`. Nothing new here yet.

---

## 6. Debug and implementation observations

None yet.

---

## 7. Analysis snippets

None yet.

---

## 8. Things to verify

- [ ] ESANN 2027 special session and dates (D-004).
- [ ] Morpho-MNIST: the package installs or the skimage re-implementation is used (spec §4.3; the spec allows either).
- [ ] Shape-oracle accuracy thresholds on real images (≥ 98%) and reconstructions (≥ 95%) at G2.
- [ ] The VAE checkpoint size is below 5 MB (EVALUATION §12).

---

## 9. Promotion queue

Empty.

---

## 10. Promotion rules

Follow the template rules.

---

## 11. Source references

- `docs/source/delta_ce_spec.md` (spec v1.0)
- `research/proposals/delta_ce_proposal_v1.md` (proposal v1, superseded where they differ)
- `docs/GOAL.md`, `docs/EVALUATION.md`, `docs/CONSTRAINTS.md`, `docs/ENVIRONMENTS.md`

---

## 12. Confidence labels

Use the template labels (`low`, `medium`, `high`).

---

## 13. Cleanup policy

Delete items once promoted or resolved. Keep open questions.

---

## 14. What not to store

Final results, decisions, or literature conclusions. Those belong in the other research files.

---

## 15. Suggested lightweight note format

```text
### <short title>
- Observed:
- Confidence: low | medium | high
- Next: <action or promotion target>
```

---

## 16. End-of-session cleanup

Before ending a session: move resolved items out, update `research/STATE.md` §8 (next action).

---

## 17. Completion checklist

- [x] Kick-off inconsistencies recorded.
- [x] Repository observations recorded.
- [ ] Items in §8 verified.
