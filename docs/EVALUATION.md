# EVALUATION.md

> Purpose: Define exactly how research results are evaluated, compared, validated, and accepted.
>
> Ownership: Project specification. Treat this file as read-only during autonomous research unless the user explicitly requests a change.
>
> This file is the source of truth for whether an experiment is scientifically valid and whether the project has met its success criteria.
>
> Source protocol: `docs/source/delta_ce_spec.md` §4–§9 as amended by `docs/source/delta_ce_spec_amendment_v1.1.md` (A1–A7) and `docs/source/delta_ce_spec_amendment_v1.2.md` (B1–B5, after G0 exp_0001). Where this file differs from those sources, this file governs. All gates run on validation; the test split is touched once, in the final procedure (§6, D-011).

---

## 1. Evaluation objective

Measure whether a Δ-CE witness, found within a fixed distance budget `r`, identifies **which factor** a known, injected model update (drift) changed, and **where**, better than CE differencing (B1) and gradient difference (B3) on the same points. Non-local search baselines (B5, DIG-style; A1, DeepXplore-style) are reported alongside for context.

Pair-level detection of *whether* a model changed (H3) is reported descriptively, next to the disagreement-rate baseline B0, and is not part of the acceptance rule (amendment A2).

The property is **decision-behaviour change on the data manifold**, not change of internal rules. Pairs whose internal rules changed without any decision change are outside the scope of the primary measure (see GOAL.md, behaviour-vs-mechanism limit).

---

## 2. Primary metric

```text
Metric: Recovery@r
Direction: maximize
Unit: fraction of evaluation points (0–1)
Computation: (1/N) Σ_i v_i · 1[ top_i = k* ],
             where v_i = 1 iff a valid witness (or CE for baselines) with d(z0,z*) ≤ r exists for point i;
             top_i = argmax_k |Δφ̃_ik| over the factor-change vector Δφ̃_i; k* is the injected factor.
             Invalid or NOT_FOUND points count as 0.
Evaluation population: N = 300 (Tier 0) or N = 200 (Tier 1) valid points per model pair,
             drawn from the TEST split with seed 5000 + pair_id, using the same points for all methods of a pair.
```

Primary decision rule (single, unambiguous): the **acceptance rule in §15** (Route A) is written over H4 and H1, H2a, H2b (differences in Recovery@r). H3 is descriptive.

### Multi-objective projects

Secondary metrics (§3) are reported but do not change the acceptance rule.

---

## 3. Secondary metrics

| Name | Direction | Unit | Purpose |
|---|---|---|---|
| Found-rate | report | fraction | share of points with a valid witness within `r`; needed to interpret Recovery@r |
| Share of k* | maximize | fraction | `mean_{i: v_i=1} |Δφ̃_ik*| / Σ_k |Δφ̃_ik|`; magnitude of attribution |
| Pair-level score S | report | scalar | Δ-CE: found-rate (N ∪ O directions); B1, B1-det, B3: median ‖Δφ̃_i‖₂; B0: disagreement rate (below). Used for descriptive H3 |
| B0 disagreement rate | report | fraction | Monte Carlo estimate of the discrepancy volume (Hammer & Hüllermeier 2021): share of **all** test samples of the tier (unfiltered) where argmax f_old ≠ argmax f_new. The filtered evaluation points cannot be used, since both models agree on them by construction (D-012) |
| AUROC(S) per method (H3) | report | scalar | drift vs null pairs, for Δ-CE, B0, B1, B3; labelled "conditional on G0-qualified pairs". B0 is a **global** measure over the data distribution; Δ-CE found-rate is a **local** measure around the filtered evaluation points. They answer different questions; their AUROCs are shown side by side for reference, never to rank one above the other (A7) |
| SAR (systematic attribution rate) | report | fraction of null pairs | share of N0/N1 pairs with `max_k mean_{v_i=1} share_ik ≥ 0.5` and found-rate ≥ 0.2 |
| Plausibility (Tier 1) | report | fraction, distance | shape-oracle confidence ≥ 0.9 on decoded witnesses; 5-NN distance in shape-oracle feature space to real test images |
| Runtime | report | seconds per point, device stated | efficiency; CPU or GPU recorded per run |
| Flip_k | diagnostic | fraction | sensitivity of predictions to re-sampling factor k (gate G0 only) |

---

## 4. Benchmark and dataset

**Tier L — linear sanity (H4 only).** Synthetic Gaussian linear models, `p ∈ {5, 20}`, 1000 random pairs. No external data.

**Tier 0 — Ellipses renderer (synthetic).** 32×32×3 images from a 5-factor differentiable renderer (aspect, size, colour, background, horizontal position), factors sampled i.i.d. uniform on [0,1]. Label `y = 1[φ1 > 0.5]` with 5% label noise in the training split only. Version: spec §4.2 (renderer v1.0, data seed 1234). Sizes: train 20 000, validation 5 000, test 5 000. Ground-truth factors are exact.

**Tier 1 — Colored MorphoMNIST, 4 vs 9 (real digits, synthetic colour/thickness).** Source: MNIST digits 4 and 9 from torchvision (public download, allowed). Official train split → 90% train / 10% validation (seed 1234). Official test split (digits 4 and 9) → test. Colour and background are sampled; thickness is re-measured on each image (`2 × mean distance-transform on skeleton`). Ground-truth factors: colour φc and background φb (exact, sampled), thickness (measured, the measurement is the ground truth), shape (shape-oracle label). Tier 1 factor values are read through the analytic oracle of spec §5.3, validated at gate G2 (spec §9).

**Known exclusions.** None beyond the stated filters. Pixels or images are never removed after seeing results.

---

## 5. Data split and permitted usage

```text
Train (Tier 0: 20 000; Tier 1: 90% of MNIST 4/9 train):
- fits f_old and the VAE (Tier 1) and generates adaptation data for f_new.
- may be inspected during development.

Validation (Tier 0: 5 000; Tier 1: 10% of MNIST 4/9 train, ≈ 1 180 images):
- all gates: G0 (Tier 0: 2 000 clean validation points; Tier 1: the whole validation split), G1 pilot, G2;
- σ_k normalisation (Tier 1), calibration of r, classifier accuracy requirements;
- hypothesis selection, restart/λ schedule checks, sample-level error analysis, diagnostics;
- the ≤ 2 pre-declared G0 adjustments (spec §4.4) are decided on validation only.

Test (Tier 0: 5 000 renderer samples; Tier 1: official MNIST 4/9 test split):
- evaluated ONLY once per locked candidate, in the final procedure (§19);
- not used for any gate, tuning, diagnostic, hypothesis generation, or implementation decision.
```

Adaptation sets (10 000 Tier 0 / 8 000 Tier 1) are drawn from the training split and are not used for evaluation.

---

## 6. Test-set blindness and information-flow rule

The test set is blind only if the evaluated method was developed without any information derived from it. Forbidden flows are as listed in the template: aggregate test metrics, per-sample test predictions, subgroup test performance, test-failure inspection, and any method or hyperparameter change motivated by test results.

**Project-specific rule (D-011, amendment A1).** Every gate and calibration uses validation:

| Gate or step | Split |
|---|---|
| G0 (Flip_k, classifier accuracy, decisive disagreement DD — v1.2 B3) | validation (Tier 0: 2 000 clean points; Tier 1: whole validation split) |
| G1 (H4 check) | Tier L synthetic |
| G1 (pilot, seed 0, N = 100, all methods) | validation points, sampled with seed 5000 + pair_id |
| G2 (oracle correlation, shape-oracle accuracy, G0-Tier 1) | validation images |
| σ_k normalisation of φ̃ (Tier 1) | clean validation images, D(μ(x)) |
| r calibration | validation, T1 seed 0, 300 points |
| H1–H4, P-tie, descriptive H3, B0, B5, ablations | test, final procedure only |

Pilot discipline (restrictions of D-009 kept, applied to validation; amendment A1): only the pilot's found-rate and runtime are read. Its Recovery@r, share, attribution vectors, and per-sample outputs are not inspected. After a failed G1, only the solver-repair options named in spec §9 (restarts, λ schedule, initialisation from the ordinary CE) may be changed.

### Reusing the same test set for a later method

The same Tier 0 and Tier 1 test sets may be reused for a later method only if that method was developed independently of prior test results (provenance via Git commits and `research/DECISIONS.md`). Any change motivated by test results voids the blindness for the changed method and must be reported as exploratory analysis only (spec §13).

### Burden of proof

If it is unclear whether prior test information influenced a method, the test set is treated as non-independent for that method.

---

## 7. Baseline evaluation

```text
Baseline (primary): B1 — CE differencing (Artelt et al. Definition 3).
Reference source: Artelt et al., arXiv:2104.02459 (2021), NPL 2022 (to be read in full at G-lit).
Implementation: re-implemented under src/dce/baselines/b1_diff.py, same optimiser and budget as Δ-CE.
Checkpoint/version: the same f_old and f_new checkpoints as the candidate pair.
Required reproduction conditions:
  - R = 5 restarts, seeds seed+0 and seed+1 for the two independent CE searches;
  - same proximity/Reg terms, same r, same point set, same φ̃ normalisation;
  - B1-det (R = 1, no noise) reported alongside.
Secondary baselines: B3 (gradient difference, Eq. 14 of Artelt), B4 (Tier 0 only, global permutation importance,
  2 000 test points, 5 permutations).

Non-local search baselines (amendment A4; in the main table, NOT in the Holm family):
  B5 (DIG-style, after Renard et al.): from z0, a line search along the straight segments z0 → z_k for the
      K = 20 nearest training points z_k in the same space Z with metric d (Tier 0: factor vectors of the training
      split; Tier 1: μ(x) of training images), 50 equally spaced steps per segment. Returns the point in F^N_κ with the
      smallest d(z0, ·), if ≤ r; otherwise NOT_FOUND. Δφ̃ = φ̃(D(z)) − φ̃(D(z0)).
  A1 "DeepXplore-style" (λ = 0, after Pei et al.): one-sided disagreement without proximity; reported next to B5.

Detection baseline for descriptive H3 (amendment A2):
  B0 — disagreement rate (Hammer & Hüllermeier 2021), see §3.
```

Shared controls for all methods: same dataset/split, same classifiers, same point set per pair, same `r`, same metric code, same φ̃ normalisation.

---

## 8. Evaluation procedure

```text
1. Select a locked experiment commit and configuration (configs/experiments/exp_NNNN.yaml).
2. Verify the data version: Tier 0 renderer hash + data seed 1234; Tier 1 MNIST source and digit filter.
3. Verify split usage: test access only for run kind "final" (§19); validation otherwise.
4. Build the classifier pairs (f_old^(s), f_new^(pair)); verify clean accuracy ≥ 95% (Tier 0) / ≥ 95% (Tier 1, amendment v1.7 H1; was 97%) on validation (spec §5.1 "test sạch" → validation, A1).
5. For each pair: sample N valid points (§3.2 of spec: both margins ≥ κ, label matches) with seed 5000 + pair_id.
6. Run every method (Δ-CE N, Δ-CE O where specified, B1, B1-det, B3, B4, B5, ablations incl. A1) on the same points; compute B0 on the unfiltered test split.
7. Confirm expected counts: 17 pairs × 300 points (Tier 0); 11 pairs × 200 points (Tier 1).
8. Compute Recovery@r, found-rate, share, S, B0, descriptive AUROC(S), SAR, plausibility, runtime.
9. Compute the statistics in §11 on the test split (final run only).
10. Apply the acceptance rule in §15.
11. Write the reporting record (§18) and the per-seed table.
```

Ablations A1–A3 are run on Tier 0 for all pairs and on Tier 1 for seed 0 only; they are reported, not used in the acceptance rule.

---

## 9. Seeds and repetitions

```text
Tier L: seeds 0–999 for pair generation; not applicable to acceptance.
Tier 0 classifier seeds s ∈ {0,…,4}: one f_old^(s) per seed, shared by all conditions of seed s.
  - T1, T3, T4: seeds 0–2 (3 pairs each).
  - N0: seeds 0–4 (5 pairs). Different data order from f_old, same clean distribution.
  - N1: seeds 0–2 (3 pairs). φ2 ~ U[0.5, 1], labelling rule unchanged.
Tier 1 seeds: T1 and T3 seeds 0–2 (3 pairs each); N0 seeds 0–4 (5 pairs). Ablations: seed 0 only.
Point sampling: seed 5000 + pair_id. Solver restarts: seed_restart = 10 000 + pair_id.
What is re-seeded: classifier initialisation, data order, adaptation sampling, point sampling, restart noise.
What remains fixed: renderer, labels, split membership, κ, r, λ schedule, T, R.
```

Seeds are fixed in the config before any run. No seed is dropped or replaced after seeing results. Per-seed results are always reported.

---

## 10. Aggregation

```text
Within a pair: Recovery@r over its N points (points are the resampling unit for bootstrap).
Across pairs in a condition: mean of pair-level Recovery@r (pair stratification, §11).
Reported central statistic: pair-averaged Recovery@r and the difference Δ-CE − baseline.
Reported dispersion: 95% percentile CI (paired bootstrap), and the per-seed differences.
Seed consistency: for H1, the difference averaged over the (T1_s, T3_s) pair of each seed s must be positive for s = 0, 1, 2.
Invalid runs: any pair that fails validity (§16) is reported and excluded only if the exclusion is recorded before
  the test run; otherwise the run is invalid and the experiment is rerun from the same commit.
AUROC (descriptive H3): computed on pair-level scores S for Δ-CE, B0, B1, B3: Tier 0 9 drift vs 8 null pairs;
  Tier 1 6 vs 5. Exact values reported, labelled "conditional on G0-qualified pairs".
```

---

## 11. Uncertainty and statistical evidence

```text
Statistical test required: yes
Reason: the acceptance rule depends on confidence intervals of paired differences.
Method: paired bootstrap, 10 000 resamples of points, stratified by pair; per-pair means averaged across pairs.
Confidence level / alpha: 95% two-sided percentile CI; family of three tests (H1, H2a, H2b) controlled by Holm (α = 0.05).
Holm reading (D-028): the two-sided p of each test comes from its own bootstrap distribution, p = 2·min(P(d* ≤ 0), P(d* ≥ 0)); a hypothesis passes Holm iff its adjusted p < 0.05. The 95% CI is reported unadjusted. If a test is dropped, Holm runs over the remaining ones.
Bootstrap RNG seeds: 6000 + index (H1 = 0, H2a = 1, H2b = 2, B5-T0 = 3, B5-T1 = 4), one generator per comparison.
Uncertainty reporting: CI for every difference; per-seed table; AUROC without CI (pair counts are too small).
```

The test choice is fixed here and must not change after the test run.

---

## 12. Checkpoint and model selection

```text
Selection metric: none. Checkpoints are final-epoch checkpoints.
Selection split: not applicable.
Selection frequency: not applicable.
Tie-breaking rule: not applicable.
Early-stopping rule: none.
```

f_old: 30 epochs, Adam lr 1e-3 with cosine decay to 0 (v1.2 B1; Tier 1 epochs per its own G0), final checkpoint. f_new: fine-tuning on an adaptation set with 5% symmetric label noise (v1.2 B2); 3 epochs at lr 3e-4 for T1, N0, N1; T3 and T4 use the knobs of v1.2 B4 (knob 1: 5 epochs at lr 1e-3; knob 2: 10 epochs). The final recipe per condition is reported in the paper.

### Persistent artifact and final-evaluation policy

```text
Checkpoint retention location: runs/experiments/exp_NNNN/artifacts/ (models are small, < 5 MB each; fetched locally).
  Tier 1 VAE and shape oracle are retained in the same way.
Selection rule: fixed-epoch checkpoint as above.
Final evaluation: reuse the selected checkpoint. No retraining before test.
Permitted data during any retraining: not applicable (no retraining is planned).
When final test access occurs: only in the final procedure (§19).
```

Each checkpoint is recorded with its SHA256 in `artifact_manifest.json`. A later evaluation loads a checkpoint only after verifying its hash; a mismatch invalidates the evaluation.

---

## 13. Diagnostic and confirmation evaluation

Diagnostic analysis: **permitted, on the validation split only.**

### Diagnostic evaluation

```text
Diagnostic data or split: validation split (Tier 0: 5 000; Tier 1: 10% of train).
Sample-level inspection allowed: yes (validation only).
Permitted information: sample IDs, factor values, predictions, margins, witness distances, decoded witness images,
                        per-sample attributions, failure patterns.
Purpose: identify failure modes (e.g. restarts, λ schedule, not-found cases) and generate hypotheses.
```

Diagnostic results must not be used to choose individual points, and improvement on inspected validation points is not evidence of general improvement. No test-split information of any kind is used before §19.

### Confirmation evaluation

```text
Confirmation data, split, or procedure: a new predefined pair set on validation (seeds unchanged) for any
  hypothesis derived from validation diagnostics, and then the final test procedure (§19) for the claim.
Sample-level access allowed: none on test; full on validation.
Permitted information: aggregate metrics on test only in §19.
Aggregate metric(s): Recovery@r (primary), found-rate; S-based AUROC descriptive.
Seeds/repeats: as §9.
Candidate-promotion criterion: §15 acceptance rule, applied to the test split once.
```

### Candidate promotion

A candidate is promoted only if it satisfies §15 on the test split. Validation improvement alone is not promotion.

---

## 14. Comparison rules

```text
Dataset/split: identical per pair and per tier for all methods.
Preprocessing: identical factor normalisation φ̃ = φ̂ / σ_k (Tier 0: σ_k = 1; Tier 1: σ_k from clean validation images of D(μ(x)), A1, D-011).
Model initialisation: same f_old and f_new for all methods of a pair.
Training budget: fixed per spec §5.1 and §4.4.
Evaluation budget: same R = 5, T = 400, lr = 0.05, λ0 = 1.0, σ = 0.05 for Δ-CE and CE searches; B3 is closed-form;
  B5 uses K = 20 segments × 50 steps and the same r and κ.
Attack parameters: not applicable.
Random seeds: as §9.
Hardware: Tier L and Tier 0 on local CPU; Tier 1 training (VAE, shape oracle, f_old, f_new) on Kaggle T4; every Tier 1 solver and baseline runs on local CPU from downloaded, hash-checked checkpoints, so all methods share one device (v1.6 G2). Runtime comparisons are only within the same device class and are labelled.
Batch size: all N points of a pair per batch (CPU); the solver uses one code path for every method.
Precision: float32.
Runtime measurement procedure: wall-clock and CPU time per point, averaged over the N points of one pair, device recorded; phase-2 CPU time reported separately for points with and without a witness (v1.6 G3).
Dependency versions: recorded in the run metadata (lock file hash).
Other: r is calibrated once on T1 seed 0 validation points (2 × median CE distance of f_new) and then frozen in the config.
```

Intentional differences (e.g. CPU vs GPU runtime) are documented in `research/DECISIONS.md`.

---

## 15. Exact acceptance criteria

```text
Primary condition (Route A):
  H4 passes (implementation check), AND at least one of:
      H1 (Tier 1, pooled T1 + T3 pairs): Recovery@r(Δ-CE) − Recovery@r(B1) is > 0 after Holm (adjusted p < 0.05, §11),
          and is positive for seeds 0, 1, 2;
      H2a (Tier 0, T3 pairs): Recovery@r(Δ-CE) − Recovery@r(B3) is > 0 after Holm (adjusted p < 0.05, §11),
          and is positive for seeds 0, 1, 2;
      H2b (Tier 1, T3 pairs): same as H2a with B3 on Tier 1.

Required secondary conditions (reported, not part of the decision):
  - all pairs in §9 are completed and valid (§16);
  - P-tie for T1 and T4 (Tier 0): |Recovery(Δ-CE) − Recovery(B3)| and |Recovery(Δ-CE) − Recovery(B1-det)|;
    a value > 0.10 is reported as a contradiction of Proposition 4;
  - descriptive H3: AUROC(S) for Δ-CE, B0, B1, B3, "conditional on G0-qualified pairs";
  - B5 and A1 (DeepXplore-style) Recovery@r in the main table;
  - SAR and found-rate on N0/N1.

Solver claim rule versus B5 (amendment A4 as revised by A7; outside Holm; does not change Route A):
  Let d_t = Recovery@r(Δ-CE) − Recovery@r(B5) on T3 of tier t ∈ {0, 1}, and CI_t its 95% paired-bootstrap CI
  (same procedure as §11, no Holm correction).
  (a) The claim "Δ-CE finds better witnesses than DIG-style search" is allowed ONLY IF d_t > 0.10 AND CI_t > 0
      in BOTH tiers.
  (b) In every other case: no solver claim; the claim rests on the theory (L1, P2, P3) and the benchmark.
  (c) If in ANY tier d_t < −0.10 AND CI_t < 0: state explicitly in the results and the discussion that B5 is better.
      This must not be omitted or moved to a footnote.
  (d) Tier-specific outcomes are reported as they are, per tier.

Required repetitions:
  3 classifier seeds for drift pairs; 5 for N0; 3 for N1 (§9).

Required uncertainty/statistical condition:
  paired bootstrap 10 000 resamples, 95% CI, Holm over H1, H2a, H2b only (§11).

Constraint checks:
  - compute budget (docs/CONSTRAINTS.md §2): cumulative GPU-hours ≤ 10;
  - runtime by projection (v1.6 G1): Tier 0 final run ≤ 36 h and Tier 1 evaluation ≤ 36 h wall-clock locally, measured at G1 and reported; actual runtime reported per method, split by points with and without a witness (G3);
  - no test-set use before the candidate is locked (§6, §19).

Route consequences (spec §10 as amended by A3):
  Route A — H4 ∧ (H1 ∨ H2a ∨ H2b): Δ-CE witness paper with the benchmark (claims limited by the B5 rule).
  Route B — H4 ∧ ¬(H1 ∨ H2a ∨ H2b): benchmark + analysis paper (former B′ merged here).
  Stop — G0 fails after the 2 pre-declared adjustments.
```

A single favourable run or a single tier is not sufficient.

---

## 16. Validity requirements

A result is scientifically valid only if all applicable requirements hold.

- [ ] Correct benchmark and version (Tier 0 renderer hash, data seed 1234; Tier 1 MNIST 4/9 filter).
- [ ] Correct split: test only in the final procedure; validation for every gate and calibration.
- [ ] Test-set blindness preserved (§6).
- [ ] Expected pair count and point count completed (17 × 300 for Tier 0; 11 × 200 for Tier 1).
- [ ] Intended config used (hash in run metadata).
- [ ] Run corresponds to the intended Git commit.
- [ ] Seeds §9 completed; no seed dropped after results.
- [ ] Required artifacts exist, with SHA256 in `artifact_manifest.json`.
- [ ] Metrics parse; no NaN/Inf in required metrics.
- [ ] No leakage between adaptation data and evaluation points.
- [ ] No result-dependent change of r, κ, λ schedule, split, or benchmark.
- [ ] No seed, checkpoint, or point cherry-picking.
- [ ] Clean accuracy thresholds met (≥ 95% Tier 0, ≥ 95% Tier 1 since amendment v1.7) on validation.
- [ ] For Tier 1: shape oracle accuracy ≥ 98% on real images and ≥ 95% on reconstructions (gate G2).

A completed remote job is not automatically a valid research result.

---

## 17. Invalid experiment conditions

An experiment is invalid evidence if, for example:

- a method or config was changed after its test result was seen;
- test samples were used for any gate, calibration, diagnostic, or implementation choice;
- r, κ, or the λ schedule were changed after results;
- the Kaggle job was cut at the hard timeout before all pairs completed;
- a checkpoint hash does not match the manifest;
- the restart seed or point-sampling seed differs from §9;
- a bug changed the intended solver or baseline (e.g. B1 restarts not independent).

Invalid experiments may be used for debugging but not to support or reject a hypothesis.

---

## 18. Reporting requirements

```text
Experiment ID:
Hypothesis ID:
Git commit (experiment) and result commit:
Configuration: configs/experiments/exp_NNNN.yaml (hash)
Dataset/split: tier, split name, data version
Seed(s): per §9
Primary metric: Recovery@r per pair and per condition
Secondary metrics: found-rate, share, S, B0, descriptive AUROC(S), SAR, plausibility, runtime; B5 and A1 in the main table
Aggregation: §10
Uncertainty: §11 (CI, Holm, per-seed table)
Requested hardware: (e.g. Kaggle T4 / RTX)
Observed hardware: from inside the runtime
Runtime: seconds per point
Validity status: valid | provisional | invalid (with reason)
Artifacts: manifest path and SHA256 values
Notes: deviations, exclusions, and any exploratory analysis labelled as such
```

Evidence is stored under `runs/experiments/exp_NNNN/` (see `docs/RESEARCH_PROTOCOL.md` §11).

---

## 19. Final test procedure

Before the final test evaluation:

- candidate method and hyperparameters are fixed (λ schedule, κ = 0.5, r calibrated once);
- checkpoint rule is fixed (§12);
- evaluation code and the exact metric definitions are fixed;
- validation-based checks (H4, G0–G2, pilot) have passed;
- the candidate commit and config hash are recorded in `research/STATE.md` under "Test-set state".

Then:

```text
lock candidate → run final test (once per candidate, run kind "final") → validate → report
```

Any change to the candidate after this point makes the test non-blind for the changed method (§6). Such a change requires a new hypothesis and a new locked candidate; it cannot reuse the same test claim.

---

## 20. Final verification

Before declaring success (route A):

```text
identify winning commit/config
→ reproduce from a clean checkout (fresh environment, same config hash)
→ rerun one drift pair and one null pair from scratch and compare Recovery@r within the bootstrap CI
→ verify B1, B3 and B5 are reproduced with the same code path
→ verify test-set independence (§6 provenance)
→ verify benchmark integrity (renderer and MNIST hashes)
→ check leakage and cherry-picking (§16)
→ verify constraints (GPU-hours ≤ 10)
→ apply §15 exactly
```

The reproduction step in the source spec (§15, 9–12/11) is part of this procedure.

---

## 21. Completion checklist

- [x] Exactly one primary metric (Recovery@r) and one unambiguous acceptance rule (§15, Route A = H4 ∧ (H1 ∨ H2a ∨ H2b)).
- [x] Metric direction and computation are explicit.
- [x] Benchmark/dataset versions identified.
- [x] Train/validation/test usage explicit.
- [x] Test-set blindness defined by information flow.
- [x] Reuse rule for the test set explicit.
- [x] Diagnostic access policy defined (validation only).
- [x] Confirmation procedure and promotion criterion predefined.
- [x] Test data excluded from every gate, calibration, and diagnostic (D-011).
- [x] Baseline reproduction/evaluation defined (B1 primary; B3, B5, A1, B0 secondary).
- [x] Seeds and repetitions defined.
- [x] Aggregation rules fixed.
- [x] Statistical testing required (paired bootstrap, Holm).
- [x] Uncertainty reporting defined.
- [x] Checkpoint selection fixed (final epoch, no selection).
- [x] Persistent-artifact retention and final-evaluation policy fixed.
- [x] Fair-comparison controls defined.
- [x] Exact acceptance criteria written before research begins.
- [x] Validity and invalidation conditions explicit.
- [x] Reporting requirements defined.
- [x] Final-test procedure fixed.
- [x] Final-verification requirements explicit.
