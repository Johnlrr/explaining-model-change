# EVALUATION.md

> Purpose: Define exactly how research results are evaluated, compared, validated, and accepted.
>
> Ownership: Project specification. Treat this file as read-only during autonomous research unless the user explicitly requests a change.
>
> This file is the source of truth for whether an experiment is scientifically valid and whether the project has met its success criteria.
>
> Source protocol: `docs/source/delta_ce_spec.md` §4–§9. Where this file differs from that source, this file governs. The gate procedure (G0–G2) follows the source spec, including its use of the test split for gate checks (§6 below; decision D-009 in `research/DECISIONS.md`).

---

## 1. Evaluation objective

Measure whether a Δ-CE witness, found within a fixed distance budget `r`, identifies the factor that was changed by a known, injected model update (drift), and whether it separates drift pairs from null pairs at the level of model pairs. Also measure whether it does this better than CE differencing (B1) and gradient difference (B3) on the same points.

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

Primary decision rule (single, unambiguous): the project is successful iff the **acceptance rule in §15** holds. Recovery@r is the primary metric; the rule is written over H1, H2a, H2b (differences in Recovery@r) and H3 (AUROC of the pair-level score).

### Multi-objective projects

Secondary metrics (§3) are reported but do not change the acceptance rule.

---

## 3. Secondary metrics

| Name | Direction | Unit | Purpose |
|---|---|---|---|
| Found-rate | report | fraction | share of points with a valid witness within `r`; needed to interpret Recovery@r |
| Share of k* | maximize | fraction | `mean_{i: v_i=1} |Δφ̃_ik*| / Σ_k |Δφ̃_ik|`; magnitude of attribution |
| Pair-level score S | report | scalar | Δ-CE: found-rate (N ∪ O directions); B1, B1-det, B3: median ‖Δφ̃_i‖₂. Used for H3 |
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

Validation (Tier 0: 5 000; Tier 1: 10% of MNIST 4/9 train):
- hypothesis selection, calibration of r, restart/λ schedule checks, sample-level error analysis, diagnostics.

Test (Tier 0: 5 000 renderer samples; Tier 1: official MNIST 4/9 test split):
- gate checks listed in §6 (aggregate statistics only);
- the final evaluation in §19, once per locked candidate;
- no sample-level inspection, no tuning, and no method-comparison metric (Recovery@r, share, AUROC) before the candidate is locked.
```

Adaptation sets (10 000 Tier 0 / 8 000 Tier 1) are drawn from the training split and are not used for evaluation.

---

## 6. Test-set blindness and information-flow rule

The test set is blind only if the evaluated method was developed without any information derived from it. Forbidden flows are as listed in the template: aggregate test metrics, per-sample test predictions, subgroup test performance, test-failure inspection, and any method or hyperparameter change motivated by test results.

**Project-specific rule (D-009).** The source spec applies the gates to test samples. This project follows the spec. The test split may therefore be read for the benchmark-qualification and solver-mechanics checks below, and only for the aggregate statistics named there:

| Gate or step | Split | Statistics that may be read |
|---|---|---|
| G0 (phenomenon check) | test, 2 000 clean points (spec §9) | Flip_k, classifier accuracy, agreement f_new = f_old |
| G1 (H4 check) | Tier L synthetic | H4 tolerances |
| G1 (pilot, seed 0, N = 100, all methods) | test points, spec §7 sampling | found-rate and runtime only |
| G2 (oracle correlation, shape-oracle accuracy, G0-Tier 1) | test images (spec §9) | Spearman, accuracy, agreement, Flip_k |
| σ_k normalisation of φ̃ (Tier 1) | clean test images (spec §5.3) | σ_k |
| r calibration | validation, T1 seed 0, 300 points | median CE distance |
| H1–H4 and P-tie tests | test, final procedure only | all |

Restrictions that keep the comparison honest: the pilot's Recovery@r, share, attribution vectors, and per-sample outputs are not inspected, and only the solver-repair options named in spec §9 (restarts, λ schedule, initialisation from the ordinary CE) may be changed after a failed G1. The ≤ 2 pre-declared G0 adjustments (spec §4.4) are decided on the G0 statistics above. The paper reports that gates used the test split.

Consequence: the gate statistics are test-derived information about the benchmark and solver, not about the method comparison. Any change motivated by a method-comparison metric on test voids blindness for the changed method (see §6 "Reusing the same test set").

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
```

Shared controls for all methods: same dataset/split, same classifiers, same point set per pair, same `r`, same metric code, same φ̃ normalisation.

---

## 8. Evaluation procedure

```text
1. Select a locked experiment commit and configuration (configs/experiments/exp_NNNN.yaml).
2. Verify the data version: Tier 0 renderer hash + data seed 1234; Tier 1 MNIST source and digit filter.
3. Verify split usage: test access only for the gate statistics in §6 and for run kind "final" (§19); validation otherwise.
4. Build the classifier pairs (f_old^(s), f_new^(pair)); verify clean accuracy ≥ 95% (Tier 0) / ≥ 97% (Tier 1) on validation.
5. For each pair: sample N valid points (§3.2 of spec: both margins ≥ κ, label matches) with seed 5000 + pair_id.
6. Run every method (Δ-CE N, Δ-CE O where specified, B1, B1-det, B3, B4, ablations) on the same points.
7. Confirm expected counts: 17 pairs × 300 points (Tier 0); 11 pairs × 200 points (Tier 1).
8. Compute Recovery@r, found-rate, share, S, SAR, plausibility, runtime.
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
AUROC (H3): computed on pair-level scores S: Tier 0 9 drift vs 8 null pairs; Tier 1 6 vs 5. Exact values reported.
```

---

## 11. Uncertainty and statistical evidence

```text
Statistical test required: yes
Reason: the acceptance rule depends on confidence intervals of paired differences.
Method: paired bootstrap, 10 000 resamples of points, stratified by pair; per-pair means averaged across pairs.
Confidence level / alpha: 95% two-sided percentile CI; family of three tests (H1, H2a, H2b) controlled by Holm (α = 0.05).
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

f_old: 15 epochs (Tier 0) / 10 epochs (Tier 1), final checkpoint. f_new: fine-tuning for 3 epochs (or 5 epochs if the single pre-declared G0 knob is used, §4.4 of spec), final checkpoint.

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

Diagnostic results must not be used to choose individual points, and improvement on inspected validation points is not evidence of general improvement. Test-split gate statistics (§6) are aggregate only; no sample-level test inspection is permitted.

### Confirmation evaluation

```text
Confirmation data, split, or procedure: a new predefined pair set on validation (seeds unchanged) for any
  hypothesis derived from validation diagnostics, and then the final test procedure (§19) for the claim.
Sample-level access allowed: none on test; full on validation.
Permitted information: aggregate metrics on test only in §19.
Aggregate metric(s): Recovery@r (primary), S-based AUROC, found-rate.
Seeds/repeats: as §9.
Candidate-promotion criterion: §15 acceptance rule, applied to the test split once.
```

### Candidate promotion

A candidate is promoted only if it satisfies §15 on the test split. Validation improvement alone is not promotion.

---

## 14. Comparison rules

```text
Dataset/split: identical per pair and per tier for all methods.
Preprocessing: identical factor normalisation φ̃ = φ̂ / σ_k (Tier 0: σ_k = 1; Tier 1: σ_k from clean test images of D(μ(x)), spec §5.3, D-009).
Model initialisation: same f_old and f_new for all methods of a pair.
Training budget: fixed per spec §5.1 and §4.4.
Evaluation budget: same R = 5, T = 400, lr = 0.05, λ0 = 1.0, σ = 0.05 for Δ-CE and CE searches; B3 is closed-form.
Attack parameters: not applicable.
Random seeds: as §9.
Hardware: Tier L and Tier 0 on local CPU; Tier 1 on Kaggle GPU. Runtime comparisons are only within the same device class and are labelled.
Batch size: 300 points per batch on GPU; CPU runs are unbatched or batched by the same code path.
Precision: float32.
Runtime measurement procedure: wall-clock per point, averaged over the N points of one pair, device recorded.
Dependency versions: recorded in the run metadata (lock file hash).
Other: r is calibrated once on T1 seed 0 validation points (2 × median CE distance of f_new) and then frozen in the config.
```

Intentional differences (e.g. CPU vs GPU runtime) are documented in `research/DECISIONS.md`.

---

## 15. Exact acceptance criteria

```text
Primary condition:
  (1) H3: AUROC of pair-level score S (Δ-CE) ≥ 0.90 on Tier 0 (9 drift vs 8 null pairs) AND
      ≥ 0.90 on Tier 1 (6 drift vs 5 null pairs); AND
  (2) at least one of the following holds:
      H1 (Tier 1, pooled T1 + T3 pairs): Recovery@r(Δ-CE) − Recovery@r(B1) has 95% CI > 0 after Holm,
          and is positive for seeds 0, 1, 2;
      H2a (Tier 0, T3 pairs): Recovery@r(Δ-CE) − Recovery@r(B3) has 95% CI > 0 after Holm,
          and is positive for seeds 0, 1, 2;
      H2b (Tier 1, T3 pairs): same as H2a with B3 on Tier 1.
  (3) H4 passes (implementation check; a failure is a bug, not a result).

Required secondary conditions:
  - all pairs in §9 are completed and valid (§16);
  - the predicted tie (P-tie) is reported for T1 and T4 (Tier 0): |Recovery(Δ-CE) − Recovery(B3)| and
    |Recovery(Δ-CE) − Recovery(B1-det)| are reported; a value > 0.10 is reported as a contradiction of Proposition 4;
  - SAR and found-rate on N0/N1 are reported.

Required repetitions:
  3 classifier seeds for drift pairs; 5 for N0; 3 for N1 (§9).

Required uncertainty/statistical condition:
  paired bootstrap 10 000 resamples, 95% CI, Holm over H1, H2a, H2b (§11).

Constraint checks:
  - compute budget (docs/CONSTRAINTS.md §2): cumulative GPU-hours ≤ 10;
  - runtime per pair ≤ 5 minutes for Δ-CE (G1 requirement, reported);
  - test-set use limited to the §6 gate statistics before the candidate is locked (§6, §19).

Route consequences (spec §10, mirrored in research/DECISIONS.md):
  Route A — conditions (1) and (2) hold: claim the Δ-CE method with the benchmark.
  Route B — H4 holds; H1, H2a, H2b all fail; H3 holds: report theory plus evidence that B1/B3 suffice for local changes.
  Route B′ — H3 fails: report the benchmark and negative controls only.
  Stop — G0 fails after the 2 pre-declared adjustments.
```

A single favourable run or a single tier is not sufficient.

---

## 16. Validity requirements

A result is scientifically valid only if all applicable requirements hold.

- [ ] Correct benchmark and version (Tier 0 renderer hash, data seed 1234; Tier 1 MNIST 4/9 filter).
- [ ] Correct split: test only for the §6 gate statistics and the final procedure; validation otherwise.
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
- [ ] Clean accuracy thresholds met (≥ 95% Tier 0, ≥ 97% Tier 1) on validation.
- [ ] For Tier 1: shape oracle accuracy ≥ 98% on real images and ≥ 95% on reconstructions (gate G2).

A completed remote job is not automatically a valid research result.

---

## 17. Invalid experiment conditions

An experiment is invalid evidence if, for example:

- a method or config was changed after its test result was seen;
- test samples were used for a sample-level diagnostic, or a gate statistic beyond those listed in §6;
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
Secondary metrics: found-rate, share, S, SAR, plausibility, runtime
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
→ verify B1 and B3 are reproduced with the same code path
→ verify test-set independence (§6 provenance)
→ verify benchmark integrity (renderer and MNIST hashes)
→ check leakage and cherry-picking (§16)
→ verify constraints (GPU-hours ≤ 10)
→ apply §15 exactly
```

The reproduction step in the source spec (§15, 9–12/11) is part of this procedure.

---

## 21. Completion checklist

- [x] Exactly one primary metric (Recovery@r) and one unambiguous acceptance rule (§15).
- [x] Metric direction and computation are explicit.
- [x] Benchmark/dataset versions identified.
- [x] Train/validation/test usage explicit.
- [x] Test-set blindness defined by information flow.
- [x] Reuse rule for the test set explicit.
- [x] Diagnostic access policy defined (validation only).
- [x] Confirmation procedure and promotion criterion predefined.
- [x] Test use for gates limited to the aggregate statistics in §6 (D-009); no sample-level test diagnostics.
- [x] Baseline reproduction/evaluation defined (B1 primary, B3 secondary).
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
