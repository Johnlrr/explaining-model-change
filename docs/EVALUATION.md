# EVALUATION.md

> Purpose: Define exactly how research results are evaluated, compared, validated, and accepted.
>
> Ownership: Project specification. Treat this file as read-only during autonomous research unless the user explicitly requests a change.
>
> This file is the source of truth for whether an experiment is scientifically valid and whether the project has met its success criteria.

---

## 1. Evaluation objective

<REQUIRED — State what the evaluation is intended to measure.>

Describe the property that matters scientifically, for example:

```text
Evaluate whether the proposed method improves adversarial robustness relative to the reproduced baseline while preserving clean performance.
```

Keep this aligned with `docs/GOAL.md`.

---

## 2. Primary metric

<REQUIRED — Define exactly one primary metric or one unambiguous primary decision rule.>

Specify:

```text
Metric:
Direction: maximize | minimize
Unit:
Computation:
Evaluation population:
```

Example:

```text
Metric: robust accuracy
Direction: maximize
Unit: percentage points
Computation: correctly classified adversarial examples / total evaluated examples
Evaluation population: official held-out evaluation split under the specified attack protocol
```

### Multi-objective projects

Multiple metrics may be reported, but the project must still define one unambiguous primary decision rule.

Prefer:

```text
maximize robust accuracy
subject to clean accuracy drop <= 1 percentage point
```

instead of leaving several metrics equally important and allowing the agent to decide after seeing results.

---

## 3. Secondary metrics

<OPTIONAL — Define metrics that must be reported but do not independently determine success.>

Examples:

- clean accuracy;
- runtime;
- memory;
- calibration;
- reconstruction quality;
- robustness under secondary attacks;
- variance across seeds.

For each metric specify:

```text
Name:
Direction:
Unit:
Purpose:
```

---

## 4. Benchmark and dataset

<REQUIRED — Identify the benchmark and all data used for evaluation.>

Specify:

```text
Dataset / benchmark:
Version:
Source:
Task:
Relevant subset(s):
Required preprocessing:
Known exclusions:
```

If an external benchmark has an official evaluation protocol, identify the exact version or ruleset.

Do not silently change benchmark composition during research.

---

## 5. Data split and permitted usage

<REQUIRED — Define the role of every split and what information may flow from it.>

Recommended structure:

```text
Train:
- may be used for fitting model parameters;
- may be inspected during development.

Validation:
- may be used for hypothesis selection;
- may be used for hyperparameter tuning;
- may be used for checkpoint selection;
- may be used for error analysis.

Test:
- must remain blind during method development;
- must not be used for tuning, hypothesis generation, checkpoint selection, or implementation decisions;
- may only be evaluated after the candidate method/configuration is locked.
```

If the project has no natural train/validation/test split, define an equivalent protocol explicitly.

---

## 6. Test-set blindness and information-flow rule

The test set is considered blind only if the evaluated method was developed independently of all information derived from that test set.

Forbidden information flow includes:

- aggregate test metrics;
- per-sample test predictions;
- test-set error analysis;
- subgroup test performance;
- qualitative inspection of test failures;
- test-set examples used to motivate implementation changes;
- hyperparameter changes influenced by test performance;
- checkpoint selection influenced by test performance;
- architecture or method changes motivated by prior test results;
- any derived statistic that reveals information about test behavior.

### Reusing the same test set for a later method

A later method may be evaluated on the same test set **if and only if** it was developed independently of all prior information derived from that test set.

Valid pattern:

```text
Method A
→ lock implementation/configuration
→ evaluate on test

Method B
→ developed only from train/validation/literature/theory
→ no decisions influenced by Method A's test result
→ lock implementation/configuration
→ evaluate on the same test
```

Invalid pattern:

```text
Method A
→ evaluate on test
→ inspect weakness on test subset X
→ design Method B to fix X
→ evaluate Method B on the same test
```

In the invalid pattern, that test set is no longer an independent final test for Method B.

### Burden of proof

If it is unclear whether prior test information influenced the later method, treat the test set as non-independent for that method unless provenance is sufficient to establish independence.

Relevant provenance may include:

- timestamps;
- Git commits;
- hypothesis history;
- locked configs;
- experiment records;
- documented development decisions.

---

## 7. Baseline evaluation

<REQUIRED — Define how the primary baseline is evaluated.>

Specify:

```text
Baseline:
Reference source:
Implementation:
Checkpoint/version:
Required reproduction conditions:
```

The baseline should normally be reproduced under the same evaluation conditions used for candidate methods.

Baseline and candidate comparisons should use the same, when applicable:

- dataset and split;
- preprocessing;
- metric implementation;
- attack/evaluation parameters;
- sample count;
- seed policy;
- checkpoint-selection policy;
- compute constraints;
- hardware class if runtime or efficiency is compared.

If exact equality is impossible, document the difference and its impact on comparability.

---

## 8. Evaluation procedure

<REQUIRED — Define the sequence required for a valid evaluation.>

Template:

```text
1. Select a locked experiment commit and configuration.
2. Verify the required data/version.
3. Verify split usage and test blindness.
4. Run the specified evaluation pipeline.
5. Confirm the expected sample/task count.
6. Compute the primary metric.
7. Compute required secondary metrics.
8. Validate outputs and artifacts.
9. Aggregate across seeds/repetitions if required.
10. Apply the acceptance rule.
11. Record the complete result and provenance.
```

Project-specific steps should be added here.

---

## 9. Seeds and repetitions

<REQUIRED — Define how randomness is handled.>

Specify:

```text
Number of independent runs:
Seed values or generation rule:
What is re-seeded:
What remains fixed:
```

If the experiment is deterministic, state:

```text
Seeds/repetitions: not applicable
Reason:
```

Do not choose favorable seeds after seeing results.

---

## 10. Aggregation

<REQUIRED — Define how multiple samples, tasks, seeds, or repetitions are combined.>

Specify as applicable:

```text
Within-run aggregation:
Across-run aggregation:
Reported central statistic:
Reported dispersion:
Handling of failed/invalid runs:
```

Examples:

- mean ± standard deviation;
- median and interquartile range;
- weighted mean across tasks;
- worst-case metric;
- macro average.

Aggregation rules must be fixed before final evaluation.

---

## 11. Uncertainty and statistical evidence

<REQUIRED — Explicitly state whether statistical testing is required.>

Use:

```text
Statistical test required: yes | no
Reason:
Method:
Confidence level / alpha:
Uncertainty reporting:
```

Statistical significance is **not universally mandatory**.

However, when repeated measurements are meaningful and the project has sufficient compute/data, uncertainty estimation and appropriate statistical testing should be used where they materially strengthen the conclusion.

If no statistical test is required, still define how uncertainty or run-to-run variability will be reported when applicable.

Do not introduce a statistical test after seeing the final result merely because it produces a favorable conclusion.

---

## 12. Checkpoint and model selection

<REQUIRED when model/checkpoint selection exists. Otherwise mark not applicable.>

Define exactly how the final model/checkpoint is selected.

Selection must use only permitted development information, normally train/validation data.

Specify:

```text
Selection metric:
Selection split:
Selection frequency:
Tie-breaking rule:
Early-stopping rule:
```

Never select a checkpoint based on final test-set performance.

### Persistent artifact and final-evaluation policy

<REQUIRED when persistent trained artifacts exist. Otherwise mark not applicable.>

Specify:

```text
Checkpoint retention location:
Selection rule:
Final evaluation: reuse selected checkpoint | predefined retraining protocol
Permitted data during any retraining:
When final test access occurs:
```

Default: reuse the validation-selected checkpoint. If retraining is intended,
define the train/validation data and complete protocol before final test access.
Later evaluation must identify the selected remote artifact through its manifest
and verify its SHA256 before loading it.

For an important remote producer, also record its unique kernel slug, pushed
version, manifest SHA256, and artifact checkpoint SHA256 values. Define the
consumer's required preflight version check and mounted manifest/checkpoint hash
verification; any mismatch invalidates evaluation rather than permitting retrain.

---

## 13. Comparison rules

<REQUIRED — Define what must remain controlled for a fair comparison.>

Specify applicable controls:

```text
Dataset/split:
Preprocessing:
Model initialization:
Training budget:
Evaluation budget:
Attack parameters:
Random seeds:
Hardware:
Batch size:
Precision:
Runtime measurement procedure:
Dependency versions:
Other:
```

Not every project requires all of these to match.

If a difference is intentional, document why the comparison remains scientifically meaningful.

---

## 14. Exact acceptance criteria

<REQUIRED — Define the exact rule for project success before autonomous research begins.>

The rule must be mechanically interpretable enough that success is not decided after seeing results.

Template:

```text
Primary condition:
<exact threshold / improvement rule>

Required secondary conditions:
- <condition>
- <condition>

Required repetitions:
<number>

Required uncertainty/statistical condition:
<rule or not required>

Constraint checks:
- <constraint>
- <constraint>
```

Example:

```text
Primary condition:
mean robust accuracy >= reproduced baseline + 2.0 percentage points

Required secondary conditions:
- clean accuracy drop <= 1.0 percentage point
- no increase in attack failure rate due to evaluation errors

Required repetitions:
3 independent seeds

Required uncertainty/statistical condition:
95% confidence interval reported

Constraint checks:
- runtime within project limit
- no benchmark or test-set leakage
```

A single favorable run is not sufficient unless this file explicitly defines one run as sufficient.

---

## 15. Validity requirements

A result is scientifically valid only if all applicable requirements hold.

At minimum verify:

- [ ] Correct benchmark and version were used.
- [ ] Correct split was used.
- [ ] Test-set blindness was preserved.
- [ ] Expected sample/task count completed.
- [ ] Intended experiment configuration was actually used.
- [ ] Run corresponds to the intended Git commit.
- [ ] Required seeds/repetitions completed.
- [ ] Required artifacts exist.
- [ ] Metrics parse correctly.
- [ ] No required metric is NaN/Inf.
- [ ] No accidental data leakage occurred.
- [ ] No result-dependent benchmark changes occurred.
- [ ] No seed/checkpoint/sample cherry-picking occurred.
- [ ] Any required hardware/runtime controls were satisfied.

A completed remote job is not automatically a valid research result.

---

## 16. Invalid experiment conditions

<REQUIRED — Add project-specific invalidation conditions.>

An experiment must be treated as invalid evidence if, for example:

- the wrong dataset or split was used;
- test information influenced method development;
- evaluation stopped before the required workload completed;
- metric computation was incorrect;
- required samples were silently skipped;
- a bug altered the intended method;
- wrong checkpoint/configuration was evaluated;
- data leakage occurred;
- benchmark rules were changed after seeing results;
- only favorable seeds/checkpoints/samples were reported.

Invalid experiments may be useful for debugging but must not be used to support or reject a scientific hypothesis.

---

## 17. Reporting requirements

<REQUIRED — Define what every valid full experiment must report.>

Recommended minimum:

```text
Experiment ID:
Hypothesis ID:
Git commit:
Configuration:
Dataset/split:
Seed(s):
Primary metric:
Secondary metrics:
Aggregation:
Uncertainty:
Requested hardware:
Observed hardware:
Runtime:
Validity status:
Artifacts:
Notes:
```

Store evidence according to `docs/RESEARCH_PROTOCOL.md`.

---

## 18. Final test procedure

<REQUIRED — Define when and how the final test set may be evaluated.>

Before final test evaluation:

- the candidate method must be fixed;
- hyperparameters must be fixed;
- checkpoint-selection rule must be fixed;
- evaluation code must be fixed;
- required validation checks must pass;
- the exact candidate commit/config must be recorded.

Then:

```text
lock candidate
→ run final test
→ validate result
→ report result
```

Do not modify the candidate in response to final test information and then treat the same test evaluation as blind unless the later method satisfies the independence rule in Section 6.

---

## 19. Final verification

Crossing the acceptance threshold once does not automatically complete the project.

Before declaring success:

```text
identify winning commit/config
→ reproduce from clean state
→ run all required seeds/repetitions
→ verify baseline comparison
→ verify test-set independence
→ verify benchmark integrity
→ check leakage/cherry-picking
→ verify constraints
→ apply exact acceptance rule
```

Any stricter verification rule defined here overrides the generic procedure in `docs/RESEARCH_PROTOCOL.md`.

---

## 20. Completion checklist

Before autonomous research begins, confirm:

- [ ] Exactly one primary metric or unambiguous primary decision rule is defined.
- [ ] Metric direction and computation are explicit.
- [ ] Benchmark/dataset version is identified.
- [ ] Train/validation/test usage is explicit.
- [ ] Test-set blindness is defined by information flow.
- [ ] Rules for reuse of a previously evaluated test set are explicit.
- [ ] Baseline reproduction/evaluation procedure is defined.
- [ ] Seeds/repetitions are defined or explicitly not applicable.
- [ ] Aggregation rules are fixed.
- [ ] Statistical testing is explicitly required or not required.
- [ ] Uncertainty reporting is defined where applicable.
- [ ] Checkpoint/model selection rules are fixed.
- [ ] Persistent-artifact retention and final-evaluation checkpoint/retraining policy are fixed when applicable.
- [ ] Fair-comparison controls are defined.
- [ ] Exact acceptance criteria are written before research begins.
- [ ] Validity and invalidation conditions are explicit.
- [ ] Reporting requirements are defined.
- [ ] Final-test procedure is fixed.
- [ ] Final-verification requirements are explicit.
