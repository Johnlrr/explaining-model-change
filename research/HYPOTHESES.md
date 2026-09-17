# HYPOTHESES.md

> Purpose: Track the complete lifecycle of research hypotheses across the project.
>
> Ownership: Mutable research state. The agent should update this file whenever hypotheses are proposed, tested, reprioritized, concluded, or abandoned.
>
> This file preserves research reasoning across time. Unlike `research/STATE.md`, it is intentionally historical.

---

## 1. Hypothesis index

Keep this table current.

| ID | Short name | Status | Priority | Parent | Main experiment | Novelty status |
|---|---|---|---|---|---|---|
| H-001 | <short name> | proposed | high | — | — | candidate |

Allowed status values:

```text
proposed
queued
testing
supported
rejected
inconclusive
abandoned
```

Allowed priority values:

```text
high
medium
low
```

Suggested novelty status values:

```text
candidate
supported
uncertain
not_novel
not_applicable
```

---

## 2. Status definitions

### `proposed`

The hypothesis has been formulated but is not yet scheduled for testing.

### `queued`

The hypothesis is considered worth testing and is waiting for execution.

### `testing`

An experiment is currently being prepared, run, or validated.

### `supported`

Current valid evidence is consistent with the hypothesis.

This does **not** mean the hypothesis has been proven true.

### `rejected`

Valid evidence contradicts the hypothesis strongly enough that the current formulation should no longer guide research.

### `inconclusive`

The experiment produced valid evidence, but the evidence is insufficient to support or reject the hypothesis.

### `abandoned`

The hypothesis has not necessarily been falsified, but it is no longer worth pursuing.

Possible reasons include:

- excessive cost;
- low expected information value;
- dependency became unavailable;
- project scope changed;
- a stronger hypothesis superseded it;
- prior work made it non-novel or irrelevant.

Do not use `abandoned` to hide an unfavorable valid result.

---

## 3. Priority policy

Priority indicates the expected value of testing the hypothesis now.

### `high`

Use when the hypothesis:

- could materially change project direction;
- directly targets the primary objective;
- has strong evidence or rationale;
- has high information value;
- is required before other hypotheses can proceed.

### `medium`

Use when the hypothesis is useful but not currently critical.

### `low`

Use when the hypothesis is speculative, expensive relative to expected value, or mainly useful as a later ablation/follow-up.

Priority may change as evidence accumulates.

Material reprioritization should be justified in the hypothesis entry or `research/DECISIONS.md`.

---

## 4. Hypothesis template

Copy the following section for each new hypothesis.

---

## H-XXX — <short descriptive name>

### Status

```text
Status: proposed | queued | testing | supported | rejected | inconclusive | abandoned
Priority: high | medium | low
Created:
Last updated:
```

---

### Origin

<REQUIRED — State what motivated this hypothesis.>

Possible sources:

- observation from a previous experiment;
- baseline weakness;
- literature finding;
- theoretical reasoning;
- failure pattern;
- implementation behavior;
- metric tradeoff.

Reference relevant experiment IDs, literature entries, or decisions when possible.

If sample-level diagnostics inspired the hypothesis, record the observed pattern and diagnostic source. The inspected examples themselves must not be treated as confirmation evidence. If diagnostics were not used, write `N/A` for the diagnostic-specific fields.

```text
Origin:
Diagnostic source, if any:
Observed failure pattern:
```

Example:

```text
Origin:
exp_0004 showed that robustness improved only on samples where restoration area was small.
This suggests the restoration mechanism may be over-editing already-correct regions.

Diagnostic source, if any: exp_0004 validation diagnostics
Observed failure pattern: failures concentrate among samples with large restoration areas
```

---

### Hypothesis

<REQUIRED — State one precise, falsifiable claim.>

Prefer:

```text
Reducing restoration outside high-confidence adversarial regions will improve robust accuracy without materially decreasing clean accuracy.
```

Avoid:

```text
Maybe masking can be improved.
```

---

### Rationale

<REQUIRED — Explain why the hypothesis is plausible.>

Keep this focused on mechanism or causal reasoning rather than repeating the hypothesis.

---

### Predictions

<REQUIRED — Write this before running the experiment.>

State in advance what diagnostic and confirmation outcomes would support the hypothesis. The confirmation prediction must describe evidence beyond the examples that inspired the change.

Example:

```text
Diagnostic prediction:
- the recurring failure pattern should change in the predicted direction;

Confirmation prediction:
- compared with the reproduced baseline, the primary robust-accuracy metric should increase;
- the improvement should satisfy the predefined confirmation criterion;
- clean-accuracy drop should remain within the allowed threshold;
- restoration area should decrease.
```

Do not rewrite the prediction after seeing the result.

---

### Falsification signals

<REQUIRED — Define what evidence would count against the hypothesis.>

State what diagnostic and confirmation outcomes would weaken or falsify the hypothesis.

Example:

```text
Diagnostic falsification signal:
The predicted failure pattern does not change on permitted diagnostic data.

Confirmation falsification signal:
The independent confirmation criterion is not met, or any gain comes only with an unacceptable clean-accuracy drop.
```

This should be defined before the full experiment.

---

### Controlled variables

<REQUIRED — List what should remain unchanged so the experiment is interpretable.>

Example:

```text
- model checkpoint;
- attack parameters;
- dataset split;
- preprocessing;
- evaluation code;
- random-seed policy;
- training budget.
```

---

### Minimal experiment

<REQUIRED — Define the smallest experiment that can meaningfully test the hypothesis.>

Template:

```text
Change:
Control:
Smoke:
Full experiment:
Primary metric:
Secondary checks:
```

Prefer minimal targeted experiments over broad uncontrolled changes.

### Confirmation procedure

<REQUIRED — Reference the predefined procedure in `docs/EVALUATION.md` and state how this hypothesis will use it.>

```text
Confirmation data/procedure:
Permitted access level:
Aggregate metric(s):
Seeds/repeats, when applicable:
Support criterion:
Outcome that weakens the hypothesis:
Outcome that falsifies the hypothesis:
```

---

### Parent and dependencies

<OPTIONAL — Use when this hypothesis depends on prior reasoning or experiments.>

```text
Parent hypothesis:
Depends on experiments:
Depends on artifacts:
Blocked by:
```

Example:

```text
Parent hypothesis: H-003
Depends on experiments: exp_0008
```

A parent relation means the new hypothesis was motivated by or derived from another hypothesis.

A dependency means testing cannot proceed until the dependency is satisfied.

---

### Prior-work relation

<REQUIRED when novelty or contribution is relevant. Otherwise mark `not_applicable`.>

```text
Relevant prior work:
Relation: baseline reproduction | replication | adaptation/extension | candidate contribution
Novelty status: candidate | supported | uncertain | not_novel | not_applicable
Literature references:
```

Do not claim novelty merely because the idea was generated independently.

If prior work materially overlaps, update `research/LITERATURE.md`.

---

### Planned experiment

```text
Primary experiment ID:
Config:
Expected environment:
Expected cost:
Expected runtime:
```

If not yet scheduled:

```text
Primary experiment ID: not assigned
```

---

### Experiments

Append only experiments that materially test this hypothesis.

Example:

```text
- exp_0007 — smoke — passed
- exp_0008 — full — valid
- exp_0009 — verification — valid
```

For each relevant experiment, reference run evidence when available.

---

### Results

<Fill after valid experiment evidence exists.>

Record:

```text
Diagnostic result:
Confirmation result:
Observed primary result:
Observed secondary results:
Comparison with diagnostic prediction:
Comparison with confirmation prediction:
Validity:
Relevant run evidence:
```

Do not treat invalid execution as scientific evidence.

---

### Conclusion

<Fill after analysis.>

```text
Conclusion: supported | rejected | inconclusive | abandoned
Reason:
Confidence / remaining uncertainty:
```

Use `supported` only when valid evidence is consistent with the prediction.

Use `rejected` only when valid evidence meaningfully contradicts the hypothesis.

Use `inconclusive` when the evidence cannot distinguish between plausible explanations.

---

### Follow-up

<REQUIRED after conclusion unless no follow-up is warranted.>

Possible outcomes:

```text
- propose child hypothesis;
- run targeted ablation;
- improve measurement;
- stop this line of inquiry;
- revisit after dependency is resolved.
```

If creating a child hypothesis, reference its ID.

---

## 5. Hypothesis lifecycle

Recommended lifecycle:

```text
observation
→ proposed
→ prior-work check when needed
→ prediction + falsification rule
→ queued
→ testing
→ valid evidence
→ predefined confirmation when required
→ supported / rejected / inconclusive
→ follow-up or closure
```

Possible alternate ending:

```text
proposed
→ abandoned
```

when the hypothesis is no longer worth testing.

---

## 6. Experiment-to-hypothesis rule

Default:

```text
one primary hypothesis per experiment
```

An experiment may provide evidence about multiple hypotheses only when interpretation remains clear.

If multiple hypotheses are affected, identify:

```text
Primary hypothesis:
Secondary hypotheses:
```

Do not create a large bundle of changes that makes causal interpretation impossible.

---

## 7. Pre-registration discipline

Before a full experiment begins, the hypothesis entry should already contain:

- hypothesis statement;
- rationale;
- diagnostic and confirmation predictions;
- diagnostic and confirmation falsification signals;
- controlled variables;
- minimal experiment;
- confirmation procedure;
- planned experiment ID/config when available.

This prevents post-hoc rewriting of the hypothesis to match observed results.

Git history may be used to verify that these fields existed before the result was observed.

---

## 8. Test-set information-flow rule

Hypotheses must obey the blindness rules in `docs/EVALUATION.md`.

A hypothesis may be developed from:

- training data;
- validation data;
- literature;
- theory;
- prior non-test experiments;
- implementation analysis.

A hypothesis must not be presented as test-independent if it was influenced by:

- final test metrics;
- test error analysis;
- test subgroup behavior;
- qualitative inspection of final test failures;
- any other information derived from the final test set.

If a hypothesis was influenced by prior test-set information, record that explicitly.

Do not conceal test-derived reasoning.

---

## 9. Negative and rejected hypotheses

Never delete a rejected hypothesis merely because it failed.

A rejected hypothesis should preserve:

- original prediction;
- experiment evidence;
- conclusion;
- useful observation;
- likely reason for failure;
- whether a narrower follow-up remains plausible.

Negative results help prevent repeated dead ends.

---

## 10. Inconclusive hypotheses

Use `inconclusive` when:

- effect size is too small relative to uncertainty;
- experiment lacks power;
- multiple explanations remain plausible;
- measurement is noisy;
- required comparison failed validation;
- evidence conflicts across valid runs.

Do not force every valid experiment into `supported` or `rejected`.

---

## 11. Abandoned hypotheses

Use `abandoned` only when the hypothesis is not worth further resource expenditure.

Record:

```text
Reason for abandonment:
Evidence already collected:
What would justify reopening:
```

An abandoned hypothesis may later be reopened if new evidence or resources change the decision.

---

## 12. Hypothesis relationships

Optional relationship fields:

```text
Parent:
Children:
Depends on:
Supersedes:
Contradicts:
Related:
```

Use these when they materially improve reasoning continuity.

Do not build relationship graphs for trivial connections.

---

## 13. Choosing the next hypothesis

When selecting among queued hypotheses, prefer those with:

1. highest relevance to the primary objective;
2. highest expected information value;
3. strongest evidence/rationale;
4. lowest cost for the information gained;
5. ability to reject major branches of the search space;
6. minimal dependency on uncertain assumptions.

Respect resource-priority rules in `docs/CONSTRAINTS.md`.

Do not automatically choose the newest hypothesis.

---

## 14. Hypothesis cleanup

Periodically review the index.

Allowed actions:

- reprioritize;
- mark stale ideas as abandoned;
- merge genuinely duplicate hypotheses;
- create narrower child hypotheses;
- add missing provenance.

Do not erase historical conclusions.

If two entries are merged, preserve references from the old IDs to the surviving hypothesis.

---

## 15. Completion checklist

For every hypothesis that enters `testing`, confirm:

- [ ] Unique hypothesis ID exists.
- [ ] Status is current.
- [ ] Priority is assigned.
- [ ] Origin is documented.
- [ ] Hypothesis is falsifiable.
- [ ] Rationale is explicit.
- [ ] Diagnostic source and observed pattern are recorded, or marked `N/A`.
- [ ] Diagnostic and confirmation predictions were written before the result.
- [ ] Diagnostic and confirmation falsification signals were written before the result.
- [ ] Controlled variables are defined.
- [ ] Minimal experiment is defined.
- [ ] Confirmation procedure and promotion criterion are identified.
- [ ] Parent/dependencies are recorded when relevant.
- [ ] Prior-work relation is recorded when relevant.
- [ ] Planned experiment is identifiable.
- [ ] Test-set information-flow rules are respected.

For concluded hypotheses, also confirm:

- [ ] Relevant experiments are linked.
- [ ] Result is summarized.
- [ ] Conclusion is one of supported/rejected/inconclusive/abandoned.
- [ ] Follow-up is recorded.
- [ ] Rejected or abandoned hypotheses were not deleted.
