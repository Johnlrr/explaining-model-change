# CONSTRAINTS.md

> Purpose: Define the operational, resource, and methodological limits that govern this research project.
>
> Ownership: Project specification. Treat this file as read-only during autonomous research unless the user explicitly requests a change.
>
> This file defines what the agent may and may not do while pursuing the goal in `docs/GOAL.md`.

---

## 1. Constraint policy

Every project-specific constraint should be classified as either:

```text
HARD
SOFT
```

### HARD

A hard constraint must not be violated autonomously.

If progress would require violating a hard constraint, the agent must stop and request explicit human approval.

Examples:

```text
Maximum GPU budget: 20 GPU-hours — HARD
External training data: not allowed — HARD
Paid external APIs: not allowed — HARD
```

### SOFT

A soft constraint is a preference rather than an absolute prohibition.

The agent should obey it by default, but may deviate when there is a clear research justification.

Any meaningful deviation should be recorded in `research/DECISIONS.md`.

Examples:

```text
Prefer experiments under 2 hours — SOFT
Prefer models below 1B parameters — SOFT
```

Do not leave the intended strength of an important constraint ambiguous.

---

## 2. Compute budget

<REQUIRED — Define available compute resources and hard/soft limits. If unconstrained, explicitly write `Not constrained`.>

Template:

```text
GPU budget:
Classification: HARD | SOFT
Limit: <value or Not constrained>

CPU budget:
Classification: HARD | SOFT
Limit: <value or Not constrained>

RAM limit:
Classification: HARD | SOFT
Limit: <value or Not constrained>

Storage limit:
Classification: HARD | SOFT
Limit: <value or Not constrained>
```

Possible units include:

- GPU-hours;
- accelerator quota;
- CPU-hours;
- GB RAM;
- GB persistent storage.

If the platform itself imposes a limit, record it even if the project does not add a stricter one.

---

## 3. Time budget

<REQUIRED — Define project-level and per-experiment time constraints.>

Template:

```text
Overall research deadline:
Classification:
Limit: <date/time or Not constrained>

Maximum full-run duration:
Classification:
Limit: <duration or Not constrained>

Preferred experiment duration:
Classification:
Limit: <duration or Not constrained>
```

Do not silently continue past a hard deadline.

---

## 4. Experiment budget

<REQUIRED — Define limits on expensive experiments if applicable.>

Template:

```text
Maximum full experiments:
Classification:
Limit: <count or Not constrained>

Maximum expensive ablations:
Classification:
Limit: <count or Not constrained>
```

Cheap local checks and smoke runs do not need to count toward the same budget unless explicitly specified.

---

## 5. Concurrency override

The generic concurrency policy is defined in `docs/RESEARCH_PROTOCOL.md`.

Default:

```text
Maximum concurrent Kaggle GPU jobs: 2
Concurrent versions of the same kernel slug: not allowed
```

Project-specific override:

```text
Override: none
```

If overriding:

```text
Constraint:
Classification: HARD | SOFT
New limit:
Reason:
```

Do not duplicate the generic rule here unless the project changes it.

---

## 6. Execution timeout override

Generic timeout behavior is defined in `docs/RESEARCH_PROTOCOL.md`.

Project-specific override:

```text
Smoke timeout override: none
Full-run timeout override: none
Local command timeout override: none
```

If a project requires exact values, specify:

```text
Constraint:
Classification: HARD | SOFT
Limit:
Reason:
```

A local command timeout must not be interpreted as proof that the remote job was cancelled.

---

## 7. Retry override

The generic retry policy is defined in `docs/RESEARCH_PROTOCOL.md`.

Default:

```text
Maximum repair/retry attempts per failure class: 5
```

Project-specific override:

```text
Override: none
```

If changed, specify:

```text
Constraint:
Classification: HARD | SOFT
Retry limit:
Reason:
```

---

## 8. Allowed resources

<REQUIRED — Explicitly define what external resources may be used.>

For each category, state whether it is:

```text
Allowed
Restricted
Not allowed
Not constrained
```

Recommended categories:

```text
Pretrained models:
External checkpoints:
External datasets:
Synthetic data:
External repositories/code:
Internet access:
Open-source libraries:
Hosted APIs:
Paid APIs/services:
Competition-provided assets:
Human-generated labels:
Other:
```

If usage is restricted, define the boundary clearly.

Example:

```text
External pretrained checkpoints:
Allowed only if publicly available before the project start date.

External datasets:
Not allowed.

Internet access:
Allowed for literature search and dependency acquisition.
```

---

## 9. Forbidden resources and actions

<REQUIRED — Explicitly list resources or actions that must never be used in this project.>

Examples:

```text
- Hidden/private evaluation labels.
- Undocumented external data.
- Paid APIs without prior approval.
- Manual modification of benchmark labels.
- Test-set-guided tuning.
- Submission strategies that violate competition rules.
```

Classify each meaningful prohibition as HARD unless explicitly intended otherwise.

---

## 10. Data constraints

<REQUIRED when data usage matters. Otherwise write `Not constrained`.>

Specify constraints such as:

```text
Allowed training data:
Allowed validation data:
External data:
Synthetic data:
Data augmentation:
Pseudo-labeling:
Test-time data access:
Data retention/storage:
```

Example:

```text
External training data:
Classification: HARD
Rule: Not allowed.

Synthetic augmentation:
Classification: SOFT
Rule: Allowed, but must be generated only from permitted training data.
```

Do not duplicate train/validation/test information-flow rules already defined in `docs/EVALUATION.md`; reference them instead.

---

## 11. Method constraints

<REQUIRED when the project restricts the solution space. Otherwise write `Not constrained`.>

Possible constraints include:

```text
Model family:
Backbone:
Parameter count:
Training method:
Inference method:
Ensembling:
Test-time adaptation:
Retraining:
Fine-tuning:
Quantization:
External retrieval:
Human intervention:
```

Example:

```text
Backbone changes:
Classification: HARD
Rule: Not allowed.

Parameter count:
Classification: SOFT
Rule: Prefer no more than 10% increase over baseline.

Ensembling:
Classification: HARD
Rule: Not allowed.
```

These constraints should prevent the research process from obtaining gains through methods that violate the intended project scope.

---

## 12. Comparison-resource constraints

<OPTIONAL — Use when efficiency, runtime, memory, or compute fairness matters.>

Specify whether compared methods must use the same:

```text
Hardware:
Precision:
Batch size:
Training budget:
Inference budget:
Attack/evaluation budget:
Wall-clock measurement procedure:
Memory measurement procedure:
```

Classify each as HARD or SOFT.

Detailed metric definitions belong in `docs/EVALUATION.md`.

---

## 13. Research integrity constraints

The generic integrity rules in `docs/RESEARCH_PROTOCOL.md` and `docs/EVALUATION.md` always apply.

At minimum, the project must not permit:

- test-set leakage;
- cherry-picking seeds;
- cherry-picking checkpoints;
- selectively dropping unfavorable samples;
- changing benchmark rules after seeing results;
- presenting invalid runs as scientific evidence;
- presenting reproduced prior work as a new contribution;
- hiding negative results that materially affect interpretation.

Add project-specific integrity constraints here only when needed.

---

## 14. Human-approval boundaries

The agent may autonomously perform normal research actions that are permitted by the project specifications.

For example, normal Kaggle smoke and full jobs do not require approval when they remain within the defined constraints.

The agent must stop and request human approval before:

- modifying any project specification under `docs/`;
- violating or increasing a HARD budget;
- using a resource whose permission is undefined;
- using data or models outside the permitted scope;
- incurring real monetary cost that has not already been authorized;
- changing the evaluation protocol;
- changing acceptance criteria;
- weakening test-set blindness rules;
- bypassing benchmark or competition rules;
- performing an action explicitly marked as requiring approval.

Project-specific additions:

```text
<OPTIONAL — Add other approval boundaries or write None.>
```

---

## 15. Stop conditions

<REQUIRED — Define any project-specific conditions that force autonomous research to stop.>

Generic stop conditions are defined in `docs/RESEARCH_PROTOCOL.md`.

Possible project-specific conditions:

```text
- GPU budget exhausted.
- Overall research deadline reached.
- Maximum number of full runs reached.
- Required resource becomes unavailable.
- All viable hypotheses under the allowed scope are exhausted.
- A blocker requires changing a HARD constraint.
```

Template:

```text
Condition:
Classification: HARD
Action:
```

A stop condition should not be silently bypassed.

---

## 16. Resource priority under scarcity

<REQUIRED — Define what the agent should prioritize when remaining resources are limited.>

Recommended default:

```text
1. Preserve enough budget to validate the current best result.
2. Prefer experiments with high information value.
3. Prefer experiments capable of rejecting major hypotheses.
4. Avoid broad low-information parameter sweeps.
5. Reserve final resources for required reproducibility and success verification.
```

Project-specific priority:

```text
<REQUIRED — Confirm the default above or replace it with project-specific priorities.>
```

This prevents the research loop from spending the entire budget on exploratory runs and leaving no resources for final verification.

---

## 17. Constraint conflicts

If two constraints conflict:

1. HARD constraints override SOFT constraints.
2. More specific project constraints override generic defaults.
3. `docs/EVALUATION.md` controls evaluation validity.
4. `docs/GOAL.md` controls project intent.
5. The agent must not invent a resolution when two HARD constraints are genuinely incompatible.

If the conflict cannot be resolved without changing a specification, stop and request human input.

---

## 18. Exceptions and overrides

A HARD constraint may only be changed through explicit human instruction.

A SOFT constraint may be exceeded autonomously only when:

- there is a clear research justification;
- no HARD constraint is violated;
- the decision is recorded in `research/DECISIONS.md`;
- the deviation does not invalidate the evaluation protocol.

When an exception is approved, record:

```text
Constraint:
Original rule:
Approved exception:
Reason:
Scope:
Date / relevant commit:
```

Do not treat a one-time exception as a permanent rule change unless the specification itself is updated.

---

## 19. Completion checklist

Before autonomous research begins, confirm:

- [ ] Every important constraint is classified as HARD or SOFT.
- [ ] Unconstrained fields explicitly say `Not constrained`.
- [ ] Compute budget is defined.
- [ ] Time budget is defined.
- [ ] Experiment budget is defined or explicitly unconstrained.
- [ ] Concurrency overrides are explicit or `none`.
- [ ] Timeout overrides are explicit or `none`.
- [ ] Retry overrides are explicit or `none`.
- [ ] Allowed external resources are defined.
- [ ] Forbidden resources/actions are explicit.
- [ ] Data constraints are defined where relevant.
- [ ] Method constraints are defined where relevant.
- [ ] Human-approval boundaries are explicit.
- [ ] Project-specific stop conditions are defined.
- [ ] Resource priority under scarcity is defined.
- [ ] No HARD constraint conflicts with the evaluation protocol or project goal.
