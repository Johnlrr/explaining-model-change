# DECISIONS.md

> Purpose: Record durable research and operational decisions that materially affect project direction, interpretation, or execution.
>
> Ownership: Mutable research state. Update this file when an important decision is made, revised, superseded, or reversed.
>
> This file is a **decision log, not an activity log**.

---

## 1. Decision index

Keep this table current.

| ID | Short title | Status | Date | Related hypotheses | Related experiments |
|---|---|---|---|---|---|
| D-001 | <short title> | active | <date> | H-XXX | exp_XXXX |

Allowed status values:

```text
active
superseded
reversed
```

---

## 2. What belongs here

Record a decision when it has lasting consequences for the project.

Typical examples:

- selecting or changing the primary baseline;
- stopping a research direction even though it was not formally falsified;
- changing research strategy because of resource limits;
- deciding not to use a data source, model family, or external method;
- choosing an implementation convention that affects many experiments;
- accepting a known limitation;
- changing environment strategy in a way that affects comparability;
- deviating from a SOFT constraint;
- establishing a project-wide convention not already defined in `docs/`;
- choosing between multiple plausible interpretations of prior evidence.

Do not use this file for routine actions such as:

```text
run exp_0012
fix import error
fetch Kaggle output
update a config value for one experiment
```

Those belong in `research/STATE.md`, `research/HYPOTHESES.md`, run evidence, or Git history.

---

## 3. Decision template

Copy the following section for each important decision.

---

## D-XXX — <short descriptive title>

### Status

```text
Status: active | superseded | reversed
Date:
Last updated:
Relevant commit:
```

---

### Context

<REQUIRED — Explain why a decision was needed.>

Describe the situation, ambiguity, tradeoff, or evidence that forced a choice.

Keep this factual and concise.

---

### Decision

<REQUIRED — State exactly what was decided.>

Prefer a direct statement.

Example:

```text
Do not test larger backbones unless new evidence specifically supports model capacity as the limiting factor.
```

---

### Rationale

<REQUIRED — Explain why this option was chosen.>

Focus on the reasoning that distinguishes this choice from alternatives.

Possible factors:

- experimental evidence;
- literature;
- constraints;
- expected information value;
- reproducibility;
- compute cost;
- implementation risk;
- evaluation validity.

---

### Evidence

<REQUIRED when evidence exists. Otherwise write `Not evidence-driven`.>

Reference relevant sources:

```text
Hypotheses:
Experiments:
Run evidence:
Literature entries:
Constraints:
Other:
```

Prefer IDs and repository references over conversational descriptions.

---

### Alternatives considered

<REQUIRED — Record meaningful alternatives that were considered.>

Example:

```text
1. Continue scaling model size.
2. Switch focus to preprocessing.
3. Run one targeted capacity diagnostic first.
```

Do not list obviously irrelevant alternatives.

---

### Consequences

<REQUIRED — State what this decision changes going forward.>

Consider effects on:

- future hypotheses;
- experiment design;
- baseline comparison;
- implementation;
- environment selection;
- resource usage;
- interpretation of results;
- final reporting.

Example:

```text
Future hypotheses should focus on preprocessing and restoration behavior.
Large-model scaling should not consume additional GPU budget unless the revisit condition is met.
```

---

### Revisit condition

<REQUIRED — Define what evidence or situation would justify reconsidering this decision.>

Examples:

```text
Revisit if a later experiment indicates model capacity is the dominant bottleneck.

Revisit if RTX availability changes enough to make the previously rejected experiment affordable.

Revisit if new prior work introduces a directly comparable baseline.
```

Avoid vague conditions such as:

```text
revisit later
```

---

### Supersedes / superseded by

```text
Supersedes:
Superseded by:
```

Use decision IDs when applicable.

If neither applies:

```text
None
```

---

### Notes

<OPTIONAL — Add concise implementation or interpretation notes that do not belong elsewhere.>

---

## 4. Status semantics

### `active`

The decision currently governs project behavior.

### `superseded`

A newer decision replaces it.

Keep the old entry for history and reference the replacement decision.

Example:

```text
Status: superseded
Superseded by: D-009
```

### `reversed`

The project intentionally moved back from this decision because new evidence invalidated or outweighed the original rationale.

Record the evidence that caused the reversal.

Do not delete the original decision.

---

## 5. Decision lifecycle

Recommended lifecycle:

```text
important ambiguity or tradeoff appears
→ collect relevant evidence
→ consider alternatives
→ make decision
→ record consequences
→ define revisit condition
→ keep active until new evidence warrants change
```

If a later decision changes it:

```text
old decision
→ superseded or reversed
→ new decision references old decision
```

---

## 6. Evidence-driven decisions

Whenever possible, decisions should reference concrete evidence.

Good:

```text
Decision:
Stop testing larger backbones.

Evidence:
exp_0011 and exp_0014 both increased compute substantially without improving the primary metric.
```

Weak:

```text
Decision:
Stop testing larger backbones.

Reason:
Does not seem promising.
```

Not every decision can be evidence-driven, especially early project setup choices. In that case write:

```text
Evidence: Not evidence-driven
```

and explain the rationale clearly.

---

## 7. Decisions derived from negative results

Negative results may justify durable project decisions.

Example:

```text
H-006: rejected
H-008: rejected
↓
D-004:
Stop allocating GPU budget to model-capacity scaling unless new evidence appears.
```

Keep the hypothesis conclusion in `research/HYPOTHESES.md` and the durable strategic consequence here.

---

## 8. Constraint deviations

A meaningful deviation from a SOFT constraint should be recorded here.

Template:

```text
Constraint:
Original preference:
Deviation:
Reason:
Expected benefit:
Expected cost/risk:
Scope:
Revisit condition:
```

HARD constraints cannot be overridden autonomously.

If a human explicitly approves a HARD-constraint exception, record the approval and scope here after the specification is handled appropriately.

---

## 9. Environment and infrastructure decisions

Record environment/infrastructure choices here only when they affect research interpretation or many future runs.

Examples worth recording:

- switching final benchmarking from T4 to RTX;
- deciding all runtime comparisons must use one GPU class;
- accepting a platform limitation that affects artifact handling;
- changing the default remote execution strategy for the project.

Routine infrastructure fixes belong in Git history or issue tracking, not this decision log.

---

## 10. Baseline decisions

When choosing or changing a primary baseline, record:

```text
Baseline selected:
Reason:
Alternatives considered:
Comparability implications:
Required reproduction:
Revisit condition:
```

A baseline change can materially alter interpretation of all later results and should therefore be treated as a durable decision.

---

## 11. Research-direction decisions

Use a decision when the project intentionally changes direction.

Examples:

```text
Stop pursuing architecture scaling.
Focus remaining budget on preprocessing.
Abandon external retrieval because it violates the intended deployment setting.
Prioritize robustness over runtime optimization until target robustness is reached.
```

These decisions should reference the evidence that motivated them.

---

## 12. Decision conflicts

If two active decisions conflict:

1. Prefer the newer decision only if it explicitly supersedes the older one.
2. If neither supersedes the other, treat the conflict as unresolved.
3. Do not silently choose one.
4. Resolve the conflict through a new decision or human input.

A conflict involving project specifications under `docs/` must not be resolved by this file alone.

Project specifications remain authoritative.

---

## 13. Relationship to other files

Use:

```text
docs/GOAL.md
```

for what the project is trying to achieve.

Use:

```text
docs/EVALUATION.md
```

for how results are judged.

Use:

```text
docs/CONSTRAINTS.md
```

for project limits.

Use:

```text
docs/ENVIRONMENTS.md
```

for available execution environments.

Use:

```text
research/STATE.md
```

for the current snapshot and next action.

Use:

```text
research/HYPOTHESES.md
```

for scientific hypotheses and their evidence.

Use:

```text
research/LITERATURE.md
```

for prior work and novelty evidence.

Use:

```text
research/DECISIONS.md
```

only for durable choices and their rationale.

---

## 14. Cleanup and maintenance

Periodically review active decisions.

For each active decision ask:

```text
Is the rationale still valid?
Has the revisit condition been triggered?
Has another decision superseded it?
Does the project still follow it?
```

Do not delete obsolete decisions.

Update status and links instead.

---

## 15. Completion checklist

For each important decision, confirm:

- [ ] Unique decision ID exists.
- [ ] Status is current.
- [ ] Context explains why a decision was needed.
- [ ] Decision is stated precisely.
- [ ] Rationale is explicit.
- [ ] Relevant evidence is referenced.
- [ ] Meaningful alternatives are recorded.
- [ ] Consequences are clear.
- [ ] Revisit condition is defined.
- [ ] Superseded/reversed relationships are recorded when applicable.
- [ ] The entry describes a durable decision, not a routine action.
