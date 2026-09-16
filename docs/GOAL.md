# GOAL.md

> Purpose: Define what this research project is trying to achieve.
>
> Ownership: Project specification. Treat this file as read-only during autonomous research unless the user explicitly requests a change.

---

## Research question

<REQUIRED — State the main research question as precisely as possible. Prefer one primary question that can be answered by evidence rather than a broad topic description.>

Example form:

```text
Can <method/idea> improve <target property> compared with <baseline> under <evaluation setting>?
```

---

## Primary objective

<REQUIRED — State the concrete outcome this project is trying to achieve. Describe the intended research outcome, not the implementation steps.>

The objective should be specific enough that another researcher can understand what constitutes meaningful progress.

---

## Motivation

<REQUIRED — In 2–5 sentences, explain why this question is worth investigating.>

Keep this short. Focus on:

- the practical or scientific problem;
- why the current baseline or existing approach is insufficient;
- why answering the research question would be useful.

Detailed prior-work discussion belongs in `research/LITERATURE.md`.

---

## Baseline

<REQUIRED — Identify the primary baseline used as the reference point for this project.>

Include enough information to identify it unambiguously, for example:

```text
Method/model:
Reference implementation or source:
Relevant version/checkpoint:
Why this is the primary baseline:
```

The baseline should normally be reproduced under the project evaluation conditions before new contributions are judged against it.

Reproduction procedure and evaluation details belong in `docs/EVALUATION.md` and `docs/RESEARCH_PROTOCOL.md`.

---

## Target contribution

<REQUIRED — Describe the kind of contribution this project is intended to produce.>

Possible contribution types include:

- new method or algorithm;
- adaptation or extension of an existing method;
- robustness improvement;
- accuracy/performance improvement;
- efficiency improvement;
- reduction in compute, memory, or latency;
- empirical finding;
- failure analysis;
- theoretical or mechanistic insight;
- evaluation methodology;
- other clearly stated research contribution.

Write the intended contribution without assuming that novelty has already been established.

Example:

```text
Develop and validate a modification to the baseline method that improves robustness under the defined benchmark while preserving clean performance.
```

---

## Success criteria

<REQUIRED — Define success at a high level.>

Describe what must be demonstrated for the project to count as successful, without duplicating exact metric definitions, thresholds, seed counts, or statistical rules from `docs/EVALUATION.md`.

Example:

```text
The proposed contribution must outperform the reproduced baseline under the official project evaluation protocol while satisfying all required constraints and validation checks.
```

Exact acceptance criteria belong in:

```text
docs/EVALUATION.md
```

A result is not considered final until it passes the success-verification procedure in `docs/RESEARCH_PROTOCOL.md`.

---

## Secondary objectives

<OPTIONAL — List useful outcomes that are desirable but not required for primary project success.>

Examples:

- reduce runtime;
- simplify the method;
- improve interpretability;
- identify failure modes;
- understand sensitivity to important parameters.

Do not allow secondary objectives to replace the primary objective unless this file is explicitly revised.

---

## Out of scope

<REQUIRED — State what this project intentionally does not attempt to solve.>

Use this section to prevent uncontrolled scope expansion.

Examples:

```text
- Other datasets not included in the official benchmark.
- Large-scale architecture search.
- Optimizing for deployment hardware outside the defined environment.
- Solving related tasks that do not directly answer the research question.
```

---

## Non-goals

<REQUIRED — Define outcomes that may look promising but must not be treated as project success.>

Examples:

```text
- Improving a proxy metric while the primary evaluation metric does not improve.
- Reporting only the best seed or checkpoint.
- Beating an external published number without reproducing the baseline under comparable conditions.
- Obtaining gains by changing the benchmark, split, or evaluation rules.
- Reproducing an existing method without an additional research contribution, except when reproduction itself is explicitly the project objective.
```

Use this section to make misleading shortcuts explicit.

---

## Completion checklist

Before autonomous research begins, confirm:

- [ ] The research question is precise and answerable by evidence.
- [ ] The primary objective is clearly stated.
- [ ] Motivation is present and limited to roughly 2–5 sentences.
- [ ] A primary baseline is identified unambiguously.
- [ ] The intended contribution type is stated.
- [ ] High-level success criteria are defined.
- [ ] Exact evaluation details are delegated to `docs/EVALUATION.md`.
- [ ] Out-of-scope boundaries are explicit.
- [ ] Non-goals prevent obvious shortcuts or misleading success claims.
- [ ] No section depends on undocumented assumptions essential to interpreting the project goal.
