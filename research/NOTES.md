# NOTES.md

> Purpose: Serve as a lightweight research scratchpad for observations, ideas, questions, and implementation notes that are useful but not yet mature enough to become project state, hypotheses, decisions, or literature records.
>
> Ownership: Mutable research state. The agent may update this file freely during research.
>
> This file is **not a source of truth**. Important information must eventually be promoted to the appropriate canonical file.

---

## 1. Usage principles

Use `research/NOTES.md` for:

- raw observations;
- tentative interpretations;
- unresolved questions;
- candidate hypotheses;
- debugging observations;
- quick analysis snippets;
- things that need verification;
- ideas that may become useful later.

Do not use this file as the final home for:

- current project state;
- validated hypotheses;
- durable decisions;
- literature evidence;
- final experiment results.

Those belong elsewhere.

---

## 2. Inbox

Use this section for newly captured ideas or observations that have not yet been organized.

Recommended lightweight format:

```text
Date:
Source:
Note:
Confidence:
Next action:
```

Example:

```text
Date: 2026-09-16
Source: exp_0014
Note:
Failures appear concentrated among samples with large restoration areas.

Confidence:
low

Next action:
Check correlation between restoration area and robust accuracy.
```

The inbox may be messy temporarily, but should be cleaned periodically.

---

## 3. Experimental observations

Record patterns noticed during or after experiments that are not yet established conclusions.

Template:

```text
Observation ID:
Date:
Source experiment/run:
Observation:
Confidence: low | medium | high
Possible explanation:
Alternative explanation:
What would confirm/refute it:
```

Example:

```text
Observation ID: O-012
Date: 2026-09-16
Source experiment/run: exp_0014

Observation:
Large-mask samples appear to contribute disproportionately to failures.

Confidence:
medium

Possible explanation:
The restoration module may be over-editing unaffected regions.

Alternative explanation:
Large masks may simply correlate with stronger attacks.

What would confirm/refute it:
Analyze error rate against mask area while controlling for attack strength.
```

Do not promote an observation to `research/STATE.md` until it is sufficiently supported.

---

## 4. Questions

Track unresolved research questions.

Template:

```text
Q-XXX:
Question:
Why it matters:
Current evidence:
How to answer:
Status: open | answered | dropped
```

Example:

```text
Q-004:
Question:
Is restoration area itself predictive of robustness failure?

Why it matters:
If yes, it may motivate a more selective restoration mechanism.

Current evidence:
Preliminary pattern in exp_0014.

How to answer:
Run correlation/subgroup analysis on validation results.

Status:
open
```

---

## 5. Candidate hypotheses

Use this section for ideas that may become formal hypotheses but are not yet ready.

Template:

```text
Candidate:
Origin:
Rough idea:
Why plausible:
Missing before promotion:
```

Example:

```text
Candidate:
Selective restoration may outperform full-region restoration.

Origin:
O-012

Rough idea:
Only restore high-confidence adversarial pixels within the detected mask.

Why plausible:
Large restoration areas correlate with failures in current validation observations.

Missing before promotion:
- stronger evidence for the correlation;
- prior-work check;
- measurable prediction;
- falsification condition.
```

When ready, move it to `research/HYPOTHESES.md` and assign an `H-XXX` ID.

---

## 6. Debug and implementation observations

Use for technical findings that may affect experiments but are not yet durable project decisions.

Examples:

- numerical instability under one configuration;
- unexpected memory bottleneck;
- suspicious preprocessing behavior;
- environment-specific quirks;
- possible implementation bug;
- unusual runtime behavior.

Template:

```text
Date:
Source:
Observation:
Research impact:
Next check:
```

If the issue becomes a durable implementation or research decision, promote it to `research/DECISIONS.md`.

If it is purely an infrastructure fix with no research consequence, Git history or issue tracking may be sufficient.

---

## 7. Analysis snippets

Use this section for compact intermediate analysis.

Examples:

```text
exp_0011 vs exp_0014:
robust accuracy: +1.2 pp
clean accuracy: -0.4 pp
runtime: +18%

Tentative interpretation:
gain may be real, but variance across seeds is not yet known.
```

Good uses include:

- quick metric comparisons;
- rough calculations;
- subgroup summaries;
- candidate correlations;
- provisional interpretation.

Do not treat analysis snippets as final evidence until the underlying experiment has been validated.

---

## 8. Things to verify

Keep a short actionable list.

Example:

```text
- [ ] Confirm exp_0014 used the intended checkpoint.
- [ ] Verify whether L-019 already contains confidence-weighted restoration.
- [ ] Check whether the runtime increase is GPU or preprocessing bound.
```

Remove or mark items when resolved.

---

## 9. Promotion queue

Use this section to prevent important information from getting trapped in the scratchpad.

Template:

| Note / item | Promote to | Reason | Status |
|---|---|---|---|
| O-012 | HYPOTHESES.md | strong candidate mechanism | pending |
| Q-004 result | STATE.md | changes current interpretation | pending |
| environment fallback rule | DECISIONS.md | durable operational choice | pending |

Allowed destinations:

```text
research/STATE.md
research/HYPOTHESES.md
research/DECISIONS.md
research/LITERATURE.md
runs/
```

Suggested status values:

```text
pending
promoted
discarded
```

---

## 10. Promotion rules

Promote information when it becomes important enough to guide future work.

### Promote to `research/STATE.md`

When it is a current, sufficiently supported fact that affects:

- active direction;
- current best;
- next action;
- blocker;
- strongest observation.

### Promote to `research/HYPOTHESES.md`

When an idea has:

- clear origin;
- falsifiable statement;
- measurable prediction;
- falsification signal;
- minimal experiment.

### Promote to `research/DECISIONS.md`

When a choice has durable consequences for:

- research direction;
- resource use;
- environment strategy;
- baseline selection;
- project-wide implementation behavior.

### Promote to `research/LITERATURE.md`

When a source or prior-work finding materially affects:

- novelty;
- baseline;
- hypothesis design;
- interpretation.

### Promote to `runs/`

When the information is experiment evidence that must remain reproducible and immutable.

---

## 11. Source references

For important notes, include a traceable source when possible.

Useful references include:

```text
Experiment ID:
Run ID:
Hypothesis ID:
Decision ID:
Literature ID:
Git commit:
Kernel slug:
File/path:
```

Avoid unsupported notes such as:

```text
I think this happened before.
```

Prefer:

```text
exp_0014 showed this pattern on validation.
```

---

## 12. Confidence labels

Optional confidence labels:

```text
low
medium
high
```

Use them only as rough research-state indicators.

They are not substitutes for formal evidence or statistical analysis.

Suggested interpretation:

### `low`

Initial impression or weak evidence.

### `medium`

Repeated or reasonably supported observation, but alternative explanations remain.

### `high`

Strong and repeated evidence, likely ready for promotion to a canonical file.

---

## 13. Cleanup policy

Review this file periodically.

During cleanup:

- promote mature information;
- remove resolved trivial notes;
- mark discarded ideas;
- merge duplicates;
- remove stale implementation noise;
- keep unresolved high-value observations.

Do not let this file grow indefinitely.

Historical material that matters should already exist in canonical files, run evidence, or Git history.

---

## 14. What not to store

Do not use `NOTES.md` for:

- final success criteria;
- authoritative evaluation rules;
- secrets or credentials;
- complete experiment logs;
- permanent benchmark results;
- full paper summaries;
- duplicate copies of other research files;
- unbounded stream-of-consciousness notes.

The purpose is to reduce cognitive loss, not create another source of truth.

---

## 15. Suggested lightweight note format

For fast capture, use:

```text
### <short title>

Date:
Source:
Observation / idea:
Confidence:
Next:
Promotion target:
```

Not every note needs every field.

Flexibility is intentional.

---

## 16. End-of-session cleanup

Before ending a meaningful research session:

1. review new notes;
2. promote anything that became important;
3. update `research/STATE.md`;
4. ensure the next action is not trapped only in this file;
5. clean obvious temporary noise.

The agent should be able to resume the project without relying on `NOTES.md` alone.

---

## 17. Completion checklist

Periodically confirm:

- [ ] Important observations reference their source when possible.
- [ ] Mature hypotheses have been promoted to `research/HYPOTHESES.md`.
- [ ] Durable decisions have been promoted to `research/DECISIONS.md`.
- [ ] Important prior-work findings have been promoted to `research/LITERATURE.md`.
- [ ] Current facts and next actions are reflected in `research/STATE.md`.
- [ ] Experiment evidence is preserved under `runs/`.
- [ ] Resolved or low-value clutter has been removed.
- [ ] No important project knowledge exists only in this scratchpad.
