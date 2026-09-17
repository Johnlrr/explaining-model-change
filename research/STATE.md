# STATE.md

> Purpose: Maintain a concise, current snapshot of the research project so an agent can resume work without reconstructing the entire history.
>
> Ownership: Mutable research state. The agent should update this file whenever meaningful project state changes.
>
> This file is a **snapshot, not a history log**. Historical evidence belongs in Git, `research/HYPOTHESES.md`, `research/DECISIONS.md`, `research/LITERATURE.md`, and `runs/`.

---

## 1. Project status

```text
Status: initializing | baseline | researching | verifying | blocked | completed
Last updated:
Current commit:
Current branch:
```

Use the status that best describes the project now.

---

## 2. Current objective

<REQUIRED — Summarize the active research objective in 1–3 lines.>

Do not duplicate all of `docs/GOAL.md`.

Example:

```text
Improve adversarial robustness over the reproduced baseline while preserving clean accuracy under the official evaluation protocol.
```

---

## 3. Baseline status

```text
Baseline:
Status: not_started | running | reproduced | failed | unavailable
Best reproduced result:
Primary metric:
Relevant experiment ID:
Commit:
Notes:
```

If baseline reproduction is skipped or impossible, reference the corresponding decision in `research/DECISIONS.md`.

---

## 4. Current best result

Only a scientifically valid experiment should be treated as the current best.

```text
Experiment ID:
Hypothesis ID:
Primary metric:
Secondary constraints satisfied: yes | no | n/a
Comparison vs baseline:
Validity: valid | provisional | invalid
Commit:
Run evidence:
```

Use:

```text
None yet
```

if no valid candidate result exists.

A provisional or invalid result must not silently replace the current valid best.

---

## 5. Active hypothesis

```text
Hypothesis ID:
Short statement:
Status: proposed | testing | supported | rejected | inconclusive
Current experiment ID:
Why this hypothesis is active:
```

Detailed hypothesis history belongs in `research/HYPOTHESES.md`.

---

## 6. Latest experiment

```text
Experiment ID:
Hypothesis ID:
Run type: smoke | full | verification | baseline
Result:
Validity: valid | provisional | invalid
Primary observation:
Conclusion:
Relevant run evidence:
```

This section describes the latest meaningful experiment, not necessarily the current best.

---

## 7. Strongest current observations

<REQUIRED — Keep only observations that materially affect current research direction.>

Example:

```text
- H-003 improved validation robustness but caused unacceptable clean-accuracy loss.
- Runtime bottleneck is dominated by restoration rather than classification.
- Increasing model capacity did not improve the primary metric.
```

Avoid turning this section into an experiment history.

---

## 8. Next action

This should be the single most important action for the agent to take after reading this file.

```text
Exact next action:
Why:
Expected evidence:
Relevant hypothesis:
Relevant experiment/config:
```

Example:

```text
Exact next action:
Run smoke for exp_0007 testing H-004.

Why:
Implementation and local validation passed, and this is the minimal experiment that can test H-004.

Expected evidence:
Whether the modified normalization path executes correctly and produces finite metrics.
```

The next action should be concrete enough to execute without rediscovering project context.

---

## 9. In-flight remote jobs

Track all currently active remote jobs before and while waiting. These records are
recovery checkpoints for interruption or local execution limits; a job appearing
here does not mean the agent should stop normally. Under normal operation, block on
the executor wait when no independent work remains, then remove or update the entry
after reconciling the terminal result.

If none exist:

```text
In-flight jobs: none
```

Otherwise, for each job record:

```text
Experiment ID:
Run type:
Kernel slug / remote job ID:
Purpose:
Submitted from commit:
Requested hardware:
Last known status:
Last checked:
Expected outputs:
```

### Resume rule

Before submitting new remote work, the agent must inspect this section.

For every in-flight job:

```text
check remote status
→ if complete: fetch and validate outputs
→ if failed: classify failure
→ if still running: avoid duplicate submission
→ update STATE.md
```

Do not submit a replacement job merely because the previous local session ended.

When resuming after interruption, reconcile every in-flight job first: fetch and
validate completed work, classify terminal failures, and continue blocking wait for
queued or running work unless an external interruption, local execution limit,
human-intervention requirement, or explicit stop condition prevents it. Do not hand
monitoring to the human merely because a recovered job remains active.

---

## 10. Test-set state

This section is required to preserve test-set blindness and information-flow provenance.

```text
Status: untouched | evaluated
Last evaluation:
Methods/configurations evaluated:
Relevant commits:
Information revealed:
Blindness notes:
```

### `untouched`

Use only when no information derived from the final test set has been observed.

### `evaluated`

Use once final test information has been revealed.

Record enough information to determine whether future methods are independent of prior test-set information.

Example:

```text
Status: evaluated
Last evaluation: exp_0018
Methods/configurations evaluated: method-A
Relevant commits: abc123
Information revealed: aggregate primary and secondary metrics only
Blindness notes:
Method-B was already fully specified and committed before exp_0018 completed.
```

Do not reset this field to `untouched`.

---

## 11. Resource state

Keep current resource usage visible.

Template:

```text
GPU budget used:
GPU budget remaining:
Full experiments used:
Full experiments remaining:
Wall-clock deadline / time remaining:
Other constrained resources:
```

If a resource is not constrained:

```text
Not constrained
```

Use the accounting rules defined in `docs/CONSTRAINTS.md`.

---

## 12. Current blockers

If none:

```text
Blockers: none
```

Otherwise:

```text
Blocker:
Impact:
What has already been tried:
Human action required: yes | no
Next resolution attempt:
```

Do not hide blockers by repeatedly retrying the same failure without new evidence.

---

## 13. Immediate queue

Keep only a short queue of the next 1–3 likely actions.

Example:

```text
1. Fetch and validate exp_0012 when its Kaggle job completes.
2. If valid, update H-006 based on the result.
3. If H-006 is rejected, evaluate whether H-007 has enough evidence to proceed.
```

This is not the full hypothesis backlog.

The complete backlog belongs in `research/HYPOTHESES.md`.

---

## 14. Current research phase summary

<OPTIONAL — Use when a short free-form summary materially improves resume quality.>

Keep this concise.

Example:

```text
Baseline is reproduced. Two optimization hypotheses were rejected.
Current evidence points toward preprocessing as the main source of robustness loss.
H-006 is running as the next targeted test.
```

---

## 15. State update rules

Update `STATE.md` whenever any of the following changes:

- project status;
- baseline status;
- current best result;
- active hypothesis;
- latest valid experiment;
- strongest observations;
- next action;
- in-flight remote jobs;
- test-set state;
- resource usage;
- blockers.

Prefer replacing stale state over appending historical notes.

Before ending a meaningful research session, ensure this file accurately answers:

```text
Where are we?
What is currently true?
What is running?
What is the best valid result?
What information has the test set revealed?
What should happen next?
What could block it?
```

---

## 16. Completion checklist

Before relying on this file for autonomous resume, confirm:

- [ ] Project status is current.
- [ ] Current objective is concise.
- [ ] Baseline status is current.
- [ ] Current best points only to a valid result.
- [ ] Active hypothesis is identified.
- [ ] Latest experiment is recorded.
- [ ] Strongest observations reflect current evidence.
- [ ] One exact next action is defined.
- [ ] All in-flight remote jobs are tracked.
- [ ] Test-set state is current and never falsely reset.
- [ ] Resource usage is current.
- [ ] Blockers are explicit.
- [ ] Immediate queue contains no more than a few useful next actions.
- [ ] Stale history has been removed rather than accumulated.
