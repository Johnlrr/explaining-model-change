# RESEARCH_PROTOCOL.md

## Purpose

This document defines the default research procedure for all research branches created from this repository.

It is intentionally project-agnostic.

Project-specific objectives, metrics, datasets, budgets, and environments belong in:

- `docs/GOAL.md`
- `docs/EVALUATION.md`
- `docs/CONSTRAINTS.md`
- `docs/ENVIRONMENTS.md`

This protocol defines **how research should be conducted**, not what any specific project should optimize.

---

## 1. Default research lifecycle

Every research project should follow this loop:

```text
Understand the project
→ Reproduce the baseline
→ Observe
→ Research prior work when needed
→ Form hypothesis
→ Check relation to prior work
→ Make measurable prediction
→ Design minimal experiment
→ Implement
→ Local validation
→ Smoke run
→ Commit experiment state
→ Full experiment
→ Fetch and validate results
→ Analyze
→ Record evidence
→ Commit result
→ Update research state
→ Choose next hypothesis
→ Repeat
```

The agent should continue until:

- success criteria are met and final verification passes;
- a project stop condition is reached;
- resource/time budget is exhausted;
- or human intervention is required.

---

## 2. Baseline reproduction

### Default rule

A research branch should reproduce the defined baseline before using it as the primary comparison for new contributions.

The purpose is consistency:

- same environment;
- same implementation assumptions;
- same dataset/split;
- same evaluation code;
- same hardware constraints where relevant;
- same metric definitions.

Do not assume a published or externally reported number is directly comparable to the local research environment.

### Baseline reproduction may be skipped only when

- the project documentation explicitly says not to reproduce it;
- reproduction is impossible due to unavailable data, model, code, access, or compute;
- the baseline has already been reproduced and validated in the same project environment.

If baseline reproduction cannot be completed, record:

- why;
- what external number is being used instead;
- why the comparison is still meaningful;
- what uncertainty this introduces.

Store this decision in `research/DECISIONS.md`.

Baseline reproduction is allowed even when it is not itself a novel contribution.

---

## 3. Observation before hypothesis

A new hypothesis should normally be motivated by evidence.

Useful sources include:

- previous experiment results;
- failure patterns;
- model behavior;
- metric tradeoffs;
- implementation observations;
- baseline weaknesses;
- literature findings;
- theoretical reasoning.

Avoid unexplained changes such as:

```text
try another learning rate
try a larger model
try another loss
```

unless they are part of an explicitly defined search procedure.

Prefer:

```text
Observation:
validation variance rises sharply after increasing model capacity.

Hypothesis:
the current optimization settings are too aggressive for the larger model.

Prediction:
reducing the learning rate should preserve the capacity gain while reducing variance.
```

---

## 4. Hypothesis requirements

Each hypothesis should be recorded in `research/HYPOTHESES.md`.

A hypothesis should contain:

- unique ID;
- status;
- motivating observation;
- hypothesis statement;
- measurable prediction;
- intended experiment;
- controlled variables;
- result;
- conclusion;
- follow-up.

Recommended status values:

```text
proposed
testing
supported
rejected
inconclusive
```

A hypothesis should be narrow enough that its experiment provides interpretable evidence.

Prefer one coherent change at a time.

If several changes are inseparable as one method, explain why they must be tested together.

---

## 5. Literature and prior-work research

Literature/web research is not required before every small experiment.

It is required when:

- opening a substantially new research direction;
- selecting or changing an important baseline;
- adopting a known external technique;
- evaluating whether an idea may already exist;
- preparing to describe a result as a contribution;
- assessing novelty.

Record materially relevant findings in:

```text
research/LITERATURE.md
```

Distinguish:

- **baseline reproduction** — reproducing known work for comparison;
- **replication** — checking known findings;
- **adaptation/extension** — modifying known work;
- **candidate contribution** — appears meaningfully different after prior-work review.

Suggested novelty status:

```text
candidate
supported
uncertain
not_novel
```

Do not claim novelty only because an idea was independently generated.

Do not copy an existing research contribution and present reproduction as a new contribution.

---

## 6. Experiment design

Each experiment should answer a specific question.

Before implementation, identify:

```text
Hypothesis
Prediction
Primary variable changed
Controlled variables
Expected evidence
Failure interpretation
```

Prefer the smallest experiment capable of rejecting or supporting the hypothesis.

Do not expand an experiment merely because more compute is available.

---

## 7. Local validation

Before consuming remote GPU resources:

- validate configuration;
- run relevant unit/integration tests;
- check imports;
- validate paths and expected inputs;
- validate output schema where possible.

Local validation does not replace remote smoke testing.

---

## 8. Smoke testing

Every expensive remote experiment must pass a smoke run first.

Smoke and full runs should use the same underlying execution path.

Only the workload scale should change.

Examples:

### Evaluation

```text
smoke:
~5 samples

full:
complete evaluation set
```

### Training

```text
smoke:
1 epoch or a small number of batches

full:
configured training schedule
```

The smoke run should verify, when applicable:

- environment starts correctly;
- required inputs are mounted;
- dependencies load;
- model loads;
- GPU is available when required;
- preprocessing works;
- forward/backward pass works;
- metrics can be computed;
- required outputs are written;
- no NaN/Inf appears.

A smoke failure must be debugged before the full experiment starts.

---

### Authoritative source and generated staging

```text
scientific/research implementation  -> src/
experiment/executor configuration   -> configs/
generated remote staging             -> .work/
experiment evidence                  -> runs/
```

Smoke and full runs must derive from authoritative code under `src/`.
Generated Kaggle wrappers under `.work/` are disposable; a fresh clone must not
need `.work/` to reconstruct research implementation. Important full runs should
record the source path and commit that produced them.

---

## 9. Offline Kaggle dependency workflow

When a Kaggle job may use Internet:

```text
run normally
```

When the target Kaggle job must run with Internet disabled and requires unavailable dependencies/assets:

```text
preparation kernel
Internet ON
↓
download/build required dependencies/assets
↓
write them to Kaggle kernel output
↓
offline consumer kernel
Internet OFF
↓
attach preparation output through kernel_sources
↓
smoke
↓
full run after smoke passes
```

Use `kernel_sources` as the automated staging mechanism for temporary offline dependency bundles.

Do not transfer large dependency bundles Kaggle → local → Kaggle unless explicitly required.

---

## 10. Parallel experiment policy

At most **two Kaggle GPU jobs should run concurrently** by default.

Parallel experiments are allowed only when they are logically independent.

Good example:

```text
baseline-worker
hypothesis-h001
```

Bad example:

```text
H002 requires observations from H001,
but both are launched before H001 finishes.
```

### Kernel slug rule

Never run two concurrent versions of the same Kaggle kernel slug.

Reason:

- status/output operations are resolved by kernel slug;
- fetching historical output by exact running version is not part of the normal harness workflow;
- concurrent versions of the same slug can cause ambiguity about which run is being monitored or fetched.

Therefore:

```text
Allowed:
kernel-baseline
kernel-h001

Not allowed concurrently:
experiment-worker version 5
experiment-worker version 6
```

Use distinct kernel slugs for concurrent jobs.

The repository harness may enforce stronger local locking rules.

---

## 11. Repository conventions

Use consistent identifiers and paths so that another agent can understand the project without reconstructing naming conventions.

### Hypothesis IDs

Use:

```text
H-001
H-002
H-003
```

Keep the ID stable for the lifetime of the hypothesis.

### Experiment IDs

Use:

```text
exp_0001
exp_0002
exp_0003
```

Each full experiment should have a unique experiment ID.

### Smoke IDs

Use:

```text
smoke_0001
smoke_0002
```

A smoke run is not the scientific result itself. It is execution validation.

### Config layout

Recommended:

```text
configs/
├─ experiments/
│  ├─ exp_0001.yaml
│  └─ exp_0002.yaml
└─ executors/
```

Project-specific layout may differ if documented clearly.

### Run evidence layout

Recommended:

```text
runs/
├─ experiments/
│  └─ exp_0001/
│     ├─ metadata.json
│     ├─ config.yaml
│     ├─ metrics.json
│     ├─ stdout.log
│     ├─ stderr.log
│     └─ artifacts/
└─ smoke/
   └─ smoke_0001/
```

Completed run evidence should not be silently overwritten.

### Kernel naming

For remote Kaggle kernels, prefer names that expose purpose and avoid collisions, for example:

```text
<project>-baseline
<project>-h001
<project>-prep-runtime
```

Concurrent jobs must use different kernel slugs.

### Commit messages

Recommended patterns:

```text
exp(H-003): test lower learning rate
result(H-003): reject optimization-instability hypothesis
research: update baseline decision
infra: fix Kaggle status parsing
```

Do not bundle unrelated changes into one experiment commit.

### Temporary files

Temporary staging/build/download content belongs under ignored workspace paths such as:

```text
.work/
```

Do not commit credentials, generated caches, local virtual environments, or transient remote artifacts.

---

## 12. Commit before full experiment

After:

- implementation;
- local checks;
- successful smoke;

commit the exact experiment code/config before the full benchmark.

Recommended lifecycle:

```text
form hypothesis
→ implement
→ local validation
→ smoke pass
→ commit experiment
→ full run
→ analyze
→ update research records
→ commit result
```

This ensures full-run metadata can reference a reproducible code state.

Do not benchmark an important full experiment from an unidentified working-tree state unless unavoidable.

If the worktree is dirty, record that fact explicitly.

---

## 13. Reproducibility requirements

For every important full experiment, preserve enough information to reproduce it later.

Record when available:

- experiment ID;
- hypothesis ID;
- exact Git commit;
- dirty/clean worktree state;
- source hash if supported by the harness;
- experiment config;
- random seed(s);
- dataset/split identity;
- input source identities;
- executor/backend;
- requested hardware;
- observed hardware;
- environment/dependency information;
- start/end timestamps;
- metrics;
- logs;
- relevant artifacts.

Do not record requested configuration as though it were observed runtime fact.

If an important result cannot be reproduced later from the stored evidence, treat that as a research-process defect.

---

## 14. Full experiment

Only start the full experiment after the smoke path is valid.

Use:

- the committed experiment implementation;
- the evaluation conditions defined in `docs/EVALUATION.md`;
- resource limits defined in `docs/CONSTRAINTS.md`;
- an appropriate hard remote timeout.

Do not remove hard timeouts entirely.

A local CLI timeout does not mean the remote job was cancelled.

If a local command times out, query remote state before deciding what happened.

Remote waiting is execution orchestration, not a scientific experiment step. Use
the following lifecycle for remote work:

```text
submit remote job
→ persist in-flight state
→ perform useful independent work if available
→ otherwise block once on the executor wait command
→ terminal state
→ fetch
→ validate
→ analyze
→ clear or update in-flight state
```

The executor should poll internally with sleep intervals; the language model should
not issue repeated status calls. In-flight state exists for crash or session
recovery and is not, by itself, a reason to stop and transfer monitoring to the
human. A harness wait timeout does not imply remote cancellation; preserve the last
known remote state and reconcile it when execution can resume.

---

## 15. Failure classification and retry

Do not blindly rerun failures.

For each failure:

```text
inspect logs/evidence
→ classify failure
→ make minimal repair
→ run local validation
→ smoke again
```

Useful failure classes include:

- implementation bug;
- configuration error;
- dependency/input error;
- infrastructure error;
- timeout/resource exhaustion;
- numerical instability;
- invalid evaluation output;
- external Kaggle/transient error.

Default retry limit:

```text
5 repair/retry attempts per failure class
```

unless overridden in `docs/CONSTRAINTS.md`.

Escalate when:

- the retry limit is exhausted;
- access/credentials are missing;
- a competition requires manual acceptance;
- the specification is ambiguous;
- fixing the problem requires changing read-only project specs;
- resource limits would be exceeded.

---

## 16. Result validation

A completed process is not automatically a valid experiment.

Validate against `docs/EVALUATION.md`.

When applicable, confirm:

- correct dataset/split;
- expected workload completed;
- required seeds/repetitions completed;
- required artifacts exist;
- metrics parse correctly;
- metrics are finite;
- intended configuration was actually used;
- run corresponds to the intended Git commit;
- no benchmark rule changed;
- no accidental data leakage occurred;
- no cherry-picking occurred.

Invalid experiments should not be treated as evidence supporting or rejecting the scientific hypothesis.

Record them as execution failures instead.

---

## 17. Negative results

Negative scientific results are valuable.

If a valid experiment rejects a hypothesis:

- record it in `research/HYPOTHESES.md`;
- record the important observation;
- update `research/STATE.md`;
- commit the result when it materially informs future work.

Example:

```text
Hypothesis H-004: rejected

Observation:
the proposed normalization reduced accuracy consistently across all required seeds.

Implication:
the instability is unlikely to be caused by the hypothesized activation-scale issue.
```

Do not hide valid negative results.

### Bug runs

A run invalidated by a bug, broken environment, wrong configuration, or malformed evaluation is **not** a scientific negative result.

Keep enough evidence to diagnose it, but do not present it as evidence about the research hypothesis.

---

## 18. Analysis after each full experiment

After a valid full experiment, answer:

1. What happened?
2. Did the measurable prediction occur?
3. Was the hypothesis supported, rejected, or inconclusive?
4. What is the strongest observation?
5. What alternative explanation remains plausible?
6. How does this compare with baseline/current best?
7. What is the highest-value next experiment?

Do not automatically generate another experiment without analyzing the current evidence first.

---

## 19. Research-state persistence

After meaningful progress update:

### `research/STATE.md`

Keep current:

- objective;
- baseline;
- current best;
- latest experiment;
- strongest observations;
- active hypothesis;
- next action;
- remaining resource budget;
- blockers.

### `research/HYPOTHESES.md`

Update the tested hypothesis and any newly proposed hypotheses.

### `research/DECISIONS.md`

Record decisions with long-term consequences.

### `research/LITERATURE.md`

Record prior work that materially affects direction, novelty, or interpretation.

### `research/NOTES.md`

Use for temporary ideas and observations that are not yet established conclusions.

---

## 20. Success verification

Do not stop immediately after one run crosses the target.

When success criteria first appear satisfied:

```text
identify winning commit/config
→ reproduce from clean state
→ run required seeds/repetitions
→ re-evaluate baseline comparison
→ check benchmark integrity
→ check leakage/cherry-picking
→ verify constraints
→ confirm success
```

Follow any stricter requirements in `docs/EVALUATION.md`.

A result is considered final only after this verification passes.

---

## 21. Stop conditions

Stop the autonomous research loop when:

- success is verified;
- project timeout is reached;
- GPU/experiment budget is exhausted;
- explicit stop conditions are met;
- no valid next experiment can be justified;
- human intervention is required.

Before stopping, persist:

- current best;
- latest evidence;
- active blockers;
- what has already been tried;
- recommended next action.

---

## 22. Research quality principles

Always prefer:

```text
reproducibility > convenience
evidence > intuition
controlled experiment > broad uncontrolled change
validated result > attractive metric
useful negative result > hidden failure
clear uncertainty > unsupported claim
```

Do not:

- change evaluation rules to improve results;
- cherry-pick seeds/samples/checkpoints;
- claim novelty without prior-work review;
- treat implementation bugs as hypothesis rejection;
- spend expensive GPU time before smoke validation;
- rely on chat history as durable project memory.
