# ENVIRONMENTS.md

> Purpose: Define the execution environments available to this research project, what each environment provides, and how the agent should choose between them.
>
> Ownership: Project specification. Treat this file as read-only during autonomous research unless the user explicitly requests a change.
>
> This file describes **where research can run**. Resource limits and budgets belong in `docs/CONSTRAINTS.md`. Evaluation validity belongs in `docs/EVALUATION.md`.

---

## 1. Environment registry

<REQUIRED — List every execution environment available to this project.>

Use stable environment IDs.

Recommended structure:

```text
Environment ID:
Type:
Status: available | unavailable | unverified
Primary purpose:
```

Example:

```text
Environment ID: local
Type: local workstation
Status: available
Primary purpose: development, local validation, lightweight analysis

Environment ID: kaggle-gpu
Type: Kaggle kernel
Status: available
Primary purpose: smoke runs, full GPU experiments, final verification
```

Do not assume an environment is usable merely because it is documented. If status is `unverified`, verify it before relying on it.

---

## 2. Default environment selection

<REQUIRED — Define the default environment for each stage of research.>

Template:

```text
Development / coding:
Local validation:
Smoke runs:
Full experiments:
Final verification:
Literature / web research:
Artifact inspection:
```

Recommended default:

```text
Development / coding: local
Local validation: local
Smoke runs: Kaggle GPU when remote GPU is required
Full experiments: Kaggle GPU
Final verification: same class of environment required by evaluation protocol
Literature / web research: environment with Internet access
Artifact inspection: local after fetch when practical
```

Project-specific requirements may override this.

---

## 3. Local environment

<REQUIRED — Describe the local environment available to the agent.>

Template:

```text
Environment ID: local
Status: available | unavailable | unverified

Operating system:
Architecture:
Python version:
Package manager / virtual environment:
CPU:
GPU:
RAM:
Persistent storage:
Working directory:
Internet access:
Relevant tools:
Known limitations:
```

Do not include credentials, API keys, tokens, passwords, or other secrets.

The local environment is normally preferred for:

- code editing;
- unit tests;
- integration tests;
- configuration validation;
- lightweight preprocessing;
- result analysis;
- artifact inspection;
- tasks that do not require expensive remote compute.

---

## 4. Kaggle environment

<REQUIRED when Kaggle is used. Otherwise mark `Not used`.>

Template:

```text
Environment ID: kaggle-gpu
Status: available | unavailable | unverified

Execution model: batch kernel
Kaggle account context:
Python/runtime assumptions:
Internet capability:
Available accelerator classes:
Competition sources:
Dataset sources:
Kernel sources:
Model sources:
Persistent output behavior:
Known platform limitations:
```

Kaggle should be treated as a **remote batch executor**, not as an interactive SSH machine.

Remote state and artifacts should be handled through the repository Kaggle harness.

---

## 5. GPU preference policy

For general-purpose GPU research, prefer faster compatible RTX-class hardware over T4.

Default preference order:

```text
1. RTX-class GPU
2. T4
```

Use RTX-class hardware whenever:

- it is available;
- it is compatible with project requirements;
- it does not violate `docs/CONSTRAINTS.md`;
- it does not invalidate a hardware-controlled comparison.

Use T4 when:

- RTX hardware is unavailable;
- RTX quota is exhausted;
- the experiment specifically requires T4;
- a fair-comparison protocol requires T4;
- or there is another documented research/infrastructure reason.

Do not choose slower hardware merely for consistency unless evaluation fairness requires it.

---

## 6. Requested vs observed hardware

Every important remote run should distinguish:

```text
Requested accelerator:
Observed accelerator:
```

Do not record requested hardware as though it were observed runtime fact.

For important experiments, verify observed hardware from inside the execution environment when possible.

If requested and observed hardware differ:

- record the mismatch;
- determine whether hardware matters under `docs/EVALUATION.md`;
- invalidate or rerun the experiment if the mismatch affects evaluation validity;
- otherwise, the run may remain usable if the mismatch is documented and does not affect the scientific conclusion.

---

## 7. Internet policy

<REQUIRED — Define Internet availability and permitted usage for each environment.>

Template:

```text
Local:
Internet: on | off | conditional
Permitted use:

Kaggle preparation environment:
Internet: on | off | conditional
Permitted use:

Kaggle experiment environment:
Internet: on | off | conditional
Permitted use:
```

Possible permitted uses include:

- literature search;
- dependency acquisition;
- downloading public pretrained assets;
- accessing public repositories;
- fetching project-approved resources.

Internet availability does not override resource restrictions in `docs/CONSTRAINTS.md`.

---

## 8. Dependency strategy

Use the simplest valid dependency path.

### Internet-enabled execution

When the target environment may access the Internet:

```text
install/load dependencies normally
→ validate environment
→ smoke
→ full run
```

### Internet-disabled Kaggle execution

When the target Kaggle kernel must have Internet disabled and required dependencies/assets are not already available:

```text
preparation kernel
Internet ON
↓
download/build required dependencies/assets
↓
write them to preparation-kernel output
↓
consumer kernel
Internet OFF
↓
attach preparation output through kernel_sources
↓
smoke
↓
full run
```

Detailed research execution rules belong in `docs/RESEARCH_PROTOCOL.md`.

Do not perform unnecessary Kaggle → local → Kaggle transfers of large dependency bundles.

---

## 9. Input sources and mounts

<REQUIRED — List external inputs required by the project and how they are attached.>

Template:

```text
Input ID:
Type: competition_source | dataset_source | kernel_source | model_source | local
Source:
Required by:
Expected mount/path:
Availability:
Notes:
```

Examples:

```text
Input ID: competition-data
Type: competition_source
Source: <competition slug>
Required by: smoke, full, final
Expected mount/path: <path>
Availability: unverified
```

Do not hard-code a path unless it is stable and actually required.

If mount paths are discovered at runtime, record observed paths in run evidence.

---

## 10. Environment-specific assets

<OPTIONAL — Record assets that exist only in one environment.>

Examples:

- competition-provided wheels;
- cached pretrained models;
- preparation-kernel outputs;
- environment-specific checkpoints;
- local-only test fixtures.

For each asset specify:

```text
Asset:
Environment:
Source:
Purpose:
Expected lifetime:
```

Do not treat temporary remote assets as durable project memory unless their provenance is preserved.

---

## 11. Output and artifact handling

<REQUIRED — Define what each environment should produce and what should be persisted.>

Recommended policy:

### Local

Use local output for:

- test reports;
- temporary analysis;
- generated configs;
- fetched remote artifacts;
- research-state updates.

Temporary staging belongs under ignored paths such as:

```text
.work/
```

### Kaggle

Each important run should write machine-readable outputs when applicable, such as:

```text
result.json
metrics.json
run metadata
logs
required artifacts
```

After completion:

```text
remote run
→ fetch required outputs
→ validate outputs
→ store immutable evidence under runs/
→ update research state
```

Do not rely on remote notebook state alone as the only evidence of an important experiment.

---

## 12. Authoritative evidence

For a completed experiment, the authoritative record should be the persisted run evidence referenced by the repository, not conversational history.

When applicable, evidence should include:

- experiment ID;
- Git commit;
- configuration;
- requested hardware;
- observed hardware;
- input identities;
- metrics;
- logs;
- artifacts;
- timestamps;
- validity status.

See `docs/RESEARCH_PROTOCOL.md` for reproducibility requirements.

---

## 13. Authentication and secrets

Authentication may be required for remote environments.

Template:

```text
Environment:
Authentication required: yes | no
Readiness check:
Credential storage: outside repository
```

Rules:

- never store credentials in this file;
- never commit API tokens, passwords, secret keys, cookies, or credential files;
- credentials must be managed outside version control;
- readiness checks may verify authentication but should not print secrets.

Example:

```text
Environment: Kaggle
Authentication required: yes
Readiness check: Kaggle CLI authentication check
Credential storage: outside repository
```

---

## 14. Environment readiness checks

Before relying on an environment for an important experiment, verify the relevant items.

Recommended checks:

```text
Environment reachable:
Authentication valid:
Expected runtime available:
Required accelerator visible:
Observed hardware recorded:
Required inputs mounted:
Required dependencies import:
Internet state correct:
Output directory writable:
Expected artifacts can be produced:
```

A documented environment with failed readiness checks should be treated as unavailable until repaired.

---

## 15. Kaggle runtime verification

For important Kaggle runs, verify when applicable:

- actual GPU model;
- GPU count;
- CUDA availability;
- required Python/package versions;
- required mounted sources;
- expected competition/dataset/kernel/model inputs;
- Internet state if relevant;
- writable output directory;
- machine-readable result generation.

Smoke runs should catch environment mismatch before expensive full runs.

---

## 16. Environment selection rules

Choose the cheapest/simplest environment that can validly answer the experiment question.

Preferred order:

```text
local validation
→ remote smoke if needed
→ remote full experiment
```

Use local execution when:

- GPU is unnecessary;
- the test is cheap;
- only syntax/config/integration validation is needed;
- analysis can be performed from already-fetched artifacts.

Use Kaggle GPU when:

- required compute is unavailable locally;
- the benchmark must run in the Kaggle environment;
- GPU acceleration materially reduces runtime;
- competition-provided sources/assets are needed.

Do not use remote GPU for work that can be validated cheaply and reliably locally.

---

## 17. Fallback environment policy

<REQUIRED — Define whether and when the agent may switch environments.>

Recommended default:

```text
Automatic fallback: allowed only when scientific validity is preserved.
```

Fallback may be reasonable when:

- preferred RTX hardware is unavailable;
- a remote service is temporarily unavailable;
- the task is environment-independent;
- the evaluation protocol does not require hardware consistency.

Fallback must not occur silently when hardware/runtime affects the metric being compared.

If fallback changes scientifically relevant conditions, either:

- rerun under the required environment;
- or mark the run non-comparable / invalid according to `docs/EVALUATION.md`.

---

## 18. Known Kaggle platform limitations

The following should be treated as operational facts for the harness:

- Kaggle is a batch execution platform rather than an SSH environment.
- Remote execution may continue after a local CLI command times out.
- Local timeout does not imply remote cancellation.
- Concurrent versions of the same kernel slug should not be used.
- Distinct concurrent experiments should use distinct kernel slugs.
- Output retrieval is associated with the kernel slug used by the harness.
- Important artifacts should be fetched after completion and persisted locally.
- Temporary remote state should not be treated as durable project memory.

Project-specific limitations may be added here.

---

## 19. Environment-specific setup

<OPTIONAL — Add setup instructions that are stable and safe to keep in the repository.>

Examples:

```text
Local:
- activate project virtual environment
- install project in editable mode

Kaggle:
- attach required competition source
- attach preparation kernel output
- disable Internet for final benchmark
```

Do not put secrets here.

Avoid duplicating generic workflow already defined in `docs/RESEARCH_PROTOCOL.md`.

---

## 20. Environment change policy

If an important environment changes during research, record the change when it can affect reproducibility or interpretation.

Examples:

- GPU class changed;
- runtime image changed;
- Python/package version changed;
- competition-provided assets changed;
- mount source changed;
- Internet policy changed.

Material changes should be recorded in `research/DECISIONS.md` and run metadata where relevant.

Do not silently compare results across materially different environments when the difference may affect the conclusion.

---

## 21. Completion checklist

Before autonomous research begins, confirm:

- [ ] All available execution environments are listed.
- [ ] Every environment has a status: `available`, `unavailable`, or `unverified`.
- [ ] Default environment selection is defined.
- [ ] Local environment capabilities are documented.
- [ ] Kaggle environment is documented if used.
- [ ] RTX-class GPU is preferred over T4 unless evaluation constraints require otherwise.
- [ ] Requested and observed hardware are tracked separately.
- [ ] Internet policy is explicit.
- [ ] Dependency strategy is defined.
- [ ] Required input sources are listed.
- [ ] Output/artifact handling is defined.
- [ ] Authentication requirements are documented without secrets.
- [ ] Readiness checks are defined.
- [ ] Fallback rules are explicit.
- [ ] Known platform limitations are documented.
- [ ] Environment changes that affect reproducibility will be recorded.
