# ENVIRONMENTS.md

> Purpose: Define the execution environments available to this research project, what each environment provides, and how the agent should choose between them.
>
> Ownership: Project specification. Treat this file as read-only during autonomous research unless the user explicitly requests a change.
>
> This file describes **where research can run**. Resource limits and budgets belong in `docs/CONSTRAINTS.md`. Evaluation validity belongs in `docs/EVALUATION.md`.

---

## 1. Environment registry

```text
Environment ID: local
Type: local workstation (WSL2, Linux 6.18 kernel)
Status: available
Primary purpose: development, Tier L and Tier 0 experiments (CPU), unit tests, analysis, artifact inspection

Environment ID: kaggle-gpu
Type: Kaggle kernel (batch), through the repository harness (tools/kaggle)
Status: verified 2026-10-15 (smoke_0002: observed 2 × Tesla T4, offline, kernel_sources hash verified)
Primary purpose: Tier 1 smoke and full runs (GPU).

Environment ID: kaggle-prep
Type: Kaggle kernel with Internet ON
Status: verified 2026-10-15 (smoke_0001: MNIST 4/9 downloaded, 11 791 train / 1 991 test)
Primary purpose: download MNIST 4/9 and write it to kernel output for the offline Tier 1 kernels (kernel_sources).
```

Both Kaggle environments were verified by smoke runs on account `boinhbo`. Runtime seen: Python 3.13, torch 2.11 (cu128 on GPU), torchvision 0.26, numpy 2.1, scipy 1.16, scikit-image 0.25; no extra dependency bundle is needed.

---

## 2. Default environment selection

```text
Development / coding: local
Local validation: local (pytest, ruff; PyTorch CPU only where a local test needs it)
Smoke runs: local for Tier L and Tier 0; kaggle-gpu for Tier 1
Full experiments: local CPU for Tier L and Tier 0; kaggle-gpu for Tier 1 (≤ 4 jobs)
Final verification: local CPU for Tier 0 reproduction; kaggle-gpu for Tier 1 reproduction of one drift and one null pair
Literature / web research: local with Internet access
Artifact inspection: local after fetch (checkpoints are small)
```

---

## 3. Local environment

```text
Environment ID: local
Status: available

Operating system: Ubuntu-like Linux under WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2)
Architecture: x86_64
Python version: 3.12.3 (system); project uses uv (uv 0.12.7)
Package manager / virtual environment: uv, project virtual environment (.venv, git-ignored); uv.lock present
CPU: 16 logical cores
GPU: none detected (nvidia-smi not available)
RAM: 15 GB
Persistent storage: ~935 GB free on the root filesystem
Working directory: /home/toska/code/research/Agent-auto-Research
Internet access: on
Relevant tools: git, gh, uv, pytest, ruff, the Kaggle CLI (via tools/kaggle)
Known limitations: no GPU, so Tier 1 cannot run locally. PyTorch and torchvision are not installed locally at kick-off;
                   they are installed (CPU wheels) when the Tier 0 renderer and classifier are implemented at G0,
                   because Tier 0 is specified as a local CPU run.
```

Do not include credentials, API keys, tokens, passwords, or other secrets.

---

## 4. Kaggle environment

```text
Environment ID: kaggle-gpu
Status: unverified

Execution model: batch kernel (script), submitted and waited on by the repository harness only
Kaggle account context: one or more Kaggle accounts connected through `python -m tools.kaggle accounts add` (token files
                        kept outside Git). `--account auto` selects an account with GPU quota; the chosen account is
                        recorded in run metadata. Account names are not recorded in this file.
Python/runtime assumptions: Python 3.10+; PyTorch and torchvision available in the Kaggle image; scikit-image and scipy
                        installed from the staged dependency bundle if missing (see §8).
Internet capability: kaggle-prep ON; kaggle-gpu OFF (offline experiments).
Available accelerator classes: T4 (observed: two Tesla T4 devices). RTX PRO 6000 is not used: the repository's acceptance-tested RTX context needs a competition source this project cannot attach.
Competition sources: none.
Dataset sources: none. MNIST is obtained by kaggle-prep and passed through kernel_sources.
Kernel sources: kaggle-prep output (MNIST 4/9 raw files, skimage/scipy wheels if needed).
Model sources: none (no pretrained models).
Persistent output behavior: kernel output is kept remotely; checkpoints (< 5 MB) and result.json / artifact_manifest.json are fetched.
Known platform limitations: session time limit (harness sets 3600 s for full runs); GPU quota varies per account;
                        offline kernels cannot download data.
```

Kaggle is a **remote batch executor**, not an interactive machine. Remote state and artifacts are handled through the repository harness.

---

## 5. GPU preference policy

Default order: RTX-class first, then T4 (`docs/ENVIRONMENTS.md` §5 generic policy).

This project uses the T4 (D-010): the RTX example config requires a competition source this project cannot attach. The Tier 1 model is small (≈ 60k parameters) and the GPU is a convenience, not a scientific variable. If the hardware changes between pairs of one comparison, record it in `research/DECISIONS.md` and keep the same device class within each comparison (runtime is compared only within a device class).

---

## 6. Requested vs observed hardware

Every Kaggle run records:

```text
Requested accelerator: from the kernel config (machine_shape)
Observed accelerator: from inside the runtime (nvidia-smi output or torch.cuda.get_device_name())
```

Mismatch between requested and observed hardware is recorded. It invalidates a run only if it affects runtime comparisons (EVALUATION §14).

---

## 7. Internet policy

```text
Local:
Internet: on
Permitted use: literature search, dependency acquisition, public repositories for reading

Kaggle preparation environment (kaggle-prep):
Internet: on
Permitted use: download MNIST 4/9 (torchvision) and write it to kernel output. Nothing else.

Kaggle experiment environment (kaggle-gpu):
Internet: off
Permitted use: none beyond the staged inputs (kernel_sources). Experiment code must not call the network.
```

Internet availability does not override the resource restrictions in `docs/CONSTRAINTS.md` §8.

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

Inputs for this project:

```text
Input ID: mnist-4-9
Type: dataset_source (via kaggle-prep kernel output, not a Kaggle dataset)
Source: torchvision MNIST download, performed by kaggle-prep with Internet ON; digits 4 and 9 filtered
Required by: Tier 1 data build (job a in CONSTRAINTS §4)
Expected mount/path: kernel_sources output of kaggle-prep, read-only
Availability: after the prep kernel completes
Notes: no Kaggle Dataset upload loop (AGENTS.md)

Input ID: offline-wheels (only if needed)
Type: kernel_source
Source: kaggle-prep output with scikit-image/scipy wheels, only if the Kaggle image lacks them
Required by: all Tier 1 kernels
Availability: verified in the smoke run
```

Tier 0 and Tier L need no external inputs (synthetic, generated locally with seed 1234).

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

Per environment:

- `local`: generated configs, unit-test reports, analysis tables, figures, the paper draft. Tier 0 and Tier L run outputs are persisted under `runs/experiments/exp_NNNN/` (metadata, config, metrics, logs, artifacts).
- `kaggle-prep`: MNIST raw files for the offline kernels. Not persisted in Git.
- `kaggle-gpu`: `result.json`, `artifact_manifest.json` (with SHA256 of every checkpoint), stdout/stderr logs, and the small checkpoints (VAE, shape oracle, f_old, f_new). Fetched into `runs/experiments/exp_NNNN/` by the harness. Only evidence and checkpoints are fetched; the raw dataset is not.

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

This project's rules:

- Switching Tier 1 from RTX to T4 is allowed only if the switch is recorded in `research/DECISIONS.md`, and the same device class is used within each paired comparison (B1 vs Δ-CE, B3 vs Δ-CE).
- Switching Tier 1 to local CPU is not allowed for the acceptance runs (no GPU locally, and the spec budget assumes GPU).
- A switch made for an infrastructure failure must not change any scientific variable (EVALUATION §17).

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
