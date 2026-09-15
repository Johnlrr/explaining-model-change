# ARC-AGI-3 RTX PRO 6000 smoke acceptance

## Result

Passed on 2026-09-15 using repository harness version `0.1.0` and Kaggle CLI
`2.2.4`.

- Competition: `arc-prize-2026-arc-agi-3`.
- Kernel: `minhtran0105/harness-arc3-rtxpro6000-20260915-234430`, version `1`.
- Kernel visibility: private.
- Internet: disabled.
- Requested accelerator: `NvidiaRtxPro6000`.
- Actual GPU count: `1`.
- Actual GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`.
- Raw terminal status: `KernelWorkerStatus.COMPLETE`.
- Competition mount: `/kaggle/input/competitions/arc-prize-2026-arc-agi-3`.

The bounded mount sample contained:

- `environment_files`;
- `ARC-AGI-3-Agents`;
- `arc_agi_3_wheels`.

## Validation

The private kernel attached exactly the intended `competition_sources`, used no
dataset, model, or kernel inputs, and required no external download. CUDA was
available and the requested RTX PRO 6000 corresponded to the GPU reported by
the runtime.

The runner performed exactly five small CUDA matrix multiplications, verified
finite tensors, and wrote a valid `result.json` with:

- `cuda_available == true`;
- `gpu_count == 1`;
- a recorded GPU name;
- `competition_input_found == true`;
- `work.processed == 5`;
- finite `metric == 2.49755859375`.

The harness captured a numeric kernel version, reached normalized `SUCCESS`,
fetched output and logs, validated the configured JSON checks, and persisted
final timestamps, Git state, and a matching submitted-source SHA256.

## Retry and fix

One retry was required. The first temporary configuration used a title whose
Kaggle-generated slug did not equal its configured kernel ID, so submission
succeeded under a different slug and the harness could not poll the configured
ID. The first remote job reached `COMPLETE` before retrying. Only the ignored
temporary title was changed so its slugification matched the configured ID;
there was no harness code change and no accelerator fallback.

## Decision

**GO** — ARC-AGI-3 competition input, `NvidiaRtxPro6000`, Internet-disabled
execution, output/log fetching, validation, and run persistence work end to end
through the repository harness.
