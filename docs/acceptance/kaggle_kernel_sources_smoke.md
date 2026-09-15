# Kaggle kernel_sources smoke acceptance

## Result

Passed on 2026-09-15 using repository harness version `0.1.0` and Kaggle CLI
`2.2.4`.

- Preparation kernel: `minhtran0105/harness-kernel-source-prep-20260915-233516`, version `1`.
- Preparation Internet setting: enabled.
- Offline consumer kernel: `minhtran0105/harness-kernel-source-offline-20260915-233516`, version `1`.
- Consumer Internet setting: disabled.
- Consumer `kernel_sources`: `minhtran0105/harness-kernel-source-prep-20260915-233516`.
- Observed mount path: `/kaggle/input/notebooks/minhtran0105/harness-kernel-source-prep-20260915-233516/dependency_payload.json`.

## Validated behavior

The preparation kernel wrote `dependency_payload.json` with message
`kernel-source-ok` and value `42`. The harness captured a numeric version,
observed `KernelWorkerStatus.COMPLETE`, fetched the artifact and log, validated
the JSON contract, and persisted final metadata.

The Internet-disabled consumer discovered that artifact recursively under
`/kaggle/input`, verified its contents, performed exactly five processing
iterations, and wrote `result.json`. The result reported:

- `kernel_source_loaded == true`;
- `message == "kernel-source-ok"`;
- `source_value == 42`;
- `work.processed == 5`;
- finite `metric == 220.0`.

The consumer also reached `KernelWorkerStatus.COMPLETE`; its output, log,
metrics, final timestamps, submitted version, and source identity were fetched
or persisted successfully.

No harness or runner fix was required, and neither kernel required a retry.

## Decision

**GO** — an Internet-enabled preparation kernel can provide a small dependency
artifact directly to an Internet-disabled downstream kernel through
`kernel_sources` without local Dataset promotion.
