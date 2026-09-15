# Kaggle T4 smoke acceptance

## Result

Passed on 2026-09-15 using repository harness version `0.1.0` and Kaggle CLI
`2.2.4`.

The test used a private Kaggle Python script kernel with an `NvidiaTeslaT4`
request, detected a `Tesla T4`, kept Internet disabled, attached no external
inputs, and processed exactly five small CUDA matrix multiplications. The result
contained a finite metric and passed the configured JSON output checks.

## Validated executor behavior

The real end-to-end path completed successfully:

```text
self-contained package
-> submit private kernel
-> capture numeric kernel version
-> poll queued/running/complete status
-> fetch output and logs
-> validate required JSON, workload, CUDA, and finite values
-> persist final local metadata, metrics, logs, and source identity
```

The acceptance run also confirmed parsing of the Kaggle CLI 2.2 worker-status
format (`KernelWorkerStatus.RUNNING` and `KernelWorkerStatus.COMPLETE`). Source
identity included the available Git commit, dirty-worktree state, and a SHA256
matching the exact staged entrypoint.

Full runtime directories and downloaded artifacts are intentionally excluded
from the reusable template. Future research branches should run their own small
smoke check before any expensive experiment.
