# Kaggle remote-artifact handoff smoke acceptance

Date: 2026-09-17

Result: **PASS**

This CPU-only, Internet-disabled acceptance validated that a lightweight local
harness record can refer to a remote Kaggle checkpoint without downloading that
checkpoint locally.

## Producer

- Kernel: `tranmeennn/artifact-handoff-smoke-producer-20260917a`
- Kaggle version: `1`; final remote status: `KernelWorkerStatus.COMPLETE`
- Internet: off; GPU: disabled
- Remote outputs: `result.json`, `artifact_manifest.json`, and
  `checkpoints/test_checkpoint.pt`.
- Selectively downloaded locally: `result.json`, `artifact_manifest.json`, and
  the Kaggle log only. No `.pt` checkpoint was downloaded.
- Manifest SHA256:
  `cfbf2545f9452ae701be632abb656e801db254fb402f975f4d2ff40c35fa4968`
- Manifest checkpoint SHA256:
  `c1ba6d8df9a0bdb582a15d2896a191fde54986205a5a6fa03223e6bb3889c860`

`immutable_kernel: true` was tested after completion. The harness rejected the
same slug before Kaggle push with exit code `2`; the producer version stayed
`1 → 1`.

## Consumer

- Kernel: `tranmeennn/artifact-handoff-smoke-consumer-20260917a`
- Kaggle version: `1`; final remote status: `KernelWorkerStatus.COMPLETE`
- Internet: off; GPU: disabled
- `kernel_sources`: the exact producer kernel above
- Pre-submit version check: expected producer version `1` equalled current
  Kaggle producer version `1`.
- Observed mounted checkpoint path:
  `/kaggle/input/notebooks/tranmeennn/artifact-handoff-smoke-producer-20260917a/checkpoints/test_checkpoint.pt`
- Consumer validation: mounted manifest SHA matched; declared checkpoint SHA
  matched; checkpoint loaded; deterministic CPU inference succeeded (`metric:
  3.25`).
- Selectively downloaded locally: `consumer_result.json` and the Kaggle log
  only. No `.pt` checkpoint was downloaded.

The consumer implementation had no training loop, optimizer, backward pass, or
replacement checkpoint creation. A recursive check of local acceptance run and
staging directories found no checkpoint.

## Infrastructure fixes verified

Kaggle CLI 2.2.4 returned a semantic HTTP `403` for a new private slug in the
authenticated owner's namespace. The executor recognizes that narrow case as
absent while retaining distinguishable authorization failures. It also reads an
existing kernel's version from `GetKernelResponse.metadata.current_version_number`,
not the source blob. Regression tests cover both behaviors.
