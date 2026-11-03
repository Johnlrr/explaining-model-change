# exp_0050 — D-052 X4: second architecture (ResNet-8), split D — NOT DONE (registered feasibility rule)

Registered: https://osf.io/vfjtk (2026-11-02). Config configs/experiments/exp_0050.yaml (sha256 c1ea2784…; canonical
JSON sha256 859f81c0… passed to and checked by the training kernel).

**Training (Kaggle T4, offline).** Kernel dinhbaobao/dce-exp0050-train (run dir runs/experiments/exp_0050_kaggle_train),
bundle `x4_train` tree 8a2ba61e… (commit 96b1eb5), torch 2.11.0+cu130, 572 s (≈ 0.16 GPU-hours). 12 checkpoints in
models/, all matching artifact_manifest.json. f_old validation accuracy 0.992 (seed 0) and 0.992 (seed 1).
Preceded by smoke_0030 (same path, 1 epoch; passed).

**Qualification (validation, local, `x4_resnet qualify`).** qualification.json: offsets 121, 122, 123 and rotations
124–127 qualify; offset 120 fails (t̂ 0.491, dt = −0.012); nulls 128, 129 dd 0.0035 and 0 (all admitted).

**Search smoke (Kaggle CPU).** smoke_0031 (dinhbaobao/dce-smoke0031-x4; pairs 120, 128, 3 points; rc 0, both pairs;
inputs dataset dinhbaobao/dce-x4-inputs, listing 192d8f3e…; bundle tree 782d0711…, commit 64116c9). Not interpreted.

**Timing → go/no-go (Kaggle CPU, same executor as the search would use).** Kernel dinhbaobao/dce-exp0050-time
(run dir runs/experiments/exp_0050_kaggle_time; rc 0; config, inputs and bundle hashes as above). `x4_resnet time`, 5
queries each of pairs 120 (offset) and 128 (null): Δ-CE ≈ 34 s wall / 132 CPU-s per query (offset), 36 + 40 s wall /
138 + 156 CPU-s (null, types N and O). Projection of the committed code (wall-hours): **30.9 h at 200 queries per pair,
15.5 h at 100** → `go: false`. Cross-check in CPU-hours, the unit of the registration (Δ-CE solver only, no CF-diff,
no overhead): 91.3 h at 200, 45.6 h at 100. Both readings exceed 12 h at 100 queries, so the outcome does not depend on
the unit.

**Decision (registered rule, applied as written):** X4 is left out of this paper and reported as not done. No
ResNet-8 search outputs exist; nothing about ResNet-8 witnesses is claimed.
