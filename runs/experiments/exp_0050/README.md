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

## Update 2026-11-03 (D-053): search run locally as a disclosed deviation

`x4_resnet run-chunked --chunk 20`, config configs/experiments/exp_0050_run.yaml (n_points 100), commit c8e6110,
local CPU (AMD Ryzen 7 PRO 250), 50 chunks, finished 2026-11-03 15:48 (log run_chunked.log, metadata_chunked.json).
Analysis `x4_resnet analyze` → results_x4.json (registered readouts, descriptive, k of n):

- **Null witness share: 0.90** pooled (pair 128 0.91, 129 0.89; validation decisive disagreement 0.0035, 0). CNN: 0.
- Grad-diff Recovery < 0.2 on qualified α = 0 offsets: **0 of 3** (0.32–0.36). Higher at α = 0.5 than α = 0 within
  seed: **0 of 3** (rotations 0.06–0.10).
- Lowered-class detection at ResNet-null thresholds (τ_0): Δ-CE 1.00 in 3/3, s_M 0.02–0.16, CF-diff 0 → Δ-CE ≥ s_M +
  0.20 in 3 of 3. **Not at matched false positives**: Δ-CE itself flags 0.90 of null queries.

Checks (not the registered readouts): models in eval mode; witnesses re-evaluated point by point satisfy m_o ≥ 0.5,
m_n ≤ −0.5; batch vs single-point margins differ by ≤ 2.3·10⁻⁶. Margin change on nulls: max|h| over the r-ball
median 1.93 (CNN split-D nulls 0.51), share ≥ 2κ 0.99 (CNN 0.07); |h(z0)| median 0.41 (CNN 0.21). Witness rate vs
max|h| in the ball is the same on both architectures where both have data (≥ 1.5: CNN 1.00, ResNet 0.95); ResNet null
updates simply sit there. Grad-diff argmax on ResNet spreads over aspect/size/position for every pair type, nulls
included (aspect 0.20–0.30 on nulls), so 0.32–0.36 on offsets is at the null level.
