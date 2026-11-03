# exp_0049 — D-052 X3 (nulls without admission filter) + X2 (more offset pairs), split D

Registered: https://osf.io/vfjtk (2026-11-02, before split-D data were generated). Code f7487b3 (`split_d run`;
`split_d.py` unchanged up to the analysis commit 96b1eb5), config configs/experiments/exp_0049.yaml (sha256 b77fbc45…).
Models and qualification (validation): models/, artifact_manifest.json, qualification.json (8/9 offset pairs qualify,
pair 103 fails dt ≥ 0.02; nulls all admitted). Search on Kaggle only (D-052 one-machine rule): kernels
boinhbo/dce-exp0049-k{1,2,3} (run_pairs shards 0/3, 1/3, 2/3; D-044 kaggle-cpu, Intel Xeon 2.2 GHz, 4 cores, AVX2,
torch 2.14.1+cpu; bundle tree df321b1f…; inputs listing 97bbb457…; rc 0; 17/17 pairs; files match the kernel
manifests). Kernel records: runs/experiments/exp_0049_kaggle_k{1,2,3}. Analysis: `split_d analyze` → results_d.json.

**X3 — 8 nulls, all admitted (2 400 queries).** Decisive disagreement on validation 0 in 8/8. Δ-CE witness (N or O)
share: **0.000 pooled and in every pair** → prediction (≤ 0.05) **met**. CF-diff attributes a change to 0.983–0.997 of
null queries. Thresholds from these nulls: s_M τ_0 = 1.144, τ_5% = 0.661; s_M,r 1.285 / 1.060; CF-diff ‖Δφ‖ 0.503 / 0.155.

**X2 (a) — confirmatory.** Grad-diff Recovery < 0.2 in **8/8** qualified pairs (0.033–0.093); exact one-sided sign test
p = 0.0039 < 0.05 → **confirmed**. (Pair 103, not qualified: 0.033.)

**X2 (b) — detection on the lowered class (class 1) at X3 thresholds, qualified pairs (descriptive).**

| | Δ-CE type N | s_M τ_0 | s_M τ_5% | s_M,r τ_0 | CF-diff τ_0 | CF-diff τ_5% |
|---|---|---|---|---|---|---|
| range over 8 pairs | 0.17–1.00 | 0.00–0.54 | 0.08–0.94 | 0.00–0.99 | 0.00 (all) | 0.03–0.18 |
| mean | 0.77 | 0.17 | 0.42 | 0.49 | 0.00 | 0.08 |

Per pair (Δ-CE / s_M τ_0): 100 0.17/0.00, 101 0.20/0.00, 102 0.93/0.01, 104 0.84/0.00, 105 1.00/0.03, 106 1.00/0.43,
107 0.99/0.31, 108 1.00/0.54. Branch rule (registered): s_M ≥ Δ-CE − 0.05 in 0/8 pairs (not "whether is cheap");
Δ-CE ≥ s_M + 0.20 in 7/8 pairs, **not every** pair (pair 100: +0.17) → **branch (iii): numbers reported, the first,
conservative wording** (no detection-sensitivity claim for witnesses). This differs from M (exp_0046, branch (ii), on
the 7 earlier pairs); X2 is the registered readout and governs the wording.

**Machine-sensitivity check (declared in D-052 before any number; descriptive).** `dce.cli.machine_check`: the first 10
queries of pairs 110, 115, 100, 105 re-run locally (AMD Ryzen 7 PRO 250, ATen AVX512, same pinned versions) with `split_d.d_pair`
unchanged; phase-1 restart noise drawn for the full 300-point batch and cut to its first 10 rows, so the noise equals
the Kaggle run's (a 10-point batch would draw different noise for every noisy restart). Point ids, y, z0 identical.
Δ-CE found N and O, CF-diff found and Grad-diff found agree on **40/40** queries; s_M max |diff| < 10⁻⁶. CF-diff
‖Δφ‖ agrees (|diff| < 0.005) on **28/40** queries; the other 12 differ by up to 0.10 (pair means 0.058 vs 0.060,
0.060 vs 0.042, 0.078 vs 0.078, 0.067 vs 0.070). Output: machine_check/machine_check.json, run.log.
