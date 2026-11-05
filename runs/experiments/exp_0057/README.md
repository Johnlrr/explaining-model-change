# exp_0057 — D-056 X5: Δ-CE in a learned VAE latent, split F (registered https://osf.io/fr7jv)

Code ef66f09 (src unchanged since; bundle v110_tc_kaggle tree 32a5de06…), config configs/experiments/exp_0057.yaml
(sha256 83c17095…, r_vae 1.6408, n_points 100), VAE exp_0056 (vae32_beta1.0, gate passed). Search on 4 kaggle-cpu
kernels boinhbo/dce-exp0057-k{1..4} (inputs dataset boinhbo/dce-x5-inputs, listing a682804e…; rc 0; 2.6–3.9 h each;
k1–k3 Intel Xeon 2.2 GHz, k4 AMD EPYC 7B12). Kernel records: runs/experiments/exp_0057_kaggle_k{1..4}; every output
matched its kernel manifest; pairs/ holds the 16 merged pair files (100 distinct valid split-F queries each, data seed
10500 + pair). Deviation (D-056 addendum, author, before split F): 100 queries on CPU instead of a GPU. Analysis:
`x5_vae analyze` (local, clean tree) → results_x5.json.

**1. Primary (registered): witness share on null queries.** **0/800** (Δ-CE type N or O), 0 in every one of the 8
null pairs → prediction (≤ 0.05 pooled and per pair) **met**. CF-diff attributes a change to 0.85–0.94 of the same
null queries.

**2. Detection on the lowered class (class 1) at thresholds from these nulls (descriptive).**
τ_0 / τ_5%: s_M 1.136 / 0.624; s_M,r 1.384 / 0.972; CF-diff ‖z_n − z_o‖ 1.950 / 0.246.

| pair (t, seed) | Δ-CE | s_M τ0 | s_M τ5% | s_M,r τ0 | CF-diff τ0 | CF-diff τ5% |
|---|---|---|---|---|---|---|
| 100 (.53, 2) | 0.12 | 0.00 | 0.03 | 0.00 | 0.00 | 0.12 |
| 101 (.56, 2) | 0.13 | 0.00 | 0.09 | 0.00 | 0.00 | 0.06 |
| 102 (.60, 2) | 0.85 | 0.02 | 0.21 | 0.06 | 0.00 | 0.25 |
| 104 (.56, 3) | 0.66 | 0.00 | 0.23 | 0.11 | 0.00 | 0.21 |
| 105 (.60, 3) | 0.96 | 0.00 | 0.20 | 0.17 | 0.00 | 0.35 |
| 106 (.53, 4) | 0.98 | 0.48 | 0.98 | 0.77 | 0.00 | 0.52 |
| 107 (.56, 4) | 0.91 | 0.36 | 0.89 | 0.66 | 0.00 | 0.34 |
| 108 (.60, 4) | 0.95 | 0.58 | 0.93 | 0.91 | 0.00 | 0.84 |

Δ-CE ≥ 0.5 in **6/8** pairs → prediction (≥ 5/8) **met**. The two low pairs (100, 101: f_old seed 2, t ≤ 0.56) were
also the lowest on split D in factor space (0.17, 0.20).

**3. On-manifold share of witnesses (renderer fit, e_95 = 2.61e-4 from 1 000 validation reconstructions).** Nulls: no
witness to classify. Offsets (type N): 0.91 of witnesses on-manifold.

**4. Recovery (factors read by fitting the renderer; secondary, approximate).** Δ-CE 0.02–0.46 over all 100 queries
(only the lowered class, ≈ half, can have a type-N witness, as on Threshold); CF-diff 0.55–0.82. Δ-CE type O (raised
class) found 0.05–0.56.
