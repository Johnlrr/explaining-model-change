# exp_0036 — T1 2×2 (H8 design) on new data, descriptive (D-045)

Code 9433dd7 (tb-run for Tier 0 pairs, tb-describe). T1 pairs 0–2 (exp_0002 f_old, exp_0003 f_new, hash-verified),
k* = colour; new_test_set(9000 + pair_id, 5 000) — seeds 9000–9002, first use; 300 valid points; Δ-CE N; B3 and ∇m_old
at z0 and z*. Three detached CPU shards, 24 min each, no interruption. tb-describe (bootstrap seed 6602, 10 000) →
results_2x2.json. No hypothesis, no Holm.

| cell (projection, B3) | gap Δ-CE − B3 on the support | CI90 |
|---|---|---|
| (z0, z0) reference | −0.166 | [−0.192, −0.140] |
| (z*, z0) | −0.024 | [−0.041, −0.007] |
| (z*, z*) | −0.028 | [−0.039, −0.017] |
| (z0, z*) | −0.188 | [−0.213, −0.164] |

T = |gap(z0,z0)| − 2|gap(z*,z0)| = 0.118 (CI95 lower 0.086); Rec_proj(B3) at (z*, z0) 0.96. Supports 250, 256, 247
(found-rate 0.82–0.85). Per pair, Rec_proj (Δ-CE, B3): (z0,z0) 0.75–0.82 vs 0.96; (z*,z0) 0.93–0.96 vs 0.96–0.97.
Locality: median ‖z* − z0‖ 0.29, median 1 − cos 0.12 (≈ 28°); Spearman 1 − cos ~ distance +0.14, deviation ~ distance −0.14.

Reading: the grid result (exp_0033, H8) repeats on the shortcut design: taking the old-model normal at the witness
narrows the gap by more than half (−0.166 → −0.024); it does not close it fully (CI90 below 0). Moving B3 to z* alone
does not help. The normal turns at all distances.
