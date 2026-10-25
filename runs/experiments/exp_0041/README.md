# exp_0041 — Δ-CE type N, type O and N ∪ O on Threshold (T4), new data, descriptive (D-048 E1)

Code 8bae341 (`dce.cli.review_d048 e1-run`, 3 shards, then `e1-describe`; config `configs/experiments/exp_0041.yaml`).
Tier 0 T4 pairs 6–8 (exp_0002 f_old, exp_0003 f_new, hash-verified), k* = aspect; data `new_test_set(9100 + pair_id, 5 000)`
(seeds 9106–9108, first use), 300 valid points per pair (900 queries: 496 class 0, 404 class 1), locked solver, 1.5 h wall.
N ∪ O = nearer witness when both exist. Bootstrap seed 6603, 10 000, paired by pair. No test, no Holm. smoke_0026 = same path on validation.

| method | Recovery@r [95% CI] | found-rate | class 0: found / Rec | class 1: found / Rec | per pair Recovery |
|---|---|---|---|---|---|
| Δ-CE N | 0.449 [0.416, 0.481] | 0.449 | 0.000 / 0.000 | 1.000 / 1.000 | 0.453, 0.443, 0.450 |
| Δ-CE O | 0.551 [0.519, 0.584] | 0.551 | 1.000 / 1.000 | 0.000 / 0.000 | 0.547, 0.557, 0.550 |
| Δ-CE N ∪ O | 1.000 [1.000, 1.000] | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.0, 1.0, 1.0 |
| CF-diff (B1-det) | 0.768 [0.740, 0.794] | 0.884 | 0.798 / 0.698 | 0.990 / 0.854 | 0.780, 0.737, 0.787 |
| Grad-diff (B3) | 0.070 [0.054, 0.087] | 1.000 | 1.000 / 0.109 | 1.000 / 0.022 | 0.047, 0.070, 0.093 |
| CF-new (ordinary CE of f_new) | 0.891 [0.870, 0.910] | 0.891 | 0.802 / 0.802 | 1.000 / 1.000 | 0.927, 0.847, 0.900 |

Reading: type N has a witness for every class-1 query and for none of class 0 (the update lowers class 1: aspect
∈ (0.5, 0.6)), type O the converse; the nearer witness has k* = aspect on all 900 queries. Table 2 of the paper (locked
split, type N only, 0.47) is unchanged; the paper now reports N ∪ O next to it. CI [1.000, 1.000] is a ceiling effect.
