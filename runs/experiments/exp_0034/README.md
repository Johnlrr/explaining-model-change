# exp_0034 — v1.10 T-C abstention recall (D-043 §4, descriptive)

31 pairs (config configs/experiments/exp_0034.yaml, unchanged throughout): G1 null 9–16, G2 T4 class 0 6–8, G3 grid
20–27 (new_test_set(9000 + pair_id)), G4 H7 |b| ∈ {1.1, 1.5} (exp_0032 points and Δ-CE). Δ-CE N on the first 100
valid points; up to 50 NOT_FOUND queries per pair; M = 100 000 accepted uniform samples in B(z0, r) ∩ [0,1]^5, then
projected Adam ascent (16 starts, 200 steps); per-query generator (9500 + pair_id)·10 000 + query.

## Execution (disclosed)
- First local launch (018e51d) died with the local session at 0 complete pairs; relaunched from a91e7c6 with
  per-query checkpoints and per-query generators (values do not depend on where or when a query runs).
- Local, 2 static shards × 8 threads, then (62cd8a1) dynamic claiming × 4 threads: pairs 9–12 and G4. Local 4 vs 8
  threads give bit-identical values (smoke_0021 comparison).
- D-044 (user's authorisation 2026-10-21): pairs 13–16, 6–8, 20–27 on 5 Kaggle CPU kernels (commit 4e52443; run
  records exp_0034_kaggle_k1..k5; Intel Xeon 2.2 GHz, AVX2, pinned torch 2.14.1+cpu / numpy 2.5.4 / scipy 1.18.1,
  4 threads; 5.2–6.4 h each, no deadline stop). Every manifest hash, config hash, input hash and bundle tree hash
  checked; files copied unchanged into pairs/. Pair 13 had one query run locally before it was moved to Kaggle (kept
  in .work only; Kaggle value used).
- G4: Δ-CE found a witness for every lowered-class query (exp_0032), so G4 has no query.
- Cost ≈ 2× the amendment's estimate (≈ 2 h per 50-query pair on 4 cores).

## Cross-machine agreement and the D-044 guard
Data digests, z0, y and the Δ-CE found vector are identical across machines (pairs 9, 13). The sampling stage agrees to
≤ 1.5e-7. The ascent stage can end at another local optimum: |Δ s_ascent| up to 9.6e-3 (pair 13 query 0, 22 q14),
larger than the smoke's 3.5e-4. Registered guard (|s_max| < 1e-3 → re-run locally): pair 20 q33, Kaggle −9.8e-5 →
local −6.8e-3; no change. Sensitivity (added after seeing the results, descriptive): all 14 Kaggle queries with
|s_max| < 0.02 re-run locally (`tc-recheck`, a9b7c70; control pair 9 q0 reproduced exactly): one sign differs,
pair 20 q27 (Kaggle −3.1e-3, local +1.8e-3). tc_recheck_{a..d}.json.

## Result (results_tc.json, registered values)
| group | pairs | queries | false negatives (s ≥ 0) | rate [CP95] | max s: median, max |
|---|---|---|---|---|---|
| G1 null | 9–16 | 400 | 0 | 0 [0, 0.009] | −0.38, −0.15 |
| G2 T4 class 0 | 6–8 | 150 | 0 | 0 [0, 0.024] | −1.00, −0.82 |
| G3 offsets | 20–27 | 395 | 3 (4 with the local re-run) | 0.008 [0.002, 0.022] | −0.62, +0.048 |
| G4 H7 | 12 pairs | 0 | — | — | — |

All accepted counts = 100 000. All false negatives are on pair 20 (grid null cell, t̂ − t̂_old = +0.039), class 1, the
class where Δ-CE does find witnesses (33/100): queries 18, 26, 64 (+ 27 locally), max s 0.003–0.048 (thin feasible
regions). On the queried G3 points Δ-CE recovers ≥ 246 of 249 (250) witness-holding points.

Reading: as stated in §4, no false negatives where no witness is expected (null pairs; the class an offset cannot
flip), with s bounded away from 0 (G1 max −0.15). Misses occur only where the feasible band is thin (a small offset).
