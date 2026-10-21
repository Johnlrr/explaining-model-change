# exp_0035 — v1.10 T-D found-rate against κ (D-043 §5, descriptive)

Code a91e7c6 (td-run, resumable per κ). 9 pairs: H7 seed 0 with b ∈ {−2, −1.1, −0.5, 0.5, 1.1, 2.0} (exp_0032 points),
grid pair 20 (α = 0, t = 0.5), grid pair 26 (α = 0, t = 0.6), Tier 0 N0 pair 9; first 100 κ = 0.5 valid points of
new_test_set(9000 + pair_id) (reused after their confirmatory test, §6.1). Δ-CE N at κ ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}.
One detached CPU process; the first launch died with the local session (0 pairs complete) and was relaunched
from a91e7c6 (per-κ checkpoints); no pair was run twice to completion. 5.7 h wall. td-figure → results_td.json,
found_vs_b.pdf.

Found-rate (lowered-class / other-class share in brackets for H7):

| pair | model change | κ=0 | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|
| 60 | b = −2.0 | 0.37 | 0.37 | 0.37 | 0.37 | 0.37 | 0.01 |
| 62 | b = −1.1 | 0.49 | 0.49 | 0.49 | 0.49 | 0 | 0 |
| 64 | b = −0.5 | 0.51 | 0.51 | 0 | 0 | 0 | 0 |
| 65 | b = +0.5 | 0.40 | 0.40 | 0 | 0 | 0 | 0 |
| 67 | b = +1.1 | 0.37 | 0.37 | 0.37 | 0.37 | 0 | 0 |
| 69 | b = +2.0 | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 | 0 |
| 20 | grid α=0, t=0.5 | 0.80 | 0.68 | 0.64 | 0.33 | 0 | 0 |
| 26 | grid α=0, t=0.6 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.17 |
| 9 | Tier 0 N0 | 0.78 | 0.46 | 0.13 | 0 | 0 | 0 |

Reading (descriptive):
- H7: found-rate is the lowered-class share (witnesses on 100% of lowered-class points, 0% elsewhere) while 2κ < |b|,
  and 0 once 2κ > |b|, exactly as Lemma 1. At 2κ = |b| (b = ±0.5 at κ = 0.25; b = ±2 at κ = 1) the band
  [κ, |b| − κ] is a single value: 0 found, except 1 point for b = −2 at κ = 1 (feasible within the SLSQP tolerance).
- Pair 26 (label threshold 0.6) behaves like a logit offset of 1.5 < |b| < 2: one class only, flat to κ = 0.75, 0.17 at 1.0.
- Pair 9 (N0 retrain, no label change) decays with κ: retraining noise is a small-margin difference, gone at κ = 0.5.
- Pair 20 (grid null cell α = 0, t = 0.5) keeps 0.33 at κ = 0.5, class 1 only, 0 at 0.75: the fine-tune left a small
  threshold offset (exp_0029: t̂ − t̂_old = +0.039; locked exp_0026 found-rate 0.21, all class 1). A one-class κ profile
  is the logit-offset signature (H7), here at a small |b|.
