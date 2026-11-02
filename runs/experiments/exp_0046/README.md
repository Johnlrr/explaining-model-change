# exp_0046 — margin control for detection at matched null false positives (D-052 M), descriptive

Saved z0 plus hash-verified checkpoints (no solver run). Code 12c618b (`review_0211 m-analyze`); `results_m.json` says
dirty = true because `src/dce/cli/review_0211.py` gained the E/X1 stages while this run was in progress — `m_analyze`
and its helpers were not changed. Identity checks passed: every saved z0 is a valid query (margin ≥ κ) for the loaded
checkpoints, and |h| = |b| on every exp_0032 logit-shift pair.

Scores: s_M = |h(z0)|; s_M,r = max |h| over 256 uniform points of the r-ball ∩ [0, 1]^5 plus z0. Thresholds on the 2 400
exp_0017 null queries: s_M τ_0 = 1.650, τ_5% = 0.840; s_M,r τ_0 = 2.070, τ_5% = 1.407; CF-diff τ_0 = 0.391, τ_5% = 0.150.

Fine-tuned offsets, lowered class (Δ-CE type N at 0/2 400 null witnesses):

| pair | n | Δ-CE | s_M τ_0 | s_M,r τ_0 | CF-diff τ_0 | s_M τ_5% | CF-diff τ_5% |
|---|---|---|---|---|---|---|---|
| 6 | 136 | 1.000 | 0.103 | 0.419 | 0.000 | 0.574 | 0.279 |
| 7 | 133 | 1.000 | 0.075 | 0.662 | 0.000 | 0.662 | 0.278 |
| 8 | 135 | 1.000 | 0.030 | 0.074 | 0.000 | 0.644 | 0.170 |
| 24 | 155 | 1.000 | 0.000 | 0.000 | 0.000 | 0.187 | 0.097 |
| 25 | 134 | 0.246 | 0.000 | 0.000 | 0.000 | 0.082 | 0.052 |
| 26 | 141 | 1.000 | 0.014 | 0.099 | 0.000 | 0.270 | 0.184 |
| 27 | 130 | 1.000 | 0.000 | 0.000 | 0.000 | 0.377 | 0.238 |

Constant logit offsets, lowered class: s_M at τ_0 detects only |b| = 2.0 (1.000; the null maximum is 1.65); at τ_5% from
|b| = 0.9 (1.000). Δ-CE from |b| = 1.1 (1.000), none below. CF-diff τ_5%: 0.06–0.76.

**D-052 rule → branch "witness more sensitive"** (Δ-CE ≥ s_M + 0.20 in every pair; s_M ≥ Δ-CE − 0.05 in 0 of 7).
Reading: null fine-tunes move margins by up to 1.65 without moving decisions, so a margin score needs a high threshold
to stay silent on nulls; the witness tests decisive decision change and is insensitive to this rescaling.
Caveat (D-051): the null pairs were admitted with decisive disagreement 0 on validation; X3 (exp_0049) re-tests the
false-positive side on nulls admitted without that filter.
