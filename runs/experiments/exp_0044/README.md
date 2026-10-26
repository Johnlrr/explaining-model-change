# exp_0044 — detection at matched null false positives on fine-tuned offsets (D-050 E4), descriptive

Saved outputs only (code aa4dafc, `review_d048 e4-analyze`). Null reference and thresholds as exp_0042 (exp_0017 nulls
9–16; τ_0 = 0.391 = max null score, τ_5% = 0.150). Offsets: Threshold pairs 6–8 (exp_0041, split C) and grid α = 0 pairs
24–27 (exp_0026, locked grid split, read only). Lowered class = class 1 (saved y checked against 1[aspect > 0.5]).

| run | pair | lowered n | Δ-CE (type N) lowered | CF-diff τ_0 lowered | CF-diff τ_5% lowered | all: Δ-CE / τ_0 / τ_5% |
|---|---|---|---|---|---|---|
| exp_0041 | 6 | 136 | 1.000 | 0.000 | 0.279 | 0.453 / 0.000 / 0.277 |
| exp_0041 | 7 | 133 | 1.000 | 0.000 | 0.278 | 0.443 / 0.000 / 0.317 |
| exp_0041 | 8 | 135 | 1.000 | 0.000 | 0.170 | 0.450 / 0.000 / 0.157 |
| exp_0026 | 24 | 155 | 1.000 | 0.000 | 0.097 | 0.517 / 0.000 / 0.080 |
| exp_0026 | 25 | 134 | 0.246 | 0.000 | 0.052 | 0.110 / 0.000 / 0.057 |
| exp_0026 | 26 | 141 | 1.000 | 0.000 | 0.184 | 0.470 / 0.000 / 0.173 |
| exp_0026 | 27 | 130 | 1.000 | 0.000 | 0.238 | 0.433 / 0.000 / 0.177 |

D-050 rule: min(Δ-CE − CF-diff(τ_0)) over the 7 pairs = 0.246 ≥ 0.20 → branch "generalises" (descriptive; not R, since
formed after the constant-offset result). Δ-CE at 0 null false positives exceeds CF-diff even at 5% in every pair.
