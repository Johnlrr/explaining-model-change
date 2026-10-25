# exp_0042 — CF-diff with a detection threshold at matched false-positive rate (D-048 E2), descriptive

Saved outputs only, nothing re-run (code 8bae341, `dce.cli.review_d048 e2-analyze`, config `configs/experiments/exp_0042.yaml`).
Score of a query = ‖Δφ̃‖₂ of B1-det (0 if no counterfactual pair). Null reference: the 8 null pairs of exp_0017 (pairs 9–16,
2 400 queries, locked split read from saved outputs). Detection: the 30 logit-offset pairs of exp_0032 (300 queries each),
per |b|, both signs, 3 seeds of f_old. Thresholds: τ_0 = 0.391 (largest null score; FPR 0/2 400), τ_1% = 0.219, τ_5% = 0.150.

| \|b\| | queries | CF-diff @FPR 0 | @1% | @5% | AUROC vs null | Δ-CE found-rate | lowered class: CF-diff @0 / 1% / 5% | Δ-CE, lowered class |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1800 | 0.000 | 0.011 | 0.062 | 0.696 | 0.000 | 0.000 / 0.012 / 0.060 | 0.000 |
| 0.9 | 1800 | 0.000 | 0.017 | 0.106 | 0.798 | 0.000 | 0.000 / 0.020 / 0.102 | 0.000 |
| 1.1 | 1800 | 0.000 | 0.020 | 0.156 | 0.817 | 0.441 | 0.000 / 0.030 / 0.172 | 1.000 |
| 1.5 | 1800 | 0.000 | 0.049 | 0.353 | 0.831 | 0.425 | 0.000 / 0.068 / 0.403 | 1.000 |
| 2.0 | 1800 | 0.001 | 0.213 | 0.614 | 0.775 | 0.369 | 0.002 / 0.286 / 0.756 | 1.000 |

Rule of D-048 E2: detection of CF-diff at τ_0 on |b| ∈ {0.5, 0.9} is 0.000 (< 0.05), so the paper says a thresholded
CF-diff abstains there as well and the contrast is threshold versus no threshold. Reading beyond the rule: at the Δ-CE
operating point (no null query flagged) CF-diff detects almost nothing even at |b| = 2 where Δ-CE has a witness for every
lowered-class query; at 5% false positives CF-diff detects more than Δ-CE's overall found-rate at |b| = 2 (0.61 vs 0.37),
so the ranking depends on the operating point. τ_0 is the maximum of 2 400 null scores, a conservative cut that depends on
the null tail. Order of events (disclosed): the rules were written to DECISIONS.md before the script ran, but the script
was committed (8bae341) after its first output was read; one correction of the null source (exp_0019 holds B1, not
B1-det → exp_0017) was made before any score was read.
