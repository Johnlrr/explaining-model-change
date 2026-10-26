# exp_0043 — linear Δ-CE recovery condition at the query vs Δ-CE Recovery on CNN rotations (D-049, review A5), descriptive

Code a1da6e9 (`dce.cli.review_d048 e3-analyze`, config `configs/experiments/exp_0043.yaml`). exp_0033 saved outputs:
24 α > 0 grid pairs, fresh data (seeds 9000 + pair_id), Δ-CE N witnesses; margins and gradients of the hash-verified
checkpoints at z0 (∇m_o reproduces the saved grad_z0, checked). P = 1[θ > π/4 or d_o(sin θ + cos θ) < d_n] at z0;
outcome = top factor of the Δ-CE Δφ̃ is colour, on the support.

| support | share P=1 | Rec(P=1) | Rec(P=0) | gap [95% CI] | agreement P = R | Rec overall |
|---|---|---|---|---|---|---|
| 3 952 | 0.435 | 0.638 | 0.040 | +0.598 [0.574, 0.622] | 0.820 | 0.300 |

Factor alignment (assumption of the linear Recovery statement): top factor of ∇m_o = aspect on 0.85 of points, of
v = ∇m_n⊥ = colour on 0.73.

D-049 rule: both groups ≥ 5% and gap ≥ 0.20 with CI lower > 0 → the condition transfers when evaluated at the query.

After the result (not declared in advance): within-pair gaps, pairs with ≥ 10 points in both groups: 21 of 24; mean
+0.541, range 0.292–0.749, positive in all 21 — the pooled gap is not a between-pair (α) artefact.
