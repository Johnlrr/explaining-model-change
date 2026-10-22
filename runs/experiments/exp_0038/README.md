# exp_0038 — solver vector error on the CNN benchmark (validation, descriptive) + worst-case bounds (saved outputs)

Code dbd1962 (`dce.cli.tangent_check`), worst case `dce.analysis.worst_case` (reproduces every original estimate at τ = 0).
Validation, ~100 valid points per pair (0, 3, 6, 36, 44, 24).

Vector error (relative, points found by both):
- Δ-CE, experiments' solver vs tight float64 reference: median 1.0e-4, p95 0.223, max 1.58 (n = 409); heavy tail:
  some points converge to a different local minimum. Top-factor changes: 2/409 (0.5%; both in pair 24).
- DeepXplore, float32 vs float64 (no optimum without proximity): median 0.54, p95 1.22; top-factor changes 98/301 (33%).

Worst case with τ = p95 (0.223 solver methods, 1.22 DeepXplore), factor √2, adversarial assignment of every near-tie:
several directions do not hold (H2a, two tie comparisons, C3, H5d, Δ-CE vs DeepXplore on Subgroup, H8 T → −0.69):
worst_case_p95.json. Per the author's rule, stopped before adding anything to proofs.md.
