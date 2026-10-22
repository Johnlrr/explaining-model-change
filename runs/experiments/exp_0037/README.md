# exp_0037 — why the P5 vector criterion failed at p = 5 (exp_0023): numerics, not the formulas

Tier L only (no test data). Code: `dce.cli.p5_diag` (this commit's parent). Fresh seeds 71 000 + p, p ∈ {2, 5, 10},
1 000 random linear instances each, generator of exp_0023. Results: results.json. Summary in
research/theory/proofs.md §6.

- KKT certificate of the closed form of Prop. 2(b): holds on every instance (residuals ≤ 1e-14, multipliers ≥ 0).
- scipy SLSQP float64: matches the closed form to 1e-6 on 100%.
- Experiments' two-phase solver: vector agreement to 1e-3 on 82.7 / 89.9 / 96.6% (float32), 82.7 / 91.3 / 97.0%
  (float64). Disagreements are 90–98% case A (one active constraint), not near the case boundary; distance error
  ≤ 7e-4 everywhere. Explanation: tangential error ε costs only ε²/2 in distance on a single active face, and the
  solver stops on the distance.
