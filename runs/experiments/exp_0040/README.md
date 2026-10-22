# exp_0040 — run-to-run variability of the reported comparisons (validation, descriptive; D-047)

Code `dce.cli.variability_check` (99de825). Units = (pair, run); runs f32_s0/s1/s2, f64_s0/s1. Local: Shortcut (0) and
Threshold (6), 300 points, all runs; H5d pairs 36, 37, 44, 45 (150 points), run f32_s0. Kaggle CPU (Intel Xeon, AVX2;
kernels dce-exp0040-r1..r4, run records exp_0040_kaggle_r*): H5d runs 1–4, one run per kernel. The harness marked the
kernels ERROR only for a manifest-format check (missing `kind`); outputs complete, hashes verified, rc 0 (entry fixed
afterwards). results.json.

Decisions (D-047, fixed before the results): Shortcut Δ-CE − CF-diff −0.173 to −0.183 (|Δ| > 0.10 in all runs) →
paper unchanged. H8: T = 0.085–0.120 > 0 in all runs → proofs.md §6 only.
