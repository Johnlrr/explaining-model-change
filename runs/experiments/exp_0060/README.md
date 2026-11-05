# exp_0060 — D-057 X6: two independent local changes (Shortcut + Subgroup), split G (registered https://osf.io/fp8uc)

Code 96f2bf3 (src unchanged since; bundle tree 9fb742e8…), config configs/experiments/exp_0060.yaml (sha256 524b8f39…).
f_new trained locally (8 pairs, ≈ 45 s each; models/, artifact_manifest.json, train_log.json). Qualification on
validation (qualification.json): **8/8** (ΔFlip colour 0.34–0.45, size 0.29–0.36, max bg/posx 0.011–0.028) → the
primary test is confirmatory. Search on 4 kaggle-cpu kernels boinhbo/dce-exp0060-k{1..4} (inputs boinhbo/dce-x6-inputs,
listing 69db73c3…; rc 0; 0.8–1.4 h each); outputs matched the kernel manifests; pairs/ = 8 × 300 valid split-G
queries (data seed 13000 + pair). Analysis `x6_multi analyze` → results_x6.json.

**1. Primary (registered): locality of Δ-CE.** L = share of witnesses whose largest |Δφ| is size, near (size ≥ 0.60)
minus far (size ≤ 0.45). Per pair: 0.59, 0.54, 0.53, 0.23, 0.52, 0.60, 0.50, 0.48 — **L > 0 in 8/8**, pooled
**L = 0.50** (≥ 0.30), exact sign test **p = 0.0039** → prediction **met**. Near: 0.46–0.75 of witnesses change size
most; far: 0.13–0.23. Δ-CE found a witness for 0.95–1.00 of queries.

**2. Controls (descriptive).** Pooled L: Grad-diff 0.09, CF-diff 0.13, Imp-diff 0 (one vector per pair, by
construction), **CF-new 0.51** (the counterfactual of f_new alone localises as well as the witness).

**3. Multi-factor recovery (descriptive), mean over pairs.**

| method | set-recovery {colour, size} | MRR colour | MRR size | top-1 colour | top-1 size |
|---|---|---|---|---|---|
| Δ-CE | 0.26 | 0.66 | 0.63 | 0.51 | 0.43 |
| CF-new | 0.24 | 0.66 | 0.63 | 0.51 | 0.43 |
| Grad-diff | 0.15 | 0.51 | 0.62 | 0.29 | 0.39 |
| CF-diff | 0.04 | 0.48 | 0.50 | 0.23 | 0.23 |
| Imp-diff | 0.00 | 0.42 | 0.33 | 0.00 | 0.00 |

Imp-diff ranks neither colour nor size first in 8/8 pairs → prediction **met**.
