# exp_0051 — D-053 IG-diff diagnostics (post hoc after X1; split D part with a prediction fixed beforehand)

Code `dce.cli.x1_ref` (commit c8e6110). Outputs: x1_ref_split_a.json, x1_ref_split_d.json, logs.

**Split A (exploratory, pairs 0–8, saved X1 points).** IG completeness max error 0.07–0.10 at 64 steps, 0.016–0.022
at 256; Recovery identical at 64 and 256 steps. Query-side reference (z̄ with aspect 0.1 / 0.9): Threshold 0.49–0.57
(mean reference 0.71–0.79, Grad-diff 0.05–0.09) → the "path crosses the moved boundary" mechanism is **not
supported**. Shortcut unchanged (0.93–0.96). With the saved X1 outputs (D-053): on the 8 split-A nulls IG-diff names
aspect on 0.34–0.88 of queries (pooled 0.57), SHAP-diff 0.21–0.93 (0.71), Grad-diff 0.02–0.11.

**Split D (prediction in D-053, fixed before IG was computed on split D).** IG-diff, mean reference, 64 steps,
saved split-D points of exp_0049. Nulls 110–117 (unfiltered): aspect rate 0.06–0.82, **pooled 0.42** (≥ 0.30
predicted); Grad-diff pooled **0.044** (≤ 0.15 predicted). Qualified offsets: 0.31–0.78, **8 of 8** within the null
range (≥ 6 predicted) → **prediction met**. The range criterion is weak (null range wide); the informative part is
0.42 vs 0.044 on nulls. Mean ‖IG-diff‖: nulls 0.08–0.45, offsets 0.37–0.77.
