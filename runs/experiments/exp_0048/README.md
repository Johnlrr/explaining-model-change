# exp_0048 — D-052 X1: added baselines on split A (descriptive)

Code f7487b3 (`review_0211 x1-run`; `review_0211.py` unchanged up to the analysis commit 96b1eb5), config
configs/experiments/exp_0048.yaml (sha256 96ff79cc…). Locked split-A checkpoints and points of exp_0017 (hash-checked).
Pairs 0–2 ran locally (before the move to Kaggle); pairs 3–5, 9–12 on kernel boinhbo/dce-exp0048-k1 and 6–8, 13–16 on
boinhbo/dce-exp0048-k2 (D-044 kaggle-cpu: Intel Xeon 2.2 GHz, 4 cores, AVX2, torch 2.14.1+cpu; bundle tree df321b1f…;
inputs listing 97bbb457…; rc 0; 7/7 pairs each; 72 output files match the kernel manifests). Mixing machines is allowed
by D-052 (every X1 method was machine-stable at the argmax in smoke_0028). Kernel records:
runs/experiments/exp_0048_kaggle_k{1,2}. Analysis: `review_0211 x1-analyze` → results_x1.json.

Recovery@r (found-rate in brackets), mean over the 3 pairs of each condition; per-pair range after the semicolon.

| Condition | IG-diff | SHAP-diff | Imp-diff(adapt) | Seg-search×10 | Grad-diff (Table 2) |
|---|---|---|---|---|---|
| Shortcut (colour) | 0.93 (1.0); 0.91–0.97 | 0.93 (1.0); 0.91–0.95 | 1.00 (1.0) | 0.10 (0.37); found 0.35–0.37 | 0.91 |
| Subgroup (size) | 0.40 (1.0); 0.23–0.64 | 0.44 (1.0); 0.13–0.70 | 1.00 (1.0) | 0.27 (0.40); found 0.32–0.47 | 0.62 |
| Threshold (aspect) | 0.75 (1.0); 0.71–0.79 | 0.73 (1.0); 0.72–0.75 | 1.00 (1.0) | 0.29 (0.29); found 0.26–0.32 | 0.07 |
| Null (8 pairs) | attributes 1.0 | attributes 1.0 | attributes 1.0 | not run (cost amendment) | 1.0 |

Predictions (D-052, fixed before any number):

- IG-/SHAP-diff behave like Grad-diff (high on Shortcut, low on Threshold): **half met.** High on Shortcut (0.93),
  but **not low on Threshold** (0.73–0.75 vs Grad-diff 0.07, 6/6 pairs ≥ 0.71). IG and SHAP integrate over a path or
  coalitions from a baseline, not at the query, so the offset blindness of Proposition 2(a) — a statement about the
  local gradient — does not carry over to them.
- Imp-diff(adapt) recovers colour on Shortcut: **met** (1.00 in 3/3; it also recovers size on Subgroup and aspect on
  Threshold, 1.00 everywhere), but it is one vector per pair (no per-query information) and attributes a change on
  every null query.
- Seg-search×10 found-rate stays ≤ 0.2: **not met** (0.26–0.47 in 9/9 pairs; Seg-search with 1× segments: 0.05–0.13).
