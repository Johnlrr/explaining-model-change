# exp_0059 - number check of paper/main.tex and paper/supplement.tex (D-056 W3)

Script `src/dce/cli/number_check.py` (`uv run python -m dce.cli.number_check --out runs/experiments/exp_0059`). Reads saved outputs only; nothing is re-run. Per claim: location, value as printed, computed value, source, status in `results_numcheck.json`. OK = equal to the source after the printed rounding.

Claims checked: 795. **OK 793, MISMATCH 2, UNSOURCED 0.**

## MISMATCH and UNSOURCED

| # | location | printed | computed | status | source | note |
|---|---|---|---|---|---|---|
| 189 | S4.4 Grad-diff 0.36 (high end, split E) | 0.36 | 0.35 | MISMATCH | same | split D (exp_0050) gives 0.32-0.36; split E, the split of the surrounding sentence, gives 0.32-0.35 |
| 546 | S-C text: grid null cell pair 21 realised threshold t_hat ~ 0.53 | 0.53 | 0.463808 | MISMATCH | exp_0026/results.json:descriptive.pairs[21].t_hat |  |

## OK with a caveat in the note

| # | location | printed | computed | status | source | note |
|---|---|---|---|---|---|---|
| 149 | S4.3 Grad-diff Recovery at alpha=0.1 | 0.36 | 0.36 | OK | exp_0026/results.json:H5.H5b.rec_b (registered contrast pairs 28, 29: t=0.5) | equals the registered H5b contrast cell (t=0.5, pairs 28/29); the mean over all 8 alpha=0.1 pairs is 0.323 (exp_0058 by_alpha) |
| 150 | S4.3 Grad-diff Recovery at alpha=0.5 | 0.93 | 0.93 | OK | exp_0026/results.json:H5.H5b.rec_a (registered contrast pairs 44, 45: t=0.5) | equals the registered H5b contrast cell (t=0.5, pairs 44/45); the mean over all 8 alpha=0.5 pairs is 0.888 (exp_0058 by_alpha) |

## Scope

Covered: abstract; Sections 3-5 text numbers; Tables 1-3 (every estimate, CI, p, pairs cell); the parameters of Sections 3 and D; supplement Section B (Table S1 parsed from the tex and compared row by row with exp_0058), Section C (all tables and text), Section E (fixed elements, results, Table X5). Sources: exp_0017, 0019, 0026, 0028, 0032, 0033, 0046-0051, 0054, 0056, 0057, 0058.

Not machine-checkable from saved outputs: proofs and prose claims without a number; registered predictions are checked only where a run README records them.

## Re-run after the two corrections (2026-11-05)

main.tex: ResNet split-E Grad-diff range 0.32–0.36 → 0.32–0.35; rotation sentence now names its cells (t = 0.5).
supplement.tex: grid null cell t̂ = 0.526 and 0.464 (was "≈ 0.53"). Script updated accordingly. Re-run:
**795 OK, 0 MISMATCH, 0 UNSOURCED** (rerun_2026-11-05/results_numcheck.json).

## Re-run after the W6 rewrite and X6 (2026-11-05, code 9be5ede)

Output in `rerun_2026-11-05b/` (the earlier outputs are unchanged). Script re-anchored to the rewritten main.tex (abstract, Table 1 with the Shift column and the "Both" row, Table 3 row "Witness localises", Section 4.1 X6 paragraph, merged OSF reference) and extended to the supplement Section X6 (design, results, Table S-X6), the split-G row and the split letters A-G of the protocol log; new sources exp_0060 (results_x6.json, qualification.json, pairs, config) and exp_0048 arrays for "none that Delta-CE misses" (Seg-found subset of DCE-found).
Claims removed because the number left main.tex: CF-diff 98.3/99.7 on split D (the 97.7-100 range now spans the three groups of the sentence and is computed over all of them), VAE "CF-diff none at tau_0", CF-diff 0.03/0.18 at 5% (split D offsets), "below 0.2 in each of the 4 pairs", Grad-diff 5% on nulls, 955/953 Seg-search counts, ResNet "100 queries per pair" in Section 4, margin score <= 0.05 and CF-diff <= 0.03 on split E, "0.5 for the CNNs" limitation text. Re-anchored: "positive for each seed" now "in each of 3 pairs" (checked on seeds and on pair-level arrays).
Result: **924 claims: OK 923, MISMATCH 1, UNSOURCED 0** (see `rerun_2026-11-05b/README.md`). The one MISMATCH: main Limitations (4) says "two disclosed exceptions"; the supplement paragraph "Disclosed exceptions" lists three items (D-041, D-055, D-053). Not interpreted here.

## Re-run after "two" → "three disclosed exceptions" (2026-11-05)

main.tex Limitations (4) now says three (D-041, D-055, D-053, as listed in the supplement). **924 OK, 0 MISMATCH,
0 UNSOURCED** (rerun_2026-11-05c/results_numcheck.json).
