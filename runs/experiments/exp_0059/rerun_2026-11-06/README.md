# exp_0059 - number check of paper/main.tex and paper/supplement.tex (D-056 W3)

Script `src/dce/cli/number_check.py` (`uv run python -m dce.cli.number_check --out runs/experiments/exp_0059`). Reads saved outputs only; nothing is re-run. Per claim: location, value as printed, computed value, source, status in `results_numcheck.json`. OK = equal to the source after the printed rounding.

Claims checked: 935. **OK 935, MISMATCH 0, UNSOURCED 0.**

## MISMATCH and UNSOURCED

none

## OK with a caveat in the note

| # | location | printed | computed | status | source | note |
|---|---|---|---|---|---|---|
| 164 | S4.3 Grad-diff Recovery at alpha=0.1 | 0.36 | 0.36 | OK | exp_0026/results.json:H5.H5b.rec_b (registered contrast pairs 28, 29: t=0.5) | equals the registered H5b contrast cell (t=0.5, pairs 28/29); the mean over all 8 alpha=0.1 pairs is 0.323 (exp_0058 by_alpha) |
| 165 | S4.3 Grad-diff Recovery at alpha=0.5 | 0.93 | 0.93 | OK | exp_0026/results.json:H5.H5b.rec_a (registered contrast pairs 44, 45: t=0.5) | equals the registered H5b contrast cell (t=0.5, pairs 44/45); the mean over all 8 alpha=0.5 pairs is 0.888 (exp_0058 by_alpha) |

## Scope

Covered: abstract; Sections 3-5 text numbers; Tables 1-3 (every estimate, CI, p, pairs cell); the parameters of Sections 3 and D; supplement Section B (Table S1 parsed from the tex and compared row by row with exp_0058), Section C (all tables and text), Section E (fixed elements, results, Table X5), Section X6 (design, results, Table S-X6), the split-G row and the split letters of the protocol log, the Table 1 shift column and 'Both' row, and the Table 3 row 'Witness localises'. Sources: exp_0017, 0019, 0026, 0028, 0032, 0033, 0046-0051, 0054, 0056, 0057, 0058, 0060.

Not machine-checkable from saved outputs: proofs and prose claims without a number; registered predictions are checked only where a run README records them.
