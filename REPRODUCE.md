# Reproducing the paper's tables and figure

Setup: `uv sync` (Python 3.12, CPU PyTorch; versions pinned in `uv.lock`). Every command below runs on CPU.

Two levels:

- **From saved outputs (minutes).** Each test run saved its per-pair outputs in `runs/experiments/<exp>/pairs/`.
  The `analyze` stages recompute every metric, CI and test from them. They refuse to overwrite an existing
  `results.json`, so the helper below copies the run to a fresh directory without its results first.
- **Full re-run (hours).** The `run` stages regenerate the data from seeds and re-solve every point with the
  checkpoints under `runs/experiments/exp_0002`, `exp_0003`, `exp_0025` (hash-verified on load). Test splits were
  used once in the study; re-running them is for verification only.

```bash
reanalyze() {   # reanalyze <module> <exp> [stage]: recompute from the saved run (outputs, metadata) into .work/repro/<exp>
  rm -rf .work/repro/$2 && mkdir -p .work/repro && cp -r runs/experiments/$2 .work/repro/$2
  rm -f .work/repro/$2/results*.json
  uv run python -m $1 ${3:-analyze} --config configs/experiments/$2.yaml --out .work/repro/$2
}
```

| Paper item | Source run | Command (from saved outputs) | Compare with |
|---|---|---|---|
| Table 2 (Recovery, found-rate; locked split) | exp_0017 | `reanalyze dce.cli.final_tier0 exp_0017` | `runs/experiments/exp_0017/results.json` |
| Table 3: Δ-CE vs gradient differences rows (H2a, P-tie) | exp_0017 | same | same |
| Table 3: C1, C2, C3 | exp_0019 | `reanalyze dce.cli.confirm_tier0 exp_0019` | `runs/experiments/exp_0019/results.json` |
| Table 3: H5a–H5d; Fig. 1 data | exp_0026 | `reanalyze dce.cli.v19_test exp_0026` | `runs/experiments/exp_0026/results.json` |
| Fig. 1 (as printed) | exp_0026 | `uv run python -m dce.analysis.v19_figure runs/experiments/exp_0026/results.json fig1.pdf` | `paper/figures/map_names.pdf` |
| Table 3: H6 (Adult) | exp_0028 | full re-run only: `uv run python -m dce.cli.v19_stage_b test --config configs/experiments/exp_0028.yaml --out .work/repro/exp_0028` (≈ 10 min) | `runs/experiments/exp_0028/h6.json` |
| Table 3: H7c | exp_0032 | `reanalyze dce.cli.v110_test exp_0032 ta-analyze` | `runs/experiments/exp_0032/results_h7.json` |
| Table 3: H8; Holm {H7c, H8} | exp_0033 | `reanalyze dce.cli.v110_test exp_0033 tb-analyze`, then `uv run python -m dce.cli.v110_test holm --config configs/experiments/exp_0033.yaml --out .work/repro/exp_0033/results_holm.json --h7 .work/repro/exp_0032/results_h7.json --h8 .work/repro/exp_0033/results_h8.json` | `runs/experiments/exp_0033/results_holm.json` |
| §4.2 Shortcut 2×2 (descriptive) | exp_0036 | `reanalyze dce.cli.v110_test exp_0036 tb-describe` | `runs/experiments/exp_0036/results_2x2.json` |
| §4.3 missed witnesses (descriptive) | exp_0034 | `reanalyze dce.cli.v110_tc exp_0034 tc-analyze` | `runs/experiments/exp_0034/results_tc.json` |
| §4.3 κ curve (descriptive) | exp_0035 | `reanalyze dce.cli.v110_tc exp_0035 td-figure` | `runs/experiments/exp_0035/results_td.json` |
| §2 numerical check of the closed forms | exp_0023, exp_0037 | `uv run python -m dce.cli.p5_diag --config configs/experiments/exp_0037.yaml --out .work/repro/exp_0037` | `runs/experiments/exp_0037/results.json` |

| §4.1 type O / N ∪ O on Threshold (split C, descriptive) | exp_0041 | `reanalyze dce.cli.review_d048 exp_0041 e1-describe` | `runs/experiments/exp_0041/results_e1.json` |
| §4.3 thresholded CF-diff, constant offsets (descriptive) | exp_0042 | `uv run python -m dce.cli.review_d048 e2-analyze --config configs/experiments/exp_0042.yaml --out .work/repro/exp_0042` | `runs/experiments/exp_0042/results_e2.json` |
| §4.2 recovery condition at the query (descriptive) | exp_0043 | `uv run python -m dce.cli.review_d048 e3-analyze --config configs/experiments/exp_0043.yaml --out .work/repro/exp_0043` | `runs/experiments/exp_0043/results_e3.json` |
| §4.3 thresholded CF-diff, fine-tuned offsets (descriptive) | exp_0044 | `... e4-analyze --config configs/experiments/exp_0044.yaml --out .work/repro/exp_0044` | `runs/experiments/exp_0044/results_e4.json` |
| per-pair counts in §4.1–4.2 | exp_0045 | `... e5-analyze --config configs/experiments/exp_0045.yaml --out .work/repro/exp_0045` | `runs/experiments/exp_0045/results_e5.json` |

### Table 3 rows → protocol IDs

The paper names tests by their claim; the protocol (`docs/source/`, `research/DECISIONS.md`, `research/HYPOTHESES.md`)
uses the IDs below. Splits: A = locked Tier 0 test (seed 1234); B = new Tier 0 test (seed 2027); C = fresh Threshold data
(seeds 9106–9108); grid test = seeds 7020–7051; "Grid, new" = seeds 9000 + pair id.

| Table 3 group / claim | Protocol ID | Holm family | When formed | Source run |
|---|---|---|---|---|
| Offsets / Grad-diff < 0.2 | H5a | ‡ {H5a–H5d} | grid (v1.9, D-038/D-039) | exp_0026 |
| Offsets / Grad-diff offset-free | H5c | ‡ | grid | exp_0026 |
| Offsets / Δ-CE > Grad-diff (Threshold, B) | C1 | * {C1, C3} | post (D-034) | exp_0019 |
| Rotations / Grad-diff rises with α | H5b | ‡ | grid | exp_0026 |
| Rotations / Proj. Δ-CE ≈ Grad-diff (Shortcut, B) | C3 | * | post (D-034) | exp_0019 |
| Rotations / Proj. Δ-CE ≈ Grad-diff (Grid) | H5d | ‡ | grid | exp_0026 |
| Rotations / Normal at z* halves gap or more | H8 | § {H7c, H8} | new (v1.10, D-043) | exp_0033 |
| Abstention / No witness on nulls | C2 (null, B); H6 (Adult) | — | post; new (D-040, D-042) | exp_0019; exp_0028 |
| Abstention / CF-diff (no cut) flags \|b\| < 2κ | H7c | § | new (v1.10, D-043) | exp_0032 |
| Δ-CE vs gradient differences / Δ-CE > Grad-diff (Subgroup) | H2a | — (m = 1) | pre (spec) | exp_0017 |
| Δ-CE vs gradient differences / Tie | P-tie | — | pre (spec) | exp_0017 |

Method names: Seg-search = B5 ("DIG-style" in the protocol); No-prox = A1 ("DeepXplore-style"); CF-diff = B1/B1-det;
Grad-diff = B3; Imp-diff = B4; CF-new = A2.

Full re-run: replace `analyze` by `run` (Tier 0 locked split: add `--final`), with `--out` a fresh directory.

Protocol: `docs/source/` (specification and amendments v1.1–v1.10), `research/DECISIONS.md` (every decision, with
its timing relative to the test data; audit in D-046), `research/HYPOTHESES.md`. Proofs: `research/theory/proofs.md`.
Unit tests: `uv run pytest tests/unit`.
