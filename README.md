# Explaining model change: a benchmark with exact ground truth

Code, data generators, protocol, decision log and run evidence for the paper

> **Benchmarking Counterfactual Explanations of Model Change with Exact Ground Truth**
> Dinh Bao Bao, Tran Cong Minh and Le Hoai Bac — Faculty of Information Technology, University of Science,
> Vietnam National University, Ho Chi Minh City. Submitted to ESANN 2027.

The paper asks which kind of model change each family of explanations can see — counterfactual differencing,
gradient differences, importance differences, disagreement search and differential counterfactuals (Δ-CE: the nearest
input on which exactly one of two models changes its decision by a margin). Images are rendered from five known
factors, so the ground truth of every injected change is exact.

## Contents

| Path | What |
|---|---|
| `paper/main.tex`, `paper/main.pdf` | the paper (6 pages) |
| `paper/supplement.tex`, `paper/supplement.pdf` | supplementary material: proofs, pair-level statistics, sensitivity, budget, X5/X6, protocol log |
| `src/dce/` | all research code (renderer, models, Δ-CE solver, baselines, experiment CLIs, analysis) |
| `configs/` | shared parameters (`tiers/`) and one config per experiment (`experiments/`) |
| `runs/` | evidence: per-pair outputs, results, manifests and checkpoints (sha256-checked on load) |
| `research/` | decision log (`DECISIONS.md`), hypotheses, literature, notes, proofs (`theory/`) |
| `docs/` | the frozen specification and its amendments |
| `tests/unit/` | unit tests |
| `REPRODUCE.md` | how to recompute every table and figure |
| `COMMIT_MAP.tsv` | old → new commit hashes (see Provenance) |

## Quick start

```bash
uv sync                      # Python 3.12, CPU PyTorch, versions pinned in uv.lock
uv run pytest tests/unit     # unit tests
```

Then follow `REPRODUCE.md`: every table can be recomputed from the saved per-pair outputs in minutes; full re-runs
regenerate the data from seeds and take hours on CPU.

## Pre-registration

The follow-up experiments were registered before their data existed:
[osf.io/vfjtk](https://osf.io/vfjtk) (X2–X4, split D), [osf.io/fr7jv](https://osf.io/fr7jv) (X5, learned latent,
split F), [osf.io/fp8uc](https://osf.io/fp8uc) (X6, two local changes, split G). The paper and `research/DECISIONS.md`
state which tests were formed after earlier results and list every deviation.

## Provenance

The project was developed on a branch of a private research workspace and moved here with its full commit history,
keeping only the project's own files. Commit dates are part of the protocol record. Moving the history changed every
commit hash: hashes quoted in `research/`, in run records under `runs/` and in the OSF registrations are the **old**
hashes, and `COMMIT_MAP.tsv` maps each of them to the commit in this repository (commits that touched only workspace
files were dropped). Internal review notes and the planning documents derived from them are not included; references
to them in `research/` are left as they are. A few comments that referred to the private workspace were reworded
throughout the history, also inside the source copies embedded in the Kaggle bundle files
(`src/dce/remote/bundles/`), so the tree hash of an old bundle can differ from the `bundle_tree_hash` recorded by the
run that used it. Run records keep the original local paths and the ids of the Kaggle kernels that executed some runs;
the generic tool that submitted those kernels is not part of this repository, but every kernel runs a module of
`src/dce` that can be run locally with the same configuration.

## License

See `LICENSE`.
