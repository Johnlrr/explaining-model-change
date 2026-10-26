# STATE.md

> Purpose: Maintain a concise, current snapshot of the research project so an agent can resume work without reconstructing the entire history.
>
> Ownership: Mutable research state. The agent should update this file whenever meaningful project state changes.
>
> This file is a **snapshot, not a history log**. Historical evidence belongs in Git, `research/HYPOTHESES.md`, `research/DECISIONS.md`, `research/LITERATURE.md`, and `runs/`.

---

## 1. Project status

```text
D-048 (2026-10-26): external review (argument-review + conference-reviewer) checked against saved outputs. exp_0041
  (type N/O/N∪O on Threshold, new data): N 0.449, O 0.551, N∪O 1.000. exp_0042 (CF-diff thresholded at matched FPR, saved
  outputs): flags none below 2κ at FPR 0. paper/main.tex: W6 number/label fixes, DeepXplore → No-prox, Table 3 column K
  (D/R), type-N statement, CI caveat, Theorem 1 of [1] stated; 6 pages, 0 overfull. Follow-ups, not done: a "where" measure
  (A1), Δ-CE Recovery vs d_o(sin θ+cos θ)/d_n (A5). NEXT: author reads the diff; ESANN deadline 18/11.
D-049 (2026-10-26): A1 → writing only (a "where" metric would be definitional for Δ-CE); A5 → exp_0043: the linear
  recovery condition at the query predicts Δ-CE witness direction on CNN rotations (Rec 0.64 vs 0.04, +0.60 [0.57, 0.62]).
  Contribution (iii) rewritten accordingly; paper 6 pages, 0 overfull. NEXT: author reads the diff.
2026-10-26: ESANN 2027 author guidelines (esann.org/node/5): "The review process is single blind"; ≤ 6 pages incl.
  figures and references; supplementary material may be linked from the paper and is not reviewed. Hence NO anonymous
  copy is needed (D-046's .work/anon_release is obsolete). NEXT (author): replace "[author names and affiliation withheld
  for review]" with names and affiliations, and "[repository link]" with the public repository URL (make it public,
  or link a release); then recompile (check still 6 pages) and submit by 18/11/2026.
D-050 (2026-10-26): second review (6/10) handled: exp_0044 (zero-FP detection on fine-tuned offsets: Δ-CE 1.0 in 6/7
  pairs, CF-diff 0 in 7/7), exp_0045 (pair counts 4/4); paper reframed, splits A/B/C named; 6 pages, page images checked.
  Special session option (distributional change, Bielefeld) recorded for the author.
v1.10 (D-043, 2026-10-20) APPROVED by the author (with changes; see D-043). H-010 = H7c (logit offset, detection
  threshold), H-011 = H8 (registered cell (z*, z0)). Dev on validation: exp_0030 (H8: gap −0.19 → +0.01 at (z*, z0)),
  exp_0031 (T-C cost); smoke_0018 (T-A pipeline on validation) passed. exp_0032 DONE (0e4ab96): U1–U3 pass; H7c before Holm met (B1-det 0.961 [0.955, 0.967]); Δ-CE sharp threshold
  (0 witnesses below 2κ; 100% of lowered-class queries above). RUNNING: exp_0033 (H8, 24 pairs, two detached CPU shards;
  resume: `uv run python -m dce.cli.v110_test tb-run --config configs/experiments/exp_0033.yaml --out
  runs/experiments/exp_0033 --shard i/2`). NEXT: tb-analyze → holm → exp_0034 (T-C; G4 will be empty, since every
  lowered-class query has a witness) → exp_0035 (T-D) → summary to the author. Locked H5d −0.352 = geometry −0.168 +
  found-rate −0.184 (NOTES). ESANN deadline 18/11.
v1.10 COMPLETE; author approved the summary with corrections (2026-10-21). exp_0036 T1 2×2 (D-045, descriptive):
  gap −0.166 → −0.024 with the normal at the witness. paper/main.tex edited per summary §5 (abstract, contributions,
  Prop. 2 remark, protocol, H8 + logit-offset/abstention paragraphs, discussion, limitations; Section 5.3 → 4.2 ref
  fix); 6 pages, 0 overfull/underfull. B5 citation [5] verified correct (no change). Author's 7 corrections applied
  (545fe9c, 6 pages).
  NEXT: co-authors read and adjust wording; content judged submission-ready by the author (2026-10-21).
  Readability rewrite (09e25c2) and the author's final round (348c459): 6 pages, 0 overfull. D-046: protocol-log
  audit passed (exceptions disclosed), DeepXplore = Δ-CE without proximity, anonymous copy builder `dce.release`
  (.work/anon_release, 104 MB with checkpoints, tests pass). NEXT: author checks ESANN rules on anonymous links,
  then upload and replace "[anonymous link]" in paper/main.tex; co-author page to be refreshed after approval.
v1.9 (D-036, branch research/delta-ce-v19, 2026-10-19): amendment committed 1997732; code 19d7331.
  Pilot exp_0024 (pair 40: α=0.25, t=0.56, seed 0; validation, qualification only) FAILS §2.3:
  α̂ = 0.233 (ok), accuracy 0.929 (ok), t̂ = 0.630 (|t̂−t| = 0.070 > 0.02). Grid NOT started (v1.9 §6).
  → v1.9.1 (D-037, 67de538): offset checks replace the t̂ tolerance. exp_0025 (679d6a2): 32 pairs fine-tuned and
  qualified on validation; stop rule 2/32 (pairs 22, 23 fail 2a) → not triggered; null cell DD 0.
  → D-038 (2b reading; wall-hours; step 3 authorised), D-039 (test statistics, declared before the run).
  exp_0026 DONE (32 pairs, CPU, locked solver; consistency passed). H-008: H5a, H5b supported; H5c, H5d not
  supported (Holm m = 4; D-041 analysis-exclusion bug fixed to D-038, superseded output kept). See research/results_v19.md.
  Stage B (D-040, D-042): 5/5 Adult null pairs qualify; r_B = 0.8693; H6 CONFIRMED (exp_0028: Δ-CE N∪O 0.000 on all,
  B1-det 0.977–1.000) → H-009. Null-offset diagnostic exp_0029 (descriptive): offset alone does not separate witness /
  no witness at |offset| ≈ 0.03–0.04 (recipe differs). Next: the author rewrites paper/ (6 pages, map replaces Fig. 1).
  GPU used by v1.9: 0 h.
MERGED 2026-10-20 into research/delta-ce: research/delta-ce-p5-check (exp_0023 P5 numerical check, Figure 1 code,
  paper history) then research/delta-ce-v19. Paper: paper/main.tex = esann_main_v5.tex, 6 pages, 0 overfull, Fig. 2 =
  exp_0026 map at print size; commit "paper: ESANN submission version (v5)". Next: the author's read-through of the PDF;
  submission before the ESANN deadline. No experiment pending.
Status: TIER 0 FINAL DONE (exp_0017) — H2a NOT supported → Route B (benchmark + analysis). Previously: G1 PASSED (H4 exp_0007; found-rate exp_0009/0011; runtime projections exp_0011). Next: evaluation code + G2 (Tier 1)
Last updated: 2026-10-16
Current commit: see `git log -1` (kick-off commit on research/delta-ce)
Current branch: research/delta-ce (from main)
```

---

## 2. Current objective

Test whether Δ-CE (minimal on-manifold witnesses on which exactly one model changes its decision) recovers *which* factor a known model change affected, better than CE differencing (B1) and gradient difference (B3), with DIG-style (B5) and DeepXplore-style (A1) searches reported alongside. Detection (H3) is descriptive. Acceptance: Route A = H4 ∧ (H1 ∨ H2a ∨ H2b), `docs/EVALUATION.md` §15.

G0-Tier 0 closed 2026-10-15 (deadline was 20/10). Next gate: G1 (H4 + solver + pilot, validation). Spec frozen 2026-10-15 (D-016); amendments v1.2 and v1.3 are post-G0-result gate changes, v1.3 is the last G0 criterion change.

---

## 3. Baseline status

```text
Baseline: B1 — CE differencing (Artelt et al., Definition 3). Secondary: B1-det, B3, B4.
Status: not_started
Best reproduced result: none
Primary metric: Recovery@r (EVALUATION §2)
Relevant experiment ID: none
Commit: none
Notes: baseline to be re-implemented under src/dce/baselines/ after the solver (G1).
```

---

## 4. Current best result

```text
None yet
```

---

## 5. Active hypothesis

```text
Hypothesis ID: H-004 (solver correctness; a verification step, not a scientific claim)
Short statement: the Δ-CE solver reproduces the linear closed form of Proposition 3 and the cos-alignment of Proposition 4 on Tier L.
Status: supported (exp_0007; see research/HYPOTHESES.md H-004)
Current experiment ID: exp_0007
Why this hypothesis is active: it was the first gate (G1), now passed. Without it no result can be trusted.
```

---

## 6. Latest experiment

```text
Experiment ID: exp_0011 (G1 pilot rerun after speed-up (A), 8 cores; commit e9f8e15)
Result: found-rates identical to exp_0009 for every pair and method; 1.7 h wall (was 3.4 h)
Projections (v1.6 G1): Tier 0 final 16.5 h ≤ 36; Tier 1 evaluation 5.8 h (conservative 18.6 h) ≤ 36; GPU ≪ 10 h
Conclusion: G1 passes. Runtime split: Δ-CE phase-2 CPU 4.5–5.3 s/point with a witness, 16–20 s without
Relevant run evidence: runs/experiments/exp_0011/ ; research/NOTES.md §3 ; D-027
```

## 7. Strongest current observations

Only theoretical results exist so far; none is empirical:

- Band containment (Lemma 1, spec §3.2, proposal §3.2): every witness satisfies h(z) ≤ −2κ and |h(z)| ≥ m_o(z) + κ. Proved.
- Null certificate (Proposition 2): under a Lipschitz bound on h, no witness exists near z0 when h(z0) + 2κ > 0. Proved for the idealised Lipschitz case only; not certified for deep networks.
- Linear closed form (Proposition 3): cases A and B. Proved.
- Direction recovery (Proposition 4): Δ-CE and CE differencing both recover span{PΔw} in the linear case. Proved, and it means Δ-CE is not expected to beat B1 on local linear changes. This is a stated prediction (P-tie), not a result.

Source: `docs/source/delta_ce_spec.md` §3, `research/proposals/delta_ce_proposal_v1.md` §3.

---

## 8. Next action

1. G-lit, Kaggle smoke, G0-Tier 0, G1: done (G1 passed 2026-10-16, exp_0007/0009/0011).
2. **Evaluation code — written and committed 2026-10-16 (6e4fe41)** (D-028): `src/dce/eval/{metrics,stats,rules}.py`; 26 synthetic tests (`tests/unit/test_eval_metrics.py`, `test_eval_rules.py`): null coverage 94.5%, planted 0.15 detected, hand-worked metrics, Holm and B5/Route rules on constructed cases. Full unit suite 133 passed. Still to do before the lock: runtime split, B0 on unfiltered split, A1/B5 wiring, and the bootstrap RNG seed in the config. Lock = commit hash recorded here at candidate lock (§19).
   - **No exception** to §19: the Tier 0 full run waits for G2, the code lock, and the candidate lock (Research Lead, 2026-10-16).
3. **G2 closed: Tier 1 dropped (exp_0016, see §12).** Next: lock the Tier 0 candidate (evaluation-code commit + config hash in §10), then the Tier 0 final run (≈ 16.5 h, local). Original plan:  Kaggle T4 trains VAE, shape oracle, f_old (5 seeds), f_new (11 pairs) via a D-026 bundle with `tier1_data.npz` (exp_0010) as kernel source; local CPU: oracle checks, σ_k, G0-Tier 1 (C1 for T3 from the start, v1.3 C3), r calibration Tier 1, GPU-vs-CPU logit gap.
4. **Final run (§19)** only after G2 passes and the candidate is locked: Tier 0 test ≈ 16.5 h, Tier 1 test ≈ 6–19 h, local 8 cores.

## 9. In-flight remote jobs

```text
None. exp_0020 (D-032) analyzed; exp_0021 G0-T5 failed → D-035 stopped (no test set generated). Summary: research/results_summary.md.
```

### Resume rule

If a Kaggle job is recorded here as in flight, resume by waiting on it with the harness (`docs/RESEARCH_PROTOCOL.md` §8, remote wait), not by resubmitting.

---

## 10. Test-set state

```text
Tier 0 test (5 000 renderer samples): EVALUATED once (exp_0017, 2026-10-17; run 17.05 h from 39753b4 = lock ede7374 + STATE only; analysis 2026-10-17)
  candidate / code commit: ede7374efc36c30c3c9816af61c18679f4aeaec6 (final program 24b7651; evaluation code 6e4fe41)
  configs/experiments/exp_0017.yaml sha256 80c34f2e2b52a7573865bc47a425b781b70c78645478b101f660b3ba72462bae
  configs/tiers/tier0.yaml          sha256 35ea879391bd66749d51083d394e8f9c9d573f69cdbf21b0cb0a45e7b4e231a4
  docs/EVALUATION.md                sha256 c4af3580ac59967523c547500a576b1f0e165bdaf888085b73236838e69bc7ac
  run: exp_0017 (final), 8 threads; any later change to src/, configs/ or docs/EVALUATION.md before analysis voids the lock
Tier 1 test (MNIST 4/9 official test): untouched; Tier 1 dropped (v1.8 I6). Touched once by the D-032 exploratory N0 analysis, after the Tier 0 final run.
Status rule: a test split becomes `evaluated` only in the final procedure (EVALUATION §19), once per locked candidate.
```

### Pre-lock checklist (Tier 0), 2026-10-16

| # | Item | Status |
|---|---|---|
| 1 | EVALUATION/GOAL follow the Tier 1 drop | done, dcf3420 |
| 2 | Wiring + final program: MethodOutput, B0 on the unfiltered split, runtime split by witness, schema-only tests, run/analyze stages | done, 6e4fe41 (records) + 24b7651 (final program); smoke on validation 2 pairs × 3 points |
| 3 | Tie rule + zero-vector test (T4, k* = 0) | done, `tests/unit/test_eval_metrics.py::test_zero_vector_and_ties_have_no_top_factor_and_score_zero` |
| 4 | Bootstrap seeds in the config | done, `configs/experiments/exp_0017.yaml` (`bootstrap_seeds`, checked against the code) |
| 5 | r frozen | `configs/tiers/tier0.yaml` r = 0.5412657260894775 (exp_0008) |
| 6 | Lock record (commit, config sha256, EVALUATION sha256) | done, see above (D-032 declared first, ede7374) |
| 7 | §20 reproduction | done: exp_0018 bit-identical to exp_0017 on pairs 3 and 9 (2026-10-18) |

### §20 reproduction plan (declared before the final run)

After the final run and analysis: from a clean checkout of the lock commit, in a fresh environment, rerun pair 3 (T3 seed 0, drift) and pair 9 (N0 seed 0, null) with the same config (`run_kind: final`, `pair_ids` restricted to the pair), and compare their pair-level Recovery@r (pair 3) and found-rate (both) with the final run; they must agree within the pair's paired-bootstrap 95% CI (they are expected to be identical: the pipeline is deterministic on CPU). Also check B1, B3 and B5 come from the same code path (they do: one program) and the checkpoint hashes (load_pair verifies them).

### Launch plan

`setsid nohup uv run python -m dce.cli.final_tier0 run --config configs/experiments/exp_0017.yaml --out runs/experiments/exp_0017 --final` (survives the session; resumable per pair), then `analyze`. Expected ≈ 16.5 h on 8 threads. No metric is printed during the run.

## 11. Resource state

```text
GPU budget (HARD, 10 GPU-hours): used 0.0 / 10
Kaggle accounts: 3 connected (86.4 GPU-hours remaining in total before the smokes). kaggle-gpu verified: 2 × Tesla T4, offline, kernel_sources OK. Project GPU use so far: ≈ 0.03 h (smoke_0002, smoke_0011, smoke_0012, exp_0012: ≈ 60 s of compute plus startup each; measured ≈ 56 s for the full Tier 1 training)
Local compute: 16 CPU cores, no GPU; Tier 0 and Tier L run on CPU
Tier 1 Kaggle jobs used: 1 / 4 (exp_0012; smoke_0011/0012 are infrastructure smokes)
```

---

## 12. Current blockers

1. **Tier 1 dropped (G2 round two exp_0016, pre-declared rule v1.8 I6 / v1.7 H6).** VAE β = 0.1, shape oracle, all f_old ≥ 0.98, N0 pass, but G0 T1 fails in every variant (ΔFlip_color ≤ 0.081) and T3 too. Consequence: Route A = H4 ∧ H2a, Holm m = 1, no B5 solver claim; H1 and H2b untested. No round three. Awaiting the author's acknowledgement before the Tier 0 candidate lock.
2. None from the user: spec amendment v1.1 (A1–A6) applied on 2026-10-15 (D-011–D-015).
3. Residual G-lit risk: NPL 2022 journal text and Muschalik 2022 not readable (access blocked); arXiv v2 of Artelt covers the same scope.

---

## 13. Immediate queue

Plan to G0 (deadline 20/10), Tier 0, local CPU:

- [x] Environment: `research` extra in `pyproject.toml` (torch CPU 2.14 locally; Kaggle image has torch 2.11).
- [x] Renderer `src/dce/data/renderer.py` + `tests/unit/test_renderer.py` (render → read factors back: colour and background exact; aspect/size/posx via α moments within measured bounds).
- [x] Drift data `src/dce/data/drift.py` (clean splits, T1/T3/T4/N0/N1 adaptation sets, 17 pair ids) + tests.
- [x] Classifier `src/dce/models/cnn.py` (≈ 56k parameters) + tests.
- [x] G0 check `src/dce/eval/g0.py`, CLI `src/dce/cli/g0_tier0.py`; smoke_0003 passed the pipeline.
- [x] exp_0001: full G0 — FAILED (see §6).
- [x] Amendment v1.2 implemented (cosine f_old, noisy adaptation sets, DD, per-condition recipe); smoke_0004 passed the pipeline.
- [x] exp_0002: f_old, N0, N1 pass; T1, T3, T4 fail.
- [x] exp_0003: T1 (knob 1) and T4 (knob 2) pass; T3 fails with no knob left → waiting for the author.
- [x] Amendment v1.3 (T3 criterion C1) and exp_0004 re-evaluation: G0-Tier 0 passes.
- [ ] Then G1: Tier L test (H4), Δ-CE solver, ordinary CE, r calibration, pilot (validation).

## 14. Current research phase summary

Phase: W1 → W2: G0-Tier 0 and G1 passed; evaluation code and G2 (Tier 1) next.

---

## 15. State update rules

Update this file after each meaningful change: gate passed or failed, hypothesis status change, experiment completed, test split touched, Kaggle job submitted or completed, blocker raised or resolved.

---

## 16. Completion checklist

- [x] Current objective stated.
- [x] Baseline status stated.
- [x] Test-set state stated.
- [x] Next action stated.
- [x] Blockers stated.
