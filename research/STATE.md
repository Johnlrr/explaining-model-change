# STATE.md

> Purpose: Maintain a concise, current snapshot of the research project so an agent can resume work without reconstructing the entire history.
>
> Ownership: Mutable research state. The agent should update this file whenever meaningful project state changes.
>
> This file is a **snapshot, not a history log**. Historical evidence belongs in Git, `research/HYPOTHESES.md`, `research/DECISIONS.md`, `research/LITERATURE.md`, and `runs/`.

---

## 1. Project status

```text
Status: G1 PASSED (H4 exp_0007; found-rate exp_0009/0011; runtime projections exp_0011). Next: evaluation code + G2 (Tier 1)
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
None. exp_0012 (Tier 1 training) finished and all 18 checkpoints are local and hash-verified (vae.pt recovered by the author as vae.zip, renamed). exp_0013 (local G2 checks) done.
```

### Resume rule

If a Kaggle job is recorded here as in flight, resume by waiting on it with the harness (`docs/RESEARCH_PROTOCOL.md` §8, remote wait), not by resubmitting.

---

## 10. Test-set state

```text
Tier 0 test (5 000 renderer samples): untouched — NOT YET LOCKED (waiting for the author's decision on D-032, which must precede the lock)
Tier 1 test (MNIST 4/9 official test): untouched; Tier 1 dropped (v1.8 I6). Used only if D-032 (exploratory N0) is approved.
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
| 6 | Lock record (commit, config sha256, EVALUATION sha256) | pending D-032 |
| 7 | §20 reproduction plan | below |

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
