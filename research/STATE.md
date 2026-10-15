# STATE.md

> Purpose: Maintain a concise, current snapshot of the research project so an agent can resume work without reconstructing the entire history.
>
> Ownership: Mutable research state. The agent should update this file whenever meaningful project state changes.
>
> This file is a **snapshot, not a history log**. Historical evidence belongs in Git, `research/HYPOTHESES.md`, `research/DECISIONS.md`, `research/LITERATURE.md`, and `runs/`.

---

## 1. Project status

```text
Status: G1 — H4 passed, r frozen, pilot done (exp_0009): found-rate passes, RUNTIME FAILS; awaiting the author
Last updated: 2026-10-15
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
Status: proposed
Current experiment ID: none
Why this hypothesis is active: it is the first gate (G1). Without it no result can be trusted.
```

---

## 6. Latest experiment

```text
Experiment ID: exp_0009 (G1 pilot, Tier 0 validation, seed-0 pairs, N = 100, every method; commit 7c6401a)
Result: Δ-CE found-rate T1 0.81 (≥ 0.6, pass); runtime per pair at N = 100: T1 4.9, T3 4.8, T4 9.6 min (16 cores),
  N0 25.5, N1 24.9 min (8 cores) → runtime ≤ 5 min FAILS. F5: standard solver misses no witness within r.
Cause: SLSQP runs to maxiter (≈ 480 evaluations) on points without a witness (nulls, much of T4)
Relevant run evidence: runs/experiments/exp_0009/ ; research/NOTES.md §3
Previous: exp_0010 (Tier 1 data), exp_0008 (r = 0.5413), exp_0007 (H4 passes)
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

1. G-lit, Kaggle smoke, G0-Tier 0: done (G0 closed by exp_0004 under v1.3, D-020).
2. **Done for G1 so far:** r = 0.5413 frozen (exp_0008, 300/300 CE found, float64 gap 1.5e-6); pilot exp_0009 running.
   Tier 1 data prepared (exp_0010, `boinhbo/dce-tier1-data-v1`, D-025).
   **Open infra question for G2:** Tier 1 GPU jobs need the whole `dce` package, but the harness stages one file.
   Options: install from GitHub at a pinned commit (needs a push, the author's call) or a generated single-file bundle.
   **G1 pilot plan (validation only):** Tier 0 margin wrappers (float32 model for evaluation, float64 copy for SLSQP); `r` calibration (spec §6.3: 2 × median CE(f_new) distance, 300 validation points of T1 seed 0, then frozen); baselines B1, B1-det, B3, B4, B5, A1 (`src/dce/baselines/`); pilot seed 0, N = 100, all methods — read only found-rate (per condition) and runtime (≤ 5 min per pair); F5 completeness check. H4 done (exp_0007).
   Pilot adds the F5 completeness check (v1.5, D-023): 50 points of T1 seed 0, standard solver vs SLSQP from 50 random box starts.
   Pre-declared before G1 (D-021): Flip_k only for re-sampleable factors; Recovery|reach reported (descriptive). Pilot reads found-rate per condition; a low T3 found-rate does not justify changing `r`.
3. Then G2 (Tier 1 on Kaggle; T3-Tier1 uses C1 from the start, fallback bg > q85, then drop — v1.3 C3), G3.

## 9. In-flight remote jobs

```text
None.
```

### Resume rule

If a Kaggle job is recorded here as in flight, resume by waiting on it with the harness (`docs/RESEARCH_PROTOCOL.md` §8, remote wait), not by resubmitting.

---

## 10. Test-set state

```text
Tier 0 test (5 000 renderer samples): untouched
Tier 1 test (MNIST 4/9 official test): untouched
Status rule: a test split becomes `evaluated` only in the final procedure (EVALUATION §19), once per locked candidate.
```

---

## 11. Resource state

```text
GPU budget (HARD, 10 GPU-hours): used 0.0 / 10
Kaggle accounts: 3 connected (86.4 GPU-hours remaining in total before the smokes). kaggle-gpu verified: 2 × Tesla T4, offline, kernel_sources OK. Project GPU use so far: ≈ 0.01 h (smoke_0002, 14 s of compute plus startup)
Local compute: 16 CPU cores, no GPU; Tier 0 and Tier L run on CPU
Tier 1 Kaggle jobs used: 0 / 4
```

---

## 12. Current blockers

1. **G1 runtime criterion fails** (exp_0009): SLSQP on witness-less points dominates; D-024 says report, do not lower maxiter. Needs the author.
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

Phase: W1: G0-Tier 0 passed (exp_0004); G1 H4 passed (exp_0007, v1.5); G1 pilot next.

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
