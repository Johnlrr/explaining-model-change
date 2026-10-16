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
2. **Evaluation code — written 2026-10-16, uncommitted** (D-028): `src/dce/eval/{metrics,stats,rules}.py`; 26 synthetic tests (`tests/unit/test_eval_metrics.py`, `test_eval_rules.py`): null coverage 94.5%, planted 0.15 detected, hand-worked metrics, Holm and B5/Route rules on constructed cases. Full unit suite 133 passed. Still to do before the lock: runtime split, B0 on unfiltered split, A1/B5 wiring, and the bootstrap RNG seed in the config. Lock = commit hash recorded here at candidate lock (§19).
   - **No exception** to §19: the Tier 0 full run waits for G2, the code lock, and the candidate lock (Research Lead, 2026-10-16).
3. **G2 (24/10), Tier 1:** Kaggle T4 trains VAE, shape oracle, f_old (5 seeds), f_new (11 pairs) via a D-026 bundle with `tier1_data.npz` (exp_0010) as kernel source; local CPU: oracle checks, σ_k, G0-Tier 1 (C1 for T3 from the start, v1.3 C3), r calibration Tier 1, GPU-vs-CPU logit gap.
4. **Final run (§19)** only after G2 passes and the candidate is locked: Tier 0 test ≈ 16.5 h, Tier 1 test ≈ 6–19 h, local 8 cores.

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

1. None.
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
