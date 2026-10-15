# STATE.md

> Purpose: Maintain a concise, current snapshot of the research project so an agent can resume work without reconstructing the entire history.
>
> Ownership: Mutable research state. The agent should update this file whenever meaningful project state changes.
>
> This file is a **snapshot, not a history log**. Historical evidence belongs in Git, `research/HYPOTHESES.md`, `research/DECISIONS.md`, `research/LITERATURE.md`, and `runs/`.

---

## 1. Project status

```text
Status: initializing
Last updated: 2026-10-15
Current commit: see `git log -1` (kick-off commit on research/delta-ce)
Current branch: research/delta-ce (from main)
```

---

## 2. Current objective

Test whether Δ-CE (minimal on-manifold witnesses on which exactly one model changes its decision) recovers *which* factor a known model change affected, better than CE differencing (B1) and gradient difference (B3), with DIG-style (B5) and DeepXplore-style (A1) searches reported alongside. Detection (H3) is descriptive. Acceptance: Route A = H4 ∧ (H1 ∨ H2a ∨ H2b), `docs/EVALUATION.md` §15.

Next gate: G0 (20/10/2026). Spec freeze: 17/10/2026 23:59 ICT.

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
Experiment ID: exp_0002 (gate G0, Tier 0, validation; amendment v1.2; commit 1755879, clean tree)
Run type: full gate run (CPU, 1 644 s)
Result: G0 not yet passed — f_old all seeds ≥ 0.975; N0, N1 pass (DD = 0); T1, T3, T4 fail
Validity: valid
Primary observation: v1.2 fixed f_old and the nulls; T1 weakened (ΔFlip_color 0.12–0.15); T4 bimodal across seeds;
  T3 close for two seeds
Conclusion: continue with pre-declared knobs (exp_0003), reusing the exp_0002 f_old checkpoints
Relevant run evidence: runs/experiments/exp_0002/ ; research/NOTES.md §3
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

1. **G-lit: done 2026-10-15 (two rounds), passed.** Definition-level novelty downgraded to `uncertain` (DIG, DeepXplore); benchmark, margin-κ witness for updates, and theory remain `candidate` (`research/LITERATURE.md` §2, §6).
2. **Kaggle smoke: done** (smoke_0001 prep, smoke_0002 T4 GPU). Nothing further before G2.
3. **G0 (by 20/10):** build Tier 0 renderer, drift data, classifiers (`src/dce/`), and the Flip_k / accuracy check on 2 000 clean validation points (D-011). CPU only.
4. Then G1 (H4 + pilot), G2 (Tier 1 on Kaggle), G3.

---

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

1. None. G0 failure of exp_0001 resolved by amendment v1.2 (D-018); exp_0002 running.
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
- [ ] exp_0002: G0 with v1.2; T3/T4 at knob 1. If it fails: exp_0003 with knob 2 (T3/T4 epochs 10); then Stop or back to the author. If any f_old seed < 0.95: back to the author.
- [ ] If a condition fails: apply the pre-declared knob (spec §4.4) in a new experiment (exp_0002), max 2 per condition.
- [ ] Then G1: Tier L test (H4), Δ-CE solver, ordinary CE, r calibration, pilot (validation).

## 14. Current research phase summary

Phase: W1: spec frozen 2026-10-15; G0 implementation done, full G0 run (exp_0001) next.

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
