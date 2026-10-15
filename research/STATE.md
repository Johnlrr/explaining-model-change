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

Test whether Δ-CE (minimal on-manifold witnesses on which exactly one model changes its decision) recovers the injected factor of a known model change, and separates drift from null pairs, better than CE differencing (B1) and gradient difference (B3). Acceptance rule: `docs/EVALUATION.md` §15.

Next gate: G-lit (17/10/2026). Spec freeze: 17/10/2026 23:59 ICT.

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
None yet. No experiment has been run.
```

---

## 7. Strongest current observations

Only theoretical results exist so far; none is empirical:

- Band containment (Lemma 1, spec §3.2, proposal §3.2): every witness satisfies h(z) ≤ −2κ and |h(z)| ≥ m_o(z) + κ. Proved.
- Null certificate (Proposition 2): under a Lipschitz bound on h, no witness exists near z0 when h(z0) + 2κ > 0. Proved for the idealised Lipschitz case only; not certified for deep networks.
- Linear closed form (Proposition 3): cases A and B. Proved.
- Direction recovery (Proposition 4): Δ-CE and CE differencing both recover span{PΔw} in the linear case. Proved, and it means Δ-CE is not expected to beat B1 on local linear changes. This is a stated prediction (P-tie), not a result.

Source: `docs/source/delta_ce_spec.md` §3, `research/proposals/delta_ce_proposal_v1.md` §3.

---

## 8. Next action

1. **G-lit (by 17/10):** read full texts listed in `research/LITERATURE.md` §12 (Artelt et al. NPL 2022 persistent explanations; Hammer & Hüllermeier 2021; Muschalik et al. 2022; Hinder et al. model-based drift explanations; citing papers of Artelt 2021 and DiG-IN). Record each in `research/LITERATURE.md`. Decide: continue, narrow to benchmark + H3, or Stop.
2. **Human confirmation** of D-002 (gates on validation, not test; before G0) and D-004 (venue and dates; before the paper) — see `research/DECISIONS.md`.
3. **G0 (by 20/10):** build Tier 0 renderer, drift data, classifiers (`src/dce/`), and the flip-sensitivity check on validation. CPU only.
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
Kaggle accounts: not verified (kaggle-gpu status: unverified)
Local compute: 16 CPU cores, no GPU; Tier 0 and Tier L run on CPU
Tier 1 Kaggle jobs used: 0 / 4
```

---

## 12. Current blockers

1. Human confirmation of D-002 (validation-based gates) is needed before G0 (20/10). The agent works with the conservative reading (validation-based gates) until then. D-004 (venue, deadlines) is needed before the paper, not before the experiments.
2. Kaggle GPU has not been verified on the connected account (smoke required before Tier 1).
3. ESANN 2027 special-session name and dates are taken from the source spec and not verified.

---

## 13. Immediate queue

- [ ] G-lit: full-text reading and `research/LITERATURE.md` update (by 17/10)
- [ ] Human decisions D-002, D-004
- [ ] `src/dce/` package skeleton; add it to `pyproject.toml` packages when code starts
- [ ] Tier L test `tests/unit/test_linear_theory.py` (H-004)
- [ ] Tier 0 renderer, drift builder, classifier (G0)
- [ ] Kaggle smoke for kaggle-gpu and kaggle-prep (before G2)

---

## 14. Current research phase summary

Phase: W0 (15–17/10): literature and kick-off. Specification and research memory are in place; no code or experiments yet.

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
