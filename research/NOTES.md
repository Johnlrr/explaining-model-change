# NOTES.md

> Purpose: Temporary observations, questions, and ideas that are not yet formal conclusions.
>
> Ownership: Mutable research memory. Promote items to STATE, HYPOTHESES, DECISIONS, or LITERATURE when they are settled.

---

## 1. Usage principles

Raw observations and unresolved items only. Do not store final results here.

---

## 2. Inbox

### Kick-off inconsistencies found while mapping the spec (2026-10-15)

- Spec §9 (G0, G1, G2) and §7 use test points for development checks. Resolved as D-009: the user chose to follow the spec; restrictions in `docs/EVALUATION.md` §6.
- Spec §12–§13 layout and log path differ from AGENTS.md. Resolved as D-001.
- Spec §8 H4 tolerance is 1e-3; the proposal used 1e-4. Spec governs (D-005).
- Spec G0 criterion (Flip_color difference ≥ 0.15 for T1) is not the proposal's sensitivity criterion (S_k difference ≥ 0.20). Spec governs.

### Proposals to the user before the spec freeze — resolved 2026-10-15 by spec amendment v1.1 (D-011–D-015)

1. **D-009 wording is wrong about the spec.** The spec names the test split only for G0 (2 000 clean test points, §9), the f_old accuracy requirement (§5.1), σ_k (§5.3), and the final evaluation (§7, B4, plausibility). It does **not** name a split for the G1 pilot or for G2. D-009 and `docs/EVALUATION.md` §6 currently say the pilot and G2 use test "per the spec". Proposed: pilot and G2 on validation (the protocol default where the spec is silent).
2. **H3 needs a disagreement-rate baseline (B0).** Hammer & Hüllermeier (L-004) define model change as the discrepancy volume. The G0 null criterion (agreement ≥ 0.98 on test) is that same statistic, measured on the same test points. Proposed: add B0 = Monte Carlo disagreement rate on the evaluation points as a descriptive comparator for H3, and report H3 as conditional on G0-qualified pairs.
3. **Positioning:** the definition must be cited as an adaptation of DIG (L-022) and DeepXplore (L-023); see `research/LITERATURE.md` G-lit round 2.

### Repository state observed at kick-off

- `runs/smoke/` held about 40 untracked folders from an earlier, failed topic. Deleted on 2026-10-15 at the user's instruction (only `.gitkeep` remains).
- `src/` has only `.gitkeep`. `pyproject.toml` packages only `tools*`; add `src/dce` when code starts.
- `docs/RESEARCH_PROTOCOL.md`, `docs/PROMPTS.md`, and `docs/acceptance/` are generic and were left unchanged.
- No local GPU. `nvidia-smi` is not available.
- No Kaggle credential files were found under `~/.kaggle`. Accounts are managed through the repository harness (`tools/kaggle`).

### Points to verify (G-lit and G0)

- The spec's claim that Artelt et al. set aside the disagreement region as "may be small" (spec §2 point 1). Read the full text before using it in the paper.
- The 6-page limit, template, and special session (D-004).
- Whether `torchvision` MNIST download works in kaggle-prep (Internet ON). Check at the smoke run.
- The `r` calibration uses `2 × median` of the CE distance of `f_new` on T1 seed 0 validation points. Check that the CE search finds a valid point for most validation points before fixing `r`.

---

## 3. Experimental observations

None yet.

---

## 4. Questions

1. Does the tie in Proposition 4 also hold for the constrained solver when λ is decayed? (H-005)
2. Should N1 (virtual drift) use the O direction too? The spec reports O only for N0 and N1, so yes; confirm at G0.
3. Is B4 (permutation importance) meaningful for a 5-factor renderer where factors are sampled independently? Consider as secondary.

---

## 5. Candidate hypotheses

Captured in `research/HYPOTHESES.md`. Nothing new here yet.

---

## 6. Debug and implementation observations

None yet.

---

## 7. Analysis snippets

None yet.

---

## 8. Things to verify

- [ ] ESANN 2027 special session and dates (D-004).
- [ ] Morpho-MNIST: the package installs or the skimage re-implementation is used (spec §4.3; the spec allows either).
- [ ] Shape-oracle accuracy thresholds on real images (≥ 98%) and reconstructions (≥ 95%) at G2.
- [ ] The VAE checkpoint size is below 5 MB (EVALUATION §12).

---

## 9. Promotion queue

Empty.

---

## 10. Promotion rules

Follow the template rules.

---

## 11. Source references

- `docs/source/delta_ce_spec.md` (spec v1.0)
- `research/proposals/delta_ce_proposal_v1.md` (proposal v1, superseded where they differ)
- `docs/GOAL.md`, `docs/EVALUATION.md`, `docs/CONSTRAINTS.md`, `docs/ENVIRONMENTS.md`

---

## 12. Confidence labels

Use the template labels (`low`, `medium`, `high`).

---

## 13. Cleanup policy

Delete items once promoted or resolved. Keep open questions.

---

## 14. What not to store

Final results, decisions, or literature conclusions. Those belong in the other research files.

---

## 15. Suggested lightweight note format

```text
### <short title>
- Observed:
- Confidence: low | medium | high
- Next: <action or promotion target>
```

---

## 16. End-of-session cleanup

Before ending a session: move resolved items out, update `research/STATE.md` §8 (next action).

---

## 17. Completion checklist

- [x] Kick-off inconsistencies recorded.
- [x] Repository observations recorded.
- [ ] Items in §8 verified.
