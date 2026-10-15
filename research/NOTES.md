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

### exp_0001 — G0 on Tier 0 (validation, 2 000 points), 2026-10-15

| Condition | Pass | Key numbers (seeds 0/1/2[/3/4]) | Criterion |
|---|---|---|---|
| f_old accuracy | no | 0.925 / 0.978 / 0.983 / 0.982 / 0.972 | ≥ 0.95 all seeds |
| T1 | yes | ΔFlip_color 0.20 / 0.17 / 0.22 | ≥ 0.15 |
| T3 | no | subgroup acc (new labels) 0.35 / 0.26 / 0.10; agreement outside 0.91 / 0.94 / 0.92 | ≥ 0.85 and ≥ 0.95 |
| T4 | no | band acc (new labels) 0.76 / 0.78 / 0.34 | ≥ 0.80 |
| N0 | no | agreement 0.932 / 0.974 / 0.993 / 0.947 / 0.987; max abs ΔFlip ≤ 0.011 | ≥ 0.98 and ≤ 0.05 |
| N1 | no | agreement 0.903 / 0.979 / 0.982; max abs ΔFlip ≤ 0.018 | ≥ 0.98 and ≤ 0.05 |

Diagnosis (validation, from saved checkpoints, no extra training):

1. **Threshold jitter, not drift.** Null-pair disagreements sit at the decision threshold: 34–77% of them have |φ1 − 0.5| < 0.03, median |φ1 − 0.5| 0.013–0.043. A shift of 0.01 in φ1 is a 1.4% change of the log aspect ratio, below one pixel at 32 × 32. The ΔFlip_k part of the null criterion passes easily (≤ 0.018); only the agreement part fails. The same jitter lowers T3's "agreement outside the subgroup" (0.91–0.94).
2. **f_old is not converged / seed-sensitive.** Training loss plateaus near 0.28 at epoch 15, while the floor with 5% symmetric label noise is about 0.20. Seed 0 has error 0.47 in the band 0.02 ≤ |φ1 − 0.5| < 0.05 (its boundary is offset), so N0/N1 fine-tunes of seed 0 move the boundary back (f_new clean accuracy 0.982).
3. **Null is not a pure null (D-017 reading).** f_old trains on 5% noisy labels; N0 fine-tunes on noise-free labels, so N0 changes the training distribution, not only the data order.
4. **T3 is under-trained.** Fine-tune loss 0.53–0.60 after 3 epochs; the size × aspect interaction is not learned. T4 seed 2 also moves its boundary only partly (band acc 0.34).

### exp_0002 — G0 on Tier 0 after amendment v1.2 (knob 1 for T3/T4), 2026-10-15

| Condition | Pass | Key numbers (seeds 0/1/2[/3/4]) | Criterion |
|---|---|---|---|
| f_old accuracy | **yes** | 0.981 / 0.982 / 0.980 / 0.982 / 0.975 (final loss 0.276–0.285) | ≥ 0.95 |
| T1 | no | ΔFlip_color 0.134 / 0.120 / 0.148 (exp_0001: 0.17–0.22) | ≥ 0.15 |
| T3 (5 ep, lr 1e-3) | no | subgroup acc 0.87 / 0.82 / 0.59; DD outside 0.009 / 0.049 / 0.019 | ≥ 0.85; ≤ 0.05 |
| T4 (5 ep, lr 1e-3) | no | band acc 0.18 / **0.98** / 0.38; DD 0.000 / 0.037 / 0.000 | ≥ 0.80 |
| N0 | **yes** | DD 0.000 all; raw agreement 0.968–0.992; max abs ΔFlip ≤ 0.010 | DD ≤ 0.02, ΔFlip ≤ 0.05 |
| N1 | **yes** | DD 0.000 all; raw agreement 0.979–0.990; max abs ΔFlip ≤ 0.012 | same |

Reading:
- B1/B2/B3 fixed what they targeted: all f_old seeds pass, and null pairs have zero decisive disagreement (raw disagreement is threshold jitter only).
- T1 weakened below 0.15. Likely cause: B2 label noise dilutes the colour–label association the shortcut relies on (colour is drawn from the noise-free label, then 5% of labels flip), and the more converged f_old (cosine) moves less in 3 epochs at lr 3e-4; fine-tune loss plateaus at 0.28–0.29.
- T4 is bimodal: seed 1 moves its boundary fully (0.98), seeds 0 and 2 not at all (DD 0). The shifted band is ≈ 9% of the data, so the loss gain from moving is small; 5 epochs at lr 1e-3 are not always enough.
- T3 is close for seeds 0/1 and short for seed 2 (0.59); fine-tune loss still falling (0.37–0.50 at epoch 5).

Options for the user (resolved: author chose c.1 + c.2 and replaced c.3 by the DD criterion; amendment v1.2, D-018):

- (a) Stop, as spec §9 allows.
- (b) Apply only the pre-declared knob (T3/T4 epochs 3 → 5). This cannot fix N0/N1 or f_old seed 0, so G0 still fails.
- (c) Post-G0 amendment v1.2 (logged "after seeing G0 results"; G0 is a gate, not a main test):
  1. f_old training: 30 epochs with cosine learning-rate decay (accuracy requirement ≥ 0.95 unchanged);
  2. all adaptation sets carry the same 5% label noise as the training split, so N0 differs from f_old only in data order and the drifts differ only in the drift (revises D-017);
  3. agreement criteria (N0/N1 ≥ 0.98, T3 outside ≥ 0.95) measured on points with |φ1 − 0.5| ≥ 0.05, outside the threshold band; thresholds unchanged; ΔFlip criteria unchanged;
  4. then the pre-declared knob for T3/T4 if they still fail.
  Cost: one rerun, ≈ 35 min CPU.

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
