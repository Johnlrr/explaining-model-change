# DECISIONS.md

> Purpose: Record important long-term technical and research decisions, with reasons, so a fresh agent does not re-open them.
>
> Ownership: Mutable research memory. Decisions with `pending confirmation` need a human answer before they are relied on.

---

## 1. Decision index

| ID | Short title | Status | Date | Related hypotheses | Related experiments |
|---|---|---|---|---|---|
| D-001 | Map the delta spec onto the AGENTS.md layout | active | 2026-10-15 | all | — |
| D-002 | Development gates use validation, not test | reversed by D-009 | 2026-10-15 | H-001–H-003 | G0, G1, G2 |
| D-003 | Contribution type and primary acceptance rule | partially superseded by D-012, D-014 | 2026-10-15 | H-001–H-003 | — |
| D-004 | Venue, dates, and special session | active, confirmed by the user | 2026-10-15 | — | — |
| D-005 | The spec supersedes the proposal where they differ | active | 2026-10-15 | H-001–H-006 | — |
| D-006 | Environment split: Tier L and Tier 0 local CPU; Tier 1 Kaggle GPU | active | 2026-10-15 | H-001–H-003 | exp_TBD |
| D-007 | Baselines are re-implemented, no external code | active | 2026-10-15 | H-001, H-002 | — |
| D-008 | Research branch and kick-off commit | active | 2026-10-15 | — | — |
| D-009 | Gates follow the spec (test split used for gate statistics, with restrictions) | superseded by D-011 | 2026-10-15 | H-001–H-003 | G0, G1, G2 |
| D-010 | Tier 1 runs on Kaggle T4; RTX not used | active | 2026-10-15 | H-001–H-003 | smoke_0001, smoke_0002 |
| D-011 | All gates and calibrations on validation; test once (amendment A1) | active | 2026-10-15 | H-001–H-003 | G0, G1, G2 |
| D-012 | H3 descriptive with B0; acceptance Route A = H4 ∧ (H1 ∨ H2a ∨ H2b); B′ merged into B (A2, A3) | active | 2026-10-15 | H-001–H-004 | — |
| D-013 | Non-local baselines B5 (DIG-style) and A1 (DeepXplore-style), outside Holm, with a pre-declared claim rule (A4) | active | 2026-10-15 | H-002, H-006, H-007 | — |
| D-014 | Positioning and contribution order (A5) | active | 2026-10-15 | — | — |
| D-015 | NPL 2022 residual risk accepted (A6) | active | 2026-10-15 | — | — |
| D-016 | B0 global vs local wording, full B5 claim rule (A7); spec frozen | active | 2026-10-15 | H-003, H-007 | — |
| D-017 | Tier 0 implementation readings where the spec is silent | active | 2026-10-15 | — | exp_0001 |

Allowed status values: active, superseded, reversed.

---

## 2. What belongs here

Decisions that affect future work beyond one experiment: layout, test policy, baselines, environment, scope, deadlines.

---

## 3. Decision template

```text
## D-XXX — <title>

### Status
### Context
### Decision
### Rationale
### Evidence
### Alternatives considered
### Consequences
### Revisit condition
### Supersedes / superseded by
```

---

## D-001 — Map the delta spec onto the AGENTS.md layout

### Status
active

### Context
The source spec (`docs/source/delta_ce_spec.md` §12–§13) proposes its own layout (`dce/`, `scripts/`, `results/`, `logs/decisions.md`). AGENTS.md requires `src/`, `configs/`, `runs/`, `research/`, and no ad-hoc top-level source roots. The user asked to follow AGENTS.md and the existing repository conventions.

### Decision
| Spec path | Used path |
|---|---|
| `dce/` package | `src/dce/` (package to be added to `pyproject.toml` when code starts) |
| `dce/configs/*.yaml` | `configs/tiers/{tierL,tier0,tier1}.yaml` (shared parameters) and `configs/experiments/exp_NNNN.yaml` (one per run) |
| `dce/scripts/*.py` | `src/dce/cli/*.py` (no top-level `scripts/`) |
| `dce/tests/test_linear_theory.py` | `tests/unit/test_linear_theory.py` |
| `results/{tier}/{cond}_s{seed}.json` | `runs/experiments/exp_NNNN/metrics.json` (per run, per pair entries inside) |
| `logs/decisions.md` (spec §13) | `research/DECISIONS.md` |
| Spec documents | `docs/source/delta_ce_spec.md` (read-only) |
| Proposal | `research/proposals/delta_ce_proposal_v1.md` (historical, superseded by D-005) |

Smoke runs go to `runs/smoke/smoke_NNNN/`.

### Rationale
AGENTS.md defines the authoritative source root and evidence layout. Following it keeps this project comparable to other branches and avoids duplicate decision logs.

### Evidence
AGENTS.md "Implementation discipline" and "Git and reproducibility"; `docs/RESEARCH_PROTOCOL.md` §11.

### Alternatives considered
Keeping the spec layout as-is. Rejected: it creates a second source root and a second decision log.

### Consequences
Paths in the spec §12–§13 are read as the mapping above. Checksums and hashes in the spec are unchanged.

### Revisit condition
If the repository convention changes.

### Supersedes / superseded by
None.

---

## D-002 — Development gates use validation, not test

### Status
reversed by D-009 (the user chose to follow the spec, 2026-10-15)

### Context
The source spec applies several development checks to test data: G0 on 2 000 clean test points (spec §9); G1 pilot with N = 100 points from the test set (§9); G2 shape-oracle accuracy on test images (§9); and the seeds for the evaluation points are drawn from test (§7). `docs/EVALUATION.md` §5–§6 forbids test information from flowing into development, and the spec's own G0 knobs (≤ 2 adjustments) would then be chosen on test.

### Decision
Development checks (G0, G1, G2, pilot, r calibration, σ_k normalisation, shape-oracle qualification) use the validation split. The test split is used only in the final procedure (`docs/EVALUATION.md` §19). Point sampling for the pilot uses validation with seed 5000 + pair_id. The paper-facing evaluation points still use test with seed 5000 + pair_id.

### Rationale
The blindness rule in EVALUATION §6 is explicit; the spec's test-based gates would make the final test non-blind. The validation split is large enough (5 000 Tier 0; ~10% of train for Tier 1) for every gate.

### Evidence
`docs/EVALUATION.md` §5–§6; `docs/source/delta_ce_spec.md` §7 and §9.

### Alternatives considered
(a) Keep test for gates and declare them as "phenomenon checks" — rejected: the gates decide which knobs to use, which is development. (b) Create a third, untouched split only for gates — rejected for now: compute and data are fixed by the spec.

### Consequences
G0, G1, G2 numbers are validation numbers. The paper must say so. Test is evaluated once per locked candidate.

### Revisit condition
The human says the spec's test-based gates are intended. Then this decision is reversed (with the same caveat about blindness) and recorded here.

### Supersedes / superseded by
None.

---

## D-003 — Contribution type and primary acceptance rule

### Status
**partially superseded.** Still in force: the primary metric (Recovery@r on test) and the principle of one acceptance rule fixed before results. Superseded: the acceptance rule and routes below (by D-012: Route A = H4 ∧ (H1 ∨ H2a ∨ H2b), H3 descriptive, B′ merged into B) and the contribution wording (by D-014). Apply `docs/EVALUATION.md` §15, not the text below.

### Context
AGENTS.md requires that a contribution goal be stated explicitly and that tuning is not a contribution. The spec defines routes A, B, B′, Stop in §10.

### Decision
- Contribution type: methodological (definition, theory, solver, ground-truth benchmark).
- Primary metric: Recovery@r on the test split.
- Acceptance rule: `docs/EVALUATION.md` §15 (H3 AUROC ≥ 0.90 on both tiers and at least one of H1, H2a, H2b with CI > 0 after Holm and same sign in 3/3 seeds).
- Routes A, B, B′, Stop as in §15.

### Rationale
A single rule prevents choosing the success criterion after results.

### Evidence
Spec §8, §10; `docs/GOAL.md`.

### Alternatives considered
A single H1-only rule — rejected: it would ignore the detection claim.

### Consequences
Route B (theory + analysis) remains a valid outcome.

### Revisit condition
None before results. Changes after results are exploratory (spec §13).

### Supersedes / superseded by
None.

---

## D-004 — Venue, dates, and special session

### Status
active. Confirmed by the user on 2026-10-15: the 6-page limit is real and the target is ESANN 2027. A web check the same day found the special session announced by its organisers, with the conference at 21–23 April 2027 and a submission deadline of 18 November 2026 on esann.org; the session's own call text was not found on the ESANN special-sessions page (which lists only general rules).

### Context
The spec states the ESANN 2027 special session "From Understanding Distributional Change to Informed Action", a page budget of 6 pages including figures and references, and dates (internal submission 16/11, hard deadline 18/11/2026). The spec lists these as items to confirm (spec §16: session vs regular track, author registration commitment before 12/02/2027, GPU account and quota, arXiv policy). The proposal lists the page limit and special-session existence as unverified assumptions (proposal §11).

### Decision
Use the dates and the 6-page limit as project inputs. The special-session track is the user's choice (spec §16 item 1).

### Rationale
These claims were taken from a web reading in the spec and proposal; the repository does not contain the venue guidelines.

### Evidence
Spec header and §1 C1, §7–§16; proposal §11.

### Alternatives considered
Verify now via web search — deferred to the G-lit step, which covers literature and venue. Recorded as a blocker in `research/STATE.md` §12.

### Consequences
GOAL and CONSTRAINTS use the dates as HARD and mark them unverified.

### Revisit condition
If the ESANN author guidelines differ from the spec when the paper is drafted.

### Supersedes / superseded by
None.

---

## D-005 — The spec supersedes the proposal where they differ

### Status
active

### Context
The proposal v1 (15/10) and the spec v1.0 (15/10) differ in several places: H3 (SAR → AUROC), B1 definition, the null-pair set (N1 added), G0 criteria, Tier 0 renderer, r calibration, H4 tolerance (1e-4 → 1e-3), pair counts (14 → 17 Tier 0 pairs; 11 Tier 1), and the budget (12 → 10 GPU-hours).

### Decision
The spec governs. The proposal is kept as history in `research/proposals/`. Where the proposal's text is used (e.g. §2 novelty, §3.6 limits), it is cited as the proposal.

### Rationale
The spec was written after the proposal's full-text check and explicitly replaces it (spec §1).

### Evidence
Spec §1 table C1–C8.

### Alternatives considered
Keep both as co-equal — rejected: the protocol needs one source of truth.

### Consequences
None beyond the above.

### Revisit condition
None.

### Supersedes / superseded by
Supersedes: `research/proposals/delta_ce_proposal_v1.md` (for the listed items).

---

## D-006 — Environment split: Tier L and Tier 0 local CPU; Tier 1 Kaggle GPU

### Status
active

### Context
The local machine has 16 CPU cores and 15 GB RAM, with no NVIDIA GPU detected. The spec assigns Tier L and Tier 0 to CPU and Tier 1 (VAE, classifiers, oracle) to GPU with ≤ 4 Kaggle jobs.

### Decision
- Tier L and Tier 0: local CPU. PyTorch CPU is installed at G0 (a genuine local requirement).
- Tier 1: Kaggle GPU via the repository harness, with the prep-kernel pattern for MNIST (kaggle-prep, Internet on; kaggle-gpu, Internet off, `kernel_sources`).
- Kaggle accounts: multi-account harness; `--account auto` only for runs where the device class is recorded.

### Rationale
Matches the spec's compute plan and AGENTS.md's Kaggle rules.

### Evidence
`docs/ENVIRONMENTS.md` §1–§7; spec §11.

### Alternatives considered
Tier 1 locally on CPU — rejected for the acceptance runs (budget and runtime), allowed only for debugging.

### Consequences
A Kaggle smoke run is the first Tier 1 action; the environment remains `unverified` until then.

### Revisit condition
A smoke run fails on every account.

### Supersedes / superseded by
None.

---

## D-007 — Baselines are re-implemented, no external code

### Status
active

### Context
Artelt et al. and DiG-IN code may exist. Their licences, versions, and the need to match the benchmark are not checked.

### Decision
B1, B1-det, B3, B4 and the ablations are written in `src/dce/baselines/` and `src/dce/solvers/` from the equations in the spec. No external code is copied.

### Rationale
Same optimiser, budget, and points for every method (EVALUATION §14); avoids licence and version issues.

### Evidence
Spec §6.2–§6.4; `docs/CONSTRAINTS.md` §8.

### Alternatives considered
Using the authors' code — deferred; it can be checked later as a validation of B1 if available.

### Consequences
Baseline fidelity needs a unit test against the closed form for the linear case (H-004 covers Δ-CE; B1 gets its own check at G1).

### Revisit condition
A reviewer requires the original code.

### Supersedes / superseded by
None.

---

## D-008 — Research branch and kick-off commit

### Status
active

### Context
AGENTS.md: research happens on `research/<topic>`; `main` is the template.

### Decision
Branch `research/delta-ce` from `main`. The kick-off commit contains the docs, the research state, the moved spec and proposal, and the project `project rules file`. No code is included in the kick-off commit.

### Rationale
Keeps `main` free of project-specific content.

### Evidence
AGENTS.md "Branch model".

### Alternatives considered
None.

### Consequences
`main` stays generic.

### Revisit condition
None.

### Supersedes / superseded by
None.

---

## D-009 — Gates follow the spec

### Status
superseded by D-011 (spec amendment v1.1, 2026-10-15)

### Context
D-002 proposed moving the spec's test-based gates to validation to keep the test split blind. The user decided to follow the spec.

### Decision
G0, the G1 pilot, and G2 use the splits written in the spec, including the test split. To keep the method comparison honest, `docs/EVALUATION.md` §6 limits what may be read from test before the candidate is locked: aggregate gate statistics only (Flip_k, classifier accuracy and agreement, oracle Spearman and accuracy, found-rate and runtime for the pilot). The pilot's Recovery@r, share, attribution vectors, and per-sample outputs are not inspected, and only the solver-repair options named in the spec (restarts, λ schedule, initialisation from the ordinary CE) may be changed after a failed G1. These two restrictions are additions by the Research Lead; the user may drop them. `r` is calibrated on validation as in the spec.

### Rationale
The user's instruction. The gate statistics describe the benchmark and solver mechanics, not the Δ-CE versus baseline comparison, so the restrictions keep the comparison metrics unread.

### Evidence
Spec §7 and §9; `docs/EVALUATION.md` §6.

### Alternatives considered
D-002 (validation-based gates). Not chosen by the user.

### Consequences
The test split is not blind with respect to the gate statistics, and the paper states this. Any change motivated by a test method-comparison metric is exploratory (spec §13).

### Revisit condition
None before the final test.

### Supersedes / superseded by
Supersedes D-002.

---

## D-010 — Tier 1 runs on Kaggle T4; RTX not used

### Status
active

### Context
The repository's RTX PRO 6000 example attaches the competition `arc-prize-2026-arc-agi-3`, and its comments allow it only when the account and project are permitted to use that context. This project has no such permission. Smoke runs on account `boinhbo` (smoke_0001 prep, smoke_0002 GPU) succeeded: offline kernel, `kernel_sources` hash verified, two Tesla T4 devices observed.

### Decision
All Tier 1 Kaggle jobs use `machine_shape: NvidiaTeslaT4`. Every run records requested and observed hardware. One device class (T4) is used for all methods.

### Rationale
The T4 is sufficient for a ≈ 60k-parameter model, and a single device class keeps runtime comparisons valid (EVALUATION §14).

### Evidence
`runs/smoke/smoke_0001/`, `runs/smoke/smoke_0002/`; `configs/experiments/smoke_000{1,2}_*.yaml`.

### Alternatives considered
RTX PRO 6000 with the ARC competition source: not permitted for this project.

### Consequences
Runtime figures are T4 figures. The kernel runtime is Python 3.13 with torch 2.11, so local PyTorch versions should be recorded when results from both are compared.

### Revisit condition
A permitted RTX context becomes available.

### Supersedes / superseded by
None.

---

## D-011 — All gates and calibrations on validation; test once

### Status
active (spec amendment v1.1, A1; author's decision)

### Context
The author states that the test-based gates in spec v1.0 were a writing error, not a design intent. Nothing has run and the spec is not frozen.

### Decision
G0 (Tier 0: 2 000 clean validation points; Tier 1: the whole validation split, ≈ 1 180 images), the G1 pilot, G2, σ_k, the classifier accuracy requirements, and r calibration all use validation. The test split is used only in the final procedure. The pilot read restrictions of D-009 are kept, applied to validation (only found-rate and runtime are read).

### Rationale
Removes any blindness caveat from the paper at no cost.

### Evidence
`docs/source/delta_ce_spec_amendment_v1.1.md` A1; `docs/EVALUATION.md` §5–§6.

### Alternatives considered
D-009 (gates on test). Withdrawn by the author.

### Consequences
`docs/EVALUATION.md` §6 has no test exceptions. The spec's "test sạch" in §5.1, §5.3 and §9 reads as validation.

### Revisit condition
None.

### Supersedes / superseded by
Supersedes D-009; returns to the intent of D-002.

---

## D-012 — H3 descriptive with B0; acceptance Route A = H4 ∧ (H1 ∨ H2a ∨ H2b)

### Status
active (amendments A2, A3)

### Context
H3 holds almost by construction: G0 keeps null pairs with agreement ≥ 0.98, while drift pairs have disagreement regions of about 10%. Δ-CE found-rate is a local estimator of the discrepancy volume (Hammer & Hüllermeier 2021).

### Decision
- H3 becomes descriptive: report AUROC(S) for Δ-CE, B0, B1, B3, labelled "conditional on G0-qualified pairs".
- B0 = Monte Carlo disagreement rate. **Implementation clarification by the Research Lead:** B0 is computed on the whole test split of the tier (unfiltered), not on the evaluation points, because the evaluation points are filtered so that both models agree with margin ≥ κ, which makes the disagreement rate on them 0 by construction.
- Route A = H4 ∧ (H1 ∨ H2a ∨ H2b), Holm over H1, H2a, H2b as before. Route B = H4 ∧ ¬(H1 ∨ H2a ∨ H2b): benchmark + analysis paper; B′ merged into B. Stop unchanged.

### Rationale
The value of Δ-CE is in *what* and *where*, not *whether*.

### Evidence
Amendment A2, A3; `research/LITERATURE.md` L-004.

### Alternatives considered
Keep H3 in the rule and add B0 as a comparator — rejected: H3 would remain near-tautological.

### Consequences
`docs/EVALUATION.md` §1–§3, §10, §15; `research/HYPOTHESES.md` H-003.

### Revisit condition
None.

### Supersedes / superseded by
Supersedes the H3 part of D-003.

---

## D-013 — Non-local baselines B5 and A1, outside Holm, with a claim rule

### Status
active (amendment A4)

### Context
Comparing Δ-CE only against B3 on T3 could be read as a straw-man comparison, since B3 cannot see non-local change by design. DIG and DeepXplore are the closest prior art.

### Decision
- B5 (DIG-style): from z0, line search along segments z0 → z_k for the K = 20 nearest training points in Z (metric d), 50 steps per segment; return the point in F^N_κ with the smallest d ≤ r, else NOT_FOUND.
- A1 is renamed "DeepXplore-style" (λ = 0) and reported next to B5.
- Both appear in the main table and are not in the Holm family; the acceptance rule is unchanged.
- Solver claim rule (revised by A7, see D-016 and `docs/EVALUATION.md` §15): a solver claim only if d_t > 0.10 and CI_t > 0 on T3 in both tiers; no solver claim otherwise; B5 better (d_t < −0.10 and CI_t < 0 in any tier) is reported explicitly; tier-specific outcomes reported per tier.

### Rationale
Answers "how does it compare with DIG?" before a reviewer asks.

### Evidence
Amendment A4; `research/LITERATURE.md` L-022, L-023.

### Alternatives considered
Re-implementing DIG's full graph method — not chosen: B5 keeps the same space Z and budget r.

### Consequences
`docs/EVALUATION.md` §7, §8, §14, §15; `research/HYPOTHESES.md` H-006, H-007.

### Revisit condition
None before results.

### Supersedes / superseded by
None.

---

## D-014 — Positioning and contribution order

### Status
active (amendment A5)

### Decision
Δ-CE is presented as an adaptation of DIG (L-022) and DeepXplore (L-023). P4 cites Artelt et al. Theorem 1 and is not claimed. Contributions in order: (1) the ground-truth benchmark, (2) the controlled comparison of explanation families with P-tie, (3) the margin-κ witness with L1, P2, P3.

### Evidence
Amendment A5; `docs/GOAL.md` "Target contribution".

### Consequences
Route B remains a submittable paper.

---

## D-015 — NPL 2022 residual risk accepted

### Status
active (amendment A6)

### Decision
The journal text of Artelt et al. NPL 2022 is not required to proceed. Try the author's page before 17/10; it does not block G0. The arXiv v2 already contains the regularisation section that matches the journal title.

---

## D-016 — A7 and spec freeze

### Status
active (author's decision, 2026-10-15)

### Decision
- B0 is computed on the whole, unfiltered test split. The paper states that B0 is a global measure over the data distribution, while Δ-CE found-rate is local around the filtered evaluation points; their AUROCs are shown side by side for reference only.
- The B5 rule is replaced by the four-clause rule (a)–(d) in `docs/EVALUATION.md` §15: a solver claim only if d_t > 0.10 and CI_t > 0 in both tiers; B5 being better (d_t < −0.10 and CI_t < 0 in any tier) must be reported explicitly; per-tier outcomes reported as they are. Outside Holm; Route A unchanged.
- Spec v1.0 + amendment v1.1 (A1–A7) is frozen on 2026-10-15, before the planned 17/10. Later changes follow spec §13 (logged here with "before/after seeing results").

### Evidence
`docs/source/delta_ce_spec_amendment_v1.1.md` (A7, freeze).

### Supersedes / superseded by
Supersedes the claim rule in D-013.

---

## D-017 — Tier 0 implementation readings where the spec is silent

### Status
active (Research Lead, 2026-10-15, before any result; spec §13 log: "before seeing results")

### Decision
| Spec point | Reading |
|---|---|
| §4.4 adaptation sets: label noise | none; the 5% noise of §4.2 applies to the clean training split only |
| §4.4 T1: colour–label correlation 0.95 | applied to all 10 000 adaptation samples, including the 30% outside the ambiguous band |
| §4.4 N0: "khác seed thứ tự dữ liệu" | a fresh clean adaptation sample (seed 2000 + pair_id) with its own order and noise seeds; same distribution as the training split |
| §4.2 pixel noise | added per batch during f_old training and fine-tuning; never at evaluation or in G0 |
| §9 Flip_k | the same re-sampled factor values (seed 7000 + k) are used for f_old and f_new, so ΔFlip_k is paired |
| §9 G0 points | the first 2 000 points of the clean validation split (D-011) |
| §5.1 accuracy requirement | on all 5 000 clean validation labels (D-011) |
| Seeds | `configs/tiers/tier0.yaml` (init 0+s, order 100+s, noise 200+s, adaptation data 2000+pair_id, order 3000+pair_id, noise 4000+pair_id) |

### Rationale
Each reading is the simplest one consistent with the spec text; none is tuned.

### Evidence
`src/dce/data/drift.py`, `src/dce/cli/g0_tier0.py`, `configs/tiers/tier0.yaml`.

---

## 4. Status semantics

`active`: in force. `superseded`: replaced by a later decision (name it). `reversed`: the opposite was decided (name it).

---

## 5. Decision lifecycle

Create when a decision affects more than one experiment. Mark `pending confirmation` when a human must answer. Supersede rather than edit history.

---

## 6. Evidence-driven decisions

None yet. Decisions that follow results go here with the run ID and the validity status.

---

## 7. Decisions derived from negative results

None yet.

---

## 8. Constraint deviations

None.

---

## 9. Environment and infrastructure decisions

D-006 and D-001 above.

---

## 10. Baseline decisions

D-007 above.

---

## 11. Research-direction decisions

D-003 above.

---

## 12. Decision conflicts

Conflicts between the spec and the protocol are resolved in favour of `docs/EVALUATION.md` (D-011, spec amendment v1.1) and the layout in AGENTS.md (D-001).

---

## 13. Relationship to other files

`docs/` holds the spec; this file holds decisions taken under it. Hypothesis-level detail stays in `research/HYPOTHESES.md`.

---

## 14. Cleanup and maintenance

Do not delete decisions. Mark them superseded.

---

## 15. Completion checklist

- [x] Decisions D-001 to D-008 have status, context, decision, rationale.
- [x] D-002 reversed by D-009; D-009 superseded by D-011 (amendment v1.1); D-004 confirmed by the human.
- [x] Spec frozen 2026-10-15 (D-016).
