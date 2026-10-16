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
| D-017 | Tier 0 implementation readings where the spec is silent | partially superseded by D-018 (label noise) | 2026-10-15 | — | exp_0001 |
| D-018 | Amendment v1.2 after G0 exp_0001: f_old cosine 30 ep, noisy adaptation sets, DD criterion, T3/T4 lr knob | active | 2026-10-15 | — | exp_0001, exp_0002 |
| D-019 | G0 knob log (spec §4.4 / v1.2 B4) and f_old reuse in exp_0003 | active | 2026-10-15 | — | exp_0002, exp_0003 |
| D-020 | Amendment v1.3 after G0 exp_0002/exp_0003: T3 criterion C1 (ΔFlip_k* ≥ 0.15, exclusivity, DD outside); last G0 criterion change; Tier 1 T3 pre-declared | active | 2026-10-15 | — | exp_0003, exp_0004 |
| D-021 | Pre-G1 clarifications: Flip_k only for re-sampleable factors (D1); descriptive Recovery\|reach (D2) | active | 2026-10-15 | — | exp_0004 |
| D-022 | Amendment v1.4 after H4 exp_0005: two-phase solver (Adam + R1, then SLSQP per restart) for Δ-CE and ordinary CE; H4 extended to CE; one H4 run | partially superseded by D-023 (E1 phase 2 starts, E4) | 2026-10-15 | — | exp_0005, exp_0006 |
| D-023 | Amendment v1.5 after exp_0006: SLSQP from z0 and every restart end; κ+1e-4 target, float32 confirmation; H4 on two seed sets; E4 replaced (deadline 17/10); F5 completeness check at pilot | active | 2026-10-15 | — | exp_0006, exp_0007 |
| D-024 | Pre-pilot declarations (G1): parallel phase 2, wall+CPU time, r calibration over found CEs, float64 model check, pilot readings | active | 2026-10-15 | — | exp_0008, exp_0009 |
| D-025 | Tier 1 data preparation: morphology ladder with measured thickness, clean splits and factors (prep kernel, CPU) | active | 2026-10-15 | — | smoke_0008, exp_0010 |
| D-026 | Kaggle packaging: generated single-file bundles of src/dce, checked against src by a unit test | active | 2026-10-15 | — | smoke_0009 |
| D-027 | Amendment v1.6 after the G1 pilot: projected-runtime criterion (36 h / 36 h / 10 GPU-h), Tier 1 evaluation local CPU, runtime split by witness, (B) only after (A), (A) bit-identical | active | 2026-10-16 | — | exp_0009 |
| D-028 | Evaluation code before the final run: Holm and bootstrap readings, per-seed rule, share/zero-sum, B5 outcome labels; no Tier 0 test exception | active | 2026-10-16 | — | synthetic tests only |

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
partially superseded by D-027 (amendment v1.6 G2): Tier 1 evaluation (solvers, baselines) runs on local CPU; Kaggle T4 only trains. Original text: active

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
partially superseded by D-027 (amendment v1.6 G2): Tier 1 evaluation (solvers, baselines) runs on local CPU; Kaggle T4 only trains. Original text: active

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
| §4.4 adaptation sets: label noise | ~~none~~ — **superseded by D-018 (v1.2 B2): 5% symmetric noise, as the training split** |
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

## D-018 — Amendment v1.2 after G0 exp_0001

### Status
active (author's decision, 2026-10-15; spec §13 log: **after seeing G0 results**; G0 is a gate, test untouched, H1–H4 unaffected)

### Context
exp_0001 failed G0 (T3, T4, N0, N1, f_old seed 0). Diagnosis in `research/NOTES.md` §3: null disagreements are threshold jitter; f_old not converged (loss ≈ 0.28 vs floor ≈ 0.20); N0 fine-tuned on noise-free labels (D-017 reading) so it was not a null; T3 under-trained.

### Decision
`docs/source/delta_ce_spec_amendment_v1.2.md` B1–B5: f_old 30 epochs with cosine decay; 5% label noise on every adaptation set; decisive disagreement DD (argmax differs and both |margins| ≥ κ) replaces raw agreement in G0 (N0/N1: DD ≤ 0.02; T3 outside subgroup: DD ≤ 0.05; raw agreement still reported); T3/T4 knobs: knob 1 = 5 epochs and lr 1e-3 (used at once in exp_0002), knob 2 = 10 epochs. The Research Lead's proposal c.3 (|φ1 − 0.5| ≥ 0.05) was rejected by the author: not transferable to Tier 1, filters on task ground truth, and unrelated to L1.

### Consequences
Fine-tune recipes differ across conditions; reported in the paper. If any f_old seed still misses 0.95 after B1, G0 fails and goes back to the author.

### Supersedes / superseded by
Supersedes the label-noise row of D-017 and the G0 agreement criteria of spec §9.

---

## D-019 — G0 knob log and f_old reuse

### Status
active (Research Lead, within pre-declared knobs; spec §13 log: after seeing exp_0002)

### Knob log (max 2 per condition)

| Condition | Adjustment 1 | Adjustment 2 |
|---|---|---|
| T1 | exp_0003: colour–label correlation 0.95 → 0.99 (spec §4.4) | free: ambiguous fraction 0.7 → 0.9, or epochs 3 → 5 |
| T3 | exp_0002: 5 epochs, lr 1e-3 (v1.2 B4 knob 1) | exp_0003: 10 epochs, lr 1e-3 (knob 2) — last |
| T4 | exp_0002: 5 epochs, lr 1e-3 (v1.2 B4 knob 1) | exp_0003: 10 epochs, lr 1e-3 (knob 2) — last |

### Rationale for the T1 knob
exp_0002 lowered ΔFlip_color to 0.12–0.15 after B2 added 5% label noise. The colour is drawn from the noise-free label, so the noise weakens the colour–label association the shortcut needs; raising the correlation targets that cause directly. The fine-tune loss had plateaued (0.28–0.29), so more epochs at lr 3e-4 are less likely to help.

### f_old reuse
exp_0003 loads the five f_old checkpoints of exp_0002 after verifying their SHA256 against its manifest, and reruns only the nine drift pairs. N0 and N1 passed in exp_0002 with these same checkpoints, so the G0 verdict combines exp_0002 (f_old, N0, N1) and exp_0003 (T1, T3, T4). A smoke run reproduced exp_0002's f_old seed-0 validation accuracy exactly.

### Consequence
If T3 or T4 fails in exp_0003, no knob remains: Stop or back to the author (v1.2 B5).

## D-020 — Amendment v1.3: T3 G0 criterion C1, last G0 criterion change

### Status
active (author's decision, 2026-10-15; spec §13 log: **after seeing G0 results** exp_0002 and exp_0003; G0 is a gate, test untouched, H1–H4 unaffected). **This is the last change to the G0 criteria** (both tiers).

### Context
exp_0003 used the last T3 knob (D-019). Subgroup accuracy w.r.t. the new labels stayed below 0.85 (0.866 / 0.814 / 0.754), but the change is real and size-specific (ΔFlip_size ≈ +0.32, DD inside 0.61–0.75, outside ≤ 0.04). The learned size boundary is soft and seed-shifted (`research/NOTES.md` §3, exp_0003).

### Decision
`docs/source/delta_ce_spec_amendment_v1.3.md` C1–C4:
- T3 passes G0 iff, on every seed, (i) ΔFlip_{k*} ≥ 0.15 (k* = size on Tier 0, thick on Tier 1), (ii) max_{k≠k*} |ΔFlip_k| ≤ 0.05, (iii) DD outside the subgroup ≤ 0.05.
- C2: the old accuracies (raw and decisive-margin) are still computed and reported in the paper, with the limitation that the soft boundary makes T3 less non-local than designed, which hurts Δ-CE in H2.
- C3: Tier 1 T3 uses C1 from the start (pre-declared). Knobs: §4.4 (epochs; quantile 0.85 → 0.80) and the v1.2 B4 lr knob, max 2. Then fallback T3-Tier1 = flip by bg > q85, k* = bg, same C1. If that fails too: drop T3-Tier1; H2b untested, H1 on T1-Tier1 only, Holm over the remaining tests.
- C4: T1, T4, N0, N1 unchanged from v1.2.
- T3-Tier 0 is not retrained; G0 is recomputed on the saved checkpoints in exp_0004.

The Research Lead proposed (A) without (ii); the author added the exclusivity condition (ii). Option (B), dropping T3, was rejected: it loses H2a and H2b.

### G0 history (for the paper)
| Step | Trigger | Change | Type |
|---|---|---|---|
| exp_0001 | — | spec §9 criteria | baseline |
| v1.2 (D-018) | exp_0001 failed | f_old cosine 30 ep; 5% noise on adaptation sets (N0 was not a null); DD replaces raw agreement; lr knob for T3/T4 | amendment, after results |
| exp_0002, exp_0003 (D-019) | — | knobs: T1 1/2 used (corr 0.99); T3 2/2 (5 ep → 10 ep, lr 1e-3); T4 2/2 (same) | pre-declared knobs |
| v1.3 (this) | exp_0003 T3 failed | T3 criterion C1 | amendment, after results; last |

### Consequences
T3-Tier 0 enters the benchmark if exp_0004 confirms C1 on all seeds with exact numbers. H2 is tested on a less non-local change than designed (against Δ-CE).

### Supersedes / superseded by
Supersedes the T3 rows of spec §9 and of v1.2 B3.

## D-021 — Pre-G1 clarifications: Flip_k domain and Recovery|reach

### Status
active (author's decision, 2026-10-15). **Declared before any result of any method** (no solver, baseline, `r` calibration or pilot output exists). D1 is a clarification of a definition, not a G0 amendment; D2 is a new descriptive metric outside the acceptance rule and outside Holm. Neither changes `docs/` or any locked parameter.

### Decision
```text
D1. Flip_k is defined only for re-sampleable factors. Tier 1: {color, bg, thick};
    shape has no Flip. Condition (ii) of C1 (v1.3) on Tier 1 ranges over {color, bg, thick} \ {k*}
    (k* = thick; for the bg fallback, k* = bg).
D2. New descriptive metric (outside the acceptance rule, outside Holm):
    Recovery|reach = Recovery computed on the subset of points where Δ-CE finds a witness within r,
    applied to EVERY method on the SAME subset.
    Reported next to Recovery@r for every condition; found-rate reported per condition.
    Purpose: separate "reaching the change within the budget" from "right factor once reached".
```

### Rationale
- D1: shape cannot be re-sampled while keeping the digit, so Flip_shape is undefined (spec §9 defines Flip_k by re-sampling factor k). Writing this now avoids it looking like a criterion change at G2.
- D2: `r` is calibrated on T1 (2 × median CE distance) and is locked. T3 witnesses must move size into the subgroup, so many T3 points may be out of budget; Recovery@r scores NOT_FOUND as 0 while B3 always returns a direction, so H2 could fail because of the budget rather than the method. `r` is **not** changed; the pilot's found-rate (allowed to be read, EVALUATION §8) shows early whether this risk is real. A low T3 found-rate is information for reading the final result, not a reason to tune `r`.

### Caveat (Research Lead)
The D2 subset is selected on Δ-CE's own success, so within it Δ-CE has no NOT_FOUND by construction while other methods may still have theirs. Recovery|reach is therefore not a neutral comparison and must never be used as evidence for H1/H2; the paper presents it as a decomposition of Δ-CE's Recovery@r, with the subset size (found-rate) next to it.

### Consequences
The G3 evaluation code computes Recovery|reach and per-condition found-rate in addition to the metrics of EVALUATION §8 step 8.

## D-022 — Amendment v1.4: two-phase solver, single H4 run

### Status
active (author's decision, 2026-10-15; after H4 exp_0005, **before any method result on data**; Tier L is synthetic and has no test split). H4 tolerances not relaxed.

### Context
exp_0005: the spec solver (penalised hinge + Adam, lr 0.05) fails H4. λ0 = 1 is above the exact-penalty threshold (found-rate), and Adam at fixed lr has a ≈ 1e-2 precision floor (no allowed repair changes it). Option (B), relaxing the tolerance, was rejected as moving the goalposts of a correctness gate.

### Decision
`docs/source/delta_ce_spec_amendment_v1.4.md` E1–E5:
- Phase 1 = spec §6.1 + R1 (λ halved every T/8 while the restart has no feasible iterate), R = 5, T = 400. Phase 2 = SLSQP (maxiter 100, ftol 1e-10) from each restart's best feasible candidate, on the exact problem; Tier 0 adds box bounds [0, 1]^5. The result replaces the phase-1 candidate only if SLSQP succeeds and the point passes the forward-pass feasibility check. Smallest distance over restarts; NOT_FOUND if none; r applied last.
- The same two phases for Δ-CE and every ordinary CE (B1, B1-det, A2, A3, r calibration). B3, B5 unchanged; A1 (λ = 0) phase 1 only.
- H4 also checks the ordinary CE against u = −(m(z0)+κ)/‖w‖² w, same tolerances (fidelity of B1, D-007).
- **H4 runs once** (exp_0006). Pass → G1 continues; fail → stop and report. Only a documented bug fix (code not solving the declared problem) allows a rerun.
- Runtime ≤ 5 min per pair is measured at the G1 pilot with the new solver.

### Implementation readings (declared before exp_0006)
1. **SLSQP constraints are tightened by ε = 1e-6** (g ≥ ε inside SLSQP): the exact optimum lies on the boundary and would otherwise fail the hard re-check by rounding. Distance cost ≈ ε/‖∇m‖, far below the 1e-3 tolerance.
2. **Phase 2 is evaluated in float64.** Bug found in the smoke test before exp_0006: with float32 constraint evaluation (noise ≈ 1e-7) SLSQP at ftol 1e-10 stopped at the iteration limit or failed its line search in 9 of 18 runs, so the code did not solve the declared problem; in float64 18/18 succeed with relative error ≈ 1e-6. Consequence for Tier 0/1: margin functions must accept float64 inputs in phase 2 (a float64 copy of the CNN/decoder), to be implemented for the pilot.
3. Tier L models are evaluated in float64 throughout (they are exact linear functions, as in the closed forms).
4. A successful, feasible SLSQP result replaces its restart's candidate even if it is not shorter (E1 literal); the final choice is the minimum over restarts.
5. CE check (E3) for both models: CE(f_old) and CE(f_new), each against its own closed form; the direction check is |cos(u, u*)| > 0.999.

### Caveat on D-021 (author, 2026-10-15)
Recovery|reach is biased in favour of Δ-CE (the subset is selected on its own success). It may only be used to explain a low Recovery@r, never as evidence.

### Consequences
exp_0005 is reproducible only from commit 157320b (its variant flags were removed). The Tier 0/1 pilot cost grows by the SLSQP runs (≤ R per point).

## D-023 — Amendment v1.5: SLSQP from every start, two H4 seed sets, E4 replaced

### Status
active (author's decision, 2026-10-15, after exp_0006; before any method result on data). H4 tolerances unchanged.

### Context
exp_0006 (v1.4): phase 2 exact on every found witness (≤ 4.8e-6), ordinary CE 1.000, but Δ-CE p=5 found-rate 0.967 because phase 1 found no feasible start for 3.3% of pairs (far witnesses). v1.4 E4 forced a stop. The author judged E4 misplaced for a technical check on synthetic data.

### Decision
`docs/source/delta_ce_spec_amendment_v1.5.md` F1–F5:
- F1: SLSQP from z0 and from the final phase-1 iterate of each restart (R + 1 = 6 starts), feasible or not. Result: smallest distance among confirmed-feasible points. Phase 1 unchanged (v1.4 R1, T = 400).
- F2: SLSQP targets κ' = κ + 1e-4; feasibility confirmed with the original κ by the float32 model used in evaluation. Same for Δ-CE and the ordinary CE.
- F3: H4 on seeds 0–999 and a holdout set 1000–1999, both dims; must pass on both.
- F4 (replaces E4): solver repairs allowed for H4, never relaxing tolerances; every algorithmic change needs a reason independent of the specific failing cases and is logged here; deadline end of 17/10, then G1 fails and goes back to the author.
- F5: at the G1 pilot (Tier 0 validation, diagnostic only), on 50 points of pair T1 seed 0, compare the standard solver with SLSQP from 50 random starts in the box; report the share of witnesses (within r) the standard solver misses and the distance gap.

Rationale for F1 independent of the failing cases: SLSQP is an SQP method and handles infeasible starts; on the linear case the problem is a convex QP, so SLSQP from z0 solves it exactly whatever phase 1 does.

### Implementation readings (declared before exp_0007)
1. **Candidate pool** = all confirmed-feasible SLSQP outputs **plus** the phase-1 best feasible iterates (the v1.4 fallback); the answer is the minimum distance over the pool. F1 does not remove the phase-1 candidates, and including them can only shorten the answer.
2. **SLSQP's success flag is not required** (F1: "nghiệm được xác nhận khả thi"): any output confirmed feasible is a candidate. This replaces v1.4's success requirement.
3. κ' implemented as g ≥ 1e-4 on the constraint functions g (−κ − m_n, m_o − κ, −κ − m), which equals κ' = κ + 1e-4. On Tier L its distance cost is ≤ 1e-4 relative (a = m(z0) + κ ≥ 2κ = 1).
4. Tier L models are float32 tensors (the "evaluation models" of F2); SLSQP evaluates them at float64 inputs (float32 weights promoted). Closed forms use the float64 weights; the rounding gap is ≈ 1e-7 relative.
5. Restart noise: one generator per seed set, seed 10000 + first pair seed of the set.
6. Pass rule unchanged: per (set, dim, method ∈ {Δ-CE, CE(f_old), CE(f_new)}), share with rel. error < 1e-3 and share with |cos| > 0.999 both ≥ 0.99; a pair without a witness fails both.

### Consequences
Phase 2 now runs 6 SLSQP problems per point (cost measured at the pilot, E5). F5 is added to the pilot plan.

## D-024 — Pre-pilot declarations for G1 (before any pilot or calibration output)

### Status
active (author: items 1–3, 2026-10-15; Research Lead: implementation readings). Declared before `r` calibration and the pilot. ε = 1e-4 (v1.5 F2) unchanged: on Tier 0/1 it moves a witness by ≈ ε/‖∇m‖, far below the |Δφ̃| that Recovery uses.

### Author's declarations
1. **Runtime.** Phase 2 may be parallelised over points (16 cores), with one code path for every method that uses the solver. Report wall-clock and CPU time per point. If the pilot still exceeds 5 min per pair after parallelisation, that is reported as a result; `maxiter` is not lowered to fit.
2. **r calibration.** Median over the points where CE(f_new) is found; the found share is reported; r is computed once and frozen. r is a Euclidean distance in the normalised factor space [0, 1]^5.
3. **float64 model.** Before use, max |logit64 − logit32| is measured on the validation split for every model and recorded. The pilot counts the cases "SLSQP feasible in float64 (κ') but rejected by float32 (κ)".
4. Tier 1 data preparation may start on Kaggle (prep kernel) while the Tier 0 pilot runs locally.
5. Discipline: no Recovery, share or attribution vector is read anywhere in G1.

### Research Lead's implementation readings
- **Valid points (spec §3.2, EVALUATION §8 step 5):** validation indices permuted with `default_rng(5000 + pair_id)`; the first N with m_o(z0) ≥ κ and m_n(z0) ≥ κ for y = y_clean(z0) are kept. Margins are for this y throughout the search.
- **Margins on Tier 0:** m(z) = margin_y(CNN(render(z))). The float64 path uses a float64 copy of the CNN and the renderer at float64; it is used only inside SLSQP. Phase 1 and every feasibility check use the float32 model.
- **r calibration (spec §6.3):** the 300 valid validation points of pair 0 (T1 seed 0) from the sampler above; CE(f_new) with the v1.5 solver, restart seed 10000 + 0; r = 2 × median distance over the found CEs.
- **Restart seeds:** Δ-CE 10000 + pair_id; B1 CE(f_old) 10000 + pair_id + 0 and CE(f_new) 10000 + pair_id + 1 (spec §6.4 "seed+0 / seed+1"); B1-det R = 1 without noise (start z0) and phase 2 (E2).
- **B4** in the pilot uses the 2 000 G0 validation points (test only in the final run); permutation importance = accuracy drop w.r.t. y_clean when factor k is permuted, 5 permutations (seeds 0–4), difference f_new − f_old.
- **B5:** K = 20 nearest training-split factor vectors to z0 (Euclidean), points z0 + t(z_k − z0) for t = 1/50, …, 50/50; feasibility in F^N_κ with the float32 model; smallest d ≤ r.
- **A1:** λ0 = 0, phase 1 only (v1.5 E2), same R, T, restart seeds as Δ-CE.
- **Δ-CE type O** is run for N0 and N1 only (spec §3.2).
- **Pilot pairs:** seed 0 of each condition: pair 0 (T1), 3 (T3), 6 (T4), 9 (N0), 14 (N1); N = 100 valid validation points each.
- **Pilot outputs persisted:** per pair and method, found-rate, wall-clock and CPU time per point, polish counts and the float64/float32 rejection count; nothing per point, no Δφ̃. The G1 runtime check is on Δ-CE (EVALUATION §15), extrapolated from N = 100 to N = 300 and labelled as such.
- **F5 (v1.5):** the first 50 pilot points of pair 0; reference = SLSQP (same settings) from 50 starts uniform in [0, 1]^5 (seed 20000), best confirmed-feasible; report the share of points with a reference witness within r that the standard Δ-CE misses, and the relative distance gap (d_std − d_ref)/d_ref where both are found. Diagnostic only.

## D-025 — Tier 1 data preparation (spec §4.3)

### Status
active (Research Lead, 2026-10-15; started during the G1 pilot as the author suggested, D-024 item 4). Implementation readings of spec §4.3, declared before any Tier 1 data exists.

### Decision
- Kaggle prep kernel (Internet ON, CPU, no GPU budget), `src/dce/remote/tier1_prep.py`; full run on the immutable slug `boinhbo/dce-tier1-data-v1` (exp_0010) after smoke_0008. It is data staging, not one of the four Tier 1 GPU jobs (CONSTRAINTS §4).
- **Morphology ladder:** per MNIST 4/9 image, grayscale erosion/dilation with disk(1), levels −2 … +3 (negative = erosion), each measured as 2 × mean distance transform on the skeleton of α > 0.5 (the same mask threshold as the oracle, spec §5.3). Empty mask or skeleton → invalid level.
- "Lặp cho tới gần t nhất" is read as: t ~ U[1.5, 5.0] per image, and the valid level with the nearest measured thickness is used. The measured thickness of that level is the ground truth. Keeping the whole ladder lets later steps re-sample thickness (Flip_thick at G0-Tier 1, T3) without re-running morphology.
- **Splits:** official train 4/9 → `default_rng(1234)` permutation, first 10% validation, rest train; official test 4/9 → test. Label y = 1 for digit 9.
- **Clean factors:** per split, `default_rng(1235 / 1236 / 1237)` (train / val / test) draws t, then φc, φb ~ U[0, 1]. Colouring I = αc + (1 − α)b·1 is applied when images are built (`colorize`).

### Consequences
The thickness distribution is discrete per image (one value per ladder level); the measured value, not the target, is the factor. Whether this suffices for the G2 oracle criterion (Spearman ≥ 0.8 for thick) is checked at G2, not assumed.

## D-026 — Kaggle packaging of the `dce` package: generated single-file bundles

### Status
active (the user's choice, 2026-10-15, option 3). Options rejected: installing from GitHub (the repository is private; making it public or adding a token to Kaggle Secrets was not wanted).

### Context
The Kaggle harness stages exactly one tracked code file under `src/`. Tier 1 GPU jobs need the whole package (CNN, VAE, oracle, drift, solver, baselines), and the solver must be the same code as on Tier 0 (fairness, v1.4 E2).

### Decision
- `python -m dce.remote.bundle <name> <entry.module>` writes `src/dce/remote/bundles/<name>.py`: every `src/dce/**/*.py` (except the bundles) in a deterministic base64 zip, plus a loader that extracts it, checks the tree hash, puts it first on `sys.path`, and runs the entry module as `__main__`. The tree hash is exported as `DCE_BUNDLE_TREE_HASH` and must be written into each run's result.
- Bundles are generated **and committed**. `tests/unit/test_bundle.py` fails when a committed bundle differs from what `src/` would produce, so any change under `src/dce` requires rebuilding the bundles in the same commit. The entry modules (authoritative code) stay under `src/dce/remote/`.
- The same test runs a bundle in isolated mode outside the repository and checks that `dce` is imported from the extracted copy, not the editable install.

### Consequences
Bundles are ≈ 60 kB of base64 in Git per job; the source of truth remains `src/dce`. A Tier 1 run is identified by its git commit and its bundle tree hash.

## D-027 — Amendment v1.6: runtime criterion by projection; Tier 1 evaluation on local CPU

### Status
active (author's decision, 2026-10-16; after the G1 pilot exp_0009, **before** any measurement of the speed-up (A)). No change to the algorithm or to maxiter.

### Decision
`docs/source/delta_ce_spec_amendment_v1.6.md` G1–G5:
- G1: "≤ 5 min per pair" is replaced by projections from the runtime measured after (A) × points × pairs × methods: (i) Tier 0 final run ≤ 36 h wall on the local machine; (ii) Tier 1 evaluation ≤ 36 h wall locally; (iii) cumulative GPU-hours ≤ 10. All three → G1 passes on runtime.
- G2: Kaggle T4 only trains (VAE, shape oracle, f_old, f_new on Tier 1); every solver and baseline on Tier 1 runs locally on CPU from downloaded, hash-checked checkpoints. D-006 and D-010 are amended accordingly.
- G3: runtime is a secondary metric, wall and CPU time per point per method, split by points with and without a witness, with the note that Δ-CE's cost is concluding "no witness within budget".
- G4: (B), early stopping of SLSQP, only if G1 fails after (A); its rule must have a reason independent of the pilot data, logged here before running; accepted only if per-pair found-rates and the set of point IDs with a witness match the current solver on the pilot pairs (no attribution vectors read).
- G5: (A) must reproduce the current solver bit for bit on the pilot pairs; otherwise a purely numerical cause must be shown and the set of point IDs with a witness must be unchanged.

### Research Lead's readings (declared before measuring (A))
1. **Projection basis = 8 cores**: the machine is shared and capped at 8 logical cores (the user's request, 2026-10-15). Projections use runtimes measured on 8 cores.
2. **Projection model:** per pair, wall time = Σ over methods of (measured wall per point after (A), from the pilot pair of the same condition) × N, with N = 300 (Tier 0) / 200 (Tier 1); summed over all pairs of the tier (Tier 0: 17; Tier 1: 11). Methods = those of the final run: Δ-CE N (+ O on null pairs), B1, B1-det, B3, B4, B5, A1, A2 (no m_o ≥ κ), A3 (κ ∈ {0, 1.0}; κ = 0.5 is Δ-CE). Ablations on Tier 1 for seed 0 only (CONSTRAINTS §4). Conditions without a pilot pair are costed at the most expensive measured condition.
3. **Tier 1 evaluation cost** before real weights: decoder + CNN at the spec architectures with random weights, float64 forward/backward on CPU (the cost does not depend on the weight values); the Tier 1 projection scales the Tier 0 per-point solver cost by the measured ratio of per-evaluation cost (Tier 1 / Tier 0) and by the dimension (16 / 5) for phase 1 where relevant. Labelled as an estimate until G2.
4. **Device consistency (G2):** Tier 1 f_old/f_new are trained on Kaggle (torch 2.11, GPU) and evaluated locally (torch 2.14, CPU). Every evaluation quantity (valid points, margins, G0-Tier 1 numbers used by the benchmark) is recomputed locally from the downloaded checkpoints; max |logit_GPU − logit_CPU| on validation is measured and recorded.

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

## D-028 — Evaluation code before the final run: readings, no test exception

### Status
active (Research Lead, 2026-10-16). **Declared before any Tier 0 or Tier 1 metric is computed on any split.** The code was checked only on synthetic inputs with known answers (tests/unit/test_eval_metrics.py, test_eval_rules.py, test_eval_records.py: 36 tests). No spec file under `docs/` is changed.

### Decision
1. **No exception to EVALUATION §19.** The Tier 0 full run is not started before G2 passes, the evaluation code is fixed, and the candidate is locked. Order: this evaluation code → G2 (Tier 1 on Kaggle T4 for training, local CPU for checks) → lock (code commit hash and candidate config hash in STATE "Test-set state") → final test run.
2. **Holm reading (§11 "CI > 0 after Holm").** Accepted by the Research Lead 2026-10-16: a hypothesis is supported iff Holm-adjusted p < 0.05, the pair-averaged difference is > 0, and the difference is > 0 for seeds 0, 1, 2. The 95% percentile CI is reported unadjusted. The two-sided p comes from the bootstrap distribution itself: p = 2·min(P(d* ≤ 0), P(d* ≥ 0)). The family is H1, H2a, H2b; if one is dropped, Holm runs over the remaining two (tested).
3. **Bootstrap p.** p = 2·min(P*(stat ≤ 0), P*(stat ≥ 0)), floored at 1/B. Statistic = mean over pairs of the per-pair mean difference. Resampling is stratified by pair with the same indices for both methods. B = 10 000; the RNG seed is fixed in the config at lock.
4. **Per-seed sign rule (§10, §15).** H1: the pair-averaged difference of the (T1_s, T3_s) pairs is > 0 for s = 0, 1, 2. H2a, H2b: the difference of the T3_s pair is > 0 for s = 0, 1, 2.
5. **Pair scores.** S for Δ-CE = found-rate pooled over the N and O directions (where O is run). S for B1, B1-det, B3 = median ‖Δφ̃‖ over found points. B0 is computed on the unfiltered test split, as §3 states.
6. **SAR** is evaluated for Δ-CE: found-rate ≥ 0.2 and max over k of the mean share ≥ 0.5, where the mean is over found points.
7. **Share.** Points with zero Σ_k |Δφ̃_ik| are excluded from the share mean. This is not stated in §3; it is declared here. Not-found points count as 0 for Recovery, not for share.
8. **B5 rule (§15, A4/A7).** Uses the unadjusted 95% CI. Outcomes are computed per tier and overall: claim_allowed only if both tiers satisfy (a); report_b5_better if any tier satisfies (c); otherwise no_solver_claim. Per-tier results are always reported.
9. **Recovery|reach (D-021 D2)** is computed on the Δ-CE found set, for every method on the same subset, and is descriptive only.
10. **Tie rule (Research Lead, 2026-10-16, declared before any number).** If max_k |Δφ̃_ik| is attained at more than one factor (difference ≤ 1e-12) or the vector is zero, top_i is undefined and the point scores 0. A zero vector is never credited to column 0 (Tier 0 T4: k* = index 0). Factor order is the `factors` list of the tier config (Tier 0: aspect, size, color, bg, posx), not a hard-coded list.
11. **Bootstrap seeds:** 6000 + index with H1 = 0, H2a = 1, H2b = 2, B5-T0 = 3, B5-T1 = 4; one generator per comparison (`rules.BOOTSTRAP_SEEDS`).
12. **Wiring** (`src/dce/eval/records.py`): Δφ̃ = φ̃(z_new) − φ̃(z_ref) (Δ-CE, B5: z0; B1: CE of f_old), B3 direction as is, not-found rows zeroed, B0 on the unfiltered split, runtime with the phase-2 split by witness, count checks 17 × 300 / 11 × 200. Integration tests assert schema, counts and finiteness only; no Recovery, share or attribution value is printed before the lock.

### Synthetic checks (before any data)
- Zero true difference: 95% CI contains 0 in 94.5% of 200 replications; two-sided p < 0.05 in 5.5% (B = 1000, 6 pairs × 300 points).
- Planted difference 0.15: detected, CI > 0, p < 0.001.
- Hand-worked metric example (N = 10): Recovery 0.5, found-rate 0.8, share and SAR as computed by hand.
- Holm and the decision rules tested on constructed cases (including a seed-sign failure and a Holm-marginal case).

### Rationale
The final run is blind only if its code is fixed before it sees test data (§6, §19). Points 2–9 are choices or readings of text that the spec leaves open; declaring them now means they cannot be chosen after the result is known.

### Consequences
- `src/dce/eval/{metrics,stats,rules}.py` is the evaluation code. `src/dce/remote/bundles/bundle_smoke.py` was regenerated (D-026 rule: any change under `src/dce` rebuilds the bundles).
- Committed 2026-10-16 as an ordinary commit, not the lock. The lock is the commit hash recorded in STATE at candidate lock (§19); any later edit to `src/dce/eval/` before the lock is logged here.

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
