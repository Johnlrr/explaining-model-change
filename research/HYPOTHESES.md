# HYPOTHESES.md

> Purpose: Record every hypothesis: its origin, prediction, falsification condition, experiment, result, and conclusion.
>
> Ownership: Mutable research memory. Scientific conclusions are recorded here by the Research Lead only.

---

## 1. Hypothesis index

| ID | Short name | Type | Status | Priority | Parent | Main experiment | Novelty status |
|---|---|---|---|---|---|---|---|
| H-001 | Δ-CE beats CE differencing on Tier 1 drift (paper H1) | contribution | untested (Tier 1 dropped) | high | — | exp_TBD (Tier 1, T1+T3, 3 seeds) | candidate |
| H-002 | Δ-CE beats gradient difference on non-local drift T3 (paper H2a/H2b) | contribution | rejected (H2a, exp_0017) | high | — | exp_TBD (Tier 0 and Tier 1, T3) | candidate |
| H-003 | Pair-level detection of drift versus null (paper H3; descriptive since D-012) | other (descriptive) | reported (descriptive) | low | — | exp_TBD (Tier 0 and Tier 1, all pairs) | not claimed |
| H-004 | Solver reproduces linear closed forms (paper H4) | other | supported (exp_0007) | high (gate G1) | — | exp_0005–exp_0007 (Tier L) | not applicable |
| H-005 | Local drifts give a predicted tie (Proposition 4; paper P-tie) | mechanistic | rejected (exp_0017) | medium | H-001 | exp_TBD (Tier 0, T1 and T4) | candidate |
| H-006 | Proximity term is needed for locality (A1, DeepXplore-style) | ablation | proposed | medium | H-001 | exp_TBD (Tier 0, A1) | not applicable |
| H-007 | Minimal-witness search versus DIG-style line search (B5) on T3 | other (descriptive, pre-declared claim rule) | reported, no claim | medium | H-002 | exp_TBD (Tier 0 and Tier 1, T3) | uncertain |

Allowed type values: contribution, mechanistic, tuning, ablation, reproduction, other.

Status values: proposed, queued, testing, supported, rejected, inconclusive, abandoned.

Priority: high = on the acceptance path; medium = reported analysis; low = optional.

---

## 2. Status definitions

Use the status definitions in the template (`proposed`, `queued`, `testing`, `supported`, `rejected`, `inconclusive`, `abandoned`). A hypothesis is `supported` or `rejected` only after its test-set evaluation (EVALUATION §19) or, for H-004 and gate checks, after the validation check defined in its entry.

---

## 3. Priority policy

H-004 is the first hypothesis: its result gates all later work (G1). H-001 and H-002 (H1, H2a, H2b) with H-004 form the acceptance rule (EVALUATION §15, D-012). H-003, H-005, H-006, H-007 are reported regardless of direction.

---

## H-001 — Δ-CE beats CE differencing on Tier 1 drift

### Status
untested — Tier 1 dropped (v1.8 I6, exp_0016)

### Origin
Theory: Proposition 4 (both Δ-CE and CE differencing recover span{PΔw} in the linear case), so any advantage must come from the witness property, the budget, and non-uniqueness of CE, not from direction alone. Source: spec §6.4 (B1), §8 (H1).

### Contribution quality
Methodological. Δ-CE is a different object (a witness on which the models disagree) from a CE difference. A positive result would show the object is better for recovering the injected factor on realistic images.

### Hypothesis
On Colored MorphoMNIST (Tier 1), for drift pairs T1 (colour shortcut) and T3 (thickness subgroup), Δ-CE recovers the injected factor more often than CE differencing (B1) within budget r.

### Rationale
B1 explains two independent CE points, whose difference can point to a factor the models do not disagree on, and can fail on non-unique CE. Δ-CE searches directly for disagreement.

### Predictions
- Recovery@r(Δ-CE) − Recovery@r(B1), pooled over T1 and T3 Tier 1 pairs (6 pairs, 200 points each), has 95% paired-bootstrap CI > 0 after Holm.
- The difference is positive for classifier seeds 0, 1, 2 (per-seed averages over T1_s and T3_s).

### Falsification signals
- CI includes 0 or is negative.
- Sign differs for any seed.
- Found-rate of Δ-CE is below 0.6 on T1 at G1 (then the solver is repaired first, not the hypothesis).

### Controlled variables
Classifiers, points, r, φ̃, normalisation, R = 5 restarts, κ = 0.5, λ schedule.

### Minimal experiment
Tier 1, T1 and T3, seeds 0–2, 200 points each, Δ-CE N and B1 and B1-det. Smoke: Tier 1 seed 0, 20 points, same code path.

### Confirmation procedure
G1 pilot per EVALUATION §6 (found-rate and runtime only), then the final test procedure (EVALUATION §19) once for the locked candidate.

### Parent and dependencies
Depends on H-004 (solver correct) and the G0 gate (drift is real).

### Prior-work relation
Artelt et al. (2021, 2022): B1 is their method. Hinder & Hammer (2020): CE for data drift, not model change. See `research/LITERATURE.md`.

### Planned experiment
Tier 1 job (c) in CONSTRAINTS §4, after the classifier job (b).

### Experiments
exp_0016 (G2 round two) dropped Tier 1 before any test.

### Results
Not tested.

### Conclusion
Not testable within this project's pre-declared rules.

### Follow-up
If supported: report with H-002, H-003. If rejected: report B1 as sufficient on Tier 1 and go to Route B.

---

## H-002 — Δ-CE beats gradient difference on non-local drift

### Status
rejected for H2a (exp_0017, 2026-10-17); H2b untested (Tier 1 dropped, v1.8 I6)

### Origin
Proposal §3.6 and spec §2 point 3: gradient difference at x (Artelt Eq. 14) sees only local change. T3 (label flip on a subgroup, size > 0.75 / thickness > q85) is non-local by construction.

### Contribution quality
Methodological; the prediction is falsifiable and the baseline is the published approximation.

### Hypothesis
On T3 drift pairs, Δ-CE recovers the injected factor more often than gradient difference (B3), on Tier 0 and on Tier 1.

### Rationale
Where the decision change lies away from z0, ∇h(z0) can be zero while the witness set is non-empty.

### Predictions
- H2a (Tier 0, T3, 3 pairs): Recovery@r(Δ-CE) − Recovery@r(B3) has CI > 0 after Holm; positive for seeds 0–2.
- H2b (Tier 1, T3, 3 pairs): same with B3 on Tier 1.

### Falsification signals
- CI includes 0 or is negative on either tier.
- Found-rate of Δ-CE on T3 at or below 0.3 (witness rarely in budget; then the claim is about budget, not locality).

### Controlled variables
As H-001, B3 closed form.

### Minimal experiment
Tier 0 T3 seeds 0–2, 300 points; Tier 1 T3 seeds 0–2, 200 points.

### Confirmation procedure
Validation diagnostic first; then the final test once per tier (EVALUATION §19).

### Parent and dependencies
H-004; G0 (T3 changes the model: Flip/accuracy check, EVALUATION §6).

### Prior-work relation
Artelt et al. Eq. 14 (B3). Muschalik et al. 2022 (global feature importance, B4).

### Experiments
exp_0017 (Tier 0 final, test, locked candidate ede7374). Reproduction §20: exp_0018 (in progress).

### Results
H2a: Recovery@r(Δ-CE) − Recovery@r(B3) on T3 = −0.091, 95% CI [−0.133, −0.050], p = 1e-4 (Holm m = 1); seeds −0.087 / −0.123 / −0.063. Δ-CE found-rate on T3 0.81 (falsification signal 2 not triggered).

### Conclusion
Rejected: B3 beats Δ-CE on Tier 0 T3. Route A fails → Route B. Any explanation (e.g. T3 less non-local than designed, v1.3 C2) is post hoc and must be labelled so.

### Follow-up
If supported on both tiers: strongest claim (Route A). If one tier only: report the tier that supports it. If none: Route B. In all cases the solver claim is limited by H-007 (B5 rule).

---

## H-003 — Pair-level detection of drift versus null (descriptive)

### Status
reported (descriptive, exp_0017) — **descriptive since D-012** (amendment A2). Not part of the acceptance rule. Reported next to B0 (disagreement rate on the unfiltered test split), B1 and B3, labelled "conditional on G0-qualified pairs". Expected: B0 reaches AUROC ≈ 1.0, because G0 selects null pairs by agreement ≥ 0.98. The predictions below are kept for reporting.

### Origin
Spec §1 (C5): H3 replaced SAR by AUROC because SAR depends on a magnitude threshold.

### Contribution quality
Methodological; a detection claim, which the other baselines are not designed to make.

### Hypothesis
The pair-level score S (Δ-CE found-rate over both directions) separates drift pairs (T1, T3, T4 on Tier 0; T1, T3 on Tier 1) from null pairs (N0, N1 on Tier 0; N0 on Tier 1), with AUROC ≥ 0.90 on each tier.

### Rationale
A drift changes the decision region, so witnesses with disagreement exist within budget. Null pairs have only Rashomon-level disagreement, which should be rare within r.

### Predictions
- Tier 0: AUROC ≥ 0.90 (9 drift vs 8 null).
- Tier 1: AUROC ≥ 0.90 (6 drift vs 5 null).

### Falsification signals
- AUROC < 0.90 on either tier.
- Null pairs reach high found-rate (≥ 0.5) at budget r. This would show the budget is too loose, and the result is reported as such.

### Controlled variables
r fixed from calibration on T1 seed 0 validation; direction-union definition fixed in config.

### Minimal experiment
All pairs of §9 (EVALUATION) on both tiers with Δ-CE N and O.

### Confirmation procedure
Validation S values for all pairs (diagnostic); then test once.

### Parent and dependencies
H-004, G0.

### Prior-work relation
Hinder & Hammer 2020 and Muschalik 2022 do not produce a pair-level witness score of this kind (to be checked at G-lit).

### Experiments
exp_0017.

### Results
AUROC(S) = 1.0 for Δ-CE, B0, B1, B3 (conditional on G0-qualified pairs).

### Conclusion
Uninformative between methods, as anticipated in amendment A2.

### Follow-up
A failure here, with H-001/H-002 supported, is reported as a limitation. A failure with H-001/H-002 also failing means Route B′ is possible.

---

## H-004 — Solver reproduces linear closed forms

### Status
supported (validation check, exp_0007 under amendment v1.5, 2026-10-15)

### Origin
Proposal §3.6 "kiểm số trước khi viết" and spec §4.1 (Tier L).

### Contribution quality
Verification. Not a scientific claim. A failure is a bug.

### Hypothesis
On Tier L (1000 random pairs, p ∈ {5, 20}), the Δ-CE solver matches Proposition 3: relative error of δ_Δ < 1e-3; |cos(Pu*, PΔw)| > 0.999 for ≥ 99% of pairs.

### Rationale
The closed forms are exact, so the solver can be checked without data.

### Predictions
As stated; tolerances are those of spec §8 (H4), which replace the 1e-4 of the proposal.

### Falsification signals
Any pair outside tolerance after the repair budget (5 attempts, CONSTRAINTS §7).

### Controlled variables
κ = 0.5, R = 5, T = 400, lr = 0.05, λ0 = 1.0, no noise for the check.

### Minimal experiment
`tests/unit/test_linear_theory.py` (local, CPU, seconds).

### Confirmation procedure
Not applicable (no data split).

### Parent and dependencies
None.

### Prior-work relation
Artelt et al. Theorems 1–2 cover a subset of this case (no bias).

### Experiments
exp_0005 (commit 157320b): spec solver plus repair variants; closed forms checked by `tests/unit/test_linear_theory.py` (KKT + SLSQP).

### Results
Spec solver: found-rate 0.10 / 0.59 (p = 5 / 20). Best allowed repair (λ halved every T/8 while infeasible): found 0.97 / 1.00, but rel. error < 1e-3 in 11% / 22% of pairs and |cos| > 0.999 in 71% / 64%. Cause: λ0 above the exact-penalty threshold (found-rate) and the Adam lr = 0.05 noise floor (precision). Not a bug in the closed form. Details: `research/NOTES.md` §3, exp_0005.

exp_0006 (v1.4 two-phase solver, commit f45d5ba, single run): ordinary CE passes at 1.000 on both dims; Δ-CE passes at p=20 (1.000) and at p=5 every found witness is exact (rel. error ≤ 4.8e-6), but 3.3% of p=5 pairs have no feasible phase-1 candidate → shares 0.967 < 0.99. H4 fails.

exp_0007 (v1.5 solver, commit 6de9add): H4 passes on the original and holdout seed sets, both dims, Δ-CE and the ordinary CE of both models: every share 1.000. Max rel. error 5.8e-4 (Δ-CE, holdout p=5), min |cos| 0.9996; the residual error comes from the κ + 1e-4 tightening (F2).

### Conclusion
Supported. The v1.5 two-phase solver (D-022, D-023) reproduces Propositions 3–4 and the linear ordinary-CE closed form within the spec §8 tolerances on 4 000 pairs, including a holdout seed set. The path there (exp_0005 → v1.4 → exp_0006 → v1.5 → exp_0007) is logged in D-022/D-023 and goes into the paper.

### Follow-up
Pass → G1 pilot. Fail → fix the solver, not the closed form.

---

## H-005 — Local drifts give a predicted tie

### Status
rejected — prediction failed (exp_0017)

### Origin
Proposition 4 and proposal §5.3.

### Contribution quality
Mechanistic. It predicts a null-looking result, which is informative: it explains when Δ-CE adds nothing over B1 or B3.

### Hypothesis
On T1 and T4 (local drifts, Tier 0), |Recovery@r(Δ-CE) − Recovery@r(B3)| ≤ 0.10 and |Recovery@r(Δ-CE) − Recovery@r(B1-det)| ≤ 0.10.

### Rationale
Proposition 4 shows the linear directions agree. Local changes are approximately linear near z0.

### Predictions
As above, per tier and per condition.

### Falsification signals
Difference > 0.10 in either direction. This is reported as a contradiction of the linear-local argument, and a bound on Proposition 4 is discussed.

### Controlled variables
As H-001.

### Minimal experiment
Tier 0, T1 and T4, seeds 0–2.

### Confirmation procedure
Direction check on validation at G1; test once.

### Parent and dependencies
H-001.

### Prior-work relation
Artelt et al. Theorem 1 (linear case).

### Experiments
exp_0017.

### Results
P-tie |Δ| > 0.10 in all four comparisons: T1 vs B3 −0.53, T1 vs B1-det −0.17, T4 vs B3 +0.40, T4 vs B1-det −0.29.

### Conclusion
The predicted tie (Proposition 4) did not hold as operationalised. Post hoc (Research Lead, 2026-10-17): Proposition 4 concerns the direction after projecting out w_o, while Recovery takes the argmax of the unprojected Δφ; the primary metric is NOT changed after seeing results. Exploratory check declared in D-033 (b). **D-034 C3 (exp_0019, fresh data): FAILED** — projected Δ-CE − B3 on T1 = −0.304, 90% CI [−0.331, −0.278]; the projection explanation is retracted (D-034 consequence rule). D-034 C1 (T4: Δ-CE − B3 = +0.312, CI [0.273, 0.352]) passed: B3 is blind to the T4 bias change.

### Follow-up
Report as prediction and outcome, regardless of direction.

---

## H-006 — Proximity term is needed for locality (A1, DeepXplore-style)

### Status
proposed

### Origin
Spec §6.4 ablation A1 (λ = 0, similar to DiG-IN, which has no proximity term).

### Contribution quality
Ablation. Tests whether the proximity term is a necessary part of the method.

### Hypothesis
With λ = 0 (A1), Δ-CE witnesses are farther from z0 on average and have lower Recovery@r on T3 than with λ > 0.

### Rationale
Without proximity, the solver finds any disagreement point, which may be on another factor.

### Predictions
Recovery@r(A1) < Recovery@r(Δ-CE) on T3 (Tier 0), with the median witness distance of A1 greater than that of Δ-CE.

### Falsification signals
No difference or A1 better.

### Controlled variables
As H-001, only λ changed.

### Minimal experiment
Tier 0, T3 seeds 0–2, A1.

### Confirmation procedure
Validation diagnostic; reported with the main results, not used for acceptance.

### Parent and dependencies
H-001.

### Prior-work relation
DiG-IN (Augustin et al. CVPR 2024).

### Experiments
None yet.

### Results
None yet.

### Conclusion
None yet.

### Follow-up
Reported as an ablation.

---

## H-007 — Minimal-witness search versus DIG-style line search (B5)

### Status
reported (exp_0017): no solver claim (D-013, amendment A4). Outside Holm; reported in the main table.

### Origin
G-lit round 2: DIG returns the closest discrepancy intervals along segments between training points (L-022).

### Contribution quality
Determines whether a solver claim is allowed. Not an acceptance criterion.

### Hypothesis
On T3 (both tiers), the gradient-based minimal witness (Δ-CE) recovers the injected factor more often than a line search along segments to the K = 20 nearest training points (B5) within the same budget r.

### Predictions
d_t = Recovery@r(Δ-CE) − Recovery@r(B5) on T3 satisfies d_t > 0.10 and 95% CI_t > 0 in both tiers (clause (a) of the B5 rule, EVALUATION §15).

### Falsification signals
Any other outcome: no solver claim (clause b). If d_t < −0.10 and CI_t < 0 in any tier, B5 is reported as better in results and discussion (clause c). Per-tier outcomes reported as they are (clause d).

### Controlled variables
Same points, r, κ, space Z, metric d, φ̃.

### Minimal experiment
B5 on all T3 pairs, both tiers, in the same runs as H-002.

### Confirmation procedure
Final test procedure only.

### Parent and dependencies
H-002, H-004.

### Prior-work relation
DIG / DIG-CV (L-022).

### Experiments
exp_0017.

### Results
Tier 0 T3: Recovery@r(Δ-CE) − Recovery@r(B5) = +0.466, 95% CI [0.433, 0.499]; rule outcome no_solver_claim ((a) needs both tiers; Tier 1 dropped); (c) not triggered.

### Conclusion
Report the Tier 0 number per (d); the claim "Δ-CE finds better witnesses than DIG-style search" is NOT allowed.

### Follow-up
Report with H-002.

---

## H-008 — Rotation × offset detection rules from Proposition 2 (v1.9 H5a–H5d)

### ID / type / status
H-008 (H5a–H5d of spec amendment v1.9, D-036); mechanistic; **mixed: H5a, H5b supported; H5c, H5d not supported** (2026-10-20).

### Observation
exp_0017/exp_0019: B3 is blind to the T4 threshold shift (C1) but the projection explanation failed on T1 (C3). Proposition 2 (P5) predicts which change type each method family can see.

### Hypothesis and prediction (pre-declared, v1.9 §3)
H5a offset makes B3 blind (pooled B3 Recovery on α = 0, t > 0.5: CI95 upper ≤ 0.20); H5b B3 sees rotation and more with angle (Rec(0.5, 0.5) − Rec(0.1, 0.5): CI95 lower > 0); H5c B3 does not depend on offset (Rec(0.25, 0.60) − Rec(0.25, 0.50): CI90 ⊂ ±0.10); H5d after projection Δ-CE and B3 coincide on rotation (α ∈ {0.25, 0.5}, t = 0.5: CI90 ⊂ ±0.10). Holm m = 4.

### Experiments
exp_0024 (pilot), exp_0025 (32 pairs, qualification; D-037, D-038), exp_0026 (test run, CPU, locked solver; D-039; analysis bug D-041).

### Results (exp_0026, D-038 exclusions)
- §5.3 consistency (cell (0, 0.60)): B3 0.087, Δ-CE found 0.452 → pass (locked T4: 0.07, 0.47).
- H5a: 0.081, CI95 [0.066, 0.096], p_Holm 0.0004 → **supported**.
- H5b: 0.930 − 0.360 = +0.570, CI95 [0.527, 0.613], p_Holm 0.0004 → **supported**.
- H5c: 0.583 − 0.682 = −0.098, CI90 [−0.143, −0.053], p_Holm 0.944 → **not supported** (B3 loses ≈ 0.10 with offset; CI leaves the ±0.10 margin).
- H5d: −0.352, CI90 [−0.378, −0.327] (unprojected −0.545), p_Holm 1.0 → **not supported**; pre-declared reading (v1.9 §3): the linear analysis does not transfer to the CNN even for a rotation with small measured offset.

### Conclusion
The two detection rules (offset → B3 blind; rotation → B3 sees it, increasingly with angle) hold on CNNs under continuous control. The offset-invariance of B3 and the projection equivalence do not. Details: `research/results_v19.md`.

### Follow-up
Stage B → H-009. Diagnostic of measured offsets on null pairs: `research/results_v19.md` §7.

---

## H-009 — Abstention on real tabular null pairs (v1.9 H6, Adult)

### ID / type / status
H-009 (H6 of v1.9, D-036); other (external validity of C2); **supported** (2026-10-20).

### Hypothesis and prediction (pre-declared, v1.9 §4; D-040, D-042)
On 5 Adult null pairs (MLP 6-64-64-2; f_new = N0-style fine-tune), Δ-CE (N ∪ O) found-rate ≤ 0.05 on every qualified pair AND B1-det found-rate ≥ 0.90 on every qualified pair (300 valid test points, r_B frozen on validation).

### Experiments
exp_0027 (train, qualification 5/5, r_B = 0.8693), exp_0028 (test, once).

### Results
Δ-CE N ∪ O found-rate 0.000 on all five pairs; B1-det 0.977–1.000 → **confirmed**.

### Conclusion
The null-pair abstention of C2 carries over to real tabular data in this setting. Limits: decisive disagreement is rare on these pairs (validation DD 0.000); no decoder to a manifold, so abstention only; integer features treated as continuous.

---

## H-010 — Detection threshold under a logit offset (v1.10 H7c, label shift)

### ID / type / status
H-010 (H7c of v1.10, D-043); other (replication of C2 with exact ground truth); **supported** (2026-10-21; p_Holm = 0.0002, Holm m = 2 with H8, exp_0033/results_holm.json).

### Hypothesis and prediction (pre-declared, v1.10 §2.5)
On the 12 sub-threshold pairs (f_old seeds 0–2, class-1 logit bias |b| ∈ {0.5, 0.9}), which change 3–8% of decisions but none decisively, B1-det (no detection threshold) attributes a change at almost every query: pooled found-rate CI95 lower bound > 0.90, every pair ≥ 0.90, p_Holm < 0.05 (Holm with H8). Δ-CE (threshold 2κ in logit units) returns no witness — that part is the deterministic check U2, not the hypothesis. Not a ranking: the CEs of f_new and f_old really differ.

### Experiments
smoke_0018 (validation pipeline check), exp_0032 (test, once; new_test_set(9000 + pair_id)).

### Results (exp_0032, test, once)
U1–U3 passed: B3 invalid on every point of every pair; no Δ-CE witness on any of the 12 sub-threshold pairs; every witness in the lowered class with κ ≤ m_old(z*) ≤ |b| − κ. H7c: pooled B1-det found-rate 0.961, CI95 [0.955, 0.967], one-sided p = 1e-4 (unadjusted), per pair 0.927–0.997 → criterion met before Holm.
Descriptive: for |b| ≥ 2κ Δ-CE found a witness on 100% of lowered-class queries (found-rate 0.34–0.52 = the lowered-class share), and Recovery(aspect) ≈ found-rate (witnesses move along aspect); B1-det Recovery(aspect) 0.59–0.82 on all pairs.

### Conclusion
Replication of C2 with an exact offset: B1-det has no detection threshold, Δ-CE's is 2κ in logit units and is sharp (0% below, 100% of the lowered class above).

---

## H-011 — Locality of the projection explains the C3/H5d gap (v1.10 H8)

### ID / type / status
H-011 (H8 of v1.10, D-043); mechanistic; **supported** (2026-10-21).

### Hypothesis and prediction (pre-declared, v1.10 §3.3)
Proposition 2 linearises at z0; the witness lies up to r away. On the Δ-CE support set of the H5d pairs 36, 37, 44, 45, with the projection direction ŵ_o taken at the witness and B3 unchanged at z0 (cell (z*, z0)), the projected Recovery gap narrows by at least one half relative to the projection at z0 on the same support: T = |gap(z0,z0)| − 2|gap(z*,z0)|, CI95 lower > 0, p_Holm < 0.05, guard Rec_proj(B3) ≥ 0.50. Not testable if the CI90 of gap(z0,z0) contains 0. The cell was chosen after the validation run exp_0030 (gap −0.19 → +0.01).

### Experiments
exp_0030 (validation development), exp_0033 (test, once).

### Results (exp_0033, test, once)
Support 198, 183, 228, 228 (pairs 36, 37, 44, 45). gap(z0, z0) −0.190 [CI90 −0.222, −0.159] (testable); registered cell (z*, z0) +0.029 [0.006, 0.053]; T = 0.132, CI95 lower 0.065, p = 0.0006, p_Holm = 0.0006; guard Rec_proj(B3) 0.88 → **confirmed**. Descriptive: (z*, z*) −0.024; (z0, z*) −0.270 (moving B3 alone widens the gap). Projection deviation vs 1 − cos(ŵ_o(z0), ŵ_o(z*)): Spearman 0.80; vs ‖z* − z0‖: −0.10 (the mechanism prediction "increases with both" holds for the normal's rotation only). Extension (20 α > 0 pairs): at α ∈ {0.25, 0.5} the local projection brings Δ-CE to 0.77–0.98, level with B3; at α = 0.1 both are lower and the gap changes sign on some pairs.

### Conclusion
The C3/H5d failure is a failure of evaluating Proposition 2 at z0: after removing the old-model normal at the witness, the witness and the original gradient difference agree. About half of the locked −0.35 is found-rate (NOTES 2026-10-20). The cell was chosen after the validation run exp_0030 (disclosed).

---

## 4. Hypothesis template

Use the template from the earlier version of this file (preserved in git history) when a hypothesis is added. Required fields: ID, type, status, origin, contribution quality, hypothesis, rationale, predictions, falsification signals, controlled variables, minimal experiment, confirmation procedure, parent, prior-work relation, experiments, results, conclusion, follow-up.

---

## 5. Hypothesis lifecycle

`proposed → queued → testing → supported | rejected | inconclusive | abandoned`, as in the template.

---

## 6. Experiment-to-hypothesis rule

Every experiment must cite one hypothesis ID. Experiments that only check implementation (smoke runs, bug runs) cite H-004 or the hypothesis they gate.

---

## 7. Pre-registration discipline

The predictions, thresholds, and falsification signals above are written before any experiment. Changes after results are recorded in `research/DECISIONS.md` and are exploratory.

---

## 8. Test-set information-flow rule

See `docs/EVALUATION.md` §6. No hypothesis is updated from test-split information before the final procedure.

---

## 9. Negative and rejected hypotheses

Rejected hypotheses stay in this file with their results. They are not deleted.

---

## 10. Inconclusive hypotheses

Use `inconclusive` when the validity conditions (EVALUATION §16) fail or the CI overlaps the threshold in a way the rule does not decide.

---

## 11. Abandoned hypotheses

Use `abandoned` only with a decision entry in `research/DECISIONS.md`.

---

## 12. Hypothesis relationships

H-004 → H-001, H-002, H-003 (correctness gate). H-001 → H-005, H-006 (explanation of results). H-003 is independent of H-001/H-002.

---

## 13. Choosing the next hypothesis

Order: H-004 (G1), then the G0 phenomenon check, then H-001/H-002/H-003 in the order of their experiments, then H-005/H-006.

---

## 14. Hypothesis cleanup

Keep this file current. Move completed detail into the hypothesis entry, not into NOTES.md.

---

## 15. Completion checklist

- [x] Every hypothesis has ID, type, status, prediction, and falsification signal.
- [x] Predictions and thresholds match `docs/EVALUATION.md` §15.
- [ ] Experiments cited once they exist.
