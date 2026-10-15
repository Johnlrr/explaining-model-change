# GOAL.md

> Purpose: Define what this research project is trying to achieve.
>
> Ownership: Project specification. Treat this file as read-only during autonomous research unless the user explicitly requests a change.
>
> Detailed protocol source: `docs/source/delta_ce_spec.md` (spec v1.0, freeze 17/10/2026). Where that file and `docs/EVALUATION.md` differ, `docs/EVALUATION.md` governs, and the difference is recorded in `research/DECISIONS.md`.

---

## Research question

Do **differential counterfactuals** (Δ-CE) — the closest input on the data manifold on which exactly one of two models changes its decision — recover the factor responsible for an injected model change, and separate real behaviour-changing updates from null updates, more reliably than counterfactual differencing (Artelt et al.) and gradient-difference baselines?

Short form:

```text
Can minimal disagreement witnesses (Δ-CE) explain how a classifier's decision behaviour changed after adaptation, better than CE differencing or gradient-difference, under controlled injected changes with known ground truth?
```

---

## Primary objective

Produce a validated methodological contribution with three parts:

1. a formal definition of Δ-CE with the proven results in the spec (band containment, null certificate, linear closed form, linear direction recovery with its stated limits);
2. a solver and a ground-truth benchmark (Tier 0 renderer, Tier 1 colored MorphoMNIST with analytic oracles) with injected drifts T1, T3, T4 and null pairs N0, N1;
3. an empirical answer, on that benchmark, to the hypotheses H1–H4 in `docs/EVALUATION.md`, reported honestly under the route rules in §10 of the spec (Route A, B, B′, or stop).

A negative or partly negative answer is an acceptable outcome if the protocol is followed (Route B or B′).

---

## Motivation

After a classifier is adapted, practitioners need to know *what behaviour changed*, not only that accuracy moved. Current model-change explanations either compare two independent counterfactuals, which need not describe a point where the models disagree, or rank features globally, which gives no concrete input. A witness that both models classify differently, on the data manifold and within a distance budget, is a checkable object, and it may expose changes that local gradients at a point cannot see.

---

## Baseline

Primary baseline (B1, Artelt et al. Definition 3):

```text
Method/model: CE differencing. Compute z'_old = CE(z0; f_old) and z'_new = CE(z0; f_new) independently
              with the same optimizer and R restarts; explanation = φ(D(z'_new)) − φ(D(z'_old)).
Reference implementation or source: Artelt et al., arXiv:2104.02459 (2021); Neural Processing Letters 2022.
              Re-implemented in this project under src/dce/baselines/ (no external code is assumed).
Relevant version/checkpoint: not applicable (re-implemented baseline).
Why this is the primary baseline: it is the closest published method that explains model adaptation through
              counterfactuals, so H1 is defined against it.
```

Secondary baselines: B1-det (deterministic B1), B3 (gradient difference, Artelt Eq. 14), B4 (global permutation-importance difference; Tier 0 only). The baseline is reproduced under the same benchmark, splits, budget `r`, and metrics before any Δ-CE result is judged (see `docs/EVALUATION.md` §7).

---

## Target contribution

**Contribution type: methodological** (new formulation of model-change witnesses, with theory, solver, and a ground-truth benchmark). Routine tuning of `κ`, `λ`, `r`, or learning rates is supporting work only and does not satisfy this goal. Novelty is not assumed: the candidate contribution status is recorded as `candidate` in `research/LITERATURE.md` and must be re-checked after the G-lit gate.

Claims that are permitted and not permitted are listed in `research/LITERATURE.md` §6 and in the proposal's §2.2. In short:

- permitted: a distance-based null certificate (for the idealised case), the linear closed form for the constrained problem, behaviour-level detection of changes, and a benchmark with ground truth;
- not permitted: "first to explain model change with counterfactuals"; "Δ-CE recovers the change direction better than CE differencing in general" (false in the linear case, Proposition 4); a certified null for deep networks.

---

## Success criteria

The project is successful only if the primary decision rule in `docs/EVALUATION.md` §15 is satisfied on the held-out test split, after the candidate is locked, and if final verification (`docs/EVALUATION.md` §20) passes. In plain terms: Δ-CE must (a) detect real drift versus null pairs with high AUROC on both tiers, and (b) beat the baseline on at least one predeclared drift comparison with a confidence interval excluding zero.

Exact thresholds, seeds, and statistical rules are in `docs/EVALUATION.md`, not here.

---

## Secondary objectives

- Report where Δ-CE does not beat B1 or B3 (in particular the predicted tie on local drifts T1 and T4), and explain it with Proposition 4.
- Report found-rate, runtime, and plausibility on Tier 1.
- Report the behaviour-versus-mechanism limit: pairs whose internal rules changed but whose decisions did not should return "no change".

Secondary objectives must not replace the primary objective.

---

## Out of scope

- Real-world datasets, real drifts, or human-labelled audits. All ground truth is synthetic or analytic.
- Diffusion, inpainting, or generative models as the search space (the spec uses a renderer and a small VAE only).
- Architecture search, large models, or ImageNet-scale data.
- Claims about mechanisms (weights, rules) that are invisible at the decision level.
- Deployment, latency, or hardware optimisation beyond what the runtime report requires.
- Extensions to multi-class models, regression, or tabular data.

---

## Non-goals

- Reporting only the best seed, pair, or restart.
- Changing `κ`, `r`, the benchmark, the splits, or the metric after seeing test results.
- Calling a null-pair witness "wrong" merely because it exists (Rashomon effects are reported, not penalised).
- Counting a predicted tie (Proposition 4 on T1/T4) as a failure, or claiming a win from one tier only.
- Using test-split information for any development decision (see `docs/EVALUATION.md` §6).
- Presenting a proposal, theorem sketch, or literature inference as an established result.

---

## Completion checklist

- [x] The research question is precise and answerable by evidence.
- [x] The primary objective is clearly stated.
- [x] Motivation is present and short.
- [x] A primary baseline is identified unambiguously (B1).
- [x] The intended contribution type is stated (methodological).
- [x] High-level success criteria are defined.
- [x] Exact evaluation details are delegated to `docs/EVALUATION.md`.
- [x] Out-of-scope boundaries are explicit.
- [x] Non-goals prevent obvious shortcuts or misleading success claims.
- [x] Human confirmation of D-004 (venue and 6-page limit verified by the user) and of the spec-based gate procedure (D-009), 2026-10-15.
