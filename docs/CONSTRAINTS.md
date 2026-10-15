# CONSTRAINTS.md

> Purpose: Define the operational, resource, and methodological limits that govern this research project.
>
> Ownership: Project specification. Treat this file as read-only during autonomous research unless the user explicitly requests a change.
>
> This file defines what the agent may and may not do while pursuing the goal in `docs/GOAL.md`.

---

## 1. Constraint policy

HARD and SOFT are as defined in the template. Each constraint below is labelled.

---

## 2. Compute budget

```text
GPU budget:
Classification: HARD
Limit: 10 GPU-hours total for the project, counting every Kaggle GPU run (smoke, full, verification, retry).
  Estimate in the spec: ~4.5 GPU-hours. Tier 0 and Tier L run on local CPU and do not count.

CPU budget:
Classification: SOFT
Limit: local machine (16 logical cores). No CPU-hour cap; Tier 0 full runs are expected to take ~1–2 CPU-hours.

RAM limit:
Classification: SOFT
Limit: 15 GB local (WSL2). Kaggle kernel RAM follows the platform.

Storage limit:
Classification: SOFT
Limit: Not constrained beyond the repository; checkpoints are < 5 MB each (§12 of EVALUATION.md).
```

Kaggle GPU accounts are checked for quota with the repository harness before submission (`python -m tools.kaggle accounts quota`). The GPU-hour count is kept in `research/STATE.md` §11.

---

## 3. Time budget

```text
Overall research deadline:
Classification: HARD
Limit: 18/11/2026 (hard submission deadline for ESANN 2027, as stated in the source spec; venue dates are NOT
       yet verified, see research/NOTES.md). Internal deadlines: spec freeze 17/10/2026 23:59 ICT;
       G0 20/10; G1 21/10; G2 24/10; G3 1/11; internal submission 16/11.

Maximum full-run duration:
Classification: HARD
Limit: 60 minutes per Kaggle full job (remote hard timeout; see §6).

Preferred experiment duration:
Classification: SOFT
Limit: Tier 0 full run for one pair ≤ 5 minutes per pair (G1 requirement).
```

Do not continue past the submission deadline. If a gate date is missed, record the reason in `research/DECISIONS.md` and re-plan; the gate itself is not moved after the fact without a recorded decision.

---

## 4. Experiment budget

```text
Maximum full experiments:
Classification: SOFT
Limit: Not constrained beyond the 10 GPU-hour budget. Tier 1 uses at most 4 Kaggle jobs (spec §11):
       (a) data + VAE + oracle; (b) classifiers; (c) methods on T1 + T3; (d) N0 + ablations.
       Tier 0 and Tier L run locally on CPU.

Maximum expensive ablations:
Classification: SOFT
Limit: A1–A3 on Tier 0 (all pairs); A1–A3 on Tier 1 for seed 0 only.
```

Cheap local checks and smoke runs do not count toward the experiment budget but do count toward the GPU budget when run on Kaggle.

---

## 5. Concurrency override

```text
Override: none
```

Default (2 concurrent Kaggle GPU jobs, no concurrent versions of one kernel slug) applies.

---

## 6. Execution timeout override

```text
Full-run timeout override:
Constraint: Kaggle remote hard timeout for Tier 1 full jobs
Classification: HARD
Limit: 3600 seconds (timeout_seconds in the config); wait_timeout_seconds 4200; poll_interval_seconds 60.
Reason: the source spec requires each Tier 1 job to be < 60 minutes and to checkpoint.

Smoke timeout override: none (default 900 seconds).
Local command timeout override: none.
```

A local command timeout does not cancel the remote job. After any local timeout, check remote status before resubmitting.

---

## 7. Retry override

```text
Override: none
Maximum repair/retry attempts per failure class: 5 (default in docs/RESEARCH_PROTOCOL.md)
```

A retry for a valid experiment must keep the same scientific variables (`docs/EVALUATION.md` §17).

---

## 8. Allowed resources

```text
Pretrained models: Not allowed. All classifiers and the VAE are trained from scratch in this project.
External checkpoints: Not allowed.
External datasets: Restricted. MNIST (digits 4 and 9, official splits) via torchvision only, for Tier 1.
                   No other dataset.
Synthetic data: Allowed. Tier 0 renderer and Tier 1 colour/background/thickness augmentation are project code.
External repositories/code: Restricted. May be read as references (Artelt et al., Morpho-MNIST, DiG-IN, FastDiME).
                   No code is copied in; the baselines are re-implemented under src/dce/baselines/ and recorded in
                   research/LITERATURE.md. Morpho-MNIST package use is optional (skimage implementation in spec §4.3).
Internet access: Allowed for literature search, dependency acquisition, and the Kaggle preparation kernel
                   that downloads MNIST. Not allowed for the experiment kernel (see docs/ENVIRONMENTS.md §7).
Open-source libraries: Allowed: PyTorch, torchvision, numpy, scipy, scikit-image, PyYAML, pytest, ruff.
Hosted APIs: Not allowed (no LLM or hosted-model calls inside experiments).
Paid APIs/services: Not allowed.
Competition-provided assets: Not applicable.
Human-generated labels: Not allowed. All ground truth is exact (Tier 0) or analytic (Tier 1 oracle).
                   Shape oracle is trained on the project's own synthetic renders, not on human labels.
Other: Not allowed: diffusion or inpainting models in the search space (spec scope).
```

---

## 9. Forbidden resources and actions

```text
- Using test-split samples for anything except the final procedure (EVALUATION §6, D-011).
- Editing docs/ during autonomous research (AGENTS.md).
- Committing credentials; Kaggle tokens stay in the private accounts directory (README "Connected Kaggle accounts").
- Running any experiment on a Kaggle account other than the ones connected through the repository harness.
```

---

## 10. Data constraints

```text
- Tier 0 train/val/test sizes and seed (1234) are fixed (spec §4.2). Changing them is a benchmark change → forbidden after freeze.
- Tier 1 uses MNIST digits 4 and 9 only; 10% of official train becomes validation; official test is the test split.
- No augmentation other than those in spec §4.2–§4.3, and label noise (5%) only on Tier 0 training split.
- Adaptation sets (10 000 Tier 0 / 8 000 Tier 1) are sampled from the training split only.
- Thickness ground truth is the measured value of §4.3 of the spec; it is not replaced by the nominal target.
```

---

## 11. Method constraints

```text
- Δ-CE solver hyperparameters are fixed: κ = 0.5, R = 5, T = 400, lr = 0.05, λ0 = 1.0, σ = 0.05, batch 300.
  Tier 1 regulariser μ = 0.01 on ||z||²; Tier 0 uses the clamp to [0,1]^5 only.
- r is calibrated once (2 × median CE distance of f_new on 300 T1-seed-0 validation points), then frozen.
- Architecture is fixed: the classifier in spec §5.1 (≈ 60k parameters) and the VAE in spec §5.2 (latent dim 16).
- Allowed knobs for G0 only, at most two adjustments per condition: spec §4.4, with T3/T4 replaced by v1.2 B4 (knob 1: epochs 5 and lr 1e-3; knob 2: epochs 10). Any other tuning is forbidden. f_old training is fixed by v1.2 B1; if a seed misses the accuracy threshold after B1, G0 fails and goes back to the author.
- The method must not use the label of a test example, a test margin, or a test witness to choose any setting (EVALUATION §6).
```

---

## 12. Comparison-resource constraints

```text
- Baselines B1, B1-det, B3, B4 use the same classifiers, points, r, and φ̃ as Δ-CE.
- The compute budget per method is matched: B1 uses R = 5 restarts for each of its two CE searches, as does Δ-CE.
- Runtime is reported per device; CPU and GPU runtimes are never compared directly.
```

---

## 13. Research integrity constraints

```text
- Changes after a main test result are not used for that test; they are reported as exploratory (spec §13).
- Every change to spec or method after freeze (17/10/2026 23:59 ICT) is logged in research/DECISIONS.md with:
  ID, date, section, change, reason, and "before / after seeing results".
- Predicted ties (P-tie) are reported and not treated as failures.
- Negative results are recorded and not hidden.
```

---

## 14. Human-approval boundaries

The agent must stop and ask the human before:

- changing anything under `docs/` (including the frozen spec and any item in `research/DECISIONS.md` marked "pending confirmation");
- changing the venue, the deadline, the primary metric, the acceptance rule, or the test-set policy;
- spending more than 10 GPU-hours or running more than the Tier 1 job count without a recorded decision;
- using any resource in §8 marked Restricted outside its stated boundary.

---

## 15. Stop conditions

Stop and report (route in `docs/EVALUATION.md` §15):

- G0 fails after the two pre-declared adjustments (route Stop);
- H4 fails after the repair budget (implementation bug unresolved);
- the Kaggle quota or accounts become unavailable and cannot be restored;
- the submission deadline passes without a valid result;
- the GPU-hour budget is reached.

---

## 16. Resource priority under scarcity

If GPU budget is short: keep Tier L and Tier 0 (CPU, no budget) complete first; then Tier 1 seed 0 for all conditions; then the remaining Tier 1 seeds. Ablations on Tier 1 are dropped before drift pairs.

---

## 17. Constraint conflicts

If the 10 GPU-hour budget conflicts with the completeness of §9 in `docs/EVALUATION.md`, the budget wins and the affected pairs are reported as not run, not silently dropped. The decision is recorded in `research/DECISIONS.md`.

---

## 18. Exceptions and overrides

None at kick-off.

---

## 19. Completion checklist

- [x] Each constraint is labelled HARD or SOFT.
- [x] Compute budget defined.
- [x] Time budget defined (venue dates pending verification).
- [x] Experiment budget defined.
- [x] Timeout overrides defined.
- [x] Allowed resources defined.
- [x] Human-approval boundaries defined.
