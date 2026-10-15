# LITERATURE.md

> Purpose: Record prior work, novelty checks, and citations that materially influence the project.
>
> Ownership: Mutable research memory. The Research Lead writes the interpretation. Entries marked `discovered` come from web search at kick-off and have **not** been read in full.

---

## 1. Literature index

| ID | Title / source | Year | Relevance | Status | Related hypotheses | Code available |
|---|---|---:|---|---|---|---|
| L-001 | Artelt, Hinder, Vaquet, Feldhans, Hammer — *Contrastive Explanations for Explaining Model Adaptations* (arXiv:2104.02459) | 2021 | high | discovered | H-001, H-005, H-006 | unknown |
| L-002 | Artelt et al. — *Contrasting Explanations for Understanding and Regularizing Model Adaptations* (Neural Processing Letters) | 2022 | high | discovered (full text required, G-lit) | H-001, H-002 | unknown |
| L-003 | Hinder, Hammer — *Counterfactual Explanations of Concept Drift* (arXiv:2006.12822) | 2020 | high | discovered | H-003 | unknown |
| L-004 | Hammer, Hüllermeier — *Interpretable Machine Learning: On the Problem of Explaining Model Change* | 2021 | medium | discovered (full text required) | H-003 | no |
| L-005 | Muschalik, Fumagalli, Hammer, Hüllermeier — *Agnostic Explanation of Model Change based on Feature Importance* (KI) | 2022 | medium | discovered | H-002 (B4) | unknown |
| L-006 | Augustin et al. — *DiG-IN: Diffusion Guidance for Investigating Networks* (CVPR; arXiv:2311.17833) | 2024 | high | discovered | H-006 | unknown |
| L-007 | Weng et al. — *FastDiME* (ECCV; arXiv:2312.14223) | 2024 | medium | discovered | — | unknown |
| L-008 | Hamman et al. — *Robust Counterfactual Explanations for Neural Networks With Probabilistic Guarantees* (ICML) | 2023 | medium | discovered | — | unknown |
| L-009 | *Provably Robust Bayesian Counterfactual Explanations under Model Changes* (PSCE; arXiv:2601.16659) | 2026 | medium | discovered | — | unknown |
| L-010 | *Probabilistically robust counterfactual explanations under model changes* (Artificial Intelligence) | 2025 | medium | discovered | — | unknown |
| L-011 | Pawelczyk, Broelemann, Kasneci — *On Counterfactual Explanations under Predictive Multiplicity* (UAI) | 2020 | medium | discovered | H-003 (Rashomon) | unknown |
| L-012 | Altmeyer et al. — *ECCCo* (AAAI) | 2024 | low | discovered | — | unknown |
| L-013 | Castro et al. — *Morpho-MNIST* (JMLR) | 2019 | high | discovered | Tier 1 | yes (public) |
| L-014 | *A Geometric View of Counterfactual Behavior* (arXiv:2606.04209) | 2026 | low | discovered | motivation only | unknown |
| L-015 | *Explaining Concept Drift through the Evolution of Group Counterfactuals* (arXiv:2509.09616) | 2025 | low | discovered | — | unknown |
| L-016 | Wachter, Mittelstadt, Russell — *Counterfactual Explanations without Opening the Black Box* | 2017 | medium | discovered | background | n/a |
| L-017 | Yan et al. — *Positive-Congruent Training* (CVPR) | 2021 | low | discovered; not re-checked in the proposal | negative flips | unknown |
| L-018 | Kaden, Paassen, Hammer, Schubert, Villmann — *Reliable Counterfactuals for Machine Learning Models* (ESANN 2026) | 2026 | low | discovered | venue context | unknown |

Allowed relevance values: high, medium, low.

Suggested status values: discovered, screened, reviewed, follow_up_needed, not_relevant, superseded.

**Caveat.** arXiv identifiers and years in this table come from the proposal's web search (15/10/2026). They must be checked against the source before citation (spec §14 flags Yan et al. for this).

---

## 2. Search history

| Date | Query or source | Result |
|---|---|---|
| 2026-10-15 | Proposal §2.1 web search (Artelt 2021/2022, Hinder & Hammer, Muschalik, DiG-IN, FastDiME, Hamman, PSCE, ECCCo, Geometric view, Group CE) | list L-001–L-015 |
| 2026-10-15 | Spec §2 (reading of Artelt et al. arXiv:2104.02459v2 §3, Def. 3, Thm 1–2, §4.2, experiments) | recorded in spec; full text check not repeated in this session |

Not yet done (G-lit, due 17/10): full-text reading of L-002, L-004, L-005; citing-paper search (Semantic Scholar) for L-001 and L-006 with terms "explaining model updates", "negative flip explanation", "backward-compatible explanations", "explanation of model differences counterfactual", "disagreement witness".

---

## 3. Source entry template

Use the template in the earlier version of this file (git history). Full entries are written for L-001, L-002, L-003, L-006 at G-lit.

---

## 4. Baseline literature

- **B1 (CE differencing):** Artelt et al. 2021, Definition 3.
- **B3 (gradient difference):** Artelt et al. 2021, Eq. 14 (approximation).
- **B4 (global feature importance difference):** Muschalik et al. 2022.
- **B1-det, A1–A3:** this project's ablations of B1 and Δ-CE.

---

## 5. Related-method map

| Family | Methods | Relation to Δ-CE |
|---|---|---|
| Explaining model adaptation with counterfactuals | Artelt 2021/2022 (L-001, L-002) | closest; uses CE differencing (B1) |
| Explaining drift of data | Hinder & Hammer 2020 (L-003) | explains data drift, not model change |
| Feature-importance model change | Muschalik 2022 (L-005), Hammer & Hüllermeier 2021 (L-004) | global; B4 |
| Diffusion-guided disagreement | DiG-IN (L-006) | ablation A1 (no proximity) |
| Robust CE under model change | Hamman 2023 (L-008), PSCE (L-009) | opposite goal: CE valid under both models; dual |
| Shortcut via CE | FastDiME (L-007) | one model; Δ-CE compares two |

---

## 6. Novelty map

| Claim | Novelty status | Reason |
|---|---|---|
| Δ-CE as a witness on which two models disagree, on the manifold, within budget | candidate | not found in the proposal search; to be confirmed at G-lit (L-002 full text is the main check) |
| Band containment lemma (any witness has |h| ≥ m_o + κ) | candidate | elementary; still checked against L-002 |
| Null certificate (Proposition 2) | candidate | Lipschitz form; not certified for deep networks |
| Linear direction recovery (Proposition 4) | not_novel | Artelt Theorems 1–2 cover the direction case (no bias); Proposition 4 reproduces it and is cited, not claimed |
| Linear closed form with bias and margin (Proposition 3) | candidate | Artelt's linear results exclude bias; the constrained case with margin κ is an extension to check at G-lit |
| Benchmark with exact or analytic ground truth for model change | candidate | none found in the proposal search |
| Predicted tie on local drift (P-tie) | candidate | a prediction, not a result |

Novelty values follow the template: candidate | supported | uncertain | not_novel. **No claim of novelty is made until G-lit is complete.**

---

## 7. Novelty status semantics

Use the template definitions. The current overall status is `candidate`.

---

## 8. Open research gaps

- A witness-based explanation of decision change on the data manifold, with a budget, for deep models.
- A benchmark where the change type is known and the explanation can be scored exactly.
- A test for non-local change that the local gradient approximation cannot see.

---

## 9. Contradictory evidence

- Proposition 4: in the linear case Δ-CE gives no direction advantage over CE differencing. This is a limit on the claim, not an empirical contradiction.

---

## 10. Literature-derived hypotheses

H-001 to H-003 are taken from the gaps in §8. H-005 comes from Proposition 4 (Artelt Theorem 1 in the linear case).

---

## 11. Novelty threats

- **Artelt et al. NPL 2022** may already define a disagreement-based explanation in its "persistent explanations" section. The spec states that Artelt et al. set disagreement regions aside (spec §2, citing §3 of arXiv:2104.02459). This must be checked in full text. **If a disagreement witness with a ground-truth benchmark exists, the project moves to Route B′ or Stop (spec §10).**
- **Hinder et al. (model-based explanations of concept drift)** is listed in the spec's G-lit. Not yet read.
- **Muschalik et al. 2022** may include a local variant. To be checked.

---

## 12. Search gaps and next searches

Full-text reading for G-lit (due 17/10):

1. L-002 Artelt et al. NPL 2022: the persistent-explanations section and theorems.
2. L-004 Hammer & Hüllermeier 2021.
3. L-005 Muschalik et al. 2022.
4. Hinder et al., "Model-based explanations of concept drift".
5. Citing papers of L-001 and L-006 (Semantic Scholar).
6. Keyword searches listed in §2.

---

## 13. Source quality and authority

Peer-reviewed venues (NPL, KI, JMLR, CVPR, ECCV, ICML, AAAI, UAI) are preferred. arXiv entries are used only with the version stated. Web summaries are not evidence.

---

## 14. Literature search policy

Record the query and date in §2 for each search that influences a decision.

---

## 15. Novelty review before contribution claim

No contribution claim is written into the paper until §6 is updated after G-lit, and the L-002 full text has been read.

---

## 16. Search saturation

Not reached.

---

## 17. What not to store

Full copies of papers; long quotes beyond what a decision needs.

---

## 18. Maintenance rules

Update the index, the search history, and the novelty map together.

---

## 19. Completion checklist

- [x] Index with IDs and statuses.
- [x] Search history recorded.
- [x] Novelty map with status values.
- [ ] Full-text reading of L-002, L-004, L-005, and Hinder et al. (G-lit).
- [ ] Citing-paper search done.
- [ ] arXiv identifiers verified.
