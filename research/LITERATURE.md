# LITERATURE.md

> Purpose: Record prior work, novelty checks, and citations that materially influence the project.
>
> Ownership: Mutable research memory. The Research Lead writes the interpretation. Entries marked `discovered` come from web search at kick-off and have **not** been read in full.

---

## 1. Literature index

| ID | Title / source | Year | Relevance | Status | Related hypotheses | Code available |
|---|---|---:|---|---|---|---|
| L-001 | Artelt, Hinder, Vaquet, Feldhans, Hammer — *Contrastive Explanations for Explaining Model Adaptations* (arXiv:2104.02459v2; IWANN 2021) | 2021 | high | reviewed (full text, 26 pp.) | H-001, H-005, H-006 | unknown |
| L-002 | Artelt et al. — *Contrasting Explanations for Understanding and Regularizing Model Adaptations* (Neural Processing Letters) | 2022 | high | follow_up_needed (journal full text paywalled; arXiv version of L-001 read instead) | H-001, H-002 | unknown |
| L-003 | Hinder, Hammer — *Counterfactual Explanations of Concept Drift* (arXiv:2006.12822) | 2020 | high | screened (abstract and citing search) | H-003 | unknown |
| L-004 | Hammer, Hüllermeier — *Interpretable Machine Learning: On the Problem of Explaining Model Change* | 2021 | medium | follow_up_needed (not found in search; not read) | H-003 | no |
| L-005 | Muschalik, Fumagalli, Hammer, Hüllermeier — *Agnostic Explanation of Model Change based on Feature Importance* (KI, DOI 10.1007/s13218-022-00766-6) | 2022 | medium | screened (identified; full text not read) | H-002 (B4) | unknown |
| L-006 | Augustin et al. — *DiG-IN: Diffusion Guidance for Investigating Networks* (CVPR; arXiv:2311.17833) | 2024 | high | screened (abstract and search result descriptions) | H-006 | unknown |
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
| L-019 | Hinder, Vaquet, Brinkrolf, Hammer — *Model Based Explanations of Concept Drift* (arXiv:2303.09331; Neurocomputing 2023) | 2023 | medium | reviewed (arXiv full text, keyword read) | H-003 | no |
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

| 2026-10-15 | G-lit: arXiv:2104.02459 PDF read (text extracted with pypdf): Section 2–5 and experiments | verified the points in the G-lit result below |
| 2026-10-15 | G-lit: Springer NPL 2022 page | paywalled (login redirect); journal full text NOT read |
| 2026-10-15 | G-lit: arXiv:2303.09331 PDF, keyword read (counterfactual, disagree, ground truth, characteristic) | no old-vs-new classifier witness; see L-019 |
| 2026-10-15 | G-lit: OpenAlex citing lists for Artelt 2021 (7 citers), NPL 2022 (6), DiG-IN (4); Semantic Scholar was rate-limited | no citer defines a disagreement witness; OpenAlex coverage is partial |
| 2026-10-15 | G-lit: web searches "explaining model updates / negative flips / old vs new classifier disagreement", "explaining model differences counterfactual two classifiers" | closest: DiG-IN (L-006), robust-CE papers (L-008–L-010); no witness-on-disagreement-region paper found |
| 2026-10-15 | ESANN 2027 special session query | session found via an organiser's post; conference 21–23 Apr 2027, deadline 18 Nov 2026 per esann.org (see DECISIONS D-004) |

### G-lit result (2026-10-15)

Verified in the full text of arXiv:2104.02459 (L-001):

- Section 3 states that an "obvious approach" is to explain the region {x | h(x) ≠ h'(x)}, but "this set might be small and its characterization not meaningful", so they explain generalisation rules via a comparison of contrastive explanations instead (footnote 8 calls the disagreement region "an obvious and reasonable starting point"). The spec's claim C2 is correct.
- Definition 3 compares the counterfactuals of the two models at a point; Ψ is a distance or a cosine form (Eq. 12–13). The interest function is i(x,h,h') = Ψ(CF(x,h), CF(x,h')) (Eq. 11), and Eq. 14 gives the gradient approximation CF(x,h) ≈ x − η h(x) ∇f(x). Matches B1 and B3 in the spec.
- Theorem 1 and 2 are for linear sign(w^T x) classifiers with ‖w‖ = 1 and no bias (Eq. 8). Matches spec §2.
- Experiments (Gaussian blobs, Boston housing, German credit, loan approval, coffee hyperspectral, human activity recognition; GNB, linear regression, decision tree) are judged by plotting the mean counterfactual differences. In Gaussian blobs and house prices the changed feature is known and checked by eye; for HAR the text says "we do not really know the ground truth". No quantitative recovery metric, no image data, no null pairs.
- Adaptation setup (Eq. 1) requires new and old model to agree on the labelled samples, which is why disagreement sets are small.

Not verified (limits of this G-lit): the journal text of L-002 (paywalled); L-004 (Hammer & Hüllermeier 2021; not found); L-005 full text; the citing-paper lists are from OpenAlex only. Gate status: **passed provisionally**; no witness-on-disagreement-region work or ground-truth model-change benchmark was found in what could be read. The L-002 journal text remains the main residual risk (§11).

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

Novelty values follow the template: candidate | supported | uncertain | not_novel. **G-lit passed provisionally (see §2); novelty statuses stay `candidate` until the remaining reads in §12 are done.**

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

- **Artelt et al. NPL 2022 (journal version)** was not readable (paywall). The arXiv/IWANN version confirms that disagreement regions are set aside (see G-lit result) and its "persistent explanations" section (Section 5) constrains the adapted model so that chosen local explanations stay valid; it does not search for disagreement witnesses. The journal version adds a region-finding method; if it contains a disagreement witness with a ground-truth benchmark, the project moves to Route B′ or Stop (spec §10). Re-check if full text becomes available (author preprint or institutional access).
- **Hinder et al. (L-019)** explains data drift by training models that separate the data before/after drift and applying explainers (feature importance, prototypes, counterfactuals). It does not compare an old and a new classifier. Low threat.
- **Muschalik et al. 2022** may include a local variant. To be checked.

---

## 12. Search gaps and next searches

Full-text reading for G-lit (due 17/10):

Remaining after G-lit:

1. L-002 journal full text (paywalled) — try the Bielefeld repository record or an author preprint.
2. L-004 Hammer & Hüllermeier 2021 — locate.
3. L-005 Muschalik et al. 2022 — read the KI article.
4. Re-run the citing-paper search with Semantic Scholar when the rate limit allows.
5. Verify arXiv identifiers and years in §1 before citation.

---

## 13. Source quality and authority

Peer-reviewed venues (NPL, KI, JMLR, CVPR, ECCV, ICML, AAAI, UAI) are preferred. arXiv entries are used only with the version stated. Web summaries are not evidence.

---

## 14. Literature search policy

Record the query and date in §2 for each search that influences a decision.

---

## 15. Novelty review before contribution claim

No contribution claim is written into the paper until §6 is updated and the remaining items in §12 are closed or accepted as residual risk by the human.

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
- [x] G-lit performed (provisional pass); arXiv version of L-001 read in full.
- [ ] L-002 journal text, L-004, L-005 full texts (remaining residual risk).
- [ ] Citing-paper search repeated on Semantic Scholar.
- [ ] arXiv identifiers verified.
