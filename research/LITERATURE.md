# LITERATURE.md

> Purpose: Track prior work, search history, source evidence, and novelty assessment for this research project.
>
> Ownership: Mutable research state. The agent should update this file whenever prior work is searched, reviewed, or found to materially affect hypotheses, baselines, novelty, interpretation, or experiment design.
>
> This file is **not a bibliography dump**. Record sources only when they materially affect the research.

---

## 1. Literature index

Keep this table current.

| ID | Title / source | Year | Relevance | Status | Related hypotheses | Code available |
|---|---|---:|---|---|---|---|
| L-001 | <title> | <year> | high | reviewed | H-001 | yes |

Allowed relevance values:

```text
high
medium
low
```

Suggested status values:

```text
discovered
screened
reviewed
follow_up_needed
not_relevant
superseded
```

---

## 2. Search history

Track literature searches so the agent does not repeatedly search the same space without purpose.

For each meaningful search, record:

```text
Search ID:
Date:
Research question / purpose:
Query:
Search source:
Filters:
Top relevant sources:
Outcome:
Next search:
```

Example:

```text
Search ID: S-004
Date: 2026-09-16
Research question / purpose:
Find prior work using localized restoration for adversarial defense.

Query:
"localized adversarial image restoration defense confidence mask"

Search source:
Google Scholar / arXiv / web search

Filters:
2020-present, vision, adversarial robustness

Top relevant sources:
L-007, L-011

Outcome:
Found regional restoration methods, but no direct match for confidence-weighted partial restoration.

Next search:
Search specifically for confidence-weighted or uncertainty-weighted regional purification.
```

Search history should help answer:

```text
What did we already search?
What terminology did we use?
What gaps remain?
What search would add new information?
```

Do not create entries for trivial duplicate searches.

---

## 3. Source entry template

Copy the following section for each source that materially matters.

---

## L-XXX — <title>

### Bibliographic information

```text
Title:
Authors:
Year:
Venue:
Source type: paper | preprint | benchmark | repository | blog | documentation | other
URL:
DOI / identifier:
```

Use the most authoritative source available.

---

### Relevance

```text
Relevance: high | medium | low
Status: discovered | screened | reviewed | follow_up_needed | not_relevant | superseded
Related hypotheses:
Related decisions:
Related baseline:
```

---

### Problem addressed

<REQUIRED — What problem is the source trying to solve?>

Keep this concise.

---

### Method

<REQUIRED — Summarize the core method or idea in your own words.>

Focus on the mechanism relevant to this project.

Do not reproduce large passages from the source.

---

### Experimental setting

Record what matters for comparison:

```text
Dataset / benchmark:
Model:
Training setup:
Evaluation protocol:
Primary metric:
Compute / hardware if relevant:
```

If unknown:

```text
Unknown / not reported
```

---

### Main reported findings

<REQUIRED — Summarize the findings reported by the source.>

Clearly distinguish source-reported facts from our interpretation.

Use:

```text
Reported by source:
- ...

Our interpretation:
- ...
```

Do not convert an interpretation into a claim that the source itself made.

---

### Assumptions and limitations

Record limitations relevant to this project.

Examples:

- different benchmark;
- weaker attack/evaluation;
- unavailable implementation details;
- high compute requirement;
- dependence on external data;
- incompatible deployment assumptions;
- unclear reproducibility;
- evaluation on a narrow model family.

---

### Relevance to this project

<REQUIRED — Explain why this source matters here.>

Possible roles:

```text
baseline
directly related method
supporting mechanism
contradictory evidence
evaluation reference
implementation reference
novelty threat
background only
```

Explain the relation explicitly.

---

### Relation to hypotheses

```text
Supports:
Challenges:
Motivates:
Overlaps with:
No direct relation:
```

Reference hypothesis IDs where possible.

Example:

```text
Motivates: H-006
Overlaps with: H-009
```

---

### Code / repository availability

<REQUIRED>

```text
Official code available: yes | no | unclear
Repository:
License:
Pretrained weights available: yes | no | unclear
Reproduction notes:
```

If the repository is unofficial, say so.

Code availability is especially important for baseline reproduction and fair comparison.

---

### Follow-up needed

```text
None
```

or:

```text
- inspect official implementation;
- verify exact benchmark split;
- compare attack settings;
- check whether method already includes mechanism proposed in H-007.
```

---

## 4. Baseline literature

Track the authoritative sources for the primary baseline.

For each baseline:

```text
Baseline:
Primary source:
Official implementation:
Reported benchmark:
Reported metric:
Relevant version/checkpoint:
Known reproduction notes:
Local reproduction status:
Related literature IDs:
```

Use this section to avoid ambiguity about what “the baseline” means.

Detailed reproduction results belong in run evidence and `research/STATE.md`.

---

## 5. Related-method map

Group important prior work into method families.

Example:

```text
### Purification-based defenses
- L-003
- L-008

### Detection + localized restoration
- L-011
- L-014

### Test-time adaptation
- L-017
```

For each family, summarize:

```text
Core idea:
Strengths:
Weaknesses:
How it differs from our current direction:
```

This section should make the research landscape easier to reason about than a flat bibliography.

---

## 6. Novelty map

Track novelty for every hypothesis or method intended as a contribution.

Template:

```text
Candidate:
Related hypothesis:
Closest prior work:
Key overlap:
Key difference:
Evidence supporting the difference:
Remaining uncertainty:
Novelty status: candidate | supported | uncertain | not_novel
Next search:
```

Example:

```text
Candidate:
Confidence-weighted partial restoration

Related hypothesis:
H-007

Closest prior work:
L-012, L-019

Key overlap:
Both localize adversarially affected image regions before restoration.

Key difference:
Prior work restores the full localized region, while H-007 uses confidence-weighted partial restoration.

Evidence supporting the difference:
No reviewed source currently found using the same weighting mechanism under the same defense setting.

Remaining uncertainty:
Search has not yet covered all restoration/purification work before 2020.

Novelty status:
uncertain

Next search:
Search older purification literature and uncertainty-weighted image restoration methods.
```

Do not mark a contribution `supported` merely because no exact title match was found.

---

## 7. Novelty status semantics

### `candidate`

The idea appears distinct enough to justify further novelty search.

### `supported`

Search evidence currently supports meaningful differentiation from the closest known prior work.

This is not a guarantee that no prior work exists.

### `uncertain`

The search is incomplete, terminology is ambiguous, or overlap remains unclear.

### `not_novel`

Prior work already contains the material contribution being proposed.

When this occurs:

- record the overlapping source;
- update the affected hypothesis;
- decide whether adaptation/extension remains worthwhile;
- do not present reproduction as a new contribution.

---

## 8. Open research gaps

Record gaps supported by literature evidence.

Template:

```text
Gap ID:
Observed gap:
Evidence:
Relevant sources:
Why it matters:
Related hypotheses:
Confidence: high | medium | low
```

A gap should not be created merely because the agent has not yet found a paper.

Prefer:

```text
Multiple reviewed methods improve robustness but all restore the entire detected region; none of the reviewed sources study confidence-weighted partial restoration.
```

over:

```text
No one has tried our idea.
```

---

## 9. Contradictory evidence

Use this section when sources materially disagree.

Template:

```text
Question:
Source A:
Reported result:
Source B:
Reported result:
Possible reasons:
Implication for this project:
Needed experiment/search:
```

Do not force contradictory literature into one narrative without noting the disagreement.

---

## 10. Literature-derived hypotheses

Track hypotheses that originate materially from prior work.

Example:

```text
L-008
→ observation: localized restoration reduces unnecessary image modification
→ H-006: restoration area may be the main robustness/clean-accuracy tradeoff

L-014
→ observation: uncertainty can guide region selection
→ H-007: confidence-weighted partial restoration may improve the tradeoff
```

This helps preserve intellectual provenance.

---

## 11. Novelty threats

Record sources that may invalidate or narrow the contribution claim.

Template:

```text
Threat ID:
Candidate contribution:
Potentially overlapping source:
Nature of overlap:
What must be checked:
Current assessment:
```

A novelty threat should be resolved before describing a result as a contribution.

---

## 12. Search gaps and next searches

Keep a short queue of unresolved literature questions.

Example:

```text
1. Search confidence-weighted restoration in adversarial defense.
2. Check whether L-012 has a supplementary ablation equivalent to H-007.
3. Search older purification methods before 2020.
```

Do not let this become a general reading list.

Each search should answer a specific research or novelty question.

---

## 13. Source quality and authority

Prefer:

1. original paper / official benchmark / official repository;
2. peer-reviewed or primary technical source;
3. authoritative documentation;
4. secondary summaries only when primary material is unavailable.

For each important claim, favor the original source when possible.

Do not rely on a blog or secondary article for a claim that can be verified directly from the paper or official implementation.

---

## 14. Literature search policy

Literature search is required when:

- opening a substantially new research direction;
- selecting or changing an important baseline;
- adopting an external technique;
- evaluating whether a hypothesis overlaps existing work;
- preparing to describe a result as a contribution;
- assessing novelty.

It is not necessary before every small parameter or implementation experiment.

Follow the generic policy in `docs/RESEARCH_PROTOCOL.md`.

---

## 15. Novelty review before contribution claim

Before calling a result a candidate contribution:

```text
identify candidate contribution
→ search closest method family
→ search mechanism-specific terminology
→ search alternative terminology
→ inspect closest sources
→ compare mechanism and evaluation setting
→ update novelty map
```

Before marking novelty as `supported`, verify:

- [ ] Closest known prior work is identified.
- [ ] Key overlap is explicit.
- [ ] Key difference is explicit.
- [ ] Difference is meaningful, not merely cosmetic.
- [ ] Search used more than one plausible terminology where appropriate.
- [ ] Official implementations/supplements were checked when relevant.
- [ ] Remaining uncertainty is documented.
- [ ] No known source already contains the same material contribution.

---

## 16. Search saturation

Do not search indefinitely.

A novelty search may be considered sufficiently saturated for the current stage when:

- repeated queries return the same core sources;
- several plausible terminology variants have been tried;
- closest method families have been inspected;
- recent and foundational work have both been considered where relevant;
- no unresolved high-risk novelty threat remains.

Record remaining uncertainty rather than pretending the search is exhaustive.

---

## 17. What not to store

Do not turn this file into:

- a complete bibliography of the field;
- copied paper abstracts;
- long quotations;
- generic background notes with no project relevance;
- duplicate search results;
- unsourced novelty claims;
- conversational memory.

Store only what materially helps:

- choose baselines;
- form or reject hypotheses;
- assess novelty;
- interpret experiments;
- design fair comparisons.

---

## 18. Maintenance rules

Update this file when:

- a new important source is reviewed;
- a search materially changes novelty confidence;
- a baseline source is identified or changed;
- a literature finding creates or invalidates a hypothesis;
- contradictory evidence is found;
- official code/repository availability changes the research plan;
- a novelty threat is resolved.

Do not delete important old sources merely because the research direction changes.

Mark them `not_relevant` or `superseded` when appropriate.

---

## 19. Completion checklist

Before relying on the literature state, confirm:

- [ ] Important sources have stable `L-XXX` IDs.
- [ ] Search history records meaningful queries and outcomes.
- [ ] Each important source has relevance `high`, `medium`, or `low`.
- [ ] Source-reported facts are separated from our interpretation.
- [ ] Baseline literature is identified.
- [ ] Official code/repository availability is recorded.
- [ ] Closest related-method families are mapped.
- [ ] Contribution candidates have novelty entries.
- [ ] Novelty overlap and differences are explicit.
- [ ] Novelty status is `candidate`, `supported`, `uncertain`, or `not_novel`.
- [ ] Contradictory evidence is preserved when present.
- [ ] Literature-derived hypotheses reference their sources.
- [ ] Search gaps and next searches are explicit.
- [ ] Novelty claims do not exceed the evidence available.
