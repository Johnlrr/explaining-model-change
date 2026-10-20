# Paper draft v3 — full draft with numbers (Route B, after D-035)

> **Thay đổi so với v2 (18/10):** điền toàn bộ số từ `results_summary.md`; thêm cột A2 vào Bảng 2; **sửa diễn giải T4**: ablation A2 (CE thường của f_new) đạt 0.90 trên T4, nên trên một thay đổi offset, Recovery không phân biệt "thấy thay đổi" với "giải thích f_new". C1 vì vậy được phát biểu là *B3 mù*, không phải *Δ-CE thấy*; thông tin riêng của witness trên T4 là *nơi* nó tồn tại (chỉ lớp y=1). Bỏ SAR của B1 trên Tier 0 (chưa được tính).

> **Ghi chú cho tác giả (xóa trước khi chuyển sang LaTeX)**
>
> - Bản hoàn chỉnh, thay cho v0, v1, v2. Gộp: lý thuyết (L1, P2, P3, **P5 thay cho P4**), benchmark Tier 0, kết quả exp_0017 (đã đăng ký), exp_0019 (C1–C3, khẳng định trên test mới), D-033 (mô tả), exp_0020 (Tier 1 N0, khám phá), D-035 (T5 trượt G0).
> - Ba mức bằng chứng được tách xuyên suốt: **pre-registered** (H2a, P-tie trên test đã khóa), **confirmatory on a new test set** (C1–C3), **descriptive/exploratory**. Đừng trộn khi chỉnh sửa.
> - Mọi số đã điền từ `research/results_summary.md` (commit d2984cb). CI trong Bảng 2 là bootstrap mô tả (seed 6300), trừ H2a/C1/C3 là kiểm định đã khai báo. `[verify]` là trích dẫn cần kiểm.
> - P5: chỉ viết "checked numerically against the experimental solver" **nếu** phần kiểm tra Tier L (mục 8 của file P5) đã chạy và đạt. Nếu chưa, xóa nửa câu đó.
> - Ước tính độ dài: ≈ 3 100 từ + 2 bảng + 1 hình + 14 tham khảo ≈ 6 trang ESANN. Thứ tự cắt nếu thừa: phác thảo chứng minh P3 → đoạn Tier 1 N0 → câu về chi phí → bớt Related work.

---

# Which Model Changes Can Counterfactual Explanations See? A Ground-Truth Benchmark

## Abstract

Counterfactual methods are used to explain how a classifier changed after adaptation, but without ground truth one cannot tell which method explains which kind of change. We inject known changes into classifiers on rendered images with exact factors — a spurious shortcut, a shifted decision threshold, a label flip in a subgroup — next to null updates, and compare counterfactual differencing, gradient differences, importance differences, disagreement search and differential counterfactuals (Δ-CE), minimal witnesses on which exactly one model changes its decision by a margin. No family wins everywhere. A pre-registered hypothesis that witnesses beat gradient differences on the subgroup change was rejected. Two findings were confirmed on a fresh test set: gradient differences are nearly blind to a shifted threshold (Recovery 0.10 vs. 0.41 for witnesses, difference 95% CI [0.27, 0.35]), and margin witnesses abstain on null updates, where counterfactual differencing attributes a change to about 99% of points. A linear analysis separating a rotation of the decision boundary from a shift of its offset explains the threshold result, including where witnesses exist, but not the shortcut result.

---

## 1 Introduction

Deployed classifiers are updated routinely: fine-tuned after concept drift, retrained on corrected labels, adapted to a new population. An update that improves accuracy can still change individual decisions in unwanted ways [12, verify], so explaining *how* the decision behaviour of $f_{new}$ differs from that of $f_{old}$ is a natural goal of explainable AI under distributional change [4, 13].

Counterfactual explanations [9] are the main tool proposed for this. Artelt et al. [1, 2] compute a counterfactual of the same input under each model and compare the two, or approximate this comparison with a difference of gradients; Muschalik et al. [5] compare feature importances; disagreement search [6, 7] looks directly for inputs on which two models disagree. These families make different implicit assumptions about *what* changed — a direction of the weights, a decision boundary, a region of disagreement — and, without ground truth, these assumptions are not checked.

We therefore ask a narrower question than "which method is best": **which kind of model change can each family see?** We inject known changes into classifiers trained on rendered images whose generating factors are exact, so the factor responsible for each change is known, and we measure which factor each method attributes the change to. Alongside the existing families we include *differential counterfactuals* (Δ-CE): the closest input on the data manifold at which only $f_{new}$ changes its decision while $f_{old}$ keeps it, both by a margin κ. Δ-CE adapts disagreement search [6, 7] to model updates by anchoring it at a query point; the margin makes it return nothing when two models differ only by numerical jitter.

**Contributions.** (i) A **benchmark with exact ground truth** for explaining model adaptation: five rendered factors, three injected changes and two null updates (17 model pairs), released with code and protocol. (ii) A **controlled comparison** of five explanation families under a protocol fixed before the test data were touched, reporting a rejected pre-registered hypothesis, a failed theoretical prediction, and two findings confirmed on a fresh test set. (iii) A **linear analysis** of what each family sees, which separates a rotation of the decision boundary from a shift of its offset, together with basic results for the margin witness (band containment, a distance lower bound, closed form with bias and margin).

## 2 Related work

*Explaining model change.* Hammer and Hüllermeier [4] cast model change as the region where two models disagree and its volume; Muschalik et al. [5] compare feature importances of the two models; Artelt et al. [1, 2] compare counterfactuals and show, for linear models without bias, that the comparison recovers the change of the weights. Hinder and Hammer [3] explain drift of the *data* with counterfactuals; we explain change of the *model*. *Disagreement search.* DeepXplore [7] maximises disagreement between networks and DIG [6] locates discrepancy regions between classifiers; DiG-IN [8] uses diffusion guidance to maximise classifier disagreement. None anchors the search at a query and asks for the closest decisive disagreement. *Robust counterfactuals under model change* [10, 11] seek a counterfactual valid for both models; Δ-CE seeks the opposite, a point valid for exactly one. To our knowledge, no prior work evaluates these families against injected changes with known ground truth.

## 3 Differential counterfactuals and what each family sees

**Setting.** Let $(\mathcal Z,d)$ be a search space and $D:\mathcal Z\to\mathcal X$ a decoder onto the data manifold (here an exact renderer from factor space). For a binary classifier and class $y$, the margin is $m(z)=f_y(D(z))-f_{1-y}(D(z))$; we write $m_o,m_n$ for $f_{old},f_{new}$ and $h=m_n-m_o$. A query $z_0$ is valid if $m_o(z_0)\ge\kappa$ and $m_n(z_0)\ge\kappa$, with $\kappa>0$.

**Definition 1 (Δ-CE).** With $\mathcal F^N_\kappa=\{z: m_n(z)\le-\kappa,\ m_o(z)\ge\kappa\}$, a Δ-CE of $z_0$ is $z^\star\in\arg\min_{z\in\mathcal F^N_\kappa}d(z_0,z)$, returned if $d(z_0,z^\star)\le r$ and NOT_FOUND otherwise. Swapping the roles of the models gives type O.

Every $z\in\mathcal F^N_\kappa$ is a *witness*: two forward passes verify that the models decide differently, both decisively. A difference of two counterfactuals has no such property.

**Lemma 1 (band containment).** For every $z\in\mathcal F^N_\kappa$: $h(z)\le-2\kappa$ and $|h(z)|\ge m_o(z)+\kappa$.
*Proof.* $h(z)\le-\kappa-m_o(z)\le-2\kappa$. ∎

**Proposition 2 (distance lower bound).** If $h$ is $L_h$-Lipschitz on $B(z_0,r)$ and $h(z_0)+2\kappa>0$, no witness lies within $\min\{r,(h(z_0)+2\kappa)/L_h\}$ of $z_0$.
*Proof.* Closer points have $h(z)>-2\kappa$; apply Lemma 1. ∎

Two models whose margins differ everywhere by less than $2\kappa$ have no witness at all; for deep networks $L_h$ is unknown, so we use Proposition 2 to motivate the design, not as a certificate.

**Proposition 3 (linear, with bias and margin).** For linear margins with linearly independent $w_o,w_n$, the witness is unique: either the ordinary counterfactual of $f_{new}$ (case A) or the minimum-norm point with both constraints active (case B). *(Proof: strict convexity and the two active sets; full proof at [link].)*

**What each family sees (linear case).** Write $\hat w_n=\cos\theta\,\hat w_o+\sin\theta\,v$ with $v\perp\hat w_o$, and let $d_n$ and $d_o$ be the distances from $z_0$ to $\{m_n=-\kappa\}$ and $\{m_o=\kappa\}$. In factor space, Tier 0 reads explanations directly as vectors; the gradient difference of [1] is $-(\hat w_n-\hat w_o)$, counterfactual differencing is $u_n-u_o$, and the witness is $u^\star$.

**Proposition 4.** (a) *Offset* ($\theta=0$): the gradient difference is $0$; counterfactual differencing is a multiple of $\hat w_o$; a witness exists iff $d_n\le d_o$ and then equals $-d_n\hat w_o$. (b) *Rotation* ($0<\theta<\pi/2$): the gradient difference has components in the ratio $\tan(\theta/2)<1$ along $(\hat w_o,v)$, so it points mainly along the new dependence $v$; the witness has $|u^\star\cdot\hat w_o|=\min\{d_n\cos\theta,d_o\}$, and for $\theta\le\pi/4$ its component along $v$ dominates only if $d_o(\sin\theta+\cos\theta)<d_n$, which within the budget requires $d_o<r/\sqrt2$. After projecting out $\hat w_o$, all three coincide, consistent with Theorem 1 of [1].
*Proof sketch.* Gradients of linear margins are constant. The witness constraints depend only on the components along $\hat w_o$ and $v$: (a) reduces to one dimension, (b) to the two cases of Proposition 3. ∎ *(Full proof at [link]. A numerical check against the experimental solver on linear models agrees on the distance to within $10^{-3}$ in all 2 000 draws and on B3/B1-det algebra to $10^{-15}$; the stricter pre-declared vector-error criterion failed at $p=5$ (solver direction error along a nearly flat constraint) and one parallel-constraint instance was a solver miss. This does not affect the frozen Tier 0 runs, which are box-constrained.)*

In words: gradient differences see a **rotation** of the boundary and are blind to a shift of its **offset**; counterfactual differencing and witnesses see the offset, and witnesses refuse to answer when the offset is below $2\kappa$. This analysis was formulated after the main experiment; its predictions for CNNs are tested in §5.2.

## 4 A benchmark with injected model changes

**Data.** Images are $32\times32$ RGB ellipses rendered from five factors $\phi\in[0,1]^5$ sampled independently: aspect (the task factor), size, colour (red to blue), background brightness and horizontal position. The renderer is differentiable, so $\mathcal Z=[0,1]^5$ is the exact data manifold and the factor change of any explanation is read without a learned oracle. Labels are $y=\mathbb 1[\phi_1>0.5]$ with 5% training label noise; splits have 20 000 / 5 000 / 5 000 images.

**Models and changes.** $f_{old}$ is a CNN (≈ 56k parameters, five seeds). $f_{new}$ is $f_{old}$ fine-tuned on 10 000 samples (5% label noise) from one of five conditions (Table 1).

| Code | Change | Adaptation data | $k^\star$ | Pairs |
|---|---|---|---|---|
| T1 | shortcut (rotation) | 70% near the boundary; colour correlated with label (0.99) | colour | 3 |
| T3 | subgroup flip (non-local) | labels flipped for size $>0.75$ | size | 3 |
| T4 | threshold shift (offset) | $y=\mathbb 1[\phi_1>0.6]$ | aspect | 3 |
| N0 | null | clean data, new sample and order | — | 5 |
| N1 | virtual drift | size $\sim U[0.5,1]$, rule unchanged | — | 3 |

*Table 1: injected changes. T1/N0/N1: 3 epochs, lr $3\cdot10^{-4}$; T3/T4: 10 epochs, lr $10^{-3}$.*

**Qualification.** A pair enters only if the change is visible in decisions on validation data: with $\mathrm{Flip}_k$ the share of predictions that change when factor $k$ is re-sampled, T1 needs $\Delta\mathrm{Flip}_{colour}\ge0.15$; T3 needs $\Delta\mathrm{Flip}_{size}\ge0.15$, $|\Delta\mathrm{Flip}_k|\le0.05$ for other factors and decisive disagreement outside the subgroup $\le0.05$; T4 needs accuracy $\ge0.80$ under the new labels in the shifted band; null pairs need decisive disagreement $\le0.02$. All 17 pairs qualified. The learned size boundary of T3 is soft, which makes T3 less non-local than designed.

**Metrics.** For each pair we draw $N=300$ valid test points. A method returns a factor-change vector $\Delta\phi_i$ or nothing; $\mathrm{Recovery@}r=\frac1N\sum_i v_i\,\mathbb 1[\arg\max_k|\Delta\phi_{ik}|=k^\star]$, where $v_i=1$ if a valid explanation within the budget exists, and NOT_FOUND, invalid outputs and argmax ties count as 0. We also report the found-rate and, on null pairs, the systematic attribution rate (SAR): the share of pairs on which one factor receives mean share $\ge0.5$ with found-rate $\ge0.2$.

## 5 Experiments

**Methods.** All counterfactual methods share one two-phase solver (Adam with a decaying penalty, then SLSQP on the exact constrained problem from $z_0$ and every restart), the same points and the same budget $r=0.54$, calibrated once on validation; on 2 000 random linear pairs the solver matches Proposition 3 to relative error $<10^{-3}$. We compare Δ-CE ($\kappa=0.5$); **B1** counterfactual differencing [1] and its deterministic variant **B1-det**; **B3** gradient difference [1]; **B4** global permutation-importance difference [5]; **B5** a DIG-style line search towards the 20 nearest training points [6]; **A1** a DeepXplore-style search without proximity [7].

**Protocol.** Everything was fixed before the test split was used, and the test split was evaluated once. We distinguish three kinds of evidence. *Pre-registered:* H2a (Δ-CE beats B3 on T3; paired bootstrap, 10 000 resamples stratified by pair, 95% CI and positive in all three seeds) and a predicted tie on T1 and T4 ($|\Delta|\le0.10$). *Confirmatory:* three readings formed after seeing the results were tested on a new test set from the same renderer with the same models (C1–C3, Holm over C1 and C3). *Descriptive:* everything else. Two further steps failed their pre-declared qualification and were stopped: a second tier on coloured MorphoMNIST digits, and a label-shift update (§6).

**Table 2** — Recovery@r on the locked test split (found-rate in parentheses); mean over the three pairs; 95% bootstrap CIs are within ±0.035 of every entry. A2 = ordinary counterfactual of $f_{new}$ (ablation).

| | Δ-CE | B1 | B1-det | B3 | B4 | B5 | A1 | A2 |
|---|---|---|---|---|---|---|---|---|
| T1 rotation | 0.38 (0.84) | 0.54 (0.93) | 0.54 (0.93) | **0.91** (1.0) | 0.00 | 0.02 (0.08) | 0.35 (0.63) | 0.07 (0.93) |
| T3 subgroup | 0.53 (0.81) | 0.32 (1.0) | 0.32 (0.99) | **0.62** (1.0) | 0.00 | 0.06 (0.10) | 0.29 (0.58) | 0.30 (1.0) |
| T4 offset | 0.47 (0.47) | 0.77 (0.90) | 0.76 (0.90) | 0.07 (1.0) | **1.00** | 0.07 (0.08) | 0.31 (0.40) | 0.90 (0.90) |
| Null: found-rate | **0.000** | 0.995 | 0.994 | 1.0 | 1.0 | 0.000 | 0.000 | 0.996 |

*Null row: N0 and N1 pooled (8 pairs); Δ-CE counts types N and O.*

### 5.1 Pre-registered results

**H2a is rejected.** On T3, Recovery(Δ-CE) − Recovery(B3) $=-0.091$, 95% CI $[-0.133,-0.050]$, $p=10^{-4}$, negative in all three seeds ($-0.09$, $-0.12$, $-0.06$). Witnesses do better than counterfactual differencing on T3 (+0.21, descriptive) and than the DIG-style search (+0.47, CI [0.43, 0.50]), which rarely reaches the disagreement region within the budget (found-rate ≤ 0.11, 0.10 on T3); as this holds on one benchmark only, we make no claim about the solver. B1 and B1-det agree to within 0.01 everywhere, so the non-uniqueness of counterfactuals plays no role here.

**The predicted tie fails.** All four pre-declared comparisons exceed 0.10: T1 vs B3 $-0.53$, T1 vs B1-det $-0.17$, T4 vs B3 $+0.40$, T4 vs B1-det $-0.29$. The linear prediction of a tie does not hold for these CNNs.

**Null updates.** On all eight null pairs Δ-CE returns no witness (found-rate 0.000, SAR 0), while B1 produces an attribution for 99.5% of points. With κ = 0, witnesses reappear on null pairs (found-rate 0.80 on N0, 0.89 on N1), as Lemma 1 predicts: without the margin, threshold jitter between two nearly identical models counts as disagreement. All methods separate drift from null pairs perfectly at the pair level (AUROC 1.0), so the difference lies in what each method says about a null update, not in detection.

### 5.2 Confirmed on a new test set

- **C1 (offset blindness) — confirmed.** On T4, Recovery(Δ-CE) − Recovery(B3) $=+0.31$, 95% CI $[0.27,0.35]$, $p_{Holm}=2\cdot10^{-4}$, positive in all seeds (+0.36, +0.27, +0.31); Recovery is 0.41 for Δ-CE, 0.10 for B3 and 0.76 for B1-det. This is the CNN counterpart of Proposition 4(a). Its content is the *failure of B3*: on an offset change the changed factor is the task factor, and even the ordinary counterfactual of $f_{new}$ alone (A2) recovers it (0.90, Table 2). Recovery on T4 therefore shows which methods are blind, not which ones see the change.
- **C2 (abstention) — confirmed.** Δ-CE returns no witness on all eight null pairs; B1 attributes a change to 99.1% (N0) and 99.8% (N1) of points.
- **C3 (rotation, after projection) — not confirmed.** Proposition 4(b) predicts that Δ-CE and B3 coincide on T1 once the old gradient direction is projected out. They do not: 0.64 [0.61, 0.67] vs 0.95 [0.93, 0.96], difference $-0.30$ (90% CI $[-0.33,-0.28]$), with the same gap in every seed; counterfactual differencing behaves like Δ-CE (0.65). We therefore withdraw this reading; the T1 outcome remains unexplained.

### 5.3 Descriptive analyses

*Where witnesses exist.* On T4, a witness is found for every class-1 point and for no class-0 point, in all three pairs and on both test sets (e.g. 136/136 vs 0/164 for one pair); every class-0 failure is infeasible, not over budget. This is what Proposition 4(a) implies for an offset that lowers the margin of one class only; the class-wise statement was written down before the split was computed (the found-rate of 0.47 on T4 was already known and was not part of that prediction). It is the information a witness adds on T4 beyond the direction: it marks the side of the boundary on which decisions actually flipped. On T1 and T3, 16–17% of points have a witness beyond the budget and 0–2% none; halving the budget halves the found-rate (T1: 0.84 → 0.41; T3: 0.81 → 0.41).

*Real digits (exploratory).* On five null pairs of CNNs trained on coloured MNIST digits 4 vs 9 (decoded from a VAE latent), Δ-CE returned no witness on four pairs; on the fifth, which had the highest disagreement rate on test data (1.2%, about three times the others), it found witnesses for 47.5% of points, each a verified decisive disagreement between the two models. B1 attributed a change to 97–99% of points on all five pairs and a systematic attribution on all five (SAR 1.0, vs. 0 for Δ-CE restricted to the new-model type N; 0.2 if the swapped type O is included).

*Cost.* Concluding that no witness exists is the expensive case: 16–20 s of CPU per point, against 4.5–5.3 s when one exists.

**Figure 1** *(to make)*: T4 in the (aspect, colour) plane — the two decision boundaries, a class-1 query with its witness along aspect, a class-0 query with no feasible witness, and the zero gradient-difference explanation; right, rendered images.

## 6 Discussion and limitations

**What the benchmark shows.** Which family explains a model change depends on the kind of change. Gradient differences recover a change of direction (T1) and are nearly blind to a change of offset (T4, confirmed). On an offset change the direction is uninformative for every other method, since it is the task direction; what distinguishes them is whether they notice the change and where. Counterfactual differencing cannot say that nothing changed. Margin witnesses are the only family that abstains on null updates (confirmed) and that returns checkable inputs, but they are limited by feasibility and budget, and they lose on the subgroup change. A practical reading, not tested here as a procedure, is to use witnesses to decide *whether* a decision change exists near a query and other families to describe *how* the boundary moved.

**Limitations.** (1) *Synthetic data.* The benchmark is rendered. A second tier on coloured digits failed its pre-declared qualification: once the VAE and classifier were strong enough, fine-tuning no longer produced the injected shortcut and subgroup changes, so the tier was excluded; only an exploratory null analysis remains. (2) *Offset updates in practice.* Proposition 4(a) implies that the update optimal under label shift, a pure offset [14, verify], is invisible to gradient differences. Our attempt to inject such an update failed qualification: fine-tuning on an 80/20 class prior shifted predictions towards the majority class but created no decisive disagreements. Whether real label-shift adaptation produces offset changes large enough to explain remains open. (3) *T1 is unexplained:* the linear rotation analysis did not transfer (C3). (4) *Behaviour, not mechanism:* witnesses see only decision changes on the manifold within the budget. (5) *No certificate* for deep networks. (6) *Post-hoc readings:* C1–C3 were formed after the main experiment; we report the one that failed. (7) *Recovery on offset changes* rewards any explanation that follows the decision boundary of $f_{new}$ (A2: 0.90), so it measures blindness rather than sensitivity to the change; a metric that also scores *where* an explanation is returned would be needed.

**Process.** The protocol went through seven amendments, all logged, and two pre-declared stopping rules were applied; the full history, proofs and code are at [link].

---

## References (draft)

1. A. Artelt, F. Hinder, V. Vaquet, R. Feldhans, B. Hammer. Contrastive explanations for explaining model adaptations. arXiv:2104.02459, 2021.
2. A. Artelt et al. Contrasting explanations for understanding and regularizing model adaptations. *Neural Processing Letters*, 2022. [verify]
3. F. Hinder, B. Hammer. Counterfactual explanations of concept drift. arXiv:2006.12822, 2020.
4. B. Hammer, E. Hüllermeier. Interpretable machine learning: on the problem of explaining model change. 2021. [verify venue]
5. M. Muschalik, F. Fumagalli, B. Hammer, E. Hüllermeier. Agnostic explanation of model change based on feature importance. *KI – Künstliche Intelligenz*, 2022.
6. X. Renard, T. Laugel, M. Detyniecki. Understanding prediction discrepancies in machine learning classifiers. arXiv:2104.05467, 2021.
7. K. Pei, Y. Cao, J. Yang, S. Jana. DeepXplore: automated whitebox testing of deep learning systems. *SOSP*, 2017.
8. M. Augustin et al. DiG-IN: diffusion guidance for investigating networks. *CVPR*, 2024.
9. S. Wachter, B. Mittelstadt, C. Russell. Counterfactual explanations without opening the black box. *Harvard J. Law & Technology*, 2017.
10. F. Hamman et al. Robust counterfactual explanations for neural networks with probabilistic guarantees. *ICML*, 2023.
11. M. Pawelczyk, K. Broelemann, G. Kasneci. On counterfactual explanations under predictive multiplicity. *UAI*, 2020.
12. S. Yan, Y. Xiong, K. Kundu, S. Yang, S. Deng, M. Wang, W. Xia, S. Soatto. Positive-congruent training: towards regression-free model updates. *CVPR*, 2021. (authors verified on arXiv:2011.09161; venue from memory, check)
13. M. Kaden, B. Paassen, B. Hammer, R. Schubert, T. Villmann. Reliable counterfactuals for machine learning models — current aspects and perspectives. *ESANN*, 2026.
14. M. Saerens, P. Latinne, C. Decaestecker. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Computation* 14(1):21–41, 2002. (verified)

---

## Checklist trước khi nộp

- [x] Đã điền số từ `results_summary.md`.
- [ ] Quyết định có giữ cột A2 trong Bảng 2 không (khuyên giữ: nó là lý do của giới hạn (7) và reviewer sẽ hỏi).
- [ ] Hình 1.
- [ ] P5 đã vào repo; kiểm tra số trên Tier L đạt → giữ câu "numerical check"; nếu không, xóa.
- [ ] Đối chiếu câu về Artelt §3 (v0) nếu đưa lại vào Introduction.
- [ ] Kiểm 4 trích dẫn `[verify]`.
- [ ] Đếm trang trên template `esannV2.cls`.
