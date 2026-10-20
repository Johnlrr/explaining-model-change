> **Nguồn gốc (ghi bởi tác giả, 18/10/2026).** File này được viết trong phiên làm việc ngày 18/10/2026 lúc 07:54 (giờ Bangkok), trước khi `results_summary.md` có bảng tách lớp T4. Mục P5-iv là dự đoán có trước dữ liệu cho phần tách lớp của T4. Con số found-rate 0.47 của T4 đã biết từ exp_0017 và không thuộc dự đoán.
>
> **Đối chiếu với lịch sử git (ghi bởi trợ lý, không do tác giả viết):** (i) bảng tách lớp T4 (136/136 và 0/164) lần đầu được commit lúc **12:01** (`d2984cb`), không phải 15:47; (ii) kết quả phân tích D-034 (`8752597`) được commit lúc 09:27 và không tính tách lớp; (iii) file được chép vào repo lúc 15:44 và commit lúc 15:51 (`56b76e0`). Mốc 07:54 không kiểm chứng được từ repo. Mốc đáng tin là lúc commit: mọi số tách lớp đều xuất hiện sau 12:01.
>
> Trong paper, P5-iv chỉ được nêu như dự đoán khi mốc 07:54 được xác nhận bằng bản gốc trong Project; nếu không, nó là mô tả hậu nghiệm.

# Proposition 5 — what each explanation family sees: rotation vs. offset

> **Ghi chú cho tác giả**
>
> - Viết 18/10/2026, **sau** exp_0017 và **trước** khi có bất kỳ kết quả nào của D-034. Mệnh đề là toán học, nên đúng hay sai không phụ thuộc dữ liệu. Nhưng *ý tưởng* của nó đến từ việc nhìn exp_0017, nên paper phải ghi rõ: "formulated after the main experiment; its predictions for CNNs are tested on a fresh test set (C1, C3, C4)".
> - Mục 6 liệt kê các dự đoán **ghi trước khi thấy kết quả D-034**. Nên nhờ the coding tool chép mục đó vào `research/NOTES.md` kèm thời điểm, trước khi chạy `analyze` cho D-034.
> - Mục 7 là bản rút gọn để đưa vào §2 của paper. P5 bao trùm P4, nên trong paper có thể thay P4 bằng P5 để tiết kiệm chỗ.
> - Kiểm tra số (mục 8) nên giao the coding tool **sau khi D-034 xong**, trên một nhánh riêng, để không đụng vào commit đã khóa.

---

## 1 Setting and notation

Linear margins on $\mathcal Z=\mathbb R^p$ with Euclidean distance:
$$m_o(z)=w_o^\top z+b_o,\qquad m_n(z)=w_n^\top z+b_n,\qquad \rho_o=\|w_o\|,\ \rho_n=\|w_n\|.$$
A query $z_0$ is valid: $m_o(z_0)\ge\kappa$, $m_n(z_0)\ge\kappa$, $\kappa>0$.

**Geometry of the update.** Write $\hat w=w/\|w\|$ and decompose
$$\hat w_n=\cos\theta\,\hat w_o+\sin\theta\,v,\qquad v\perp\hat w_o,\ \|v\|=1,\ \theta\in[0,\pi/2).$$
The **rotation** is $\theta$ (direction of the new dependence $v$). The **offset** enters only through the margins at $z_0$. All quantities below are expressed with three distances:

| Symbol | Definition | Meaning |
|---|---|---|
| $d_n=\dfrac{m_n(z_0)+\kappa}{\rho_n}$ | distance from $z_0$ to $\{m_n=-\kappa\}$ | ordinary counterfactual distance of $f_{new}$ |
| $d_o=\dfrac{m_o(z_0)-\kappa}{\rho_o}\ \ (\ge 0)$ | distance from $z_0$ to $\{m_o=\kappa\}$ | how far the old model's decision band is |
| $d_o'=\dfrac{m_o(z_0)+\kappa}{\rho_o}=d_o+\dfrac{2\kappa}{\rho_o}$ | distance from $z_0$ to $\{m_o=-\kappa\}$ | ordinary counterfactual distance of $f_{old}$ |

**Explanations in the linear case** (Tier 0 reads factor changes in $\mathcal Z$ directly, so $\Delta\phi=$ the vector below).

| Method | Explanation vector $e$ |
|---|---|
| Δ-CE (type N) | $u^\star=\arg\min\{\|u\|:\ m_n(z_0+u)\le-\kappa,\ m_o(z_0+u)\ge\kappa\}$ |
| B1-det (CE differencing) | $u_n-u_o$ with $u_n=-d_n\hat w_n$, $u_o=-d_o'\hat w_o$ (ordinary min-norm counterfactuals) |
| B3 (gradient difference) | $-(\hat w_n-\hat w_o)$ (gradients of linear margins are constant) |

**Recovery in the factor-aligned case.** Assume $\hat w_o=e_t$ (the task factor) and $v=e_{k^\star}$ (the factor the update made the model depend on), and note that every explanation above lies in $\mathrm{span}\{e_t,e_{k^\star}\}$. Then $\arg\max_k|e_k|=k^\star$ iff $|e\cdot v|>|e\cdot\hat w_o|$; equality is a tie and counts as a failure, as in the benchmark.

---

## 2 Statement

**Proposition 5.** In the setting above:

**(a) Offset only** ($\theta=0$, i.e. $w_n=\lambda w_o$, $\lambda>0$; biases arbitrary).
1. B3 returns $e=0$ for every $z_0$.
2. B1-det returns $e=(d_o'-d_n)\,\hat w_o$, which is non-zero iff $d_n\ne d_o'$.
3. A type-N witness exists iff $d_n\le d_o$, and then $u^\star=-d_n\hat w_o$ with $\|u^\star\|=d_n$. For $\lambda=1$ the existence condition is $b_n-b_o\le-2\kappa$, independently of $z_0$.

**(b) Rotation** ($\theta\in(0,\pi/2)$; norms and biases arbitrary).
1. B3: $|e\cdot\hat w_o|/|e\cdot v|=\tan(\theta/2)<1$, independently of $z_0$, the biases and the norms.
2. Δ-CE: $u^\star\in\mathrm{span}\{\hat w_o,v\}$ with
$$|u^\star\cdot\hat w_o|=\min\{d_n\cos\theta,\ d_o\}.$$
Case A ($d_n\cos\theta\le d_o$): $u^\star=-d_n\hat w_n$. Case B ($d_n\cos\theta> d_o$): $u^\star=-d_o\,\hat w_o-\dfrac{d_n-d_o\cos\theta}{\sin\theta}\,v$.
3. B1-det: $e\cdot\hat w_o=d_o'-d_n\cos\theta$ and $e\cdot v=-d_n\sin\theta$.

**Corollary 5.1 (recovery, factor-aligned).**
- *Offset only:* B3 never recovers the changed factor; B1-det recovers $k^\star=t$ whenever $d_n\ne d_o'$; Δ-CE recovers $k^\star=t$ exactly when it finds a witness, i.e. iff $d_n\le\min\{d_o,r\}$.
- *Rotation:* B3 always recovers $k^\star$. Δ-CE recovers $k^\star$ iff
$$\theta>\pi/4\qquad\text{or}\qquad d_o(\sin\theta+\cos\theta)<d_n .$$
If $\theta\le\pi/4$, a recovering witness has $\|u^\star\|>\sqrt2\,d_o$, so it lies within the budget only if $d_o<r/\sqrt2$. B1-det recovers $k^\star$ iff $d_n\sin\theta>|d_o'-d_n\cos\theta|$.
- *After projection* with $P=I-\hat w_o\hat w_o^\top$ (the projection used in C3), all three explanations are multiples of $v$ in (b) and are zero in (a) (Proposition 4).

---

## 3 Proof of (a)

Let $\hat w=\hat w_o=\hat w_n$ and write $m_n(z_0+u)=m_n(z_0)+\rho_n\,\hat w^\top u$, $m_o(z_0+u)=m_o(z_0)+\rho_o\,\hat w^\top u$.

1. $\hat w_n-\hat w_o=0$.
2. $u_n-u_o=-d_n\hat w+d_o'\hat w$.
3. The constraints read $\hat w^\top u\le-d_n$ and $\hat w^\top u\ge-d_o$. Both involve only $s=\hat w^\top u$, so the feasible set is non-empty iff $-d_o\le-d_n$, i.e. $d_n\le d_o$. For a feasible $u$, $\|u\|\ge|s|\ge d_n$, with equality for $u=-d_n\hat w$, which is feasible when $d_n\le d_o$; it is the unique minimiser because the components orthogonal to $\hat w$ only add norm. For $\lambda=1$: $\rho_n=\rho_o=\rho$ and $d_n\le d_o\iff m_n(z_0)+\kappa\le m_o(z_0)-\kappa\iff h(z_0)\le-2\kappa$, and $h\equiv b_n-b_o$ is constant. ∎

This is the parallel case excluded from Proposition 3 and agrees with Lemma 1.

## 4 Proof of (b)

Work in the orthonormal pair $(\hat w_o,v)$; write $u=u_1\hat w_o+u_2v+u_\perp$ with $u_\perp\perp\{\hat w_o,v\}$. Then
$$m_o(z_0+u)=m_o(z_0)+\rho_o u_1,\qquad m_n(z_0+u)=m_n(z_0)+\rho_n(u_1\cos\theta+u_2\sin\theta).$$

**1. B3.** $\hat w_n-\hat w_o=(\cos\theta-1)\hat w_o+\sin\theta\,v$, so the ratio of the absolute components is $(1-\cos\theta)/\sin\theta=\tan(\theta/2)$, which is $<1$ for $\theta<\pi/2$.

**2. Δ-CE.** The constraints are
$$\text{(N)}\ \ u_1\cos\theta+u_2\sin\theta\le-d_n,\qquad \text{(O)}\ \ u_1\ge-d_o .$$
They do not involve $u_\perp$, so $u_\perp^\star=0$. Since $\hat w_o,\hat w_n$ are linearly independent for $\theta\in(0,\pi/2)$, Proposition 3 applies; in the present notation its case condition $a\,w_o^\top w_n/\|w_n\|^2\le c$ reads $\rho_o a\cos\theta/\rho_n\le c$, i.e. $d_n\cos\theta\le d_o$.

*Case A* ($d_n\cos\theta\le d_o$): $u^\star=-a\,w_n/\rho_n^2=-d_n\hat w_n$, so $u_1^\star=-d_n\cos\theta$ and $u_2^\star=-d_n\sin\theta$.

*Case B* ($d_n\cos\theta>d_o$): both constraints are active, so $u_1^\star=-d_o$ and, from (N) with equality, $u_2^\star=-(d_n-d_o\cos\theta)/\sin\theta$. This quantity is negative because $d_n-d_o\cos\theta>d_n(1-\cos^2\theta)>0$.

In both cases $|u_1^\star|=\min\{d_n\cos\theta,d_o\}$, since the case condition decides which of the two is smaller.

**3. B1-det.** $u_n-u_o=-d_n(\cos\theta\,\hat w_o+\sin\theta\,v)+d_o'\hat w_o$. ∎

## 5 Proof of Corollary 5.1

*Offset only* follows from (a): B3 gives the zero vector, which is a tie; B1-det and $u^\star$ are multiples of $e_t$.

*Rotation, Δ-CE.* Recovery means $|u_2^\star|>|u_1^\star|$.
- Case A: $d_n\sin\theta>d_n\cos\theta\iff\theta>\pi/4$.
- Case B: $(d_n-d_o\cos\theta)/\sin\theta>d_o\iff d_n>d_o(\sin\theta+\cos\theta)$.

To merge the cases, use $\cos\theta(\sin\theta+\cos\theta)=1-\sin\theta(\sin\theta-\cos\theta)$, which is $>1$ for $\theta<\pi/4$ and $<1$ for $\theta>\pi/4$.
- If $\theta>\pi/4$: in case A recovery holds; in case B, $d_o<d_n\cos\theta<d_n/(\sin\theta+\cos\theta)$, so recovery holds as well.
- If $\theta<\pi/4$: case A gives no recovery, and in case A also $d_n\le d_o/\cos\theta<d_o(\sin\theta+\cos\theta)$, so the condition $d_n>d_o(\sin\theta+\cos\theta)$ fails there too. In case B recovery holds iff that condition holds.
- If $\theta=\pi/4$: case A gives a tie; in case B, $\cos\theta=1/(\sin\theta+\cos\theta)$, so the case-B condition $d_o<d_n\cos\theta$ is the recovery condition.

Hence recovery $\iff\theta>\pi/4$ or $d_o(\sin\theta+\cos\theta)<d_n$.

*Budget.* For $\theta\le\pi/4$ a recovering witness is in case B with $|u_2^\star|>|u_1^\star|=d_o$, so $\|u^\star\|^2=d_o^2+(u_2^\star)^2>2d_o^2$.

*B1-det.* Direct from (b)3.

*Projection.* $P\hat w_o=0$ and $Pv=v$, so the projected explanations are multiples of $v$ in (b) and vanish in (a). ∎

---

## 6 Predictions for the CNN benchmark (written before any D-034 result)

These transfer the linear statements through a local linearisation, which has no error bound; they are predictions, not consequences.

| ID | Prediction | Source | Status in protocol |
|---|---|---|---|
| P5-i | On T4 and T5 (offset-type updates) B3 recovers poorly, while B1-det and Δ-CE (where found) recover the task factor | (a) | tested by **C1** (T4, Δ-CE vs B3) and **C4** (T5, B1-det vs B3) |
| P5-ii | On T1 (rotation towards colour) B3 recovers well; Δ-CE recovers only in a window, so Recovery(Δ-CE) is well below its found-rate | (b), Cor. 5.1 | descriptive |
| P5-iii | After projecting out $\nabla m_o(z_0)$, Δ-CE and B3 agree on T1 | Cor. 5.1, P4 | tested by **C3** |
| P5-iv | On T4 and T5, type-N witnesses exist essentially only for points of the class whose margin the update *lowers*: on T4 the class $y=1$ (the threshold moves up), on T5 the class $y=0$ (the prior of $y=1$ goes up). So the Δ-CE found-rate is bounded by the share of that class among valid points, and is close to 0 on the other class | (a), Lemma 1 | **descriptive, new**: report found-rate split by class |
| P5-v | Under a pure prior shift with unchanged class-conditionals, the Bayes-optimal update is a pure offset of the logit difference by $\log\frac{\pi_1'\pi_0}{\pi_0'\pi_1}$ [Saerens et al. 2002, verify]; for $P(y{=}1)$: $0.5\to0.8$ this is $\ln 4\approx1.39>2\kappa=1$, so witnesses can exist on class 0 | (a) + label-shift adjustment | motivation of T5 |

**Post-hoc consistency check (not evidence):** on exp_0017, Δ-CE on T4 had found-rate 0.47 and Recovery 0.47. P5-iv gives a reason: only the class-1 points (about half) can have a witness, and every witness found points along aspect. This was noticed only after formulating P5; the split by class on the *new* test set (P5-iv) is the actual check.

---

## 7 Paper-ready version (for §2, ≈ 0.35 page; replaces Proposition 4)

> **Proposition 4 (what each family sees; linear case).** Let $m_o,m_n$ be linear, $\hat w_n=\cos\theta\,\hat w_o+\sin\theta\,v$ with $v\perp\hat w_o$, and let $d_n$ and $d_o$ be the distances from $z_0$ to $\{m_n=-\kappa\}$ and $\{m_o=\kappa\}$.
> (a) *Offset* ($\theta=0$): the gradient difference is $0$; counterfactual differencing is a multiple of $\hat w_o$; a witness exists iff $d_n\le d_o$ and then equals $-d_n\hat w_o$.
> (b) *Rotation* ($0<\theta<\pi/2$): the gradient difference has components in the ratio $\tan(\theta/2)<1$ along $(\hat w_o,v)$; the witness has $|u^\star\cdot\hat w_o|=\min\{d_n\cos\theta,d_o\}$, and if $\theta\le\pi/4$ its component along $v$ dominates only if $d_o(\sin\theta+\cos\theta)<d_n$, which within a budget $r$ requires $d_o<r/\sqrt2$.
> In both cases all three explanations coincide after projecting out $\hat w_o$, consistent with Theorem 1 of [1].
>
> *Proof sketch.* Gradients of linear margins are constant, which gives the gradient statements. For the witness, the two constraints depend only on the components along $\hat w_o$ and $v$; (a) reduces to one dimension and (b) to the two cases of Proposition 3. ∎

> **Reading.** Gradient differences see a *rotation* of the decision boundary but are blind to a shift of its *offset*, which is exactly the update that label shift calls for [Saerens et al., verify]. Counterfactual differencing and witnesses see the offset; witnesses additionally refuse to answer when the offset is below $2\kappa$, but under a rotation they mix in the old task direction unless the old decision band is close to the query.

*(Gợi ý: khi thay P4 cũ bằng bản này, giữ một câu "Theorem 1 of [1] is the special case without bias", và chuyển chứng minh đầy đủ lên GitHub.)*

---

## 8 Numerical check to hand to the coding tool (after D-034 finishes)

On Tier L (no test data involved), on a separate branch:

1. Draw 1 000 random instances per dimension $p\in\{5,20\}$: $w_o\sim\mathcal N(0,I)$; $v$ random unit vector orthogonal to $w_o$; $\theta\sim U(0.05,1.5)$; $\rho_n/\rho_o\sim U(0.5,2)$; random biases; reject until $z_0$ is valid.
2. Compare the solver's $u^\star$ (the locked two-phase solver, without the budget) with the closed form of (b)2: relative error $<10^{-3}$ on $\ge 99\%$ of instances, as in H4.
3. Check the recovery criterion of Corollary 5.1 against $\arg\max$ on the solver output in the factor-aligned case ($\hat w_o=e_1$, $v=e_2$): agreement on 100% of instances away from ties ($|\,|u_1|-|u_2|\,|>10^{-6}$).
4. Repeat 2 for (a) with $w_n=\lambda w_o$, $\lambda\sim U(0.5,2)$: existence iff $d_n\le d_o$, and $u^\star=-d_n\hat w_o$.

This turns P5 from "proved on paper" into "proved and checked against the same solver used in the experiments".

---

## 9 New reference

- M. Saerens, P. Latinne, C. Decaestecker. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Computation* 14(1), 2002. [verify]
