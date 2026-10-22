# Proofs for Section 2 of "Which Model Changes Can Counterfactual Explanations See?"

Numbering follows the paper: Lemma 1, the linear case (here Lemma 2), Proposition 2 (a) and (b), and the
statement after projection. Derived from `research/theory/P5_rotation_offset.md` (internal name "Proposition 5",
written 18/10/2026, after the main experiment and before the grid). Numerical checks: Section 6.

## 0 Setting

Search space $\mathcal Z=\mathbb R^p$ with the Euclidean norm and no box constraints (the experiments use
$[0,1]^5$; box constraints can change the minimiser, so the closed forms below are for the unbounded case). For a query $z_0$ with label $y$, the margins are
$m_o,m_n$ and $h=m_n-m_o$. A query is valid if $m_o(z_0)\ge\kappa$ and $m_n(z_0)\ge\kappa$, $\kappa>0$.
A witness is a point of $\mathcal F^N_\kappa=\{z:\ m_n(z)\le-\kappa,\ m_o(z)\ge\kappa\}$; the $\Delta$-CE is
$z^\star=z_0+u^\star$, $u^\star\in\arg\min\{\|u\|:\ z_0+u\in\mathcal F^N_\kappa\}$ (NOT_FOUND if $\|u^\star\|>r$).

## 1 Lemma 1

**Statement.** Every witness has $h(z)\le-\kappa-m_o(z)\le-2\kappa$. Hence models whose margins differ by less
than $2\kappa$ everywhere have no witness.

**Proof.** At a witness $m_n(z)\le-\kappa$ and $m_o(z)\ge\kappa$, so $h(z)=m_n(z)-m_o(z)\le-\kappa-m_o(z)\le-2\kappa$.
If $|h|<2\kappa$ everywhere, no $z$ satisfies this. ∎

**Corollary 1 (constant logit offset).** Let $f_{new}$ add a constant $b$ to the logit of class 1 of $f_{old}$
(e.g. the prior correction of Saerens et al. [13]). The margin of class $y$ is $f_y-f_{1-y}$, so for queries with label $y=1$
$h\equiv+b$ and for queries with label $y=0$ $h\equiv-b$, at every $z$. By Lemma 1 a witness needs $h\le-2\kappa$:
it exists only for the class whose margin $b$ lowers (class 0 if $b>0$, class 1 if $b<0$), only if $|b|\ge2\kappa$,
and then exactly at the points with $\kappa\le m_o(z)\le|b|-\kappa$ (this set may still be empty in a bounded
$\mathcal Z$ or beyond the budget $r$). The gradient difference is
$\nabla h\equiv0$. No linearity is needed. ∎

*Remark (the threshold itself).* At $|b|=2\kappa$ the feasible set is the level set $\{m_o=\kappa\}$ of the
lowered class: non-empty, but of measure zero. A solver that must land on it exactly finds it only by chance
(exp_0035: 0 or 1 of 100 points at $|b|=2\kappa$). Hence the paper's wording "once $|b|$ reaches $2\kappa$".

## 2 Linear margins

$m_o(z)=w_o^\top z+b_o$, $m_n(z)=w_n^\top z+b_n$, $\rho_o=\|w_o\|$, $\rho_n=\|w_n\|$, $\hat w=w/\|w\|$. Write
$\hat w_n=\cos\theta\,\hat w_o+\sin\theta\,v$ with $v\perp\hat w_o$, $\|v\|=1$, $\theta\in[0,\pi/2)$, and

- $d_n=(m_n(z_0)+\kappa)/\rho_n>0$: distance from $z_0$ to $\{m_n=-\kappa\}$;
- $d_o=(m_o(z_0)-\kappa)/\rho_o\ge0$: distance from $z_0$ to $\{m_o=\kappa\}$;
- $d_o'=(m_o(z_0)+\kappa)/\rho_o=d_o+2\kappa/\rho_o$: distance from $z_0$ to $\{m_o=-\kappa\}$.

The explanation vectors (factor changes are read in $\mathcal Z$ directly):

| Method | Vector |
|---|---|
| Witness ($\Delta$-CE, type N) | $u^\star$ |
| Counterfactual differencing | $u_n-u_o$, with $u_n=-d_n\hat w_n$, $u_o=-d_o'\hat w_o$ the minimum-norm counterfactuals of each model |
| Gradient difference [1] | $-(\hat w_n-\hat w_o)$ (normalised gradients of the margins; constant for linear margins) |

The witness constraints in $u$ are
$$\text{(N)}\ \ \hat w_n^\top u\le-d_n,\qquad\text{(O)}\ \ \hat w_o^\top u\ge-d_o .$$

### Lemma 2 (the witness for linearly independent normals)

**Statement.** If $\theta\in(0,\pi/2)$ the nearest witness ($\Delta$-CE) is unique: it is $u_n=-d_n\hat w_n$ (the ordinary
counterfactual of $f_{new}$) if $u_n$ satisfies (O), i.e. if $d_n\cos\theta\le d_o$; otherwise it is the
minimum-norm point with both constraints active.

**Proof.** The feasible set is a non-empty closed convex polyhedron (the normals are independent, so both
half-spaces intersect) and $\|u\|^2$ is strictly convex, so the minimiser exists and is unique.
(N) is active at the minimiser: otherwise the minimiser would also minimise $\|u\|$ under (O) alone, which gives
$u=0$ because $d_o\ge0$, but $u=0$ violates (N) since $d_n>0$.
If (O) is inactive, $u^\star$ minimises $\|u\|$ under (N) alone, so $u^\star=u_n$; this is consistent iff $u_n$
satisfies (O): $\hat w_o^\top u_n=-d_n\cos\theta\ge-d_o$, i.e. $d_n\cos\theta\le d_o$.
Otherwise both are active; the KKT conditions put $u^\star$ in $\mathrm{span}\{\hat w_o,\hat w_n\}$, and the two
equalities determine it uniquely there. ∎

## 3 Proposition 2 (a): offset, $\theta=0$

**Statement.** The gradient difference is $0$; counterfactual differencing is a multiple of $\hat w_o$; a witness
exists iff $d_n\le d_o$, and then $u^\star=-d_n\hat w_o$.

**Proof.** $\hat w_n=\hat w_o=:\hat w$.
1. $\hat w_n-\hat w_o=0$.
2. $u_n-u_o=(d_o'-d_n)\,\hat w$.
3. With $s=\hat w^\top u$, (N) reads $s\le-d_n$ and (O) reads $s\ge-d_o$. The feasible set is non-empty iff
$d_n\le d_o$. Every feasible $u$ has $\|u\|\ge|s|\ge d_n$, with equality only for $u=-d_n\hat w$, which is
feasible when $d_n\le d_o$. ∎

*Remark.* Here $h(z)=(w_n-w_o)^\top z+(b_n-b_o)$ with $w_n=\lambda w_o$, $\lambda=\rho_n/\rho_o$. It is
constant only if $\lambda=1$; then $d_n\le d_o\iff h\le-2\kappa$, as in Lemma 1. For $\lambda\ne1$ the condition is
on the distances, not on $h$.

## 4 Proposition 2 (b): rotation, $0<\theta<\pi/2$

**Statement.** (i) The gradient difference has components in the ratio $\tan(\theta/2)<1$ along
$(\hat w_o,v)$. (ii) The witness satisfies $|u^\star\cdot\hat w_o|=\min\{d_n\cos\theta,\ d_o\}$ and, for
$\theta\le\pi/4$, its component along $v$ dominates only if $d_o(\sin\theta+\cos\theta)<d_n$.

**Proof.** Write $u=u_1\hat w_o+u_2v+u_\perp$ with $u_\perp\perp\{\hat w_o,v\}$. Then (N) reads
$u_1\cos\theta+u_2\sin\theta\le-d_n$ and (O) reads $u_1\ge-d_o$.

(i) $\hat w_n-\hat w_o=(\cos\theta-1)\hat w_o+\sin\theta\,v$, and $(1-\cos\theta)/\sin\theta=\tan(\theta/2)<1$.

(ii) The constraints do not involve $u_\perp$, so $u_\perp^\star=0$. By Lemma 2:
- Case A ($d_n\cos\theta\le d_o$): $u^\star=-d_n\hat w_n$, so $u_1^\star=-d_n\cos\theta$, $u_2^\star=-d_n\sin\theta$.
- Case B ($d_n\cos\theta>d_o$): $u_1^\star=-d_o$ and $u_2^\star=-(d_n-d_o\cos\theta)/\sin\theta<0$
(since $d_n-d_o\cos\theta>d_n(1-\cos^2\theta)>0$).

In both cases $|u_1^\star|=\min\{d_n\cos\theta,d_o\}$.

Dominance of $v$ means $|u_2^\star|>|u_1^\star|$. In case A this is $\tan\theta>1$, impossible for
$\theta\le\pi/4$. In case B it is $(d_n-d_o\cos\theta)/\sin\theta>d_o\iff d_n>d_o(\sin\theta+\cos\theta)$.
So for $\theta\le\pi/4$ dominance requires $d_o(\sin\theta+\cos\theta)<d_n$. The condition is also sufficient:
for $\theta\le\pi/4$, $(\sin\theta+\cos\theta)\cos\theta-1=\sin\theta(\cos\theta-\sin\theta)\ge0$, so
$d_o(\sin\theta+\cos\theta)<d_n$ implies $d_o<d_n\cos\theta$, i.e. case B, where it is the dominance condition.
(The paper states the necessary direction only.) ∎

*Remark (budget).* For $\theta\le\pi/4$ a witness dominated by $v$ is in case B with $|u_2^\star|>d_o$, so
$\|u^\star\|>\sqrt2\,d_o$; within a budget $r$ this needs $d_o<r/\sqrt2$.

*Remark (counterfactual differencing).* $u_n-u_o=(d_o'-d_n\cos\theta)\hat w_o-d_n\sin\theta\,v$.

## 5 After projection

**Statement.** With $P=I-\hat w_o\hat w_o^\top$, all three explanations are positive multiples of $-v$ in (b) (they coincide
up to a positive scale), and vanish in (a) (where the witness may not exist); this is consistent with Theorem 1 of [1].

**Proof.** $P\hat w_o=0$, $Pv=v$, $Pu_\perp=u_\perp$. In (b): $P(-(\hat w_n-\hat w_o))=-\sin\theta\,v$;
$Pu^\star=u_2^\star v$ with $u_2^\star<0$ in both cases ($u_\perp^\star=0$); $P(u_n-u_o)=-d_n\sin\theta\,v$. All
point along $-v$. In (a) all three are multiples of $\hat w_o$ and project to $0$. ∎

*Remark (CNNs).* The statement uses one normal $\hat w_o$, constant for linear margins. For curved margins the
normal at $z_0$ and at $z^\star$ differ; the paper's H8 tests which one makes the statement hold approximately.

## 6 Numerical checks

**exp_0023** (P5 check, Tier L, 1 000 random linear instances per dimension): with the experiments' two-phase solver,
distances agree with the closed forms to $10^{-3}$ on every instance; the stricter pre-declared vector criterion
(relative error of $u^\star$ below $10^{-3}$ on $\ge99\%$ of instances) failed at $p=5$ (90.1%; max 1.1%) and passed at
$p=20$.

**exp_0037** (diagnosis, fresh seeds, $p\in\{2,5,10\}$, 1 000 instances each): the failure is numerical, not in the
formulas.
1. *Certificate.* For every instance the closed form satisfies the KKT conditions in float64: feasibility and
stationarity residuals $\le10^{-14}$, multiplier of (N) $>0$, multiplier of (O) $\ge0$ in case B. The problem is a
strictly convex QP, so this certifies the closed form as the unique minimiser, without any solver.
2. *Independent solver.* scipy SLSQP in float64 ($\mathrm{ftol}=10^{-15}$) matches the closed form to $10^{-6}$ on
100% of instances.
3. *Experiments' solver.* Vector agreement to $10^{-3}$: 82.7%, 89.9%, 96.6% ($p=2,5,10$); nearly unchanged in float64
(82.7%, 91.3%, 97.0%), so precision is not the cause. 90–98% of the disagreements are in case A (one active
constraint), and they are not concentrated near the case boundary or at extreme $\theta$. On a single active face
$\|u\|^2=d_n^2+\|u_{\mathrm{tan}}\|^2$, so a tangential error $\varepsilon$ changes the distance only by
$\approx\varepsilon^2/(2d_n)$; in relative terms, a relative distance error $\delta$ allows a relative vector error
$\varepsilon/d_n\approx\sqrt{2\delta}$. The solver's relative distance error is $\le7\cdot10^{-4}$ on every instance, which
allows relative vector errors up to $\approx0.037$; the observed maximum is 0.012. (This assumes the solver's point lies
on the active face, as its feasibility check enforces up to the tolerance.) The solver's distances, which is what the experiments use, are
accurate; its vectors carry a tangential error of order $10^{-3}$–$10^{-2}$.

### 6.1 The CNN benchmark (validation, descriptive; exp_0038–exp_0040)

On CNNs the problem is non-convex, so the linear argument above does not carry over; what matters is whether the
reported comparisons depend on which solution the solver returns.

- **Re-solve change rate (main per-point measure; exp_0038).** Re-solving the same validation points with a tight
  float64 reference changes the top factor of Δ-CE on 0.5% of points (2/409; CP95 upper 1.8%), of Δ-CE projected at
  $z_0$ on 2.2% and at $z^\star$ on 0.4%, of CF-diff on 10.9% (61/561), of DeepXplore (float32 vs float64) on 33%
  (98/301). Grad-diff is analytic.
- **Many minima.** The relative vector error of Δ-CE has median $10^{-4}$ but p95 0.22 and max 1.6: a few points
  converge to a different local minimum. A gap-threshold bound ($\|\delta\|\le\tau\|u\|$, $\tau$ = p95) therefore does
  not apply when solutions differ; computed anyway (as a record only) it reverses several directions
  (`runs/experiments/exp_0038/worst_case_p95.json`). A bound that assigns every observed change against the reported
  direction (CP95 upper rates) keeps every direction except the Shortcut tie against CF-diff, and is also pessimistic
  (`flip_rates.json`).
- **Run-to-run variation (main measure for the comparisons; exp_0039, exp_0040).** Five runs per method (restart seeds
  and float32/float64), the comparisons recomputed per run:
  - Δ-CE vs DeepXplore, ~100 points per change: Recovery SD ≤ 0.005 (Δ-CE), 0.013–0.028 (DeepXplore); Δ-CE −
    DeepXplore in every run: Subgroup +0.19 to +0.25, Threshold +0.08 to +0.15.
  - Δ-CE vs CF-diff, 300 points: Shortcut −0.173 to −0.183, Threshold −0.300 to −0.330 (|Δ| > 0.10 in every run).
  - H8 on the four H5d pairs (150 points each): gap (z0, z0) −0.177 to −0.193, gap (z*, z0) +0.036 to +0.047,
    $T$ = 0.085 to 0.120 > 0 in every run (runs 1–4 on Kaggle CPU, run 0 locally; each run on one machine).
  No reported direction changes across runs.
