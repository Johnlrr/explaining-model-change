# Proofs for Section 2 of "Which Model Changes Can Counterfactual Explanations See?"

Numbering follows the paper: Lemma 1, the linear case (here Lemma 2), Proposition 2 (a) and (b), and the
statement after projection. Derived from `research/theory/P5_rotation_offset.md` (internal name "Proposition 5",
written 18/10/2026, after the main experiment and before the grid). A numerical check of the closed forms against
the experiments' solver is in `runs/experiments/exp_0023` (distances agree to 1e-3; a stricter pre-declared
vector criterion failed at p = 5, as stated in the paper).

## 0 Setting

Search space $\mathcal Z=\mathbb R^p$ with the Euclidean norm. For a query $z_0$ with label $y$, the margins are
$m_o,m_n$ and $h=m_n-m_o$. A query is valid if $m_o(z_0)\ge\kappa$ and $m_n(z_0)\ge\kappa$, $\kappa>0$.
A witness is a point of $\mathcal F^N_\kappa=\{z:\ m_n(z)\le-\kappa,\ m_o(z)\ge\kappa\}$; the $\Delta$-CE is
$z^\star=z_0+u^\star$, $u^\star\in\arg\min\{\|u\|:\ z_0+u\in\mathcal F^N_\kappa\}$ (NOT_FOUND if $\|u^\star\|>r$).

## 1 Lemma 1

**Statement.** Every witness has $h(z)\le-\kappa-m_o(z)\le-2\kappa$. Hence models whose margins differ by less
than $2\kappa$ everywhere have no witness.

**Proof.** At a witness $m_n(z)\le-\kappa$ and $m_o(z)\ge\kappa$, so $h(z)=m_n(z)-m_o(z)\le-\kappa-m_o(z)\le-2\kappa$.
If $|h|<2\kappa$ everywhere, no $z$ satisfies this. ∎

**Corollary 1 (constant logit offset).** Let $f_{new}$ add a constant $b$ to the logit of class 1 of $f_{old}$
(e.g. the prior correction of Saerens et al. [13]). The margin of class $y$ is $f_y-f_{1-y}$, so $h\equiv+b$ on
points of class 1 and $h\equiv-b$ on points of class 0, for every $z$. By Lemma 1 a witness needs $h\le-2\kappa$:
it exists only for the class whose margin $b$ lowers (class 0 if $b>0$, class 1 if $b<0$), only if $|b|\ge2\kappa$,
and then exactly at the points with $\kappa\le m_o(z)\le|b|-\kappa$. The gradient difference is
$\nabla h\equiv0$. No linearity is needed. ∎

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

**Statement.** If $\theta\in(0,\pi/2)$ the witness is unique: it is $u_n=-d_n\hat w_n$ (the ordinary
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
So for $\theta\le\pi/4$ dominance requires $d_o(\sin\theta+\cos\theta)<d_n$. ∎

*Remark (budget).* For $\theta\le\pi/4$ a witness dominated by $v$ is in case B with $|u_2^\star|>d_o$, so
$\|u^\star\|>\sqrt2\,d_o$; within a budget $r$ this needs $d_o<r/\sqrt2$.

*Remark (counterfactual differencing).* $u_n-u_o=(d_o'-d_n\cos\theta)\hat w_o-d_n\sin\theta\,v$.

## 5 After projection

**Statement.** With $P=I-\hat w_o\hat w_o^\top$, all three explanations are multiples of $v$ in (b), with the same
sign, and vanish in (a); this is consistent with Theorem 1 of [1].

**Proof.** $P\hat w_o=0$, $Pv=v$, $Pu_\perp=u_\perp$. In (b): $P(-(\hat w_n-\hat w_o))=-\sin\theta\,v$;
$Pu^\star=u_2^\star v$ with $u_2^\star<0$ in both cases ($u_\perp^\star=0$); $P(u_n-u_o)=-d_n\sin\theta\,v$. All
point along $-v$. In (a) all three are multiples of $\hat w_o$ and project to $0$. ∎

*Remark (CNNs).* The statement uses one normal $\hat w_o$, constant for linear margins. For curved margins the
normal at $z_0$ and at $z^\star$ differ; the paper's H8 tests which one makes the statement hold approximately.
