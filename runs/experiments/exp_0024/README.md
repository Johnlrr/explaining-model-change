# exp_0024 — v1.9 §6 pilot (D-036): qualification only, validation only

Code/config commit 19d7331 (clean). Pair 40 = cell (α = 0.25, t = 0.56), f_old seed 0 (exp_0002, hash-verified).
Fine-tune: 10 000 samples, 5% label noise, 10 epochs, Adam lr 1e-3, batch 256 (T4 recipe of exp_0003); local CPU, 45.8 s.
No explanation method was run.

| α̂ | t̂ | accuracy (new rule) | qualified |
|---|---|---|---|
| 0.233 (ok, tol 0.125) | 0.630 (FAIL, |t̂ − t| = 0.070 > 0.02) | 0.929 (ok) | **no** |

Fit: unpenalised logistic, converged in 12 Newton steps, not separable, linear-fit accuracy 0.982; ridge 1e-3 gives the
same α̂, t̂. Direct evaluation of f_new along aspect with the other factors at 0.5 crosses at 0.644, so the shift is
in the model, not in the fit.

Reference measurement (diagnostic, validation, same measure on existing models; `diagnostic_reference_models.txt`):
f_old s0–s2 give t̂ = 0.487, 0.496, 0.486 (rule t = 0.5); locked T4 f_new pairs 6–8 (exp_0003) give
t̂ = 0.598, 0.621, 0.591 (rule t = 0.6) — pair 7 would fail |t̂ − t| ≤ 0.02 by 0.001.
