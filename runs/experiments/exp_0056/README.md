# exp_0056 — D-056 X5: VAE training, gate and r_VAE calibration (validation only)

Registered: https://osf.io/fr7jv (2026-11-04). Code 78fd1f9 (`x5_vae train-vae`, `calibrate-r`), run from a clean
worktree of that commit (git_state dirty = False in both records), config configs/experiments/exp_0057.yaml
(sha256 517ae35d… with r_vae = null). Local CPU (plan v2 had foreseen a Kaggle T4 for training; D-056 fixes no machine;
training took 353 s).

**VAE.** 32×32 variant of `dce.models.vae.VAE`, latent 16, β = 1, 30 epochs, Adam 1e-3, batch 256, seed 11000, on
50 000 clean rendered images (data seed 11000). Checkpoint models/vae32_beta1.0.pt (sha256 303ca0e2…). Final loss
1744.69 per image (BCE + KL).

**Gate (validation, first 2 000 images; registered rule ≥ 0.90 for every f_old seed).** Accuracy of f_old seeds 0–4 on
D(μ(x)): 0.9775, 0.9815, 0.9785, 0.9845, 0.9740 → **passed at β = 1**; the β = 0.5 fallback was not used.
Reconstruction MSE 2.28e-4. vae_gate.json.

**r_VAE (validation, pair 100, 200 valid points, point seed 5100, restart seed 10100).** Ordinary counterfactual of
f_new in the latent (μ = 0.01, no box): found 200/200, median distance 0.8204 → **r_VAE = 1.6408**, frozen in
exp_0057.yaml before split F was generated. Mean std of μ(x) over latent dims on validation: 0.327. r_calibration.json.
