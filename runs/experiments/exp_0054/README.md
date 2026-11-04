# exp_0054 — D-054/D-055: κ calibration on validation nulls per architecture, test on fresh split E

Configs: configs/experiments/exp_0054.yaml (D-054, grid 0.5–2.0), configs/experiments/exp_0055.yaml (D-055, grid
extended to 2.5, 3.0 after the D-054 grid failed; disclosed, exploratory). Rule and readings fixed in D-054.

- calibration_cnn.json: CNN κ* = 0.5 (prediction met).
- calibration_resnet_D054.json: ResNet-8 null share 0.88/0.77/0.56/0.20/0.07 at κ 0.5/0.75/1.0/1.5/2.0 → no κ.
- calibration_resnet.json (D-055): κ = 2.5 → 0/100 → κ* = 2.5 (3.0 not run).
- test/ (split E, data_seed_base 9800, 100 queries/pair, pairs 120–129, commit 9261024): results_x4_kstar.json.
  Null witness share 0/200; Δ-CE lowered-class witness 0.78/0.975/1.00 on offsets 121–123 → "calibration works".
  At τ_0: s_M 0.00–0.05, CF-diff 0.00–0.025. Grad-diff Recovery 0.32–0.35, Δ-CE Recovery 0.36–0.40 (offsets).
