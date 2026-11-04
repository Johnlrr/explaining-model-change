# smoke_0035 — X5 on Kaggle CPU (D-056, D-044 executor): same path as exp_0057, 3 validation queries, pairs 110, 100

Kernel boinhbo/dce-smoke0035-x5, code ef66f09 (bundle v110_tc_kaggle, tree 32a5de06…), inputs dataset
boinhbo/dce-x5-inputs (listing a682804e…), config smoke_0035_x5.yaml (sha256 fef444ca…). rc 0, both pairs done,
391 s. Intel Xeon 2.2 GHz, 4 threads, torch 2.14.1+cpu, ATen AVX2. Wall per query ≈ 51–56 s (Δ-CE N 21–25 s, O 22–23 s,
CF-diff 5–6 s) → ≈ 6 h per kernel of 4 pairs × 100 queries. Outputs not interpreted (validation).
