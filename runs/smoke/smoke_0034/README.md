# smoke_0034 — D-056 X5 timing and chunk check (validation only)

Code 226a684 (`x5_vae time`, config exp_0057.yaml with r_vae frozen, n_points 200), laptop (AMD Ryzen 7 PRO 250,
8 torch threads, 8 solver workers). 10 validation queries of pairs 100 (offset) and 110 (null), as rows 0–9 of a
200-query run. timing.json.

- Wall per query 15.8 s (pair 100), 17.3 s (pair 110); projection 14.7 wall-h at 200 queries, 7.4 at 100.
- CPU-s per query: Δ-CE N 39.4 / 44.1, Δ-CE O 39.1 / 43.1, CF-diff wall 1.4 (CPU not recorded) → ≈ 86 CPU-s;
  ≈ 76 CPU-h at 200, ≈ 38 at 100 (registered limit 24). Phase 2 (SLSQP, float64): ≈ 25 CPU-s per query.
- Decision (D-056 addendum, author): 100 queries per pair, on kaggle-cpu instead of a GPU.
- Chunk check (scratch script, not kept): rows 3–5 computed as chunk [3, 6) equal rows 3–5 of chunk [0, 6) for pairs
  100 and 110 — every found vector, z*, distance and CF endpoint identical; s_M differs by ≤ 4.2e-7 (batch rounding).
No result was read beyond timing; validation only.
