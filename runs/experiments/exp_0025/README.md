# exp_0025 — v1.9 Stage A: 32 fine-tunes + qualification under v1.9.1 §2 (D-036, D-037)

Commit 679d6a2 (clean). Local CPU, 8 threads, 25 min wall (fine-tunes 1 446 s). Validation only; no explanation
method, no test set. Pair 40 is bit-identical to the exp_0024 pilot checkpoint (sha256 match).
Full table: `qualification.json` (per-pair fit, ridge sensitivity, DD on the 2 000 G0 points); t̂_old = 0.487 (s0), 0.496 (s1).

- Stop rule (1)/(3)/(2a): 2 / 32 fail (pairs 22, 23: α = 0, t = 0.53, (2a) t̂ − t̂_old = −0.058, −0.016) → not triggered.
- (1) and (3) pass on all 32; every fit converged, none separable; ridge 1e-3 changes α̂, t̂ by ≤ 0.001.
- Null cell (pairs 20, 21): DD = 0.000, 0.000 → null.
- (2b) order violations (adjacent steps, both pairs listed): 20–22, 29–31, 33–35, 40–42, 41–43.
- (2c): t̂(0.25, 0.60) − t̂(0.25, 0.50) = 0.077 (s0), 0.120 (s1) → both seeds pass.
- Exclusions under the "both pairs of a violating step" reading: H5a {22, 23}, H5b {29}, H5c {42, 43} (= every pair of
  cell (0.25, 0.60) → H5c not testable under this reading), H5d none. The (2b) reading is open (asked of the author).
