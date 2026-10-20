# exp_0032 — v1.10 T-A test run (D-043 §2, H-010 / H7c)

Code f05d127 (ta-* unchanged since). 30 H7 pairs (f_old seeds 0–2 × b), data new_test_set(9000 + pair_id, 5 000), 300
valid points, Δ-CE N, B1-det, B3. Two CPU shards (`--shard i/2`). The shards were interrupted twice when the the coding tool
session ended (after 20 and 24 pairs; no partial pair files) and resumed; each restart is in metadata_shard*.json
git_runs. Analysis: `ta-analyze` → results_h7.json (U1–U3 passed; H7c before Holm). Holm with H8: exp_0033.
