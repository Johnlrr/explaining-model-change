# smoke_0032 — equivalence of `x4_resnet run-chunked` (D-053) with `x4_resnet run`, local CPU. Not interpreted.

Config configs/experiments/smoke_0031_x4.yaml (pairs 120 offset and 128 null, 3 points, exp_0050 ResNet-8 checkpoints).
`run` vs `run-chunked --chunk 2` (chunks 0–2, 2–3). compare.py over every saved array: point_ids, y, z0, Δ-CE type N
(and O on 128) found / dphi / distance, CF-diff found / dphi, s_M, s_M,r bit-identical; Grad-diff dphi within
3.6·10⁻⁷, found identical. Logs: plain.log, chunk.log. Array outputs were in .work/eqtest1 (scratch, not kept).
