# Notes for the extended (rank-A) version — not for the ESANN paper

Ideas that came up during the v1.10 work. None of these is a hypothesis of the ESANN paper.

## #5 — gradient difference at the witness (2026-10-20, from exp_0030, validation, descriptive)

On the four H5d pairs (50 validation points each), the C3/H5d gap between Δ-CE and B3 after projection closes when the
projection uses ŵ_o at the witness z* instead of at z0 (cell (z*, z0): +0.01; (z*, z*): −0.01; (z0, z0): −0.19).
Computing B3 at z* instead of z0 changes almost nothing. So the useful object may be "the witness location + the
old-model normal there", rather than "B3 re-evaluated at the witness". A method for the extended version could be:
find a witness (whether), then report the projection of the change onto the old-model level set at the witness (what).
Needs its own pre-registration over the whole grid against B3 and B1-det.

## Alternatives if H8 fails on test data
Curvature of h along the path z0 → z*; solver bias (where on the feasible set the minimum-distance point lands); the
metric JᵀJ of the renderer (Euclidean distance in Z is not the image-space geometry).
