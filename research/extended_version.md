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

## #6 — the old-model normal turns even for near witnesses (2026-10-21, exp_0033 test, descriptive)

H8 held on test data (gap −0.190 → +0.029 at (z*, z0)). On the 837 support points of the H5d pairs the normal
ŵ_o turns between z0 and z* by a median 32°; 27° already for the nearest quartile (‖z* − z0‖ ≤ 0.21 ≈ 0.4 r), 36° for
the farthest. Rotation vs distance: Spearman +0.17 (weak); projection deviation vs distance: −0.10. So "the witness is
far" is not the mechanism; m_old is curved in Z through the renderer and the CNN, at the scale of r. The author's
initial mechanism ("witness far from z0") is half wrong. Directions: (a) measure the curvature of m_old along z0 → z*
(second-order term vs angle); (b) redo the analysis in the pull-back metric JᵀJ of the renderer, where Euclidean
distance in Z is replaced by image-space distance — if the normal is nearly constant in that metric, the "rotation"
is a coordinate effect of Z; (c) links to #5: the method "witness + normal at the witness" is needed precisely because
the normal is not stable within the budget r. Correlation deviation ~ 1 − cos (0.80) is near-definitional, not
evidence.

## #7 — the proximity term makes the explanation determined (2026-10-22, validation, descriptive)

exp_0038: re-solving the same points (float32 vs float64, same seed) changes the top factor of DeepXplore on 33% of
points (98/301) but of Δ-CE on 0.5% (2/409); CF-diff (two ordinary CEs) changes on 10.9% (61/561). exp_0039: the mean
Recovery is nevertheless stable across five runs (DeepXplore SD 0.013–0.028, Δ-CE ≤ 0.005). Argument for #5: the
proximity term pins the explanation (a unique nearest point), whereas a search without it returns an arbitrary feasible
point. For the extended version, measure this on test data with all pairs (attribution stability as a metric next to
Recovery).
