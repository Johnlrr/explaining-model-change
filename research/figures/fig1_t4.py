"""Figure 1 (paper): T4 (threshold shift, offset-type update) on one pair. Left panel only.

Pair: T4 seed 0 = pair 6 (exp_0017, Tier 0 final run, test split). Factor plane: x = aspect (k*, changed by the
update), y = colour (a factor the update does not touch). The other factors (size, bg, posx) are fixed at the
values of the WITNESS of query 155, so the witness lies exactly on the plane. The query is drawn at its own aspect
and colour (its other factors differ by ≤ 0.005). Query 155 (class 1) has a Δ-CE witness that moves aspect and, by ≤ 0.005, the other factors.
Point 278 (class 0) has no feasible witness; its other factors differ from query 155 by ≤ 0.023.

Region: F^N_κ = {m_n ≤ −κ, m_o ≥ κ} with the clean label of the query (y = 1), evaluated on a fine grid
(800 × 500 over aspect 0.3–0.7 × colour 0–1, i.e. ≥ 400 × 400 over aspect 0.4–0.7), with the witness's aspect and
colour inserted as exact grid lines, so the witness is evaluated exactly, not interpolated. Float32 models as in the solver. Shading is the grid region; lines are the level sets m_n = −κ (dashed) and m_o = κ (solid).

Checks run by this script before drawing (exit with an error if they fail):
  1. the saved witness of query 155 satisfies both constraints under the float32 models;
  2. the witness's grid node (exact coordinates on the plane) is shaded, and its grid margins equal the direct
     evaluation at the witness.

Illustrative, one query point, not a result. No experiment is rerun: saved outputs and the checkpoints of
exp_0002/exp_0003 are used.
Output: research/figures/fig1_t4.pdf (vector). Run: uv run python research/figures/fig1_t4.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dce.data.drift import clean_splits
from dce.data.renderer import render
from dce.eval.pairs import load_pair

KAPPA = 0.5
PAIR = 6
RUN = Path("runs/experiments/exp_0017")
QUERY, OTHER = 155, 278
ASPECT, COLOUR = 0, 2  # factor columns in renderer order: aspect, size, colour, bg, posx
X_RANGE, Y_RANGE = (0.3, 0.7), (0.0, 1.0)
NX, NY = 800, 500
OUT = Path(__file__).with_name("fig1_t4.pdf")


def margin_y1(model, phi: np.ndarray, batch: int = 4000) -> np.ndarray:
    """m(z) = f_1(D(z)) − f_0(D(z)) (spec §3.1 with y = 1), float32 as in the solver."""
    out = []
    with torch.no_grad():
        for i in range(0, len(phi), batch):
            lg = model(render(torch.as_tensor(phi[i : i + batch], dtype=torch.float32)))
            out.append((lg[:, 1] - lg[:, 0]).numpy())
    return np.concatenate(out)


def in_region(f_old, f_new, phi: np.ndarray) -> np.ndarray:
    return (margin_y1(f_new, phi) <= -KAPPA) & (margin_y1(f_old, phi) >= KAPPA)


def main() -> None:
    test = clean_splits(seed=1234)["test"]
    z = np.load(RUN / "pairs" / f"pair_{PAIR:02d}.npz")
    pid, Z, D, found = z["point_ids"], z["z0"], z["dce_N__dphi"], z["dce_N__found"]
    assert np.allclose(test.phi[pid], Z, atol=1e-6), "saved points do not match the test split"
    y = test.y_clean[pid]
    assert y[QUERY] == 1 and found[QUERY] and y[OTHER] == 0 and not found[OTHER]
    _, f_old, f_new, _ = load_pair(PAIR)

    q, w = Z[QUERY], Z[QUERY] + D[QUERY]
    # check 1: the saved witness satisfies both constraints
    wp = np.stack([w])
    assert margin_y1(f_new, wp)[0] <= -KAPPA and margin_y1(f_old, wp)[0] >= KAPPA, "witness violates F^N_κ"

    a = np.union1d(np.linspace(*X_RANGE, NX), [w[ASPECT]])
    c = np.union1d(np.linspace(*Y_RANGE, NY), [w[COLOUR]])
    A, C = np.meshgrid(a, c)
    plane = np.tile(w, (A.size, 1))                      # other factors = the witness's own values
    plane[:, ASPECT], plane[:, COLOUR] = A.ravel(), C.ravel()
    mn = margin_y1(f_new, plane).reshape(A.shape)
    mo = margin_y1(f_old, plane).reshape(A.shape)
    region = (mn <= -KAPPA) & (mo >= KAPPA)

    # check 2: the witness's own node is shaded and its grid margins equal the direct evaluation at the witness
    i_c = int(np.flatnonzero(c == w[COLOUR])[0])
    i_a = int(np.flatnonzero(a == w[ASPECT])[0])
    direct_n, direct_o = float(margin_y1(f_new, wp)[0]), float(margin_y1(f_old, wp)[0])
    assert abs(mn[i_c, i_a] - direct_n) < 1e-5 and abs(mo[i_c, i_a] - direct_o) < 1e-5, "grid node ≠ direct evaluation"
    assert region[i_c, i_a], "witness node is not shaded"
    print(f"checks passed: witness m_n={direct_n:.5f} (≤ {-KAPPA}), m_o={direct_o:.5f} (≥ {KAPPA}); "
          f"grid node equals direct evaluation; node shaded; shaded cells {int(region.sum())}; "
          f"grid {A.shape[1]}x{A.shape[0]} (aspect x colour), in aspect 0.4-0.7: {int(((a>=0.4)&(a<=0.7)).sum())} columns")

    plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(4.3, 3.7))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.97)
    ax.contourf(A, C, region.astype(float), levels=[0.5, 1.5], colors=["#c9d9ef"])
    ax.contour(A, C, mn, levels=[-KAPPA], colors="#b03a2e", linewidths=0.8, linestyles="--")
    ax.contour(A, C, mo, levels=[KAPPA], colors="#1f4e9a", linewidths=0.8)
    ax.scatter([q[ASPECT]], [q[COLOUR]], s=16, c="black", zorder=3)
    ax.annotate("query (class 1)", (q[ASPECT], q[COLOUR]), xytext=(-30, -12), textcoords="offset points", fontsize=6.5)
    ax.scatter([w[ASPECT]], [w[COLOUR]], s=26, marker="*", c="#1d7a3a", zorder=3)
    ax.annotate("Δ-CE witness", (w[ASPECT], w[COLOUR]), xytext=(-40, 6), textcoords="offset points", fontsize=6.5,
                color="#1d7a3a")
    o = Z[OTHER]
    ax.scatter([o[ASPECT]], [o[COLOUR]], s=20, marker="x", c="#b03a2e", zorder=3)
    ax.annotate("class 0: no feasible\nwitness", (o[ASPECT], o[COLOUR]), xytext=(4, -14), textcoords="offset points",
                fontsize=6.5, color="#b03a2e")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_xlabel("aspect ($k^\\star$, changed)")
    ax.set_ylabel("colour (unchanged)")
    ax.text(0.03, 0.03, "shaded: $F^N_{\\kappa}$\ndashed: $m_n=-\\kappa$\nsolid: $m_o=\\kappa$",
            va="bottom", fontsize=6.5, transform=ax.transAxes)
    fig.savefig(OUT, format="pdf")
    print(f"wrote {OUT}; pair {PAIR}, query position {QUERY} (point_ids={pid[QUERY]}), "
          f"class-0 position {OTHER} (point_ids={pid[OTHER]}), witness distance {float(z['dce_N_distance'][QUERY]):.4f}")


if __name__ == "__main__":
    main()
