"""Figure 1 (paper): T4 (threshold shift, offset-type update) on one pair, from saved outputs only.

Pair: T4 seed 0 = pair 6 (exp_0017, Tier 0 final run, test split). Factor plane: x = aspect (k*, changed by the
update), y = colour (a factor the update does not touch). The other factors (size, bg, posx) are fixed at the
query point 155's values. Query 155 (class 1) has a Δ-CE witness that moves only aspect (|Δ| = 0.052 on other
factors ≤ 0.005). Point 278 (class 0) has no feasible witness; its other factors differ from query 155 by ≤ 0.023.

Panels: left, the decision boundaries (logit difference = 0) of f_old (solid) and f_new (dashed), the witness region
F^N_κ = {m_n ≤ −κ, m_o ≥ κ} (shaded, clean label of the query); right, renders of the query, its Δ-CE witness, the
ordinary counterfactuals of f_new and of f_old from the same query (single-point runs of the locked ordinary-CE solver,
r = ∞, restart seeds as in B1: 10 006 + 1 for f_new, 10 006 + 0 for f_old), and the class-0 point.

ILLUSTRATIVE, ONE QUERY. The ordinary counterfactuals are recomputed for this one point only, in a one-point batch,
so restart noise differs from the 300-point batch that produced exp_0017's stored B1 values. They are not results and
enter no table. No experiment is rerun; the saved outputs and the checkpoints of exp_0002/exp_0003 are used.

Output: research/figures/fig1_t4.pdf (vector, 7.5 cm wide).
Run: uv run python research/figures/fig1_t4.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dce.cli.calibrate_r import solver_config
from dce.cli.g0_tier0 import load_config
from dce.data.drift import clean_splits
from dce.data.renderer import render
from dce.eval.pairs import load_pair
from dce.eval.tier0_margins import Tier0Margin
from dce.solvers.ordinary_ce import ordinary_ce

KAPPA = 0.5
PAIR = 6
RUN = Path("runs/experiments/exp_0017")
QUERY, OTHER = 155, 278       # positions in the valid-point list of pair 6
ASPECT, COLOUR = 0, 2  # factor columns: aspect, colour (renderer order)
OUT = Path(__file__).with_name("fig1_t4.pdf")


def logit_diff(model, phi: np.ndarray, y: int) -> np.ndarray:
    """f_y − f_{1−y} (spec §3.1 margin), float32 as in the solver's evaluation."""
    with torch.no_grad():
        lg = model(render(torch.as_tensor(phi, dtype=torch.float32)))
    return (lg[:, y] - lg[:, 1 - y]).numpy()


def ordinary_counterfactual(model, z0: np.ndarray, y: int, cfg: dict, restart_seed: int) -> np.ndarray | None:
    """Single-point ordinary CE of `model` for the clean label y (locked solver; r = ∞ for illustration)."""
    scfg = solver_config(cfg, r=float("inf"))
    res = ordinary_ce(torch.as_tensor(z0[None], dtype=torch.float32), Tier0Margin(model, np.array([y])), scfg, restart_seed)
    return res.z[0].numpy().astype(float) if bool(res.found[0]) else None


def main() -> None:
    test = clean_splits(seed=1234)["test"]
    z = np.load(RUN / "pairs" / f"pair_{PAIR:02d}.npz")
    pid, Z, found, dist, D = z["point_ids"], z["z0"], z["dce_N__found"], z["dce_N_distance"], z["dce_N__dphi"]
    assert np.allclose(test.phi[pid], Z, atol=1e-6), "saved points do not match the test split"
    y = test.y_clean[pid]
    assert y[QUERY] == 1 and found[QUERY] and y[OTHER] == 0 and not found[OTHER]
    _, f_old, f_new, _ = load_pair(PAIR)
    cfg = load_config(Path("configs/experiments/exp_0017.yaml"))
    q, w = Z[QUERY], Z[QUERY] + D[QUERY]
    seed = cfg["seeds"]["restart_base"] + PAIR           # B1 uses seed + 0 (f_old) and seed + 1 (f_new)
    ce_new = ordinary_counterfactual(f_new, q, 1, cfg, seed + 1)
    ce_old = ordinary_counterfactual(f_old, q, 1, cfg, seed + 0)
    print("CE f_new:", None if ce_new is None else ce_new.round(3), " CE f_old:", None if ce_old is None else ce_old.round(3))
    # sanity: the saved witness satisfies the constraints under the float32 models
    m_o_w, m_n_w = logit_diff(f_old, w[None], 1)[0], logit_diff(f_new, w[None], 1)[0]
    assert m_n_w <= -KAPPA + 1e-3 and m_o_w >= KAPPA - 1e-3, (m_o_w, m_n_w)

    g = np.linspace(0, 1, 121)
    A, C = np.meshgrid(g, g)
    plane = np.tile(q, (A.size, 1))
    plane[:, ASPECT], plane[:, COLOUR] = A.ravel(), C.ravel()
    d_old = logit_diff(f_old, plane, 1).reshape(A.shape)
    d_new = logit_diff(f_new, plane, 1).reshape(A.shape)
    region = (d_new <= -KAPPA) & (d_old >= KAPPA)

    plt.rcParams.update({"font.size": 6, "pdf.fonttype": 42, "axes.linewidth": 0.5})
    fig = plt.figure(figsize=(2.95, 3.3))
    ax = fig.add_axes([0.14, 0.45, 0.82, 0.52])
    ax.contourf(A, C, region, levels=[0.5, 1.5], colors=["#c9d9ef"])
    ax.contour(A, C, d_old, levels=[0.0], colors="#1f4e9a", linewidths=0.7)
    ax.contour(A, C, d_new, levels=[0.0], colors="#b03a2e", linewidths=0.7, linestyles="--")
    ax.scatter([q[ASPECT]], [q[COLOUR]], s=12, c="black", zorder=3)
    ax.annotate("query, class 1", (q[ASPECT], q[COLOUR]), xytext=(4, -9), textcoords="offset points", fontsize=5.5)
    ax.scatter([w[ASPECT]], [w[COLOUR]], s=16, marker="*", c="#1d7a3a", zorder=3)
    ax.annotate("Δ-CE witness", (w[ASPECT], w[COLOUR]), xytext=(-42, 5), textcoords="offset points", fontsize=5.5,
                color="#1d7a3a")
    o = Z[OTHER]
    ax.scatter([o[ASPECT]], [o[COLOUR]], s=14, marker="x", c="#b03a2e", zorder=3)
    ax.annotate("class 0: no feasible\nwitness", (o[ASPECT], o[COLOUR]), xytext=(-2, -16), textcoords="offset points",
                fontsize=5.5, color="#b03a2e")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("aspect (k*, changed)")
    ax.set_ylabel("colour (unchanged)")
    ax.text(0.02, 0.02, "solid: $f_{old}$ boundary\ndashed: $f_{new}$ boundary\nshaded: $F^N_\\kappa$",
            va="bottom", fontsize=5, transform=ax.transAxes)

    row = [("query", q), ("witness", w), ("CE $f_{new}$", ce_new), ("CE $f_{old}$", ce_old), ("class 0", o)]
    row = [(t, p) for t, p in row if p is not None]
    for k, (title, phi) in enumerate(row):
        a = fig.add_axes([0.01 + k * 0.198, 0.0, 0.19, 0.25])
        img = render(torch.as_tensor(phi[None], dtype=torch.float32))[0].permute(1, 2, 0).numpy()
        a.imshow(np.clip(img, 0, 1))
        a.set_title(title, fontsize=5, pad=3)
        a.axis("off")
    fig.text(0.5, 0.265, "illustrative: one query point, not a result", ha="center", fontsize=4.5, style="italic")
    fig.savefig(OUT, format="pdf")
    print(f"wrote {OUT}; pair {PAIR} (T4 s0), query position {QUERY} = point_ids[{QUERY}]={pid[QUERY]}, "
          f"class-0 position {OTHER} = point_ids[{OTHER}]={pid[OTHER]}; witness distance {dist[QUERY]:.4f}")


if __name__ == "__main__":
    main()
