"""v1.9 §5.2 map (D-036, D-038): every pair at its measured (t̂, α̂), three panels sharing one sequential scale.

Panels: Recovery(B3), Recovery(Δ-CE), found-rate(Δ-CE). Recovery uses the pair's k* (aspect on α = 0, colour on
α > 0); the null cell has no k* and is drawn hollow in the Recovery panels. Marker shape = f_old seed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = (("B3 Recovery@r", "rec_b3"), ("Δ-CE Recovery@r", "rec_dce_N"), ("Δ-CE found-rate", "found_dce_N"))
MARKERS = {0: "o", 1: "s"}


def value(row: dict, key: str):
    if key == "found_dce_N":
        return row[key]
    if row["k_star"] is None:
        return None
    return row[f"{key}_{row['k_star']}"]


def make_map(rows: list[dict], out: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True, constrained_layout=True)
    cmap, sc = plt.get_cmap("Blues"), None
    for ax, (title, key) in zip(axes, PANELS):
        for r in rows:
            v = value(r, key)
            kw = dict(s=40, marker=MARKERS[r["seed"]], linewidths=0.6, edgecolors="#333333", zorder=3)
            if v is None:
                ax.scatter(r["t_hat"], r["alpha_hat"], facecolors="none", **kw)
            else:
                sc = ax.scatter(r["t_hat"], r["alpha_hat"], c=[v], cmap=cmap, vmin=0, vmax=1, **kw)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(r"measured offset $\hat t$")
        ax.grid(True, linewidth=0.3, color="#dddddd", zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"measured rotation $\hat\alpha$")
    for s, m in MARKERS.items():
        axes[0].scatter([], [], marker=m, facecolors="none", edgecolors="#333333", label=f"f_old seed {s}")
    fig.legend(frameon=False, fontsize=7, loc="outside lower center", ncols=2)
    fig.colorbar(sc, ax=axes, shrink=0.9, label="value")
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
