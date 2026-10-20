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


PRINT = {"figsize": (4.8, 1.65), "marker_size": 16, "max_ticks": 3}   # 12.2 cm × 4.2 cm: ESANN \linewidth at 1:1


def make_map(rows: list[dict], out: Path, figsize=(7.0, 2.4), marker_size: float = 40, max_ticks: int | None = None) -> None:
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "xtick.major.size": 2, "ytick.major.size": 2})
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)
    cmap, sc = plt.get_cmap("Blues"), None
    for ax, (title, key) in zip(axes, PANELS):
        for r in rows:
            v = value(r, key)
            kw = dict(s=marker_size, marker=MARKERS[r["seed"]], linewidths=0.6, edgecolors="#333333", zorder=3)
            if v is None:
                ax.scatter(r["t_hat"], r["alpha_hat"], facecolors="none", **kw)
            else:
                sc = ax.scatter(r["t_hat"], r["alpha_hat"], c=[v], cmap=cmap, vmin=0, vmax=1, **kw)
        ax.set_title(title.replace("@r", "") if max_ticks else title, fontsize=8)
        if max_ticks:
            ax.set_xticks([0.45, 0.55, 0.65])
            ax.set_yticks([0.0, 0.25, 0.5])
        else:
            ax.set_xlabel(r"measured threshold $\hat t$")
        ax.grid(True, linewidth=0.3, color="#dddddd", zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"measured rotation $\hat\alpha$")
    for s, m in MARKERS.items():
        axes[0].scatter([], [], marker=m, facecolors="none", edgecolors="#333333", label=f"seed {s}")
    if max_ticks:   # print size: one shared x label, seed legend in the right column
        fig.supxlabel(r"measured threshold $\hat t$", fontsize=8)
        fig.legend(frameon=False, fontsize=7, loc="outside right lower", ncols=1, handletextpad=0.2, borderaxespad=0.1)
    else:
        fig.legend(frameon=False, fontsize=7, loc="outside lower center", ncols=2)
    fig.colorbar(sc, ax=axes, shrink=0.9, label=None if max_ticks else "value", ticks=[0, 0.5, 1])
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
