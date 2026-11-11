"""Figure 1 of the paper: geometry of a witness in the linear case (Lemma 1, Proposition 2).

Schematic, no experimental data. Usage: uv run python research/figures/fig_geometry.py
Writes paper/figures/geometry.pdf.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5, "pdf.fonttype": 42})
OLD, NEW, WIT, GRD, INK = "#6b6b6b", "#1f5fa8", "#2e8b57", "#d2691e", "#111111"
K = 0.08  # kappa in plot units (unit margin slope)


def panel(ax, title, n_new, b_old, b_new, z0, note=None, grad=True, u_offset=(-9, 3)):
    """Margins: m_o = b_old - x, m_n = b_new - n_new.z (both positive at z0)."""
    xs = np.linspace(0, 1, 401)
    X, Y = np.meshgrid(xs, xs)
    m_o = b_old - X
    m_n = b_new - (n_new[0] * X + n_new[1] * Y)
    wit = (m_o >= K) & (m_n <= -K)
    ax.contourf(X, Y, wit.astype(float), levels=[0.5, 1.5], colors=[WIT], alpha=0.28)
    ax.contour(X, Y, m_o, levels=[0], colors=[OLD], linewidths=1.1)
    ax.contour(X, Y, m_o, levels=[K], colors=[OLD], linewidths=0.7, linestyles="--")
    ax.contour(X, Y, m_n, levels=[0], colors=[NEW], linewidths=1.1)
    ax.contour(X, Y, m_n, levels=[-K], colors=[NEW], linewidths=0.7, linestyles="--")
    ax.plot(*z0, "o", ms=3.2, color=INK, zorder=5)
    ax.annotate("$z_0$", z0, xytext=(-8, -10), textcoords="offset points", fontsize=9)
    if wit.any():
        d = (X - z0[0]) ** 2 + (Y - z0[1]) ** 2
        d[~wit] = np.inf
        i = np.unravel_index(np.argmin(d), d.shape)
        w = (X[i], Y[i])
        ax.annotate("", w, z0, arrowprops=dict(arrowstyle="-|>", color=WIT, lw=1.3, shrinkA=1.5, shrinkB=0))
        ax.annotate("$u^\\star$", ((z0[0] + w[0]) / 2, (z0[1] + w[1]) / 2), xytext=u_offset, textcoords="offset points", color=WIT, fontsize=9)
    g = np.array(n_new) - np.array([1.0, 0.0])  # -(w_n - w_o) with w = -normal
    if grad and np.linalg.norm(g) > 1e-9:
        e = np.array(z0) + 0.22 * g / np.linalg.norm(g)
        ax.annotate("", e, z0, arrowprops=dict(arrowstyle="-|>", color=GRD, lw=1.3, shrinkA=1.5, shrinkB=0))
    if note:
        ax.text(*note[0], note[1], fontsize=8.5, color=note[2], ha="center", va="center")
    ax.set_title(title, fontsize=9, pad=2.5)
    ax.set_xlim(0, 1); ax.set_ylim(0.1, 0.9); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("old factor ($-\\hat w_o$)", fontsize=8.5, labelpad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.36))
    # null update: boundaries 0.14 apart (< 2 kappa = 0.16), so m_o >= kappa (x <= 0.52) and m_n <= -kappa (x >= 0.54)
    # never meet. For parallel boundaries, |u_n - u_o| equals the gap between the two solid lines.
    panel(axes[0], "(a) null update: $|h|<2\\kappa$", (1, 0), 0.60, 0.46, (0.20, 0.47),
          note=((0.81, 0.40), "no\nwitness", INK))
    axes[0].annotate("", (0.46, 0.80), (0.60, 0.80), arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
    axes[0].text(0.64, 0.78, "CF-diff\n$\\neq 0$", fontsize=8.5, color=INK, ha="left", va="center")
    panel(axes[1], "(b) offset: $\\theta=0$", (1, 0), 0.80, 0.45, (0.17, 0.5),
          note=((0.225, 0.24), "Grad-diff\n$=0$", GRD))
    th = np.deg2rad(52)
    n = (np.cos(th), np.sin(th))
    panel(axes[2], "(c) rotation: $\\theta>0$", n, 0.82, n[0] * 0.58 + n[1] * 0.52, (0.28, 0.24), u_offset=(5, -8))
    axes[2].text(0.03, 0.47, "Grad-diff", fontsize=8.5, color=GRD, ha="left")
    axes[0].set_ylabel("new factor ($-v$)", fontsize=8.5, labelpad=1.5)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.885, bottom=0.10, wspace=0.06)
    out = Path("paper/figures/geometry.pdf")
    fig.savefig(out)
    fig.savefig(".work/layout/geometry.png", dpi=220)
    print(out)


if __name__ == "__main__":
    main()
