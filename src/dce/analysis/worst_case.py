"""Worst-case sensitivity of the reported comparisons to the solver's tangential error (descriptive; saved outputs only).

A perturbation of an explanation u with ‖δ‖ ≤ τ‖u‖ changes the difference of two |components| by at most √2·τ‖u‖.
Per comparison A − B with a reported direction, the method the direction favours loses every correct point whose
k* leads by less than √2·τ‖u‖, and the other gains every wrong point whose k* trails the top factor by less than
that (ties included). ‖u‖ is the unprojected norm (the projection does not increase the error). τ is per method
(exp_0038: the experiments' solver vs a tight reference on validation). Estimates are pair means of per-pair mean
differences, as in every registered analysis.

    python -m dce.analysis.worst_case <tau_solver> <tau_deepxplore> <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from dce.analysis.projection import project
from dce.eval.metrics import MethodOutput, recovery_indicator

R = Path("runs/experiments")
SQ2 = np.sqrt(2.0)


def margins(d: np.ndarray, k: int) -> np.ndarray:
    """|d_k| − max_{j≠k} |d_j| (positive: k leads)."""
    a = np.abs(d)
    other = np.delete(a, k, axis=1).max(1)
    return a[:, k] - other


def adjust(ind: np.ndarray, d: np.ndarray, norm: np.ndarray, found: np.ndarray, k: int, tau: float, role: str) -> np.ndarray:
    lead = margins(d, k)
    thr = SQ2 * tau * norm
    if role == "down":
        return ind - ((ind == 1) & (lead < thr)).astype(float)
    return ind + ((ind == 0) & found & (lead > -thr)).astype(float)


def load(exp: str, pid: int) -> tuple[dict, dict]:
    name = f"pair_{pid:02d}" if exp in ("exp_0017", "exp_0019") else f"pair_{pid}"
    return json.loads((R / exp / "pairs" / f"{name}.json").read_text()), dict(np.load(R / exp / "pairs" / f"{name}.npz"))


def indicator(a: dict, m: str, k: int, g: np.ndarray | None, support: np.ndarray | None, tau: float, role: str | None):
    found = a[f"{m}__found"].astype(bool)
    o = MethodOutput(found, a[f"{m}__dphi"])
    norm = np.linalg.norm(o.dphi, axis=1)
    if g is not None:
        o = project(o, g)
    ind = recovery_indicator(o, k)
    if role:
        ind = adjust(ind, o.dphi, norm, found, k, tau, role)
    return ind if support is None else ind[support]


def estimate(exp, pids, a_m, b_m, k_of, taus, roles, grad=None, support=False):
    diffs = []
    for pid in pids:
        row, arr = load(exp, pid)
        k = k_of(row)
        g = arr[grad] if grad else None
        s = arr["dce__found"].astype(bool) if support else None
        ia = indicator(arr, a_m, k, g, s, taus[a_m], roles[0])
        ib = indicator(arr, b_m, k, g, s, taus[b_m], roles[1])
        diffs.append(ia.mean() - ib.mean())
    return float(np.mean(diffs))


def run(tau_s: float, tau_x: float) -> dict:
    taus = {m: tau_s for m in ("dce_N", "dce", "b1_det", "b3", "b3_z0")} | {"a1": tau_x}
    kstar = lambda r: r["k_star"]
    colour = lambda r: 2
    h5d_pairs = json.loads((R / "exp_0026" / "results.json").read_text())["H5"]["H5d"]["pairs"] \
        if "H5" in json.loads((R / "exp_0026" / "results.json").read_text()) else None
    if h5d_pairs is None:
        res26 = json.loads((R / "exp_0026" / "results.json").read_text())
        h5d_pairs = next(v["pairs"] for k, v in _walk(res26) if k == "H5d")
    specs = [  # name, exp, pairs, A, B, k*, grad, support, criterion
        ("H2a (Subgroup, locked)", "exp_0017", (3, 4, 5), "dce_N", "b3", kstar, None, False, "sign"),
        ("Tie: Shortcut Δ-CE vs Grad-diff", "exp_0017", (0, 1, 2), "dce_N", "b3", kstar, None, False, "abs>0.10"),
        ("Tie: Shortcut Δ-CE vs CF-diff", "exp_0017", (0, 1, 2), "dce_N", "b1_det", kstar, None, False, "abs>0.10"),
        ("Tie: Threshold Δ-CE vs Grad-diff", "exp_0017", (6, 7, 8), "dce_N", "b3", kstar, None, False, "abs>0.10"),
        ("Tie: Threshold Δ-CE vs CF-diff", "exp_0017", (6, 7, 8), "dce_N", "b1_det", kstar, None, False, "abs>0.10"),
        ("C1 (Threshold, new data)", "exp_0019", (6, 7, 8), "dce_N", "b3", kstar, None, False, "sign"),
        ("C3 (Shortcut, projected)", "exp_0019", (0, 1, 2), "dce_N", "b3", kstar, "grad_old", False, "abs>0.10"),
        ("H5d (grid, projected)", "exp_0026", tuple(h5d_pairs), "dce_N", "b3", colour, "grad_old", False, "abs>0.10"),
        ("Δ-CE vs DeepXplore: Subgroup", "exp_0017", (3, 4, 5), "dce_N", "a1", kstar, None, False, "sign"),
        ("Δ-CE vs DeepXplore: Threshold", "exp_0017", (6, 7, 8), "dce_N", "a1", kstar, None, False, "sign"),
    ]
    out = {"tau_solver": tau_s, "tau_deepxplore": tau_x, "factor": "sqrt(2)", "comparisons": []}
    for name, exp, pids, a, b, kf, grad, sup, crit in specs:
        zero = {m: 0.0 for m in taus}
        orig = estimate(exp, pids, a, b, kf, zero, (None, None), grad, sup)
        roles = ("down", "up") if orig > 0 else ("up", "down")      # move towards zero / the other sign
        worst = estimate(exp, pids, a, b, kf, taus, roles, grad, sup)
        holds = (worst > 0) == (orig > 0) and worst != 0 if crit == "sign" else abs(worst) > 0.10 and (worst > 0) == (orig > 0)
        out["comparisons"].append({"name": name, "estimate": orig, "worst": worst, "criterion": crit, "direction_holds": bool(holds)})
    # H8: T = |gap(z0,z0)| − 2|gap(z*,z0)| on the Δ-CE support of the four H5d pairs (exp_0033)
    h8 = (36, 37, 44, 45)
    g00 = estimate("exp_0033", h8, "dce", "b3_z0", colour, {m: 0.0 for m in taus}, (None, None), "grad_z0", True)
    gs0 = estimate("exp_0033", h8, "dce", "b3_z0", colour, {m: 0.0 for m in taus}, (None, None), "grad_zs", True)
    roles00 = ("down", "up") if g00 > 0 else ("up", "down")
    w00 = estimate("exp_0033", h8, "dce", "b3_z0", colour, taus, roles00, "grad_z0", True)
    ws0 = max((estimate("exp_0033", h8, "dce", "b3_z0", colour, taus, r, "grad_zs", True) for r in (("up", "down"), ("down", "up"))),
              key=abs)
    T, Tw = abs(g00) - 2 * abs(gs0), abs(w00) - 2 * abs(ws0)
    out["H8"] = {"gap_z0z0": g00, "gap_zsz0": gs0, "T": T, "worst_gap_z0z0": w00, "worst_gap_zsz0": ws0, "worst_T": Tw,
                 "direction_holds": bool(Tw > 0)}
    return out


def _walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k, v
            yield from _walk(v)


if __name__ == "__main__":
    res = run(float(sys.argv[1]), float(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
