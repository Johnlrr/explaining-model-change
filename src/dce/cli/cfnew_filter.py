"""D-059 (descriptive, after seeing results): is the witness more than a filtered counterfactual of f_new?

From saved outputs only (no new search):

* filtered CF-new: the saved counterfactual of f_new (a2, m_n <= -kappa) kept only if f_old still keeps its
  decision there, m_o >= kappa (two forward passes); NOT_FOUND otherwise.  Its points are witnesses, and the
  nearest point with m_n <= -kappa that is a witness is the nearest witness, so filtered CF-new is the part of
  Delta-CE that a check on CF-new already gives ("case A").  The rest of Delta-CE ("case B") are witnesses at
  queries whose CF-new point is not a witness.
* null-adjusted Recovery: Recovery minus the rate at which the same method names k* on the 8 null pairs of split A.

Split A (exp_0017, Table 2) and X6 (exp_0060, split G).  Models are loaded hash-verified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from dce.cli.g0_tier0 import git_state
from dce.cli.x6_multi import BY_ID as X6_BY_ID, load_models as x6_load_models, locality, set_metrics
from dce.eval.pairs import load_pair
from dce.eval.tier0_margins import Tier0Margin
from dce.analysis.qualification import qualification_row
from dce.cli.v19_grid import logits_np
from dce.data.drift import clean_splits, ro_label

METHODS = ["dce_N", "b1_det", "b3", "b4", "b5", "a1", "a2"]
COND_K = {"T1": 2, "T3": 1, "T4": 0}  # Shortcut: colour, Subgroup: size, Threshold: aspect


def names_k(found: np.ndarray, dphi: np.ndarray, k: int) -> np.ndarray:
    """Per point: valid output whose largest |dphi| is factor k (ties count 0), as Recovery@r."""
    d = np.abs(dphi)
    top = d.argmax(1)
    ties = (d == d.max(1, keepdims=True)).sum(1) > 1
    return found.astype(bool) & (top == k) & ~ties


def margins(f, z: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = Tier0Margin(f, y)
    with torch.no_grad():
        return m(torch.as_tensor(z, dtype=torch.float32), torch.arange(len(z))).numpy().astype(float)


def filtered_cfnew(f_old, f_new, a: dict, kappa: float, y: np.ndarray | None = None) -> dict:
    z0 = a["z0"]
    if y is None:  # both models decide y at a valid query with margin >= kappa > 0
        with torch.no_grad():
            y = Tier0Margin(f_old, np.zeros(len(z0), dtype=int)).logits(torch.as_tensor(z0, dtype=torch.float32)).argmax(1).numpy()
    mo0, mn0 = margins(f_old, z0, y), margins(f_new, z0, y)
    if not ((mo0 >= kappa - 1e-4).all() and (mn0 >= kappa - 1e-4).all()):
        raise SystemExit("saved queries are not valid under the loaded models (wrong models or points)")
    zc = z0 + a["a2__dphi"]
    fa = a["a2__found"].astype(bool)
    mo, mn = margins(f_old, zc, y), margins(f_new, zc, y)
    keep = fa & (mo >= kappa)
    fd = a["dce_N__found"].astype(bool)
    same = fd & fa & (np.linalg.norm(a["dce_N__dphi"] - a["a2__dphi"], axis=1) < 1e-3)
    return {"found": keep, "dphi": a["a2__dphi"], "mn_ok": bool((mn[fa] <= -kappa + 1e-3).all()),
            "same_point": same, "dce_found": fd}


def split_a(cfg) -> dict:
    src = Path(cfg["split_a_run"])
    pp = json.loads((src / "results.json").read_text())["per_pair"]
    kappa, cache, rows = cfg["kappa"], {}, []
    for pid, p in sorted(pp.items(), key=lambda t: int(t[0])):
        a = dict(np.load(src / "pairs" / f"pair_{int(pid):02d}.npz"))
        _, f_old, f_new, _ = load_pair(int(pid), cache=cache)
        fc = filtered_cfnew(f_old, f_new, a, kappa)
        row = {"pair_id": int(pid), "condition": p["condition"], "n": int(len(a["z0"])), "a2_mn_ok": fc["mn_ok"],
               "found": {m: float(a[f"{m}__found"].mean()) for m in METHODS},
               "found_a2_filtered": float(fc["found"].mean()),
               "dce_point_equals_filtered_cfnew": float((fc["same_point"] & fc["found"]).sum() / max(fc["dce_found"].sum(), 1)),
               "filtered_cfnew_found_but_dce_not": int((fc["found"] & ~fc["dce_found"]).sum())}
        ks = [COND_K[p["condition"]]] if p["condition"] in COND_K else [0, 1, 2]
        row["names"] = {str(k): {**{m: float(names_k(a[f"{m}__found"], a[f"{m}__dphi"], k).mean()) for m in METHODS},
                                 "a2_filtered": float(names_k(fc["found"], fc["dphi"], k).mean())} for k in ks}
        if p["condition"] in COND_K:
            k = COND_K[p["condition"]]
            hit = names_k(a["dce_N__found"], a["dce_N__dphi"], k)
            row["dce_recovery_case_a"] = float((hit & fc["found"]).mean())
            row["dce_recovery_case_b"] = float((hit & ~fc["found"]).mean())
            saved = p["methods"]["dce_N"]["recovery"]
            if abs(row["names"][str(k)]["dce_N"] - saved) > 1e-9:
                raise SystemExit(f"pair {pid}: recomputed Delta-CE Recovery differs from the saved value")
        rows.append(row)
    nulls = [r for r in rows if r["condition"] in ("N0", "N1")]
    summ = {}
    for c, k in COND_K.items():
        g = [r for r in rows if r["condition"] == c]
        mean = lambda f: float(np.mean([f(r) for r in g]))
        rec = {m: mean(lambda r, m=m: r["names"][str(k)][m]) for m in METHODS + ["a2_filtered"]}
        null = {m: float(np.mean([r["names"][str(k)][m] for r in nulls])) for m in METHODS + ["a2_filtered"]}
        summ[c] = {"k_star": k, "recovery": rec, "null_names_k_star": null,
                   "null_adjusted": {m: rec[m] - null[m] for m in rec},
                   "found_dce": mean(lambda r: r["found"]["dce_N"]), "found_a2": mean(lambda r: r["found"]["a2"]),
                   "found_a2_filtered": mean(lambda r: r["found_a2_filtered"]),
                   "dce_recovery_case_a": mean(lambda r: r["dce_recovery_case_a"]),
                   "dce_recovery_case_b": mean(lambda r: r["dce_recovery_case_b"]),
                   "share_dce_points_case_a": mean(lambda r: r["dce_point_equals_filtered_cfnew"])}
    summ["Null"] = {"found_a2": float(np.mean([r["found"]["a2"] for r in nulls])),
                    "found_a2_filtered": float(np.mean([r["found_a2_filtered"] for r in nulls])),
                    "found_dce": float(np.mean([r["found"]["dce_N"] for r in nulls]))}
    return {"pairs": rows, "summary": summ}


def null_boundaries(cfg) -> dict:
    """Realised threshold and rotation of the split-A null f_new and of f_old, by the grid's logistic fit (v1.9.1)
    on the same validation points as the grid qualification (exp_0025), so dt is comparable to the grid null cell."""
    phi = clean_splits(seed=cfg["grid_data_seed"])["val"].phi[: cfg["grid_n_qualify"]]
    src = Path(cfg["split_a_run"])
    pp = json.loads((src / "results.json").read_text())["per_pair"]
    cache, rows, old = {}, [], {}
    fit = lambda f: qualification_row(phi, (np.diff(logits_np(f, phi), axis=1)[:, 0] > 0).astype(np.int64),
                                      ro_label(phi, 0.0, 0.5), 0.0, 0.5)
    for pid, p in sorted(pp.items(), key=lambda t: int(t[0])):
        if p["condition"] not in ("N0", "N1"):
            continue
        meta, f_old, f_new, _ = load_pair(int(pid), cache=cache)
        if meta["seed"] not in old:
            old[meta["seed"]] = fit(f_old)
        rn = fit(f_new)
        rows.append({"pair_id": int(pid), "condition": p["condition"], "seed": meta["seed"],
                     "t_hat": rn["t_hat"], "alpha_hat": rn["alpha_hat"], "dt": rn["t_hat"] - old[meta["seed"]]["t_hat"]})
    g = {r["pair_id"]: r for r in json.loads((Path(cfg["grid_run"]) / "qualification.json").read_text())["pairs"]}
    cell = [{"pair_id": i, "dt": g[i]["dt"], "t_hat": g[i]["t_hat"], "alpha_hat": g[i]["alpha_hat"]} for i in (20, 21)]
    return {"pairs": rows, "max_abs_dt": max(abs(r["dt"]) for r in rows),
            "max_abs_alpha_hat": max(abs(r["alpha_hat"]) for r in rows), "grid_null_cell": cell}


def x6(cfg) -> dict:
    xc = yaml.safe_load(Path(cfg["x6_config"]).read_text())
    out, lc, kappa, cache, rows = Path(xc["out_run"]), xc["locality"], cfg["kappa"], {}, []
    for pid in xc["pair_ids"]:
        a = dict(np.load(out / "pairs" / f"pair_{pid}.npz"))
        f_old, f_new, _ = x6_load_models(xc, X6_BY_ID[pid], cache)
        fc = filtered_cfnew(f_old, f_new, a, kappa, y=a["y"])
        rows.append({"pair_id": pid, "a2_mn_ok": fc["mn_ok"],
                     "found_dce": float(fc["dce_found"].mean()), "found_a2": float(a["a2__found"].mean()),
                     "found_a2_filtered": float(fc["found"].mean()),
                     "share_dce_points_case_a": float((fc["same_point"] & fc["found"]).sum() / max(fc["dce_found"].sum(), 1)),
                     "a2_filtered": {**locality(fc["found"], fc["dphi"], a["z0"][:, 1], lc["near"], lc["far"]),
                                     **set_metrics(fc["found"], fc["dphi"])},
                     "dce_N": {**locality(a["dce_N__found"], a["dce_N__dphi"], a["z0"][:, 1], lc["near"], lc["far"]),
                               **set_metrics(a["dce_N__found"], a["dce_N__dphi"])}})
    pooled = {}
    for m in ("dce_N", "a2_filtered"):
        nn_, nf = sum(r[m]["n_near"] for r in rows), sum(r[m]["n_far"] for r in rows)
        pooled[m] = sum(r[m]["n_size_near"] for r in rows) / nn_ - sum(r[m]["n_size_far"] for r in rows) / nf
    return {"pairs": rows, "L_pooled": pooled,
            "mean": {k: float(np.mean([r[k] for r in rows])) for k in ("found_dce", "found_a2", "found_a2_filtered", "share_dce_points_case_a")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(cfg.get("threads", 8))
    res = {"label": "D-059 filtered CF-new and null-adjusted Recovery (descriptive, after seeing results)",
           "split_a": split_a(cfg), "x6": x6(cfg), "null_boundaries": null_boundaries(cfg), "git": git_state()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results_cfnew_filter.json").write_text(json.dumps(res, indent=1))
    s = res["split_a"]["summary"]
    for c in COND_K:
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s[c].items() if not isinstance(v, dict)})
        print("   null-adjusted", {m: round(v, 2) for m, v in s[c]["null_adjusted"].items()})
    print("Null", s["Null"])
    print("X6", res["x6"]["mean"], res["x6"]["L_pooled"])
    nb = res["null_boundaries"]
    print("nulls |dt| max", round(nb["max_abs_dt"], 4), "|alpha_hat| max", round(nb["max_abs_alpha_hat"], 4), [round(r["dt"], 4) for r in nb["pairs"]])
    print("grid null cell", nb["grid_null_cell"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
