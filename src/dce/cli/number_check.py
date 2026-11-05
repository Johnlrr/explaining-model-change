"""Number check (D-056 W3): every reported number of paper/main.tex and the numeric claims of
paper/supplement.tex (Sections B, C, E; the parameters of D) against the saved run outputs.

A hand-written table of claims, one entry per printed number: (location, value as printed, source, computed value,
rounding rule). A number is OK when it equals the source after the paper's rounding (half a unit of the last printed
digit; one unit of the leading digit for a power of ten); MISMATCH when it does not (or when the printed text is not
found in the tex file); UNSOURCED when no saved output holds it. Nothing is re-run: only saved results*.json, saved
pair arrays and configs are read. This script interprets nothing.

Usage:
    python -m dce.cli.number_check --out runs/experiments/exp_0059
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

from dce.analysis.v19 import boot_mean
from dce.cli.g0_tier0 import git_state
from dce.cli.review_0211 import k_index, load_run
from dce.eval.metrics import MethodOutput, recovery_indicator

R = Path("runs/experiments")
PAPER = {"main": Path("paper/main.tex"), "supp": Path("paper/supplement.tex")}
METH = {"dce": "dce_N", "cfdiff": "b1_det", "grad": "b3", "imp": "b4", "seg": "b5", "noprox": "a1", "cfnew": "a2"}


def J(p: str) -> dict:
    return json.loads((R / p).read_text())


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


TEX = {k: squash(v.read_text()) for k, v in PAPER.items()}


def dec_of(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def pf(s: str) -> float:
    s = s.replace("−", "-").replace("$", "").replace("\\,", "").replace(" ", "")
    return float(s)


def close(printed: str, computed: float, scale: float = 1.0) -> bool:
    """Equality after the printed rounding. Scientific strings ("2.4e-4") compare on their mantissa digits."""
    c = computed * scale
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)e(-?\d+)", printed.replace("−", "-"))
    if m:
        p, tol = float(m[1]) * 10 ** int(m[2]), 0.5 * 10 ** (int(m[2]) - dec_of(m[1]))
        return abs(c - p) <= tol * (1 + 1e-9)
    p = pf(printed)
    return abs(c - p) <= 0.5 * 10 ** (-dec_of(printed.replace("$", "").replace("\\,", ""))) + 1e-9


class Ledger:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def _add(self, loc, printed, computed, source, status, note="", rule="") -> None:
        if isinstance(computed, (np.floating, np.integer)):
            computed = computed.item()
        self.rows.append({"id": len(self.rows) + 1, "location": loc, "printed": printed, "computed": computed,
                          "status": status, "source": source, "rule": rule, "note": note})

    def text(self, tex: str, snippet: str, loc: str) -> None:
        """The printed text must be in the tex file (guards against a claim table that drifted from the paper)."""
        ok = squash(snippet) in TEX[tex]
        self._add(f"{loc} [text]", snippet, ok, PAPER[tex].name, "OK" if ok else "MISMATCH",
                  "" if ok else "printed snippet not found in the tex file", "substring present")

    def num(self, loc, printed: str, computed: float, source: str, scale: float = 1.0, note: str = "") -> None:
        ok = close(printed, computed, scale)
        self._add(loc, printed, computed * scale, source, "OK" if ok else "MISMATCH", note, "round to printed digits")

    def exact(self, loc, printed, computed, source: str, note: str = "") -> None:
        ok = printed == computed
        self._add(loc, printed, computed, source, "OK" if ok else "MISMATCH", note, "equal")

    def bound(self, loc, printed: str, computed: float, source: str, note: str = "", kind: str = "le") -> None:
        p = pf(printed)
        ok = computed <= p + 0.5 * 10 ** (-dec_of(printed)) if kind == "le" else computed >= p - 0.5 * 10 ** (-dec_of(printed))
        self._add(loc, printed, computed, source, "OK" if ok else "MISMATCH", note, f"{kind} bound at printed digits")

    def unsourced(self, loc, printed, why: str) -> None:
        self._add(loc, printed, None, "none", "UNSOURCED", why)


L = Ledger()


# ---------------------------------------------------------------------------------------------------------------------
def pair_mean(a: dict, meth: str, k: int) -> float:
    return float(recovery_indicator(MethodOutput(a[f"{meth}__found"], a[f"{meth}__dphi"]), k).mean())


def sec_params(A, B, G, D) -> None:
    tier = yaml.safe_load(Path("configs/tiers/tier0.yaml").read_text())
    drift = Path("src/dce/data/drift.py").read_text()
    s = "tier0.yaml"
    L.num("S3 budget r", "0.54", tier["r"], f"configs/tiers/{s}:r")
    L.num("S-D budget r", "0.5413", tier["r"], f"configs/tiers/{s}:r")
    L.exact("S3 kappa", 0.5, tier["kappa"], f"configs/tiers/{s}:kappa")
    L.exact("S-D kappa", 0.5, tier["kappa"], f"configs/tiers/{s}:kappa")
    L.exact("S3 label noise 5%", 5, round(tier["label_noise"] * 100), f"configs/tiers/{s}:label_noise")
    L.exact("S3 five factors", 5, len(tier["factors"]), f"configs/tiers/{s}:factors")
    L.exact("S3 f_old five seeds", 5, len({m["seed"] for m, _ in A}), "exp_0017 pairs/*.json:seed")
    n_all = {m["n_points"] for run in (A, B, G, D) for m, _ in run}
    L.exact("S3 N=300 valid points per pair (exp_0017, 0019, 0026, 0049)", {300}, n_all, "pairs/*.json:n_points")
    L.exact("S-D N=300 (supplement Sec. D)", {300}, n_all, "pairs/*.json:n_points")
    m = re.search(r"def adaptation_set\(\s*condition: str, seed: int, n: int = ([\d_]+)", drift)
    L.exact("S3 fine-tuned on 10 000 samples", 10000, int(m[1].replace("_", "")) if m else None, "src/dce/data/drift.py:adaptation_set(n=)")
    m = re.search(r"IMAGE_SIZE = (\d+)", Path("src/dce/data/renderer.py").read_text())
    L.exact("S3 32x32 images", 32, int(m[1]) if m else None, "src/dce/data/renderer.py:IMAGE_SIZE")
    for name, printed, field in (("Shortcut colour correlated with label (0.99)", 0.99, "t1_correlation"),
                                 ("Subgroup labels flipped for size > 0.75", 0.75, "t3_size_threshold"),
                                 ("Threshold y = 1[aspect > 0.6]", 0.6, "t4_threshold"),
                                 ("Null N1 size ~ U[0.5, 1]", 0.5, "n1_size_low")):
        mm = re.search(rf"{field}: float = ([\d.]+)", drift)
        got, srcf = (float(mm[1]) if mm else None), f"src/dce/data/drift.py:DriftParams.{field}"
        if field == "t1_correlation":   # G0 adjustment (spec 4.4): the Tier 0 pairs use 0.99, the dataclass default is 0.95
            mm = re.search(rf"{field}: ([\d.]+)", Path("configs/experiments/exp_0003.yaml").read_text())
            got, srcf = float(mm[1]), "configs/experiments/exp_0003.yaml:drift_params.t1_correlation (G0 adjustment; DriftParams default is 0.95)"
        L.exact(f"Table 1 {name}", printed, got, srcf)
    sv = tier["solver"]
    for label, printed, got in (("R = 5 restarts", 5, sv["R"]), ("T = 400 steps", 400, sv["T"]), ("lr 0.05", 0.05, sv["lr"]),
                                ("lambda0 = 1", 1.0, sv["lam0"]), ("halved 8 times", 8, sv["n_lam_steps"]),
                                ("sigma = 0.05", 0.05, sv["sigma"]), ("SLSQP at most 100 iterations", 100, sv["slsqp_maxiter"])):
        L.exact(f"S-D solver: {label}", printed, got, "configs/tiers/tier0.yaml:solver")
    L.exact("S-D ResNet-8 runs use 100 queries per pair", 100, yaml.safe_load(Path("configs/experiments/exp_0050_run.yaml").read_text())["n_points"],
            "configs/experiments/exp_0050_run.yaml:n_points")
    L.exact("S-D X5 uses 100 queries", 100, yaml.safe_load(Path("configs/experiments/exp_0057.yaml").read_text())["n_points"],
            "configs/experiments/exp_0057.yaml:n_points")
    K = int(re.search(r"b5x_K: (\d+)", Path("configs/experiments/exp_0048.yaml").read_text())[1])
    K0 = int(re.search(r"K: int = (\d+)", Path("src/dce/baselines/b5_line.py").read_text())[1])
    L.exact("S4.4 Seg-search 'ten times the segments'", 10, K // K0, "configs/experiments/exp_0048.yaml:b5x_K / b5_line.py:K")


def table1(A) -> None:
    res = J("exp_0017/results.json")
    cnt = {}
    for p in res["per_pair"].values():
        cnt[p["condition"]] = cnt.get(p["condition"], 0) + 1
    for cond, name in (("T1", "Shortcut"), ("T3", "Subgroup"), ("T4", "Threshold")):
        L.exact(f"Table 1 {name} pairs", 3, cnt[cond], "exp_0017/results.json:per_pair.*.condition")
    L.exact("Table 1 Null pairs resample (N0)", 5, cnt["N0"], "exp_0017/results.json:per_pair.*.condition")
    L.exact("Table 1 Null pairs size~U[0.5,1] (N1)", 3, cnt["N1"], "exp_0017/results.json:per_pair.*.condition")
    g = J("exp_0026/results.json")["descriptive"]["pairs"]
    L.exact("Table 1 Grid alpha=0 pairs", 8, sum(p["alpha"] == 0 for p in g), "exp_0026/results.json:descriptive.pairs")
    L.exact("Table 1 Grid alpha>0 pairs", 24, sum(p["alpha"] > 0 for p in g), "exp_0026/results.json:descriptive.pairs")
    L.exact("Table 1 Grid alpha values {0,.1,.25,.5}", [0.0, 0.1, 0.25, 0.5], sorted({p["alpha"] for p in g}), "exp_0026/results.json")
    L.exact("Table 1 Grid t values {.5,.53,.56,.6}", [0.5, 0.53, 0.56, 0.6], sorted({p["t"] for p in g}), "exp_0026/results.json")
    L.exact("Table 1 Grid two seeds", 2, len({p["seed"] for p in g}), "exp_0026/results.json")
    L.exact("Table 1 footnote a: four offsets enter the tests", 4, len(J("exp_0026/results.json")["H5"]["H5a"]["pairs"]), "exp_0026/results.json:H5.H5a.pairs")
    x2 = J("exp_0049/results_d.json")["x2"]
    L.exact("Table 1 Offsets pairs", 9, len(x2["pairs"]), "exp_0049/results_d.json:x2.pairs")
    L.exact("Table 1 Offsets t in {.53,.56,.6}", [0.53, 0.56, 0.6], sorted({p["t"] for p in x2["pairs"]}), "exp_0049/results_d.json")
    L.exact("Table 1 Offsets seeds 2-4", [2, 3, 4], sorted({p["seed"] for p in x2["pairs"]}), "exp_0049/results_d.json")
    L.exact("Table 1 footnote b: 8 qualified", 8, x2["n_qualified"], "exp_0049/results_d.json:x2.n_qualified")
    L.text("main", r"Null & resample / size $\sim U[0.5,1]$, same rule & --- & 5 / 3\\", "Table 1 Null row")
    L.text("main", r"Grid$^a$ & $y=\mathbb 1[\text{aspect}+\alpha(\text{colour}-0.5)>t]$ & aspect / colour & 8 / 24\\", "Table 1 Grid row")
    L.text("main", r"Offsets$^b$ & grid, $\alpha=0$, $t\in\{.53,.56,.6\}$, seeds 2--4 & aspect & 9\\", "Table 1 Offsets row")


T2_ROWS = {  # tex row -> (condition key, printed (recovery, found) per method)
    "Shortcut": ("T1", {"dce": ("0.38", "0.84"), "cfdiff": ("0.54", "0.93"), "grad": ("0.91", "1.0"), "imp": ("0.00", None), "seg": ("0.02", "0.08"), "noprox": ("0.35", "0.63"), "cfnew": ("0.07", "0.93")}),
    "Subgroup": ("T3", {"dce": ("0.53", "0.81"), "cfdiff": ("0.32", "0.99"), "grad": ("0.62", "1.0"), "imp": ("0.00", None), "seg": ("0.06", "0.10"), "noprox": ("0.29", "0.58"), "cfnew": ("0.30", "1.0")}),
    "Threshold": ("T4", {"dce": ("0.47", "0.47"), "cfdiff": ("0.76", "0.90"), "grad": ("0.07", "1.0"), "imp": ("1.00", None), "seg": ("0.07", "0.08"), "noprox": ("0.31", "0.40"), "cfnew": ("0.90", "0.90")}),
}
T2_TEX = {
    "Shortcut": r"Shortcut & 0.38 (0.84) & 0.54 (0.93) & \textbf{0.91} (1.0) & 0.00 & 0.02 (0.08) & 0.35 (0.63) & 0.07 (0.93)\\",
    "Subgroup": r"Subgroup & 0.53 (0.81) & 0.32 (0.99) & \textbf{0.62} (1.0) & 0.00 & 0.06 (0.10) & 0.29 (0.58) & 0.30 (1.0)\\",
    "Threshold": r"Threshold & 0.47 (0.47) & 0.76 (0.90) & 0.07 (1.0) & \textbf{1.00} & 0.07 (0.08) & 0.31 (0.40) & 0.90 (0.90)\\",
    "Null": r"Null & \textbf{0.000} & 0.994 & 1.0 & 1.0 & 0.000 & 0.000 & 0.996\\",
}


def table2(A) -> None:
    res = J("exp_0017/results.json")
    t47 = J("exp_0047/results_e.json")["table2_check"]
    for row, (cond, cells) in T2_ROWS.items():
        L.text("main", T2_TEX[row], f"Table 2 row {row}")
        for key, (pr, pf_) in cells.items():
            src = f"exp_0017/results.json:by_condition.{cond}.{METH[key]}"
            L.num(f"Table 2 {row} {key} Recovery", pr, res["by_condition"][cond][METH[key]]["recovery"], src + ".recovery")
            if pf_ is not None:
                L.num(f"Table 2 {row} {key} found-rate", pf_, res["by_condition"][cond][METH[key]]["found_rate"], src + ".found_rate")
    # null row: found-rate pooled over the 8 null pairs (5 + 3)
    nulls = [p for p in res["per_pair"].values() if p["condition"] in ("N0", "N1")]
    L.exact("Table 2 Null: 8 null pairs", 8, len(nulls), "exp_0017/results.json:per_pair")
    for key, pr in (("dce", "0.000"), ("cfdiff", "0.994"), ("grad", "1.0"), ("imp", "1.0"), ("seg", "0.000"), ("noprox", "0.000"), ("cfnew", "0.996")):
        v = float(np.mean([p["methods"][METH[key]]["found_rate"] for p in nulls]))
        L.num(f"Table 2 Null {key} found-rate", pr, v, f"exp_0017/results.json:per_pair[N0,N1].methods.{METH[key]}.found_rate (mean of 8)")
    # consistency of exp_0047 table2_check (same arrays) with exp_0017 by_condition
    worst = 0.0
    for cond, d in t47.items():
        for meth, (rec, fr) in d.items():
            ref = res["by_condition"][cond][meth]
            worst = max(worst, abs(rec - ref["recovery"]), abs(fr - ref["found_rate"]))
    L.bound("Table 2 reproduced by exp_0047 table2_check (max abs diff over cells)", "0.000", worst, "exp_0047/results_e.json:table2_check vs exp_0017")
    # caption: 95% CIs within +-0.035 (bootstrap not saved; recomputed from the saved arrays)
    half = 0.0
    for key in METH:
        meth = METH[key]
        for cond in ("T1", "T3", "T4"):
            groups = [recovery_indicator(MethodOutput(a[f"{meth}__found"], a[f"{meth}__dphi"]), k_index(m["k_star"]))
                      for m, a in A if m["condition"] == cond]
            _, stat = boot_mean(groups, np.random.default_rng(0), 10000)
            half = max(half, float((np.quantile(stat, 0.975) - np.quantile(stat, 0.025)) / 2))
    L.bound("Table 2 caption: 95% CIs within +-0.035 (max half-width over 21 Recovery cells)", "0.035", half,
            "recomputed: boot_mean over saved exp_0017 arrays, B=10000, rng seed 0 (CIs are not saved per cell)",
            note="bootstrap seed differs from the original run, which saved only H2a/H3 intervals")


def sec_abstain(A, B, G, D) -> None:
    """Section 4.1, Table 3 abstention rows, abstract."""
    d49 = J("exp_0049/results_d.json")
    x5 = J("exp_0057/results_x5.json")
    h6 = J("exp_0028/h6.json")
    c2 = J("exp_0019/results.json")["C2"]["pairs"]
    res17 = J("exp_0017/results.json")
    # abstract
    L.exact("Abstract: 0 of 2 400 unfiltered queries", (0, 2400), (round(d49["x3"]["witness_share_pooled"] * d49["x3"]["n_null_queries"]), d49["x3"]["n_null_queries"]),
            "exp_0049/results_d.json:x3")
    L.exact("Abstract: 0 of 800 through the VAE decoder", (0, 800), (x5["nulls"]["n_witnesses"], x5["nulls"]["n_queries"]), "exp_0057/results_x5.json:nulls")
    # Table 3 abstention rows
    n_b = sum(p["found_rate_union"] * 300 for p in c2.values())
    L.exact("Table 3 Null B: witnesses / queries", (0, 2400), (round(n_b), 300 * len(c2)), "exp_0019/results.json:C2.pairs.*.found_rate_union x 300")
    L.exact("Table 3 Null B: pairs 8/8", (8, 8), (sum(p["found_rate_union"] <= 0.05 for p in c2.values()), len(c2)), "exp_0019/results.json:C2")
    n_ad = sum(p["dce_union_found_rate"] * p["n_points"] for p in h6["pairs"])
    L.exact("Table 3 Adult: witnesses / queries", (0, 1500), (round(n_ad), sum(p["n_points"] for p in h6["pairs"])), "exp_0028/h6.json:pairs")
    L.exact("Table 3 Adult: pairs 5/5", (5, 5), (sum(p["dce_union_found_rate"] <= 0.05 for p in h6["pairs"]), len(h6["pairs"])), "exp_0028/h6.json")
    L.exact("Table 3 Null D: 0/2 400", (0, 2400), (round(d49["x3"]["witness_share_pooled"] * 2400), d49["x3"]["n_null_queries"]), "exp_0049/results_d.json:x3")
    L.exact("Table 3 Null D: pairs 8/8", (8, 8), (sum(p["witness_share_N_or_O"] <= 0.05 for p in d49["x3"]["pairs"]), len(d49["x3"]["pairs"])), "exp_0049/results_d.json:x3.pairs")
    L.exact("Table 3 Null F (VAE): 0/800", (0, 800), (x5["nulls"]["n_witnesses"], x5["nulls"]["n_queries"]), "exp_0057/results_x5.json:nulls")
    L.exact("Table 3 Null F: pairs 8/8", (8, 8), (sum(p["witness_share_N_or_O"] <= 0.05 for p in x5["nulls"]["pairs"]), len(x5["nulls"]["pairs"])), "exp_0057/results_x5.json:nulls.pairs")
    x4 = J("exp_0050/results_x4.json")
    n_res = sum(p["witness_share"] * 100 for p in x4["nulls"])
    L.exact("Table 3 ResNet-8 D: witnesses / queries", (180, 200), (round(n_res), 100 * len(x4["nulls"])), "exp_0050/results_x4.json:nulls x n_points 100")
    L.exact("Table 3 ResNet-8 D: pairs 0/2", (0, 2), (sum(p["witness_share"] <= 0.05 for p in x4["nulls"]), 2), "exp_0050/results_x4.json:nulls")
    x4e = J("exp_0054/test/results_x4_kstar.json")
    n_e = sum(p["witness_share"] * 100 for p in x4e["nulls"])
    L.exact("Table 3 ResNet-8 E: 0/200", (0, 200), (round(n_e), 100 * len(x4e["nulls"])), "exp_0054/test/results_x4_kstar.json:nulls x n_points 100")
    L.exact("Table 3 ResNet-8 E: pairs 2/2", (2, 2), (sum(p["witness_share"] <= 0.05 for p in x4e["nulls"]), 2), "exp_0054/test/results_x4_kstar.json")
    L.exact("Table 3 ResNet-8 E: kappa* = 2.5", 2.5, x4e["kappa"], "exp_0054/test/results_x4_kstar.json:kappa")
    # 4.1 text: CF-diff on nulls
    cf_a = [p["methods"]["b1_det"]["found_rate"] for p in res17["per_pair"].values() if p["condition"] in ("N0", "N1")]
    cf_ad = [p["b1_det_found_rate"] for p in h6["pairs"]]
    L.num("S4.1 CF-diff on the 8 nulls and 5 Adult nulls: low end (%)", "97.7", min(cf_a + cf_ad), "exp_0017 per_pair b1_det found_rate; exp_0028/h6.json b1_det_found_rate", scale=100)
    L.num("S4.1 CF-diff on the 8 nulls and 5 Adult nulls: high end (%)", "100", max(cf_a + cf_ad), "same", scale=100)
    cfd = [p["cfdiff_attributes"] for p in d49["x3"]["pairs"]]
    L.num("S4.1 split D CF-diff low end (%)", "98.3", min(cfd), "exp_0049/results_d.json:x3.pairs.cfdiff_attributes", scale=100)
    L.num("S4.1 split D CF-diff high end (%)", "99.7", max(cfd), "same", scale=100)
    L.exact("S4.1 split D: 8 new null pairs", 8, len(d49["x3"]["pairs"]), "exp_0049/results_d.json:x3.pairs")
    cfv = [p["cfdiff_attributes"] for p in x5["nulls"]["pairs"]]
    L.num("S4.1 VAE CF-diff low end (%)", "85", min(cfv), "exp_0057/results_x5.json:nulls.pairs.cfdiff_attributes", scale=100)
    L.num("S4.1 VAE CF-diff high end (%)", "94", max(cfv), "same", scale=100)
    L.exact("S4.1 VAE: 100 queries per pair", 100, x5["n_points"], "exp_0057/results_x5.json:n_points")
    L.exact("S4.1 VAE: 6 of 8 offsets with witnesses for at least half of the lowered-class queries", (6, 8),
            (x5["offsets"]["n_pairs_dce_lowered_ge_05"], len(x5["offsets"]["pairs"])), "exp_0057/results_x5.json:offsets")
    L.exact("S4.1 VAE: CF-diff flags none at tau_0 in all 8 pairs", True, all(p["cfdiff_lowered_fpr0"] == 0 for p in x5["offsets"]["pairs"]), "exp_0057/results_x5.json:offsets.pairs.cfdiff_lowered_fpr0")
    e4 = J("exp_0047/results_e.json")["e4"]
    L.num("S4.1 kappa=0: witnesses on 83% of split-A null queries", "83", e4["null"]["splitA_kappa0"], "exp_0047/results_e.json:e4.null.splitA_kappa0", scale=100)
    # dose-response from saved exp_0049 arrays: lowered-class (y=1) queries of the 9 offset pairs vs s_Mr (max |h| in the ball)
    ys, sm, fd = [], [], []
    for m, a in D:
        if m.get("kind") == "offset":
            low = a["y"] == 1
            ys.append(low), sm.append(a["s_Mr"][low]), fd.append(a["dce_N__found"][low])
    sm, fd = np.concatenate(sm), np.concatenate(fd)
    lo = fd[sm < 0.8].mean()
    hi = fd[sm >= 1.5].mean()
    L.num("S4.1 witness rate at smallest max|h| in the r-ball (descriptive): 0.02", "0.02", lo,
          "exp_0049/pairs/pair_10{0..8}.npz: dce_N__found on y==1 with s_Mr<0.8 (n=%d of %d)" % ((sm < 0.8).sum(), sm.size),
          note="not stored in a results json; recomputed from the saved per-query arrays (DECISIONS D-053 gives the same bins)")
    L.num("S4.1 witness rate at largest max|h| in the r-ball (descriptive): 1.00", "1.00", hi, "same with s_Mr>=1.5 (n=%d)" % (sm >= 1.5).sum())
    # sensitivity paragraph
    m46 = J("exp_0046/results_m.json")["by_b"]
    L.num("S4.1 witnesses on every lowered-class query from |b|=1.1 (rate at 1.1)", "1.0", m46["1.1"]["dce_lowered"], "exp_0046/results_m.json:by_b.1.1.dce_lowered")
    L.num("S4.1 ... none below (rate at 0.9)", "0.0", m46["0.9"]["dce_lowered"], "exp_0046/results_m.json:by_b.0.9.dce_lowered")
    cf_max = max(v["cfdiff_lowered_fpr0"] for v in m46.values())
    L.num("S4.1 CF-diff flags <= 0.002 up to |b|=2", "0.002", cf_max, "exp_0046/results_m.json:by_b.*.cfdiff_lowered_fpr0 (max)")
    L.exact("S4.1 margin score flags all from |b|=0.9 at 5% (rates at 0.9,1.1,1.5,2.0)", [1.0] * 4, [m46[b]["s_M_lowered_fpr0.05"] for b in ("0.9", "1.1", "1.5", "2.0")], "exp_0046/results_m.json:by_b.*.s_M_lowered_fpr0.05")
    L.exact("S4.1 ... and not at 0.5", 0.0, m46["0.5"]["s_M_lowered_fpr0.05"], "exp_0046/results_m.json:by_b.0.5")
    L.exact("S4.1 CF-diff cut = 2 (largest |b| in grid)", 2.0, max(float(b) for b in m46), "exp_0046/results_m.json:by_b keys")
    q = [p for p in d49["x2"]["pairs"] if p["qualified"]]
    L.exact("S4.1 8 registered fine-tuned offsets", 8, len(q), "exp_0049/results_d.json:x2.pairs[qualified]")
    L.num("S4.1 witnesses exist for 0.17 ... of lowered-class queries (min)", "0.17", min(p["dce_lowered"] for p in q), "exp_0049/results_d.json:x2.pairs.dce_lowered")
    L.num("S4.1 ... to 1.00 (max)", "1.00", max(p["dce_lowered"] for p in q), "same")
    L.num("S4.1 margin score flags 0.00 (min, tau_0)", "0.00", min(p["s_M_lowered_fpr0"] for p in q), "exp_0049/results_d.json:x2.pairs.s_M_lowered_fpr0")
    L.num("S4.1 margin score flags ... 0.54 (max, tau_0)", "0.54", max(p["s_M_lowered_fpr0"] for p in q), "same")
    L.exact("S4.1 CF-diff none at tau_0", 0.0, max(p["cfdiff_lowered_fpr0"] for p in q), "exp_0049/results_d.json:x2.pairs.cfdiff_lowered_fpr0")
    L.num("S4.1 CF-diff 0.03 at 5% (min)", "0.03", min(p["cfdiff_lowered_fpr0.05"] for p in q), "exp_0049/results_d.json:x2.pairs.cfdiff_lowered_fpr0.05")
    L.num("S4.1 CF-diff 0.18 at 5% (max)", "0.18", max(p["cfdiff_lowered_fpr0.05"] for p in q), "same")
    L.exact("S4.1 witness exceeded margin score by 0.2 in 7 of 8 pairs", (7, 8), (sum(p["dce_lowered"] - p["s_M_lowered_fpr0"] >= 0.2 - 1e-12 for p in q), len(q)),
            "exp_0049/results_d.json:x2.pairs (dce_lowered - s_M_lowered_fpr0 >= 0.2)")


def sec_offsets_rotations(A, B, G) -> None:
    g5 = J("exp_0026/results.json")["H5"]
    d49 = J("exp_0049/results_d.json")["x2"]
    q = [p for p in d49["pairs"] if p["qualified"]]
    L.num("S4.2 pure offsets of the grid: Grad-diff Recovery", "0.081", g5["H5a"]["estimate"], "exp_0026/results.json:H5.H5a.estimate")
    ext = J("exp_0058/results_s.json")["s1"]["rows"]
    row = {r["id"]: r for r in ext}
    L.exact("S4.2 ... below 0.2 in each of the 4 pairs", (4, 4), (row["gd_lt02_grid"]["sign_test"]["k"], row["gd_lt02_grid"]["sign_test"]["n"]), "exp_0058/results_s.json:gd_lt02_grid.sign_test")
    L.num("S4.2 split D offsets Grad-diff low end", "0.03", min(p["graddiff_recovery"] for p in q), "exp_0049/results_d.json:x2.pairs.graddiff_recovery")
    L.num("S4.2 split D offsets Grad-diff high end", "0.09", max(p["graddiff_recovery"] for p in q), "same")
    L.exact("S4.2 8 of 8 below 0.2", 8, d49["k_graddiff_below_02"], "exp_0049/results_d.json:x2.k_graddiff_below_02")
    L.num("S4.2 sign test p", "0.004", d49["sign_test_p"], "exp_0049/results_d.json:x2.sign_test_p")
    # Threshold B: Delta-CE 0.41 vs Grad-diff 0.10, positive for each seed
    t4 = [(m, a) for m, a in B if m["condition"] == "T4"]
    dce = float(np.mean([pair_mean(a, "dce_N", k_index(m["k_star"])) for m, a in t4]))
    gd = float(np.mean([pair_mean(a, "b3", k_index(m["k_star"])) for m, a in t4]))
    L.exact("S4.2 split B Threshold pairs", 3, len(t4), "exp_0019/pairs (condition T4)")
    L.num("S4.2 Delta-CE on Threshold, split B", "0.41", dce, "exp_0019/pairs/pair_0{6,7,8}.npz: dce_N Recovery mean")
    L.num("S4.2 Grad-diff on Threshold, split B", "0.10", gd, "exp_0019/pairs/pair_0{6,7,8}.npz: b3 Recovery mean")
    sd = J("exp_0019/results.json")["C1"]["seed_diffs"]
    L.exact("S4.2 positive for each seed", True, all(v > 0 for v in sd.values()), "exp_0019/results.json:C1.seed_diffs")
    # rotations
    L.num("S4.3 Grad-diff Recovery at alpha=0.1", "0.36", g5["H5b"]["rec_b"], "exp_0026/results.json:H5.H5b.rec_b (registered contrast pairs 28, 29: t=0.5)",
          note="equals the registered H5b contrast cell (t=0.5, pairs 28/29); the mean over all 8 alpha=0.1 pairs is %.3f (exp_0058 by_alpha)" % row["gd_rises_rotation"]["by_alpha"]["0.1"]["mean"])
    L.num("S4.3 Grad-diff Recovery at alpha=0.5", "0.93", g5["H5b"]["rec_a"], "exp_0026/results.json:H5.H5b.rec_a (registered contrast pairs 44, 45: t=0.5)",
          note="equals the registered H5b contrast cell (t=0.5, pairs 44/45); the mean over all 8 alpha=0.5 pairs is %.3f (exp_0058 by_alpha)" % row["gd_rises_rotation"]["by_alpha"]["0.5"]["mean"])
    e1a = J("exp_0047/results_e.json")["e1a_rotation"]
    L.exact("S4.3 24 pairs", 24, e1a["n_pairs"], "exp_0047/results_e.json:e1a_rotation.n_pairs")
    L.num("S4.3 Spearman rho", "0.89", e1a["spearman_rho"], "exp_0047/results_e.json:e1a_rotation.spearman_rho")
    L.num("S4.3 permutation p = 1e-4", "1e-4", e1a["perm_p_one_sided"], "exp_0047/results_e.json:e1a_rotation.perm_p_one_sided")


def sec_other_baselines(A) -> None:
    x1 = J("exp_0048/results_x1.json")
    ref_d = J("exp_0051/x1_ref_split_d.json")
    L.num("S4.4 IG-diff Shortcut", "0.93", x1["T1"]["ig"]["recovery"], "exp_0048/results_x1.json:T1.ig.recovery")
    L.num("S4.4 SHAP-diff Shortcut", "0.93", x1["T1"]["shap"]["recovery"], "exp_0048/results_x1.json:T1.shap.recovery")
    L.num("S4.4 Threshold SHAP-diff", "0.73", x1["T4"]["shap"]["recovery"], "exp_0048/results_x1.json:T4.shap.recovery")
    L.num("S4.4 Threshold IG-diff", "0.75", x1["T4"]["ig"]["recovery"], "exp_0048/results_x1.json:T4.ig.recovery")
    L.num("S4.4 Threshold Grad-diff", "0.07", J("exp_0017/results.json")["by_condition"]["T4"]["b3"]["recovery"], "exp_0017/results.json:by_condition.T4.b3.recovery")
    # nulls: aspect named by IG/SHAP/Grad-diff on the 8 split-A null queries
    Xn = [(m, a) for m, a in load_run(R / "exp_0048") if m["k_star"] is None]
    An = [(m, a) for m, a in A if m["k_star"] is None]
    L.exact("S4.4 split-A null pairs in exp_0048 and exp_0017", (8, 8), (len(Xn), len(An)), "exp_0048/pairs, exp_0017/pairs")

    def aspect_rate(runs, meth):
        out = []
        for _, a in runs:
            f = a.get(f"{meth}__found", np.ones(len(a[f"{meth}__dphi"]), bool))
            out.append(recovery_indicator(MethodOutput(np.ones_like(f, bool), a[f"{meth}__dphi"]), 0))
        return float(np.concatenate(out).mean())

    L.num("S4.4 IG-diff names aspect on % of split-A null queries", "57", aspect_rate(Xn, "ig"), "exp_0048/pairs null npz: ig__dphi, argmax==aspect, pooled", scale=100)
    L.num("S4.4 SHAP-diff names aspect on % of split-A null queries", "71", aspect_rate(Xn, "shap"), "exp_0048/pairs null npz: shap__dphi, pooled", scale=100)
    L.num("S4.4 Grad-diff names aspect on % of split-A null queries", "5", aspect_rate(An, "b3"), "exp_0017/pairs null npz: b3__dphi, pooled", scale=100)
    L.num("S4.4 IG-diff on % of split-D nulls", "42", ref_d["pairs"]["summary"]["null_ig_aspect_rate_pooled"], "exp_0051/x1_ref_split_d.json:pairs.summary.null_ig_aspect_rate_pooled", scale=100)
    # Imp-diff aspect first in all 9 change pairs
    e2 = J("exp_0047/results_e.json")["e2_baselines"]
    ch = [p for p in e2 if p["k_star"] is not None]
    L.exact("S4.4 Imp-diff on test data ranks aspect first in all 9 change pairs", (9, 9), (sum(p["argmax"] == "aspect" for p in ch), len(ch)), "exp_0047/results_e.json:e2_baselines.argmax")
    for cond in ("T1", "T3", "T4"):
        L.num(f"S4.4 Imp-diff on adaptation data recovers k* ({cond})", "1.00", x1[cond]["b4a"]["recovery"], f"exp_0048/results_x1.json:{cond}.b4a.recovery")
    # Seg-search x10
    X = [(m, a) for m, a in load_run(R / "exp_0048") if m["k_star"] is not None]
    fr = [float(a["b5x__found"].mean()) for _, a in X]
    L.exact("S4.4 Seg-search x10 over 9 change pairs", 9, len(X), "exp_0048/pairs")
    L.num("S4.4 Seg-search x10 found-rate low end", "0.26", min(fr), "exp_0048/pairs/*.npz: b5x__found mean per pair")
    L.num("S4.4 Seg-search x10 found-rate high end", "0.47", max(fr), "same")
    Aby = {m["pair_id"]: a for m, a in A}
    tot = both = near = 0
    for m, a in X:
        b = a["b5x__found"]
        ref = Aby[m["pair_id"]]
        assert np.allclose(ref["z0"], a["z0"], atol=1e-6)
        tot += int(b.sum())
        got = b & ref["dce_N__found"]
        both += int(got.sum())
        near += int((got & (ref["dce_N_distance"] <= a["b5x_distance"] + 1e-9)).sum())
    L.exact("S4.4 queries where Seg-search x10 finds a witness", 955, tot, "exp_0048/pairs b5x__found")
    L.exact("S4.4 ... Delta-CE finds one too on all of them", 955, both, "exp_0017/pairs dce_N__found & exp_0048 b5x__found")
    L.exact("S4.4 ... nearer or equal on 953", 953, near, "exp_0017/pairs dce_N_distance <= exp_0048 b5x_distance (tol 1e-9)")
    rd = Path("runs/experiments/exp_0048/README.md").read_text()
    L.exact("S4.4 Seg-search x10 prediction 'found-rate stays <= 0.2' was logged beforehand (README of exp_0048 records it)", True,
            "found-rate stays \u2264 0.2" in rd and "not met" in rd, "exp_0048/README.md (prediction text, D-052)", note="text source")


def sec_resnet() -> None:
    x4 = J("exp_0050/results_x4.json")
    x4e = J("exp_0054/test/results_x4_kstar.json")
    cal = J("exp_0054/calibration_resnet.json")
    cal0 = J("exp_0054/calibration_resnet_D054.json")
    L.exact("S4.4 ResNet split D: 100 queries per pair", 100, yaml.safe_load(Path("configs/experiments/exp_0050_run.yaml").read_text())["n_points"], "configs/experiments/exp_0050_run.yaml")
    L.num("S4.4 ResNet nulls: witnesses on 0.90 of null queries", "0.90", x4["null_witness_share_pooled"], "exp_0050/results_x4.json:null_witness_share_pooled")
    rd = (Path("runs/experiments/exp_0050/README.md")).read_text()
    m = re.search(r"max\|h\| over the r-ball\s*\nmedian ([\d.]+) \(CNN split-D nulls ([\d.]+)\)", rd)
    if m:
        L.exact("S4.4 null fine-tuning moves margins 'four times more' (ratio of median max|h| in the ball, rounded)", 4, round(float(m[1]) / float(m[2])),
                "exp_0050/README.md (median 1.93 vs CNN 0.51; text only)", note="ratio %.2f; README text, not a results json" % (float(m[1]) / float(m[2])))
    else:
        L.unsourced("S4.4 'four times more'", "four times", "README pattern not found")
    L.exact("S4.4 kappa grid 0.5..2.0 failed (share at 2.0 > 0.05) before extension", True, cal0["levels"][-1]["share"] > 0.05, "exp_0054/calibration_resnet_D054.json:levels")
    L.exact("S4.4 kappa* = 2.5", 2.5, cal["kappa_star"], "exp_0054/calibration_resnet.json:kappa_star")
    L.exact("S4.4 CNN kappa* = 0.5 (rule fixed in advance)", 0.5, J("exp_0054/calibration_cnn.json")["kappa_star"], "exp_0054/calibration_cnn.json:kappa_star")
    L.exact("S4.4 split E: no null query has a witness (0/200)", (0, 200), (round(sum(p["witness_share"] * 100 for p in x4e["nulls"])), 200), "exp_0054/test/results_x4_kstar.json:nulls")
    off = x4e["offsets"]
    L.num("S4.4 lowered-class offset witnesses, low end", "0.78", min(p["dce_lowered"] for p in off), "exp_0054/test/results_x4_kstar.json:offsets.dce_lowered")
    L.num("S4.4 ... high end", "1.00", max(p["dce_lowered"] for p in off), "same")
    L.num("S4.4 margin score <= 0.05", "0.05", max(p["s_M_lowered_fpr0"] for p in off), "exp_0054/test/results_x4_kstar.json:offsets.s_M_lowered_fpr0 (max)")
    L.num("S4.4 CF-diff <= 0.03", "0.03", max(p["cfdiff_lowered_fpr0"] for p in off), "exp_0054/test/results_x4_kstar.json:offsets.cfdiff_lowered_fpr0 (max)")
    L.num("S4.4 Recovery stays 0.36 (low end, split E)", "0.36", min(p["dce_recovery"] for p in off), "exp_0054/test/results_x4_kstar.json:offsets.dce_recovery")
    L.num("S4.4 Recovery stays 0.40 (high end, split E)", "0.40", max(p["dce_recovery"] for p in off), "same")
    L.num("S4.4 Grad-diff 0.32 (low end, split E)", "0.32", min(p["graddiff_recovery"] for p in off), "exp_0054/test/results_x4_kstar.json:offsets.graddiff_recovery")
    L.num("S4.4 Grad-diff 0.36 (high end, split E)", "0.36", max(p["graddiff_recovery"] for p in off), "same",
          note="split D (exp_0050) gives 0.32-0.36; split E, the split of the surrounding sentence, gives 0.32-0.35")
    L.num("S4.4 [split D reading of the same figures] Grad-diff high end", "0.36", max(p["graddiff_recovery"] for p in x4["offsets"]), "exp_0050/results_x4.json:offsets.graddiff_recovery")


def table3_stats() -> None:
    """Estimate / CI / p cells of Table 3 (everything not already covered by the abstention rows)."""
    g32 = J("exp_0032/results_h7.json")["H7c"]
    g33 = J("exp_0033/results_h8.json")["H8"]
    holm = J("exp_0033/results_holm.json")
    g26 = J("exp_0026/results.json")["H5"]
    c19 = J("exp_0019/results.json")
    c17 = J("exp_0017/results.json")
    d49 = J("exp_0049/results_d.json")["x2"]
    row = {r["id"]: r for r in J("exp_0058/results_s.json")["s1"]["rows"]}

    # CF-diff (no cut) flags |b|<2k
    L.num("Table 3 CF-diff no cut: estimate", "0.961", g32["estimate"], "exp_0032/results_h7.json:H7c.estimate")
    L.num("Table 3 CF-diff no cut: CI low", "0.955", g32["ci95"][0], "exp_0032/results_h7.json:H7c.ci95")
    L.num("Table 3 CF-diff no cut: CI high", "0.967", g32["ci95"][1], "exp_0032/results_h7.json:H7c.ci95")
    L.num("Table 3 CF-diff no cut: p (Holm, H7c)", "2e-4", holm["H7c"]["p_holm"], "exp_0033/results_holm.json:H7c.p_holm")
    # Grad-diff <0.2 grid
    L.num("Table 3 Grad-diff<0.2 grid: estimate", "0.081", g26["H5a"]["estimate"], "exp_0026/results.json:H5.H5a.estimate")
    L.num("Table 3 Grad-diff<0.2 grid: CI low", "0.066", g26["H5a"]["ci95"][0], "exp_0026/results.json:H5.H5a.ci95")
    L.num("Table 3 Grad-diff<0.2 grid: CI high", "0.096", g26["H5a"]["ci95"][1], "exp_0026/results.json:H5.H5a.ci95")
    L.num("Table 3 Grad-diff<0.2 grid: p (Holm, H5)", "4e-4", g26["H5a"]["p_holm"], "exp_0026/results.json:H5.H5a.p_holm")
    L.exact("Table 3 Grad-diff<0.2 grid: pairs 4/4", (4, 4), (row["gd_lt02_grid"]["sign_test"]["k"], row["gd_lt02_grid"]["sign_test"]["n"]), "exp_0058/results_s.json:gd_lt02_grid")
    # Offsets D
    L.num("Table 3 Offsets D: low end", "0.033", min(p["graddiff_recovery"] for p in d49["pairs"] if p["qualified"]), "exp_0049/results_d.json:x2.pairs")
    L.num("Table 3 Offsets D: high end", "0.093", max(p["graddiff_recovery"] for p in d49["pairs"] if p["qualified"]), "same")
    L.num("Table 3 Offsets D: sign test p", "0.004", d49["sign_test_p"], "exp_0049/results_d.json:x2.sign_test_p")
    L.exact("Table 3 Offsets D: pairs 8/8", (8, 8), (row["gd_lt02_D"]["sign_test"]["k"], row["gd_lt02_D"]["sign_test"]["n"]), "exp_0058/results_s.json:gd_lt02_D")
    # offset-free
    L.num("Table 3 Grad-diff offset-free: estimate", "-0.098", g26["H5c"]["estimate"], "exp_0026/results.json:H5.H5c.estimate")
    L.num("Table 3 Grad-diff offset-free: CI90 low", "-0.143", g26["H5c"]["ci90"][0], "exp_0026/results.json:H5.H5c.ci90")
    L.num("Table 3 Grad-diff offset-free: CI90 high", "-0.053", g26["H5c"]["ci90"][1], "exp_0026/results.json:H5.H5c.ci90")
    L.num("Table 3 Grad-diff offset-free: p (Holm)", "0.94", g26["H5c"]["p_holm"], "exp_0026/results.json:H5.H5c.p_holm")
    # Delta-CE > Grad-diff Threshold B
    L.num("Table 3 DCE>Grad Threshold B: estimate", "0.31", c19["C1"]["diff"], "exp_0019/results.json:C1.diff")
    L.num("Table 3 DCE>Grad Threshold B: CI low", "0.27", c19["C1"]["ci95"][0], "exp_0019/results.json:C1.ci95")
    L.num("Table 3 DCE>Grad Threshold B: CI high", "0.35", c19["C1"]["ci95"][1], "exp_0019/results.json:C1.ci95")
    L.num("Table 3 DCE>Grad Threshold B: p (Holm, C1-C3)", "2e-4", c19["C1"]["p_holm"], "exp_0019/results.json:C1.p_holm")
    L.exact("Table 3 DCE>Grad Threshold B: pairs 3/3", (3, 3), (row["dce_gt_gd_threshold_B"]["sign_test"]["k"], row["dce_gt_gd_threshold_B"]["sign_test"]["n"]), "exp_0058/results_s.json:dce_gt_gd_threshold_B")
    # rises with alpha
    L.num("Table 3 Grad-diff rises: estimate", "0.57", g26["H5b"]["estimate"], "exp_0026/results.json:H5.H5b.estimate")
    L.num("Table 3 Grad-diff rises: CI low", "0.53", g26["H5b"]["ci95"][0], "exp_0026/results.json:H5.H5b.ci95")
    L.num("Table 3 Grad-diff rises: CI high", "0.61", g26["H5b"]["ci95"][1], "exp_0026/results.json:H5.H5b.ci95")
    L.num("Table 3 Grad-diff rises: p (Holm, H5)", "4e-4", g26["H5b"]["p_holm"], "exp_0026/results.json:H5.H5b.p_holm")
    L.num("Table 3 Grad-diff rises: rho", "0.89", J("exp_0047/results_e.json")["e1a_rotation"]["spearman_rho"], "exp_0047/results_e.json:e1a_rotation")
    # projection
    L.num("Table 3 Proj Shortcut B: estimate", "-0.30", c19["C3"]["diff_projected"], "exp_0019/results.json:C3.diff_projected")
    L.num("Table 3 Proj grid: estimate", "-0.35", g26["H5d"]["estimate"], "exp_0026/results.json:H5.H5d.estimate")
    L.num("Table 3 Proj Shortcut B: CI90 half-width (+-0.03)", "0.03", (c19["C3"]["ci90"][1] - c19["C3"]["ci90"][0]) / 2, "exp_0019/results.json:C3.ci90")
    L.num("Table 3 Proj grid: CI90 half-width (+-0.03)", "0.03", (g26["H5d"]["ci90"][1] - g26["H5d"]["ci90"][0]) / 2, "exp_0026/results.json:H5.H5d.ci90")
    L.num("Table 3 Proj Shortcut B: p (Holm)", "1", c19["C3"]["p_holm"], "exp_0019/results.json:C3.p_holm")
    L.num("Table 3 Proj grid: p (Holm)", "1", g26["H5d"]["p_holm"], "exp_0026/results.json:H5.H5d.p_holm")
    # locality
    L.num("Table 3 Normal at z*: gap(z0,z0) = -0.19", "-0.19", g33["gap"]["z0,z0"], "exp_0033/results_h8.json:H8.gap")
    L.num("Table 3 Normal at z*: gap(z*,z0) = +0.03", "0.03", g33["gap"]["zs,z0"], "exp_0033/results_h8.json:H8.gap")
    L.num("Table 3 Normal at z*: CI90 low", "0.01", g33["gap_ci90"]["zs,z0"][0], "exp_0033/results_h8.json:H8.gap_ci90.zs,z0")
    L.num("Table 3 Normal at z*: CI90 high", "0.05", g33["gap_ci90"]["zs,z0"][1], "exp_0033/results_h8.json:H8.gap_ci90.zs,z0")
    L.num("Table 3 Normal at z*: p (Holm, H8)", "6e-4", holm["H8"]["p_holm"], "exp_0033/results_holm.json:H8.p_holm")
    L.exact("Table 3 Normal at z*: pairs 4/4", (4, 4), (row["normal_zstar"]["sign_test"]["k"], row["normal_zstar"]["sign_test"]["n"]), "exp_0058/results_s.json:normal_zstar")
    # Subgroup and tie
    h2 = c17["H2a"]
    L.num("Table 3 Subgroup DCE vs Grad: estimate", "-0.09", h2["diff"], "exp_0017/results.json:H2a.diff")
    L.num("Table 3 Subgroup: CI low", "-0.13", h2["ci95"][0], "exp_0017/results.json:H2a.ci95")
    L.num("Table 3 Subgroup: CI high", "-0.05", h2["ci95"][1], "exp_0017/results.json:H2a.ci95")
    L.exact("Table 3 Subgroup: pairs 0/3", (0, 3), (sum(v > 0 for v in h2["seed_diffs"].values()), len(h2["seed_diffs"])), "exp_0017/results.json:H2a.seed_diffs")
    ties = [abs(v["diff"]) for v in c17["P_tie"].values()]
    L.num("Table 3 Tie: |Delta| low end", "0.17", min(ties), "exp_0017/results.json:P_tie.*.diff")
    L.num("Table 3 Tie: |Delta| high end", "0.53", max(ties), "same")
    L.text("main", r"Tie ($|\Delta|\le0.1$) & Shortc., Thresh. & $|\Delta|=0.17$--$0.53$ & & & not conf. & R, pre\\", "Table 3 Tie row")
    L.exact("Table 3 caption: with <=4 pairs the exact pair-level p >= 0.0625", 0.0625, min(row[i]["sign_test"]["min_attainable_p"] for i in ("gd_lt02_grid", "normal_zstar")), "exp_0058/results_s.json min_attainable_p (4 pairs)")



MAIN_SNIPPETS = [
    # abstract and Section 3
    r"0 of 2\,400 unfiltered queries, and 0 of 800 when it searches through a learned VAE decoder",
    r"labels are $y=\mathbb 1[\text{aspect}>0.5]$ with 5\% noise", r"fine-tuned on 10\,000 samples", r"$32\times32$ ellipses",
    r"$N=300$ valid test points", r"the budget $r=0.54$, set once on validation", r"($\kappa=0.5$)",
    # Table 3 rows
    r"0/2\,400; 0/1\,500 & & 8/8; 5/5 & confirmed", r"0/2\,400 & & 8/8 & confirmed & R, reg", r"0/800 & & 8/8 & confirmed & R, reg",
    r"180/200; 0/200 & & 0/2; 2/2 & not; expl.",
    r"0.961 [0.955, 0.967] & $2{\cdot}10^{-4}\,^{\S}$ & & confirmed & D, new",
    r"0.081 [0.066, 0.096] & $4{\cdot}10^{-4}\,^{\ddagger}$ & 4/4 & confirmed$^\|$ & R, grid",
    r"0.033--0.093 & 0.004 (sign) & 8/8 & confirmed & R, reg",
    r"$-0.098$ [$-0.143$, $-0.053$]$^\dagger$ & $0.94\,^{\ddagger}$ & & not conf. & R, grid",
    r"$+0.31$ [0.27, 0.35] & $2{\cdot}10^{-4}\,^{*}$ & 3/3 & confirmed$^\|$ & D, post",
    r"$+0.57$ [0.53, 0.61] & $4{\cdot}10^{-4}\,^{\ddagger}$ & $\rho=0.89$ & confirmed & R, grid",
    r"$-0.30$; $-0.35$ (CIs $\pm0.03$)$^\dagger$ & $1\,^{*}$; $1\,^{\ddagger}$ & & not conf.",
    r"$-0.19\to+0.03$ [0.01, 0.05]$^\dagger$ & $6{\cdot}10^{-4}\,^{\S}$ & 4/4 & confirmed$^\|$ & R, new",
    r"$-0.09$ [$-0.13$, $-0.05$] & & 0/3 & rejected & R, pre",
    # Section 4
    r"97.7--100\% of points", r"again no witness on any of 2\,400 queries, CF-diff attributes a change to 98.3--99.7\%",
    r"100 queries per pair, on CPU instead of the registered GPU), no null query has a witness (0/800; CF-diff 85--94\%)",
    r"on 6 of 8 offsets witnesses exist for at least half of the lowered-class queries (CF-diff none at zero false positives)",
    r"with $\kappa=0$ witnesses appear on 83\% of the split-A null queries", r"rises from 0.02 to 1.00 with the largest $|h|$ in the $r$-ball",
    r"from $|b|=1.1$", r"CF-diff flags $\le0.002$ up to $|b|=2$", r"flags all from $|b|=0.9$ at 5\%",
    r"witnesses exist for 0.17--1.00 of lowered-class queries, the margin score flags 0.00--0.54 and CF-diff none (0.03--0.18 at 5\%)",
    r"exceeded the margin score by 0.2 in 7 of 8 pairs",
    r"their Recovery is 0.081 (below 0.2 in each of the 4 pairs), and on the 8 registered offsets of split D 0.03--0.09 (8 of 8 below 0.2; sign test $p=0.004$)",
    r"(0.41 vs.\ 0.10 on split B; positive for each seed)", r"hence 0.47 in Table~\ref{tab:results}",
    r"rises from 0.36 at $\alpha=0.1$ to 0.93 at $\alpha=0.5$ (Spearman $\rho=0.89$ over the 24 pairs, permutation $p=10^{-4}$",
    r"recover colour on Shortcut (0.93) like Grad-diff, and name aspect on Threshold (0.73--0.75 vs.\ 0.07)",
    r"IG- and SHAP-diff name aspect on 57\% and 71\% of queries (Grad-diff 5\%), and IG-diff on 42\% of the split-D nulls",
    r"ranks aspect first in all 9 change pairs", r"finds witnesses for 0.26--0.47 of queries (predicted $\le0.2$); on all 955 of them $\Delta$-CE finds one too, nearer or equal on 953",
    r"(split D, 100 queries per pair; run after its cost rule had excluded it)", r"moves its margins four times more and witnesses appear on 0.90 of null queries",
    r"gives $\kappa^\star=2.5$: on split E no null query has a witness (0/200), and witnesses exist for 0.78--1.00 of lowered-class offset queries (margin score $\le0.05$, CF-diff $\le0.03$ at zero false positives)",
    r"Recovery stays 0.36--0.40; Grad-diff is not offset-blind there (0.32--0.36)", r"(0.5 for the CNNs; 2.5 for the ResNet-8, after an extended search)",
    r"Table 2 caption: (95\% CIs within $\pm0.035$)".replace("Table 2 caption: ", ""),
]


def main_snippets() -> None:
    for i, sn in enumerate(MAIN_SNIPPETS, 1):
        L.text("main", sn, f"main.tex snippet {i}")


# ---------------------------------------------------------------------------------------------------------------------
# Supplement
def tex_cells(line: str) -> list[str]:
    line = line.strip().rstrip("\\").strip()
    return [c.strip() for c in line.split(" & ")]


def num_in(s: str) -> list[str]:
    s = s.replace("$-$", "-").replace("\\times10^{-", "e-").replace("\\times10^{", "e").replace("}$", "").replace("$", "")
    s = re.sub(r"(\d)e-(\d+)\}", r"\1e-\2", s)
    return re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s)


def supp_B() -> None:
    rows = {r["id"]: r for r in J("exp_0058/results_s.json")["s1"]["rows"]}
    order = ["abst_null_B", "abst_adult", "abst_null_D", "abst_resnet_D", "abst_resnet_E", "cfdiff_flags", "gd_lt02_grid", "gd_lt02_D",
             "gd_offset_free", "gd_rises_contrast", "gd_rises_rotation", "dce_gt_gd_threshold_A", "dce_gt_gd_threshold_B", "dce_gt_gd_threshold_AB",
             "dce_gt_gd_subgroup", "tie_shortcut_A", "tie_shortcut_B", "tie_shortcut_AB", "tie_threshold_A", "tie_threshold_B", "tie_threshold_AB",
             "proj_shortcut_B", "proj_grid", "normal_zstar"]
    txt = PAPER["supp"].read_text()
    start = txt.index(r"\endhead")
    seg = txt[start:txt.index(r"\end{longtable}")]
    lines = [ln for ln in seg.splitlines() if " & " in ln and ln.strip().endswith("\\")]
    L.exact("S-B Table S1 has 24 rows", 24, len(lines), "paper/supplement.tex longtable")
    for rid, ln in zip(order, lines):
        c = tex_cells(ln)
        r = rows[rid]
        loc = f"S-B Table S1 [{rid}]"
        src = f"exp_0058/results_s.json:s1.rows[{rid}]"
        s = r["summary"]
        L.exact(f"{loc} n", int(c[1]), s["n"], src + ".summary.n")
        lo, hi = re.match(r"(\S+) to (\S+)", c[2].replace("$-$", "-")).groups()
        L.num(f"{loc} range low", lo, s["min"], src + ".summary.min")
        L.num(f"{loc} range high", hi, s["max"], src + ".summary.max")
        mean_s = c[3].replace("$-$", "-")
        L.num(f"{loc} mean", mean_s.split(" ")[0], s["mean"], src + ".summary.mean")
        if "[--]" in mean_s:
            L.exact(f"{loc} CI absent for n<4", True, s["t_ci95"] is None, src + ".summary.t_ci95")
        else:
            a, b = re.search(r"\[(\S+), (\S+)\]", mean_s).groups()
            L.num(f"{loc} CI low", a, s["t_ci95"]["lo"], src + ".summary.t_ci95.lo")
            L.num(f"{loc} CI high", b, s["t_ci95"]["hi"], src + ".summary.t_ci95.hi")
        p = c[4]
        ns = num_in(p)
        if "perm." in p:
            sp = r["spearman"]
            L.num(f"{loc} rho", ns[0], sp["rho"], src + ".spearman.rho")
            L.num(f"{loc} perm p", ns[1], sp["perm_p_one_sided"], src + ".spearman.perm_p_one_sided")
        else:
            if p.startswith("sign"):
                st = r["sign_test"]
                L.num(f"{loc} sign p", ns[0], st["p_one_sided"], src + ".sign_test.p_one_sided")
                kk = re.search(r"\((\d+)/(\d+)\)", p)
                L.exact(f"{loc} sign k/n", (int(kk[1]), int(kk[2])), (st["k"], st["n"]), src + ".sign_test")
                if "flip" in p:
                    L.num(f"{loc} flip p", re.search(r"flip ([\d.]+)", p)[1], r["signflip"]["p"], src + ".signflip.p")
            else:
                wb = r["within_band"]
                kk = re.match(r"(\d+)/(\d+) in band", p)
                L.exact(f"{loc} in band k/n", (int(kk[1]), int(kk[2])), (wb["k"], wb["n"]), src + ".within_band")
                L.num(f"{loc} flip p", re.search(r"flip ([\d.]+)", p)[1], r["signflip"]["p"], src + ".signflip.p")
        mp = c[5]
        if "perm" in p:
            L.num(f"{loc} min p (floor 1/(n_perm+1))", num_in(mp)[0], 1 / (r["spearman"]["n_perm"] + 1), src + ".spearman.n_perm")
        else:
            key = "sign_test" if "sign_test" in r else "signflip"
            L.num(f"{loc} min p", num_in(mp)[0], r[key]["min_attainable_p"], src + f".{key}.min_attainable_p")
        ml = c[6]
        if rid in ("cfdiff_flags", "gd_lt02_D"):
            q = num_in(ml)
            co = r["mixedlm"]["coef"]["Intercept"]
            L.num(f"{loc} MixedLM intercept", q[0], co["coef"], src + ".mixedlm.coef.Intercept.coef")
            L.num(f"{loc} MixedLM SE", q[1], co["se"], src + ".mixedlm.coef.Intercept.se")
            L.num(f"{loc} MixedLM Wald p", q[2], r["mixedlm"]["wald_vs_claim"]["p_one_sided"], src + ".mixedlm.wald_vs_claim.p_one_sided")
        elif rid == "gd_rises_rotation":
            q = num_in(ml)
            co = r["mixedlm"]["coef"]["alpha_hat"]
            L.num(f"{loc} MixedLM slope", q[0], co["coef"], src + ".mixedlm.coef.alpha_hat.coef")
            L.num(f"{loc} MixedLM slope SE", q[1], co["se"], src + ".mixedlm.coef.alpha_hat.se")
            L.num(f"{loc} MixedLM slope p", q[2], co["p_two_sided_h0_zero"], src + ".mixedlm.coef.alpha_hat.p_two_sided_h0_zero")
        elif rid in ("abst_null_B", "abst_null_D"):
            L.exact(f"{loc} MixedLM not fitted (constant 0)", True, r["mixedlm"]["status"].startswith("not fitted"), src + ".mixedlm.status")
        else:
            L.exact(f"{loc} MixedLM absent (--)", "--", ml, src, note="no MixedLM saved" if "mixedlm" not in r else "saved but not printed")
    ok = all(r.get("crosscheck_all_match") in (True, None) for r in rows.values())
    L.exact("S-B exp_0058 internal crosschecks against the upstream outputs all match", True, ok, "exp_0058/results_s.json:s1.rows.*.crosscheck_all_match")
    # method text
    rot = rows["gd_rises_rotation"]
    L.num("S-B Method: Spearman rho = 0.893", "0.893", rot["spearman"]["rho"], "exp_0058/results_s.json")
    L.exact("S-B Method: 10 000 permutations", 10000, rot["spearman"]["n_perm"], "exp_0058/results_s.json:spearman.n_perm")
    L.exact("S-B Method: 7 200 points", 7200, rot["mixedlm"]["n_points"], "exp_0058/results_s.json:mixedlm.n_points")
    cf = rows["cfdiff_flags"]
    L.exact("S-B (iii): logit-offset pairs b in {-0.9,-0.5,0.5,0.9}, three seeds", ([-0.9, -0.5, 0.5, 0.9], [0, 1, 2]),
            (sorted({p["b"] for p in cf["pairs"]}), sorted({p["seed"] for p in cf["pairs"]})), "exp_0058/results_s.json:cfdiff_flags.pairs")
    nz = rows["normal_zstar"]
    T = abs(nz["gap_z0_z0_summary"]["mean"]) - 2 * abs(nz["gap_zs_z0_summary"]["mean"])
    L.num("S-B (iv): gap(z0,z0) = -0.190", "0.190", abs(nz["gap_z0_z0_summary"]["mean"]), "exp_0058/results_s.json:normal_zstar.gap_z0_z0_summary.mean")
    L.num("S-B (iv): gap(z*,z0) = 0.029", "0.029", abs(nz["gap_zs_z0_summary"]["mean"]), "exp_0058/results_s.json:normal_zstar.gap_zs_z0_summary.mean")
    L.num("S-B (iv): registered T = 0.132", "0.132", T, "|gap(z0,z0)| - 2|gap(z*,z0)| from exp_0058")
    L.num("S-B (iv): mean of T_i = 0.119", "0.119", nz["summary"]["mean"], "exp_0058/results_s.json:normal_zstar.summary.mean")
    L.num("S-B (iv): H8.T = 0.132 in exp_0033", "0.132", J("exp_0033/results_h8.json")["H8"]["T"], "exp_0033/results_h8.json:H8.T")


def supp_C() -> None:
    e = J("exp_0047/results_e.json")
    t2 = J("exp_0017/results.json")["by_condition"]
    e4 = e["e4"]
    src = "exp_0047/results_e.json"
    # budget sensitivity
    rtab = {"T1": ("0.13", "0.22", "0.37"), "T3": ("0.30", "0.40", "0.53"), "T4": ("0.33", "0.45", "0.47")}
    for cond, name in (("T1", "Shortcut"), ("T3", "Subgroup"), ("T4", "Threshold")):
        for rp, pr in zip(("0.3", "0.4", "0.54"), rtab[cond]):
            L.num(f"S-C Table r' {name} r'={rp}", pr, e4["r"][rp][cond], f"{src}:e4.r.{rp}.{cond}")
    for cond, nm, pr in (("T1", "Shortcut", "0.376"), ("T3", "Subgroup", "0.529"), ("T4", "Threshold", "0.466")):
        L.num(f"S-C text: Table 2 value {nm}", pr, t2[cond]["dce_N"]["recovery"], f"exp_0017/results.json:by_condition.{cond}.dce_N.recovery")
    for cond, nm, pr in (("T1", "Shortcut", "0.373"), ("T3", "Subgroup", "0.528"), ("T4", "Threshold", "0.466")):
        L.num(f"S-C text: r'=0.5413-rounded-to-0.54 value {nm}", pr, e4["r"]["0.54"][cond], f"{src}:e4.r.0.54.{cond}")
    # kappa-null table
    gnull = {q["pair_id"]: q for q in J("exp_0026/results.json")["descriptive"]["pairs"]}
    for pid in (20, 21):
        L.num(f"S-C text: grid null cell pair {pid} realised threshold t_hat ~ 0.53", "0.53", gnull[pid]["t_hat"], f"exp_0026/results.json:descriptive.pairs[{pid}].t_hat")
    L.exact("S-C text: kappa grids (split A {0,0.5,1}; grid {0,0.25,0.5,1})", (["0", "0.5", "1.0"], ["0", "0.25", "0.5", "1.0"]),
            (sorted(e4["splitA"], key=float), sorted(e4["grid_alpha0_offsets"], key=float)), f"{src}:e4")
    n = e4["null"]
    for k, pr in (("splitA_kappa0", "0.83"), ("splitA_kappa0.5", "0.00"), ("splitA_kappa1.0", "0.00"), ("grid_kappa0", "0.71"),
                  ("grid_kappa0.25", "0.51"), ("grid_kappa0.5", "0.22"), ("grid_kappa1.0", "0.00")):
        L.num(f"S-C Table kappa-null {k}", pr, n[k], f"{src}:e4.null.{k}")
    # kappa-rec table
    rec = {"T1": {"0": ("0.33", "0.93"), "0.5": ("0.38", "0.84"), "1.0": ("0.37", "0.71")},
           "T3": {"0": ("0.37", "0.99"), "0.5": ("0.53", "0.81"), "1.0": ("0.57", "0.70")},
           "T4": {"0": ("0.46", "0.47"), "0.5": ("0.47", "0.47"), "1.0": ("0.20", "0.27")}}
    for cond, nm in (("T1", "Shortcut"), ("T3", "Subgroup"), ("T4", "Threshold")):
        for kap, (pr, pfr) in rec[cond].items():
            L.num(f"S-C Table kappa-rec Split A {nm} kappa={kap} Recovery", pr, e4["splitA"][kap][cond]["recovery"], f"{src}:e4.splitA.{kap}.{cond}.recovery")
            L.num(f"S-C Table kappa-rec Split A {nm} kappa={kap} found-rate", pfr, e4["splitA"][kap][cond]["found"], f"{src}:e4.splitA.{kap}.{cond}.found")
    grid = {"0": ("0.47", "0.47"), "0.25": ("0.46", "0.47"), "0.5": ("0.35", "0.38"), "1.0": ("0.02", "0.04")}
    for kap, (pr, pfr) in grid.items():
        L.num(f"S-C Table kappa-rec grid alpha=0 kappa={kap} Recovery", pr, e4["grid_alpha0_offsets"][kap]["recovery"], f"{src}:e4.grid_alpha0_offsets.{kap}.recovery")
        L.num(f"S-C Table kappa-rec grid alpha=0 kappa={kap} found-rate", pfr, e4["grid_alpha0_offsets"][kap]["found"], f"{src}:e4.grid_alpha0_offsets.{kap}.found")
    # found-rate non-increasing in kappa in every group
    mono = True
    for cond in ("T1", "T3", "T4"):
        v = [e4["splitA"][k][cond]["found"] for k in ("0", "0.5", "1.0")]
        mono &= all(a >= b - 1e-12 for a, b in zip(v, v[1:]))
    v = [e4["grid_alpha0_offsets"][k]["found"] for k in ("0", "0.25", "0.5", "1.0")]
    mono &= all(a >= b - 1e-12 for a, b in zip(v, v[1:]))
    L.exact("S-C text: found-rate non-increasing in kappa in every group", True, bool(mono), f"{src}:e4")
    # rank metrics
    names = {"dce_N": 0, "b1_det": 1, "b3": 2, "b4": 3, "b5": 4, "a1": 5, "a2": 6}
    rank_tex = {
        "splitA_T1": ("Split A, Shortcut", ["0.59/0.75", "0.70/0.75", "0.95/0.98", "0.50/1.00", "0.04/0.03", "0.46/0.48", "0.48/0.81"]),
        "splitA_T3": ("Split A, Subgroup", ["0.65/0.70", "0.61/0.75", "0.78/0.84", "0.32/0.33", "0.08/0.09", "0.40/0.42", "0.60/0.73"]),
        "splitA_T4": ("Split A, Threshold", ["0.47/0.47", "0.83/0.88", "0.36/0.25", "1.00/1.00", "0.07/0.08", "0.35/0.38", "0.90/0.90"]),
        "grid_alpha0.0": ("Grid, alpha=0", ["0.30/0.30", "0.75/0.78", "0.37/0.30", None, None, None, "0.96/0.96"]),
        "grid_alpha0.1": ("Grid, alpha=0.1", ["0.15/0.13", "0.42/0.35", "0.58/0.61", None, None, None, "0.31/0.14"]),
        "grid_alpha0.25": ("Grid, alpha=0.25", ["0.31/0.39", "0.56/0.60", "0.81/0.88", None, None, None, "0.39/0.51"]),
        "grid_alpha0.5": ("Grid, alpha=0.5", ["0.53/0.72", "0.70/0.77", "0.94/0.97", None, None, None, "0.45/0.82"]),
    }
    for g, (nm, cells) in rank_tex.items():
        for (meth, i), cell in zip(names.items(), cells):
            if cell is None:
                L.exact(f"S-C Table rank {nm} {meth}: n/a (not computed)", True, meth not in e["e3_rank"][g], f"{src}:e3_rank.{g}")
                continue
            mrr, top2 = cell.split("/")
            L.num(f"S-C Table rank {nm} {meth} MRR", mrr, e["e3_rank"][g][meth]["mrr"], f"{src}:e3_rank.{g}.{meth}.mrr")
            L.num(f"S-C Table rank {nm} {meth} Top-2", top2, e["e3_rank"][g][meth]["top2"], f"{src}:e3_rank.{g}.{meth}.top2")
    order = e["e3_order"]
    L.exact("S-C text: MRR order differs from Recovery order in 6 of 7 groups", 6, sum(not v["same"] for v in order.values()), f"{src}:e3_order.*.same")
    L.exact("S-C text: same only for grid alpha=0.5", ["grid_alpha0.5"], [g for g, v in order.items() if v["same"]], f"{src}:e3_order")
    L.exact("S-C text: first two methods the same under both orders in every group", True,
            all(v["by_recovery"][:2] == v["by_mrr"][:2] for v in order.values()), f"{src}:e3_order")
    swaps = [g for g in order if g.startswith("grid") and not order[g]["same"]]
    L.exact("S-C text: on the grid Delta-CE falls one place under MRR (groups with differing order)", [1] * len(swaps),
            [order[g]["by_mrr"].index("dce_N") - order[g]["by_recovery"].index("dce_N") for g in swaps], f"{src}:e3_order")
    nm = {"dce_N": "$\\Delta$-CE", "b1_det": "CF-diff", "b3": "Grad-diff", "b4": "Imp-diff", "b5": "Seg-search", "a1": "No-prox", "a2": "CF-new"}
    printed_orders = {
        "splitA_T1": ("Grad-diff CF-diff $\\Delta$-CE No-prox CF-new Seg-search Imp-diff", "Grad-diff CF-diff $\\Delta$-CE Imp-diff CF-new No-prox Seg-search"),
        "splitA_T3": ("Grad-diff $\\Delta$-CE CF-diff CF-new No-prox Seg-search Imp-diff", "Grad-diff $\\Delta$-CE CF-diff CF-new No-prox Imp-diff Seg-search"),
        "splitA_T4": ("Imp-diff CF-new CF-diff $\\Delta$-CE No-prox Seg-search Grad-diff", "Imp-diff CF-new CF-diff $\\Delta$-CE Grad-diff No-prox Seg-search"),
        "grid_alpha0.0": ("CF-new CF-diff $\\Delta$-CE Grad-diff", "CF-new CF-diff Grad-diff $\\Delta$-CE"),
        "grid_alpha0.1": ("Grad-diff CF-diff $\\Delta$-CE CF-new", "Grad-diff CF-diff CF-new $\\Delta$-CE"),
        "grid_alpha0.25": ("Grad-diff CF-diff $\\Delta$-CE CF-new", "Grad-diff CF-diff CF-new $\\Delta$-CE"),
        "grid_alpha0.5": ("Grad-diff CF-diff $\\Delta$-CE CF-new", "Grad-diff CF-diff $\\Delta$-CE CF-new"),
    }
    for g, (by_rec, by_mrr) in printed_orders.items():
        L.exact(f"S-C Table order {g} by Recovery", by_rec.split(" "), [nm[k] for k in order[g]["by_recovery"]], f"{src}:e3_order.{g}.by_recovery")
        L.exact(f"S-C Table order {g} by MRR", by_mrr.split(" "), [nm[k] for k in order[g]["by_mrr"]], f"{src}:e3_order.{g}.by_mrr")
    sr = e["e3_set_recovery_alpha_gt0"]
    for meth, pr in (("a2", "0.49"), ("b1_det", "0.39"), ("dce_N", "0.37"), ("b3", "0.11")):
        L.num(f"S-C set-recovery {meth}", pr, sr[meth], f"{src}:e3_set_recovery_alpha_gt0.{meth}")
    # r' text: 0.54 column is the rounded budget, r is 0.5413
    tier = yaml.safe_load(Path("configs/tiers/tier0.yaml").read_text())
    L.num("S-C text: main-paper budget r = 0.5413", "0.5413", tier["r"], "configs/tiers/tier0.yaml:r")


def supp_E() -> None:
    x5 = J("exp_0057/results_x5.json")
    gate = J("exp_0056/vae_gate.json")
    cal = J("exp_0056/r_calibration.json")
    cfg = yaml.safe_load(Path("configs/experiments/exp_0057.yaml").read_text())
    d49 = J("exp_0049/results_d.json")["x2"]
    v = cfg["vae"]
    for label, printed, got in (("latent dimension 16", 16, v["latent"]), ("beta = 1", 1.0, v["beta"]), ("beta fallback 0.5", 0.5, v["fallback_beta"]),
                                ("30 epochs", 30, v["epochs"]), ("Adam 1e-3", 1e-3, v["lr"]), ("batch 256", 256, v["batch_size"]),
                                ("seed 11000", 11000, v["seed"]), ("50 000 images", 50000, v["n_images"]),
                                ("gate accuracy >= 0.90", 0.9, cfg["gate"]["min_acc"]), ("mu = 0.01", 0.01, cfg["mu"]),
                                ("kappa = 0.5 (tier0)", 0.5, yaml.safe_load(Path("configs/tiers/tier0.yaml").read_text())["kappa"]),
                                ("split F data seed 10500 + p", 10500, cfg["data_seed_base"]), ("100 valid queries per pair", 100, cfg["n_points"])):
        L.exact(f"S-E fixed element: {label}", printed, got, "configs/experiments/exp_0057.yaml")
    L.exact("S-E: 30 training epochs recorded", 30, len(gate["attempts"][0]["loss"]), "exp_0056/vae_gate.json:attempts[0].loss")
    L.exact("S-E: VAE passed its gate at beta = 1 (no retrain)", (True, 1), (gate["attempts"][0]["gate_passed"], len(gate["attempts"])), "exp_0056/vae_gate.json")
    acc = list(gate["attempts"][0]["f_old_acc_on_recon"].values())
    L.num("S-E: accuracy of f_old on reconstructions, low end", "0.974", min(acc), "exp_0056/vae_gate.json:attempts[0].f_old_acc_on_recon")
    L.num("S-E: accuracy ... high end", "0.985", max(acc), "same")
    L.num("S-E: reconstruction MSE 2.3e-4", "2.3e-4", gate["attempts"][0]["recon_mse_val"], "exp_0056/vae_gate.json:attempts[0].recon_mse_val")
    L.num("S-E: r_VAE = 1.641", "1.641", cal["r_vae"], "exp_0056/r_calibration.json:r_vae")
    L.num("S-E: median distance 0.820", "0.820", cal["median_distance"], "exp_0056/r_calibration.json:median_distance")
    L.exact("S-E: 200/200 counterfactuals found on validation", (200, 1.0), (cal["n"], cal["found_share"]), "exp_0056/r_calibration.json")
    L.exact("S-E: r_VAE locked in the X5 config = result", True, abs(cfg["r_vae"] - x5["r_vae"]) < 1e-12 and abs(cfg["r_vae"] - cal["r_vae"]) < 1e-12, "configs/experiments/exp_0057.yaml, exp_0057 results, exp_0056")
    # primary readout
    L.exact("S-E: 0/800 and 0 in each of the 8 null pairs", (0, 800, 8), (x5["nulls"]["n_witnesses"], x5["nulls"]["n_queries"], sum(p["n_witnesses"] == 0 for p in x5["nulls"]["pairs"])), "exp_0057/results_x5.json:nulls")
    cfv = [p["cfdiff_attributes"] for p in x5["nulls"]["pairs"]]
    L.num("S-E: CF-diff attributes a change to 85% (min)", "85", min(cfv), "exp_0057/results_x5.json:nulls.pairs.cfdiff_attributes", scale=100)
    L.num("S-E: ... to 94% (max)", "94", max(cfv), "same", scale=100)
    t = x5["nulls"]["tau"]
    for lab, key, sub, o, pr in (("s_M tau_0", "s_M", "fpr0", None, "1.136"), ("s_M tau_5%", "s_M", "fpr0.05", None, "0.624"),
                                 ("s_Mr tau_0", "s_Mr", "fpr0", None, "1.384"), ("s_Mr tau_5%", "s_Mr", "fpr0.05", None, "0.972"),
                                 ("CF-diff tau_0", "cfdiff", "fpr0", None, "1.950"), ("CF-diff tau_5%", "cfdiff", "fpr0.05", None, "0.246")):
        L.num(f"S-E thresholds {lab}", pr, t[key][sub], f"exp_0057/results_x5.json:nulls.tau.{key}.{sub}")
    pairs = x5["offsets"]["pairs"]
    L.exact("S-E: Delta-CE >= 0.5 of lowered-class queries in 6 of 8 pairs (prediction >= 5/8)", (6, 8), (sum(p["dce_lowered"] >= 0.5 for p in pairs), len(pairs)), "exp_0057/results_x5.json:offsets.pairs.dce_lowered")
    L.exact("S-E: CF-diff flags none at tau_0 in any pair", 0.0, max(p["cfdiff_lowered_fpr0"] for p in pairs), "exp_0057/results_x5.json:offsets.pairs")
    xids = {p["pair_id"] for p in pairs}
    sd = {p["pair_id"]: p["dce_lowered"] for p in d49["pairs"] if p["pair_id"] in xids}
    L.num("S-E: split-D factor-space rate of pair 100 (0.17)", "0.17", sd[100], "exp_0049/results_d.json:x2.pairs")
    L.num("S-E: split-D factor-space rate of pair 101 (0.20)", "0.20", sd[101], "exp_0049/results_d.json:x2.pairs")
    L.exact("S-E: pairs 100, 101 are the two lowest in X5 and on split D", True,
            sorted(sorted(sd, key=sd.get)[:2]) == [100, 101] and sorted(sorted((p["pair_id"] for p in pairs), key=lambda i: {p["pair_id"]: p["dce_lowered"] for p in pairs}[i])[:2]) == [100, 101],
            "exp_0049/results_d.json, exp_0057/results_x5.json")
    L.num("S-E: 91% of offset witnesses on the manifold", "91", x5["offsets"]["on_manifold_share"], "exp_0057/results_x5.json:offsets.on_manifold_share", scale=100)
    L.exact("S-E: none (witnesses) on nulls", 0, x5["nulls"]["n_witnesses"], "exp_0057/results_x5.json:nulls.n_witnesses")
    L.num("S-E: Recovery by inverse rendering, Delta-CE low end 0.02", "0.02", min(p["dce_recovery_fit"] for p in pairs), "exp_0057/results_x5.json:offsets.pairs.dce_recovery_fit")
    L.num("S-E: ... high end 0.46", "0.46", max(p["dce_recovery_fit"] for p in pairs), "same")
    L.num("S-E: CF-diff Recovery low end 0.55", "0.55", min(p["cfdiff_recovery_fit"] for p in pairs), "exp_0057/results_x5.json:offsets.pairs.cfdiff_recovery_fit")
    L.num("S-E: ... high end 0.82", "0.82", max(p["cfdiff_recovery_fit"] for p in pairs), "same")
    # Table S-X5
    tab = {100: ("60", "0.12", "0.00", "0.03", "0.00", "0.00", "0.12"), 101: ("54", "0.13", "0.00", "0.09", "0.00", "0.00", "0.06"),
           102: ("48", "0.85", "0.02", "0.21", "0.06", "0.00", "0.25"), 104: ("56", "0.66", "0.00", "0.23", "0.11", "0.00", "0.21"),
           105: ("46", "0.96", "0.00", "0.20", "0.17", "0.00", "0.35"), 106: ("48", "0.98", "0.48", "0.98", "0.77", "0.00", "0.52"),
           107: ("53", "0.91", "0.36", "0.89", "0.66", "0.00", "0.34"), 108: ("43", "0.95", "0.58", "0.93", "0.91", "0.00", "0.84")}
    pt = {(p["pair_id"]): p for p in pairs}
    cols = (("n_lowered", None), ("dce_lowered", 1), ("s_M_lowered_fpr0", 1), ("s_M_lowered_fpr0.05", 1), ("s_Mr_lowered_fpr0", 1), ("cfdiff_lowered_fpr0", 1), ("cfdiff_lowered_fpr0.05", 1))
    hdr = ("n", "Delta-CE", "s_M tau_0", "s_M tau_5%", "s_Mr tau_0", "CF-diff tau_0", "CF-diff tau_5%")
    L.exact("S-E Table X5 has the 8 pairs 100,101,102,104,105,106,107,108", sorted(tab), sorted(pt), "exp_0057/results_x5.json:offsets.pairs")
    for pid, cells in tab.items():
        for (key, _), pr, h in zip(cols, cells, hdr):
            if key == "n_lowered":
                L.exact(f"S-E Table X5 pair {pid} n", int(pr), pt[pid][key], f"exp_0057/results_x5.json:offsets.pairs[{pid}].{key}")
            else:
                L.num(f"S-E Table X5 pair {pid} {h}", pr, pt[pid][key], f"exp_0057/results_x5.json:offsets.pairs[{pid}].{key}")
    L.exact("S-E Table X5: pair labels (t, seed)", True,
            all((pt[i]["t"], pt[i]["seed"]) == tv for i, tv in {100: (0.53, 2), 101: (0.56, 2), 102: (0.6, 2), 104: (0.56, 3), 105: (0.6, 3), 106: (0.53, 4), 107: (0.56, 4), 108: (0.6, 4)}.items()),
            "exp_0057/results_x5.json:offsets.pairs.{t,seed}")
    q = [s for s in Path("runs/experiments").glob("exp_0057_kaggle_k*") if s.is_dir()]
    L.exact("S-E: four CPU kernels", 4, len(q), "runs/experiments/exp_0057_kaggle_k*")
    dec = Path("research/DECISIONS.md").read_text()
    m = re.search(r"200 queries .{0,3} (\d+) h, 100 queries .{0,3} (\d+) h", dec)
    cpu = re.search(r"≈ (\d+) CPU-s per query", dec)
    if m and cpu:
        s = float(cpu[1])
        L.num("S-E: 200 queries per pair ~ 76 CPU-h", "76", s * 200 * 16 / 3600, "research/DECISIONS.md D-056 feasibility (86 CPU-s per query x 3 200 queries)", note="timing is recorded in DECISIONS text (smoke_0034), not in a results json; DECISIONS states 76 h")
        L.num("S-E: 100 queries per pair ~ 38 CPU-h", "38", s * 100 * 16 / 3600, "same (1 600 queries)")
    else:
        L.unsourced("S-E: ~76 / ~38 CPU-h", "76; 38", "pattern not found in DECISIONS.md")
    L.exact("S-E: registration https://osf.io/fr7jv is recorded in the decision log", True, "osf.io/fr7jv" in dec, "research/DECISIONS.md D-056")
    L.exact("S-E: latent-space test 24 CPU-hour feasibility threshold", True, "24 kaggle-cpu hours" in dec or "24 h" in dec, "research/DECISIONS.md D-056")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    A, B, G = load_run(R / "exp_0017"), load_run(R / "exp_0019"), load_run(R / "exp_0026")
    D = load_run(R / "exp_0049")
    sec_params(A, B, G, D)
    table1(A)
    table2(A)
    sec_abstain(A, B, G, D)
    sec_offsets_rotations(A, B, G)
    sec_other_baselines(A)
    sec_resnet()
    table3_stats()
    main_snippets()
    supp_B()
    supp_C()
    supp_E()
    rows = L.rows
    cnt = {s: sum(r["status"] == s for r in rows) for s in ("OK", "MISMATCH", "UNSOURCED")}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results_numcheck.json").write_text(json.dumps({"label": "D-056 W3 number check", "counts": cnt, "n_claims": len(rows),
                                                                 "claims": rows, "git": git_state()}, indent=1, default=str, ensure_ascii=False))
    bad = [r for r in rows if r["status"] != "OK"]
    cav = [r for r in rows if r["status"] == "OK" and r["note"] and ("mean over all" in r["note"] or "split D" in r["note"])]
    lines = ["# exp_0059 - number check of paper/main.tex and paper/supplement.tex (D-056 W3)", "",
             ("Script `src/dce/cli/number_check.py` (`uv run python -m dce.cli.number_check --out runs/experiments/exp_0059`). "
              "Reads saved outputs only; nothing is re-run. Per claim: location, value as printed, computed value, source, status "
              "in `results_numcheck.json`. OK = equal to the source after the printed rounding."), "",
             f"Claims checked: {len(rows)}. **OK {cnt['OK']}, MISMATCH {cnt['MISMATCH']}, UNSOURCED {cnt['UNSOURCED']}.**", ""]
    for title, sel in (("MISMATCH and UNSOURCED", bad), ("OK with a caveat in the note", cav)):
        lines += [f"## {title}", ""]
        if not sel:
            lines += ["none", ""]
            continue
        lines += ["| # | location | printed | computed | status | source | note |", "|---|---|---|---|---|---|---|"]
        for r in sel:
            c = r["computed"]
            c = f"{c:.6g}" if isinstance(c, float) else str(c)
            lines.append(f"| {r['id']} | {r['location']} | {r['printed']} | {c} | {r['status']} | {r['source']} | {r['note']} |")
        lines.append("")
    lines += ["## Scope", "",
              ("Covered: abstract; Sections 3-5 text numbers; Tables 1-3 (every estimate, CI, p, pairs cell); the parameters of Sections 3 and D; "
               "supplement Section B (Table S1 parsed from the tex and compared row by row with exp_0058), Section C (all tables and text), "
               "Section E (fixed elements, results, Table X5). Sources: exp_0017, 0019, 0026, 0028, 0032, 0033, 0046-0051, 0054, 0056, 0057, 0058."), "",
              "Not machine-checkable from saved outputs: proofs and prose claims without a number; registered predictions are checked only where a run README records them."]
    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(cnt))
    for r in bad:
        print(f"{r['status']}: [{r['id']}] {r['location']} | printed {r['printed']} | computed {r['computed']} | {r['source']} | {r['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
