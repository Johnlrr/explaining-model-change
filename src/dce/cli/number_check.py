"""Number check (D-056 W3, updated after the W6 rewrite and X6, D-057): every reported number of paper/main.tex and the
numeric claims of paper/supplement.tex (Sections B, C, E, X6; the parameters of D; the split-G row of the protocol log)
against the saved run outputs.

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
    T1_ROWS = {
        "Shortcut": r"Shortcut & spurious & colour correlated with label (0.99) & colour & 3\\",
        "Subgroup": r"Subgroup & local concept & labels flipped for size $>0.75$ & size & 3\\",
        "Threshold": r"Threshold & concept & $y=\mathbbm 1[\text{aspect}>0.6]$ & aspect & 3\\",
        "Null": r"Null & none / covariate & resample / size $\sim U[0.5,1]$, same rule & --- & 5 / 3\\",
        "Grid": r"Grid$^a$ & concept & $y=\mathbbm 1[\text{aspect}+\alpha(\text{colour}-0.5)>t]$ & aspect / colour & 8 / 24\\",
        "Offsets": r"Offsets$^b$ & concept & grid, $\alpha=0$, $t\in\{.53,.56,.6\}$, seeds 2--4 & aspect & 9\\",
        "Both": r"Both$^c$ & spurious + local & Shortcut and Subgroup at once & colour, size & 8\\",
    }
    for nm, row in T1_ROWS.items():
        L.text("main", row, f"Table 1 {nm} row")
    L.text("main", r"$^a$$\alpha\in\{0,.1,.25,.5\}\times t\in\{.5,.53,.56,.6\}$, 2 seeds; 4 offsets tested. $^b$8 qualified. $^{b,c}$Registered.",
           "Table 1 caption footnotes")
    # Both: split G, 8 pairs (X6)
    r6 = J("exp_0060/results_x6.json")
    L.exact("Table 1 Both pairs (split G)", 8, len(r6["pairs"]), "exp_0060/results_x6.json:pairs")
    L.exact("Table 1 Both k* = colour, size", [["colour", "size"]] * 8,
            [json.loads((R / f"exp_0060/pairs/pair_{p['pair_id']}.json").read_text())["k_star"] for p in r6["pairs"]], "exp_0060/pairs/pair_*.json:k_star")
    # four shift types (Contributions (ii)): distinct entries of the Shift column of Table 1 ("none" is not a type)
    seg = PAPER["main"].read_text().split("\\midrule")[1].split("\\bottomrule")[0]
    types = set()
    for ln in seg.strip().splitlines():
        cell = ln.split(" & ")[1]
        for t in re.split(r" / | \+ ", cell):
            t = t.strip()
            types.add("local concept" if t == "local" else t)
    types.discard("none")
    L.exact("Contributions (ii): four shift types (distinct entries of the Shift column of Table 1, 'none' excluded)", 4, len(types),
            "paper/main.tex Table 1 Shift column (parsed): " + ", ".join(sorted(types)))
    L.text("main", "four shift types", "Contributions (ii) wording")
    L.exact("Contributions (ii): a 4x4 grid (4 alpha values x 4 t values)", (4, 4), (len({p["alpha"] for p in g}), len({p["t"] for p in g})), "exp_0026/results.json:descriptive.pairs")
    L.text("main", r"a $4\times4$ grid over rotation and threshold", "Contributions (ii) grid wording")


T2_ROWS = {  # tex row -> (condition key, printed (recovery, found) per method)
    "Shortcut": ("T1", {"dce": ("0.38", "0.84"), "cfdiff": ("0.54", "0.93"), "grad": ("0.91", "1.0"), "imp": ("0.00", None), "seg": ("0.02", "0.08"), "noprox": ("0.35", "0.63"), "cfnew": ("0.07", "0.93")}),
    "Subgroup": ("T3", {"dce": ("0.53", "0.81"), "cfdiff": ("0.32", "0.99"), "grad": ("0.62", "1.0"), "imp": ("0.00", None), "seg": ("0.06", "0.10"), "noprox": ("0.29", "0.58"), "cfnew": ("0.30", "1.0")}),
    "Threshold": ("T4", {"dce": ("0.47", "0.47"), "cfdiff": ("0.76", "0.90"), "grad": ("0.07", "1.0"), "imp": ("1.00", None), "seg": ("0.07", "0.08"), "noprox": ("0.31", "0.40"), "cfnew": ("0.90", "0.90")}),
}
T2_TEX = {
    "Shortcut": r"Shortcut $\uparrow$ & 0.38 (0.84) & 0.54 (0.93) & \textbf{0.91} (1.0) & 0.00 & 0.02 (0.08) & 0.35 (0.63) & 0.07 (0.93)\\",
    "Subgroup": r"Subgroup $\uparrow$ & 0.53 (0.81) & 0.32 (0.99) & \textbf{0.62} (1.0) & 0.00 & 0.06 (0.10) & 0.29 (0.58) & 0.30 (1.0)\\",
    "Threshold": r"Threshold $\uparrow$ & 0.47 (0.47) & 0.76 (0.90) & 0.07 (1.0) & \textbf{1.00}$^a$ & 0.07 (0.08) & 0.31 (0.40) & 0.90 (0.90)\\",
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
    L.text("main", r"also on unscreened nulls and through a learned decoder", "Contribution (iii) nulls wording")
    L.exact("Abstract: 0 of 2 400 unfiltered queries", (0, 2400), (round(d49["x3"]["witness_share_pooled"] * d49["x3"]["n_null_queries"]), d49["x3"]["n_null_queries"]),
            "exp_0049/results_d.json:x3")
    L.exact("Abstract: 0 of 800 when searching through a learned autoencoder (VAE decoder)", (0, 800), (x5["nulls"]["n_witnesses"], x5["nulls"]["n_queries"]), "exp_0057/results_x5.json:nulls")
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
    cfd = [p["cfdiff_attributes"] for p in d49["x3"]["pairs"]]
    # W6 wording: one range for the three groups named in the sentence (8 null pairs, five Adult nulls, 8 registered split-D nulls)
    L.text("main", r"No null query has a witness: on the 8 null pairs of Table~\ref{tab:results}, five null updates of an MLP on Adult, and 8 registered unscreened null pairs, admitted without the qualification check", "S4.1 nulls sentence (three groups)")
    L.text("main", r"CF-diff reports a change for 97.7--100\% of points", "S4.1 CF-diff range wording")
    L.num("S4.1 CF-diff on the 8 nulls, 5 Adult nulls and 8 split-D nulls: low end (%)", "97.7", min(cf_a + cf_ad + cfd), "exp_0017 per_pair b1_det found_rate; exp_0028/h6.json b1_det_found_rate; exp_0049/results_d.json:x3.pairs.cfdiff_attributes", scale=100)
    L.num("S4.1 CF-diff on the 8 nulls, 5 Adult nulls and 8 split-D nulls: high end (%)", "100", max(cf_a + cf_ad + cfd), "same", scale=100)
    L.exact("S4.1 split D: 8 new null pairs", 8, len(d49["x3"]["pairs"]), "exp_0049/results_d.json:x3.pairs")
    L.exact("S4.1 five Adult null updates", 5, len(h6["pairs"]), "exp_0028/h6.json:pairs")
    L.exact("S4.1 the 8 null pairs of split A (5 resample + 3 size~U)", 8, len(cf_a), "exp_0017/results.json:per_pair (N0, N1)")
    cfv = [p["cfdiff_attributes"] for p in x5["nulls"]["pairs"]]
    L.num("S4.1 VAE CF-diff low end (%)", "85", min(cfv), "exp_0057/results_x5.json:nulls.pairs.cfdiff_attributes", scale=100)
    L.num("S4.1 VAE CF-diff high end (%)", "94", max(cfv), "same", scale=100)
    L.exact("S4.1 VAE: 100 queries per pair", 100, x5["n_points"], "exp_0057/results_x5.json:n_points")
    L.exact("S4.1 VAE: 6 of 8 offsets with witnesses for at least half of the lowered-class queries", (6, 8),
            (x5["offsets"]["n_pairs_dce_lowered_ge_05"], len(x5["offsets"]["pairs"])), "exp_0057/results_x5.json:offsets")
    L.text("main", r"(registered), $\Delta$-CE finds no witness for a null query (0/800; CF-diff 85--94\%)", "S4.1 VAE wording")
    L.text("main", r"and on 6 of 8 offsets finds one for at least half of the lowered-class queries", "S4.1 VAE offsets wording")
    e4 = J("exp_0047/results_e.json")["e4"]
    L.num("S4.1 kappa=0: witnesses on 83% of null queries (W6 wording 'null queries'; the source is the split-A null queries)", "83", e4["null"]["splitA_kappa0"], "exp_0047/results_e.json:e4.null.splitA_kappa0", scale=100,
          note="main text no longer says 'split-A'; 0.83 is the split-A value (grid nulls at kappa=0: %.2f)" % e4["null"]["grid_kappa0"])
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
    L.exact("S4.1 witness beat the margin score by 0.2 in 7 of 8 pairs (not in all)", (7, 8), (sum(p["dce_lowered"] - p["s_M_lowered_fpr0"] >= 0.2 - 1e-12 for p in q), len(q)),
            "exp_0049/results_d.json:x2.pairs (dce_lowered - s_M_lowered_fpr0 >= 0.2)")


def sec_offsets_rotations(A, B, G) -> None:
    g5 = J("exp_0026/results.json")["H5"]
    d49 = J("exp_0049/results_d.json")["x2"]
    q = [p for p in d49["pairs"] if p["qualified"]]
    L.text("main", r"their Recovery is 0.081 on the grid's pure offsets and 0.03--0.09 on the 8 registered offsets", "S4.2 offsets wording")
    L.text("main", r"On Threshold, $\Delta$-CE beats them on new data (0.41 vs.\ 0.10; main test: 0.47 vs.\ 0.07) in 3 of 3 pairs, too few for a pair-level test", "S4.2 Threshold wording")
    L.num("S4.2 pure offsets of the grid: Grad-diff Recovery", "0.081", g5["H5a"]["estimate"], "exp_0026/results.json:H5.H5a.estimate")
    ext = J("exp_0058/results_s.json")["s1"]["rows"]
    row = {r["id"]: r for r in ext}
    L.num("S4.2 split D offsets Grad-diff low end", "0.03", min(p["graddiff_recovery"] for p in q), "exp_0049/results_d.json:x2.pairs.graddiff_recovery")
    L.num("S4.2 split D offsets Grad-diff high end", "0.09", max(p["graddiff_recovery"] for p in q), "same")
    L.exact("S4.2 8 of 8 below 0.2", 8, d49["k_graddiff_below_02"], "exp_0049/results_d.json:x2.k_graddiff_below_02")
    L.num("S4.2 sign test p", "0.004", d49["sign_test_p"], "exp_0049/results_d.json:x2.sign_test_p")
    # Threshold B: Delta-CE 0.41 vs Grad-diff 0.10, positive for each seed
    t4 = [(m, a) for m, a in B if m["condition"] == "T4"]
    dce = float(np.mean([pair_mean(a, "dce_N", k_index(m["k_star"])) for m, a in t4]))
    gd = float(np.mean([pair_mean(a, "b3", k_index(m["k_star"])) for m, a in t4]))
    L.exact("S4.2 split B Threshold pairs", 3, len(t4), "exp_0019/pairs (condition T4)")
    L.exact("S4.2 Delta-CE beats Grad-diff on Threshold in each of 3 pairs (pair-level Recovery, exp_0019 arrays)", (3, 3),
            (sum(pair_mean(a, "dce_N", k_index(m["k_star"])) > pair_mean(a, "b3", k_index(m["k_star"])) for m, a in t4), len(t4)), "exp_0019/pairs/pair_0{6,7,8}.npz: dce_N vs b3 Recovery per pair")
    L.num("S4.2 Delta-CE on Threshold, split B", "0.41", dce, "exp_0019/pairs/pair_0{6,7,8}.npz: dce_N Recovery mean")
    L.num("S4.2 Grad-diff on Threshold, split B", "0.10", gd, "exp_0019/pairs/pair_0{6,7,8}.npz: b3 Recovery mean")
    sd = J("exp_0019/results.json")["C1"]["seed_diffs"]
    L.exact("S4.2 (W6: 'in each of 3 pairs') the three seed-level differences are positive; seeds = pairs for Threshold", (True, 3), (all(v > 0 for v in sd.values()), len(sd)), "exp_0019/results.json:C1.seed_diffs")
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
    L.text("main", r"but also on 57\% and 71\% of null queries (IG-diff on 42\% of the registered nulls, as predicted), so this carries no signal", "S4.4 nulls wording")
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
    L.text("main", r"finds witnesses for 0.26--0.47 of queries (predicted $\le0.2$), none missed by $\Delta$-CE", "S4.4 Seg-search wording")
    L.exact("S4.4 'none that Delta-CE misses': queries where Seg-search x10 finds a witness and Delta-CE does not (Seg-found subset of DCE-found, exp_0048 b5x__found vs exp_0017 dce_N__found)", 0, tot - both,
            "exp_0048/pairs b5x__found & exp_0017/pairs dce_N__found, same z0 (asserted)", note="%d Seg-search x10 witnesses over %d queries; Delta-CE finds one on %d; nearer or equal on %d (numbers no longer printed in main.tex)" % (tot, sum(len(a["b5x__found"]) for _, a in X), both, near))
    rd = Path("runs/experiments/exp_0048/README.md").read_text()
    L.exact("S4.4 Seg-search x10 prediction 'found-rate stays <= 0.2' was logged beforehand (README of exp_0048 records it)", True,
            "found-rate stays \u2264 0.2" in rd and "not met" in rd, "exp_0048/README.md (prediction text, D-052)", note="text source")


def sec_resnet() -> None:
    x4 = J("exp_0050/results_x4.json")
    x4e = J("exp_0054/test/results_x4_kstar.json")
    cal = J("exp_0054/calibration_resnet.json")
    cal0 = J("exp_0054/calibration_resnet_D054.json")
    L.text("main", r"null fine-tuning moves margins four times more than on the CNNs, and witnesses appear on 0.90 of null queries", "S4.4 ResNet wording")
    L.text("main", r"(rule fixed in advance; grid extended after $\kappa\le2$ failed), $\kappa^\star=2.5$, no new null query has a witness (0/200) and witnesses exist for 0.78--1.00 of lowered-class offset queries", "S4.4 calibration wording")
    L.num("S4.4 ResNet nulls: witnesses on 0.90 of null queries", "0.90", x4["null_witness_share_pooled"], "exp_0050/results_x4.json:null_witness_share_pooled")
    rd = (Path("runs/experiments/exp_0050/README.md")).read_text()
    m = re.search(r"max\|h\| over the r-ball\s*\nmedian ([\d.]+) \(CNN split-D nulls ([\d.]+)\)", rd)
    if m:
        L.exact("S4.4 null fine-tuning moves margins 'four times more' (ratio of median max|h| in the ball, rounded)", 4, round(float(m[1]) / float(m[2])),
                "exp_0050/README.md (median 1.93 vs CNN 0.51; text only)", note="ratio %.2f; README text, not a results json" % (float(m[1]) / float(m[2])))
    else:
        L.unsourced("S4.4 'four times more'", "four times", "README pattern not found")
    L.exact("S4.4 'kappa<=2 failed': the validation grid ends at kappa=2.0", 2.0, max(l["kappa"] for l in cal0["levels"]), "exp_0054/calibration_resnet_D054.json:levels")
    L.exact("S4.4 kappa grid 0.5..2.0 failed (share at 2.0 > 0.05) before extension", True, cal0["levels"][-1]["share"] > 0.05, "exp_0054/calibration_resnet_D054.json:levels")
    L.exact("S4.4 kappa* = 2.5", 2.5, cal["kappa_star"], "exp_0054/calibration_resnet.json:kappa_star")
    L.exact("S4.4 CNN kappa* = 0.5 (rule fixed in advance; main text states kappa=0.5 for the CNNs in Sec. 3)", 0.5, J("exp_0054/calibration_cnn.json")["kappa_star"], "exp_0054/calibration_cnn.json:kappa_star")
    L.exact("S4.4 split E: no null query has a witness (0/200)", (0, 200), (round(sum(p["witness_share"] * 100 for p in x4e["nulls"])), 200), "exp_0054/test/results_x4_kstar.json:nulls")
    off = x4e["offsets"]
    L.num("S4.4 lowered-class offset witnesses, low end", "0.78", min(p["dce_lowered"] for p in off), "exp_0054/test/results_x4_kstar.json:offsets.dce_lowered")
    L.num("S4.4 ... high end", "1.00", max(p["dce_lowered"] for p in off), "same")
    L.num("S4.4 Recovery stays 0.36 (low end, split E)", "0.36", min(p["dce_recovery"] for p in off), "exp_0054/test/results_x4_kstar.json:offsets.dce_recovery")
    L.num("S4.4 Recovery stays 0.40 (high end, split E)", "0.40", max(p["dce_recovery"] for p in off), "same")
    L.text("main", r"($\Delta$-CE Recovery 0.36--0.40). Grad-diff sees neither change: it names aspect on 0.32--0.36 of offset queries, near its 0.20--0.30 on nulls, and its rotation Recovery, 0.06--0.10, is below that of the offset with equal seed and threshold in all 3 pairs.", "S4.4 ResNet Grad-diff wording")
    L.exact("S4.4 ResNet split D: 2 null, 3 offset and 4 rotation pairs (no longer printed in the text; Table 3 gives 0/2 and 0/3)", (2, 3, 4), (len(x4["nulls"]), len(x4["offsets"]), len(x4["rotations"])), "exp_0050/results_x4.json")
    _q = {q["pair_id"]: q.get("qualified", True) for q in J("exp_0050/qualification.json")["pairs"]}
    L.exact("S4.4 ResNet rotation vs offset: 3 comparable pairs (same seed and t), the fourth rotation's offset not qualified", (3, False),
            (sum(any(o["seed"] == r["seed"] and o["t"] == r["t"] for o in x4["offsets"]) for r in x4["rotations"]), _q[120]), "exp_0050/results_x4.json; exp_0050/qualification.json")
    L.num("S4.4 ResNet Grad-diff offsets, low end (split D)", "0.32", min(p["graddiff_recovery"] for p in x4["offsets"]), "exp_0050/results_x4.json:offsets.graddiff_recovery")
    L.num("S4.4 ResNet Grad-diff rotations, low end (split D)", "0.06", min(p["graddiff_recovery"] for p in x4["rotations"]), "exp_0050/results_x4.json:rotations.graddiff_recovery")
    L.num("S4.4 ResNet Grad-diff rotations, high end (split D)", "0.10", max(p["graddiff_recovery"] for p in x4["rotations"]), "exp_0050/results_x4.json:rotations.graddiff_recovery")
    _nr = []
    for _p in (128, 129):
        _z = np.load(R / "exp_0050" / "pairs" / f"pair_{_p}.npz")
        _f = _z["b3__found"].astype(bool)
        _nr.append(float((np.abs(_z["b3__dphi"][_f]).argmax(1) == 0).mean()))
    L.num("S4.4 ResNet Grad-diff names aspect on nulls, low end (split D)", "0.20", min(_nr), "exp_0050/pairs/pair_128,129.npz: b3 argmax |dphi| == aspect")
    L.num("S4.4 ResNet Grad-diff names aspect on nulls, high end (split D)", "0.30", max(_nr), "same")
    L.exact("Table 3 row: ResNet Grad-diff < 0.2 on offsets in 0/3 (registered readout)", "0 of 3", x4["k_graddiff_below_02_offsets"], "exp_0050/results_x4.json:k_graddiff_below_02_offsets")
    L.exact("Table 3 row: ResNet Grad-diff rotation above offset in 0/3 (registered readout)", "0 of 3", x4["k_rotation_above_offset"], "exp_0050/results_x4.json:k_rotation_above_offset")
    L.text("main", r"Grad-diff $<0.2$; rot.\ $>$ offset & $\times$ OSF & ResNet-8 & 0.32--0.36; 0.06--0.10 & -- & 0/3; 0/3", "Table 3 row ResNet Grad-diff offsets and rotation")
    _x2 = [q for q in J("exp_0049/results_d.json")["x2"]["pairs"] if q["qualified"]]
    _dd = [q["dce_lowered"] - q["s_M_lowered_fpr0"] for q in _x2]
    L.num("Table 3 row: witness minus margin score, low end", "0.17", min(_dd), "exp_0049/results_d.json:x2.pairs (qualified) dce_lowered - s_M_lowered_fpr0")
    L.num("Table 3 row: witness minus margin score, high end", "0.97", max(_dd), "same")
    L.exact("Table 3 row: witness >= margin score + 0.2 in 7 of 8 pairs", (7, 8), (sum(d >= 0.2 - 1e-9 for d in _dd), len(_dd)), "same")
    L.text("main", r"Witness $\ge$ margin score $+0.2$ & $\times$ OSF & Registered offsets & $+0.17$ to $+0.97$ & -- & 7/8", "Table 3 row witness vs margin score")
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
    L.text("main", r"$\Delta$-CE $\approx$ Grad-diff, CF-diff & $\times$ pre & Shortc., Thresh. & $|\Delta|=0.17$--$0.53$ & -- & --\\", "Table 3 Tie row")
    L.exact("Table 3 caption: with <=4 pairs the exact pair-level p >= 0.0625", 0.0625, min(row[i]["sign_test"]["min_attainable_p"] for i in ("gd_lt02_grid", "normal_zstar")), "exp_0058/results_s.json min_attainable_p (4 pairs)")



MAIN_SNIPPETS = [
    # Section 3 (the abstract wording and Table 1 rows are guarded where their numbers are checked)
    r"labels are $y=\mathbbm 1[\text{aspect}>0.5]$ with 5\% noise", r"fine-tuned on 10\,000 samples", r"$32\times32$ ellipses",
    r"$N=300$ valid test points", r"the test points and the budget $r=0.54$, set once on validation", r"($\kappa=0.5$)",
    # Table 3 rows
    r"No witness on nulls & $\checkmark$ post & Nulls; Adult MLP & 0/2\,400; 0/1\,500 & -- & 8/8; 5/5", r"No witness on nulls & $\checkmark$ OSF & Unscreened; decoder & 0/2\,400; 0/800", r"0/2\,400; 0/800 & -- & 8/8; 8/8",
    r"$\times$ OSF; expl. & ResNet-8, $\kappa=0.5$; 2.5 & 180/200; 0/200 & -- & 0/2; 2/2",
    r"$\checkmark$ post & Logit offsets, uncut & 0.961 [0.955, 0.967] & $2{\cdot}10^{-4}$ & --",
    r"Witness localises ($L>0$) & $\checkmark$ OSF & Two changes & $L=0.50$ & 0.004 & 8/8",
    r"($\checkmark$) pre & Grid, $\alpha=0$ & 0.081 [0.066, 0.096] & $4{\cdot}10^{-4}$ & 4/4",
    r"$\checkmark$ OSF & Registered offsets & 0.033--0.093 & 0.004 & 8/8",
    r"$\times$ pre & Grid & $-0.098$ [$-0.143$, $-0.053$] & 0.94 & --",
    r"($\checkmark$) post & Threshold & $+0.31$ [0.27, 0.35] & $2{\cdot}10^{-4}$ & 3/3",
    r"$\checkmark$ pre & Grid & $+0.57$ [0.53, 0.61] & $4{\cdot}10^{-4}$ & $\rho=0.89$",
    r"$\times$ post; pre & Shortcut; grid & $-0.30$; $-0.35$ ($\pm0.03$) & 1; 1 & --",
    r"($\checkmark$) post & Grid & $-0.19\to+0.03$ [0.01, 0.05] & $6{\cdot}10^{-4}$ & 4/4",
    r"$\Delta$-CE $>$ Grad-diff (rejected) & $\times$ pre & Subgroup & $-0.09$ [$-0.13$, $-0.05$] & -- & 0/3",
    r"CIs $\pm0.035$",
    # Section 4.1
    r"With $\kappa=0$, witnesses appear on 83\% of null queries",
    r"witnesses exist for 0.17--1.00 of lowered-class queries on the 8 registered offsets, the margin score flags 0.00--0.54 and CF-diff none",
    r"The witness beat the margin score by 0.2 in 7 of 8 pairs, short of the registered 8/8,",
    # Section 4.2, 4.3
    r"hence $\Delta$-CE's found-rate of 0.47",
    r"rises from 0.36 at $\alpha=0.1$ to 0.93 at $\alpha=0.5$ ($t=0.5$; Spearman $\rho=0.89$ over the 24 rotation pairs",
    r"recover colour on Shortcut (0.93) and, against the predicted blindness, name aspect on Threshold (0.73--0.75; Grad-diff 0.07)",
    r"ranks aspect first in all 9 change pairs", r"$k^\star$ on adaptation data", r"Seg-search with $10\times$ the segments",
    # Section 5 and 6
    r"On a registered ResNet-8", r"three follow-ups were registered before their data existed~\cite{osf}",
    r"osf.io/vfjtk, osf.io/fr7jv, osf.io/fp8uc", r"dates all test sets",
    r"with three disclosed exceptions",
]


def main_snippets() -> None:
    for i, sn in enumerate(MAIN_SNIPPETS, 1):
        L.text("main", sn, f"main.tex snippet {i}")
    # S4.1 cost sentence (added 2026-11-11): wall seconds per query on the main test set (exp_0058 S2, run A exp_0017)
    rt = J("exp_0058/results_s.json")["s2"]["runs"]["A exp_0017"]["methods"]
    src = "exp_0058/results_s.json:s2.runs.A exp_0017.methods"
    L.text("main", r"The search costs 2.4\,s per query (CF-new: 0.6\,s; Grad-diff: under 1\,ms)", "S4.1 cost sentence")
    L.num("S4.1 Delta-CE (type N) wall s per query", "2.4", rt["dce_N"]["wall_s_per_point"]["mean"], src + ".dce_N.wall_s_per_point.mean")
    L.num("S4.1 CF-new wall s per query", "0.6", rt["a2"]["wall_s_per_point"]["mean"], src + ".a2.wall_s_per_point.mean")
    L.exact("S4.1 Grad-diff under 1 ms per query", True, rt["b3"]["wall_s_per_point"]["mean"] < 1e-3, src + ".b3.wall_s_per_point.mean")


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
    for pid, pr in ((20, "0.526"), (21, "0.464")):
        L.num(f"S-C text: grid null cell pair {pid} realised threshold t_hat", pr, gnull[pid]["t_hat"], f"exp_0026/results.json:descriptive.pairs[{pid}].t_hat")
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


def sec_x6_and_protocol() -> None:
    """X6 (split G, exp_0060, D-057): main-paper abstract/Table 3/Section 4.1 'where' paragraph, supplement Section X6 with
    Table S-X6 and the split-G row of the protocol log; plus the protocol numbers added by the W6 rewrite."""
    r6 = J("exp_0060/results_x6.json")
    q6 = J("exp_0060/qualification.json")
    cfg = yaml.safe_load(Path("configs/experiments/exp_0060.yaml").read_text())
    prim, pooled = r6["primary"], r6["L_pooled_by_method"]
    pr = r6["pairs"]
    sx = "exp_0060/results_x6.json"
    dec = Path("research/DECISIONS.md").read_text()
    d57 = dec[dec.index("## D-057"):]
    d57 = d57[:d57.find("\n## D-058")] if "\n## D-058" in d57 else d57
    drift = Path("src/dce/data/drift.py").read_text()

    # ---- consistency of the analysis file with the saved per-pair numbers
    L.exact("X6 results: L_per_pair equals the per-pair dce_N.L and pooled L is above 0 (sanity)", True,
            all(abs(a - p["dce_N"]["L"]) < 1e-12 for a, p in zip(prim["L_per_pair"], pr)) and prim["L_pooled"] > 0, sx + ":primary.L_per_pair, pairs[*].dce_N.L")
    L.exact("X6 results: pair ids 130..137 as in the config", list(range(130, 138)), [p["pair_id"] for p in pr], sx + ":pairs.pair_id; configs/experiments/exp_0060.yaml:pair_ids")
    # ---- main.tex: abstract, Table 3 row, Section 4.1 paragraph
    L.text("main", r"and localises two independent changes", "Abstract X6 wording")
    L.text("main", r"witnesses change size most for 46--75\% of queries near the large-object region and 13--23\% far from it; this difference $L$ is positive in all 8 pairs (pooled 0.50; Table~\ref{tab:tests})",
           "S4.1 X6 paragraph: near/far/8 of 8/pooled/p")
    L.text("main", r"Grad-diff and CF-diff barely localise ($L=0.09$ and $0.13$), and Imp-diff, one vector per pair, cannot; CF-new localises equally well (0.51)", "S4.1 X6 paragraph: controls")
    L.text("main", r"With a colour shortcut everywhere and a label flip for large objects, witnesses", "S4.1 X6 paragraph: design")
    near = [p["dce_N"]["share_size_near"] for p in pr]
    far = [p["dce_N"]["share_size_far"] for p in pr]
    nt = "main text says 'of queries'; the saved share is among queries with a witness (n_near, n_far count found queries)"
    L.num("S4.1 X6: size most for 46% (min, near the large-object region)", "46", min(near), sx + ":pairs[*].dce_N.share_size_near", scale=100, note=nt)
    L.num("S4.1 X6: ... 75% (max, near)", "75", max(near), "same", scale=100, note=nt)
    L.num("S4.1 X6: far from it 13% (min)", "13", min(far), sx + ":pairs[*].dce_N.share_size_far", scale=100, note=nt)
    L.num("S4.1 X6: ... 23% (max, far)", "23", max(far), "same", scale=100, note=nt)
    L.exact("S4.1 X6: L > 0 in 8 of 8 pairs", (8, 8), (prim["k_L_positive"], prim["n_qualified"]), sx + ":primary.k_L_positive / n_qualified")
    L.exact("S4.1 X6: L > 0 in 8 of 8 pairs (recounted from per-pair L)", 8, sum(x > 0 for x in prim["L_per_pair"]), sx + ":primary.L_per_pair")
    L.num("S4.1 X6: pooled L = 0.50", "0.50", prim["L_pooled"], sx + ":primary.L_pooled")
    L.num("S4.1 X6: sign test p = 0.004", "0.004", prim["sign_test_p"], sx + ":primary.sign_test_p")
    L.num("S4.1 X6: Grad-diff pooled L 0.09", "0.09", pooled["b3"], sx + ":L_pooled_by_method.b3")
    L.num("S4.1 X6: CF-diff pooled L 0.13", "0.13", pooled["b1_det"], sx + ":L_pooled_by_method.b1_det")
    L.num("S4.1 X6: CF-new pooled L 0.51", "0.51", pooled["a2"], sx + ":L_pooled_by_method.a2")
    L.num("Table 3 'Witness localises' L = 0.50", "0.50", prim["L_pooled"], sx + ":primary.L_pooled")
    L.num("Table 3 'Witness localises' p = 0.004 (sign)", "0.004", prim["sign_test_p"], sx + ":primary.sign_test_p")
    L.exact("Table 3 'Witness localises' pairs 8/8", (8, 8), (prim["k_L_positive"], prim["n_qualified"]), sx + ":primary")
    L.exact("Table 3 'Witness localises' status 'confirmed': registered prediction met (all pairs L>0, pooled L >= 0.30)", True,
            bool(prim["prediction_met"]) and prim["k_L_positive"] == 8 and prim["L_pooled"] >= 0.30, sx + ":primary.prediction_met")
    L.exact("S4.1 X6: registered prediction (pooled L >= 0.30) is the one in the decision log D-057", True, "0.30" in d57 and "fp8uc" in d57, "research/DECISIONS.md D-057")

    # ---- supplement, Section X6: registration and design
    sp = lambda sn, loc: L.text("supp", sn, loc)
    sp(r"Registered at \url{https://osf.io/fp8uc} on 2026-11-05, before any X6 model was trained and before the test data (split G) existed (decision log D-057)", "S-X6 registration sentence")
    L.exact("S-X6 registration: osf.io/fp8uc and the date 2026-11-05 are in D-057 of the decision log", True,
            bool(re.search(r"osf\.io/fp8uc \(submitted 2026-11-05", dec)), "research/DECISIONS.md D-057")
    sp(r"Shortcut data (70\% of samples with aspect in $[0.4,0.6]$, colour agreeing with the noise-free label with probability 0.99) whose labels are then flipped for size $>0.75$, with 5\% label noise; 10 epochs, Adam $10^{-3}$, batch 256",
       "S-X6 design sentence")
    fr = float(re.search(r"t1_ambiguous_frac: float = ([\d.]+)", drift)[1])
    lo_ = float(re.search(r"t1_ambiguous_low: float = ([\d.]+)", drift)[1])
    hi_ = float(re.search(r"t1_ambiguous_high: float = ([\d.]+)", drift)[1])
    L.num("S-X6 design: 70% of samples in the ambiguous aspect band", "70", fr, "src/dce/data/drift.py:DriftParams.t1_ambiguous_frac (used by ss_adaptation_set)", scale=100)
    L.exact("S-X6 design: aspect band [0.4, 0.6]", (0.4, 0.6), (lo_, hi_), "src/dce/data/drift.py:DriftParams.t1_ambiguous_low/high")
    L.exact("S-X6 design: colour agrees with the noise-free label with probability 0.99", 0.99, cfg["t1_correlation"], "configs/experiments/exp_0060.yaml:t1_correlation")
    L.exact("S-X6 design: labels flipped for size > 0.75", 0.75, cfg["size_threshold"], "configs/experiments/exp_0060.yaml:size_threshold")
    L.exact("S-X6 design: 5% label noise", 5, round(cfg["adapt_label_noise"] * 100), "configs/experiments/exp_0060.yaml:adapt_label_noise")
    L.exact("S-X6 design: 10 epochs, lr 1e-3, batch 256", (10, 1e-3, 256), (cfg["recipe"]["epochs"], cfg["recipe"]["lr"], cfg["recipe"]["batch_size"]), "configs/experiments/exp_0060.yaml:recipe")
    L.exact("S-X6 design: f_new fine-tuned on 10 000 adaptation samples (main Sec. 3)", 10000, cfg["n_adapt"], "configs/experiments/exp_0060.yaml:n_adapt")
    L.exact("S-X6 design: pairs 130--137", (130, 137, 8), (min(cfg["pair_ids"]), max(cfg["pair_ids"]), len(cfg["pair_ids"])), "configs/experiments/exp_0060.yaml:pair_ids")
    sp(r"Pairs 130--137: $f_{old}$ seeds 0--4 and 0--2 with a second adaptation draw", "S-X6 pairs sentence")
    seeds = [p["seed"] for p in q6["pairs"]]
    L.exact("S-X6 design: f_old seeds 0--4, then 0--2 with a second adaptation draw", ([0, 1, 2, 3, 4], [0, 1, 2]), (seeds[:5], seeds[5:]), "exp_0060/qualification.json:pairs[*].seed")
    sp(r"Qualification (validation): $\Delta$Flip of colour and of size $\ge0.15$, of background and position $\le0.05$", "S-X6 qualification rule sentence")
    L.exact("S-X6 qualification rule: min delta 0.15, max other 0.05", (0.15, 0.05), (cfg["qualify"]["min_delta"], cfg["qualify"]["max_other"]), "configs/experiments/exp_0060.yaml:qualify")
    sp(r"Test: 300 valid queries per pair (data seed $13000+p$)", "S-X6 test sentence")
    dsd = [json.loads((R / f"exp_0060/pairs/pair_{p['pair_id']}.json").read_text()) for p in pr]
    L.exact("S-X6 test: 300 valid queries per pair", [300] * 8, [d["n_points"] for d in dsd], "exp_0060/pairs/pair_*.json:n_points")
    L.exact("S-X6 test: data seed 13000 + pair id (13000 + p; the saved seeds are 13000 + pair id)", [13000 + p["pair_id"] for p in pr],
            [d["data_seed"] for d in dsd], "exp_0060/pairs/pair_*.json:data_seed", note="config data_seed_base %d; saved data_seed = base + pair_id" % cfg["data_seed_base"])
    L.exact("S-X6 test: data_seed_base 13000 in the config", 13000, cfg["data_seed_base"], "configs/experiments/exp_0060.yaml:data_seed_base")
    sp(r"for queries with size $\ge0.60$ (near the subgroup) minus the same share for size $\le0.45$ (far); prediction $L>0$ in every qualified pair and pooled $L\ge0.30$, exact sign test over pairs", "S-X6 readout sentence")
    L.exact("S-X6 readout: near size >= 0.60, far size <= 0.45", (0.60, 0.45), (cfg["locality"]["near"], cfg["locality"]["far"]), "configs/experiments/exp_0060.yaml:locality")
    L.exact("S-X6 readout: the registered pooled threshold 0.30 is in D-057", True, bool(re.search(r"0\.30", d57)), "research/DECISIONS.md D-057")
    sp(r"Registered at \url{https://osf.io/fp8uc}", "S-X6 OSF link")

    # ---- supplement, Section X6: results
    sp(r"All 8 pairs qualified ($\Delta$Flip colour 0.34--0.45, size 0.29--0.36, background/position $\le0.03$), so the test is confirmatory.", "S-X6 results sentence 1")
    L.exact("S-X6 results: all 8 pairs qualified", (8, 8), (q6["n_qualified"], len(q6["pairs"])), "exp_0060/qualification.json:n_qualified")
    qp = q6["pairs"]
    L.num("S-X6 qualification: dFlip colour low end 0.34", "0.34", min(x["delta_flip_colour"] for x in qp), "exp_0060/qualification.json:pairs[*].delta_flip_colour")
    L.num("S-X6 qualification: dFlip colour high end 0.45", "0.45", max(x["delta_flip_colour"] for x in qp), "same")
    L.num("S-X6 qualification: dFlip size low end 0.29", "0.29", min(x["delta_flip_size"] for x in qp), "exp_0060/qualification.json:pairs[*].delta_flip_size")
    L.num("S-X6 qualification: dFlip size high end 0.36", "0.36", max(x["delta_flip_size"] for x in qp), "same")
    L.bound("S-X6 qualification: background/position change <= 0.03", "0.03", max(x["max_abs_delta_flip_bg_posx"] for x in qp), "exp_0060/qualification.json:pairs[*].max_abs_delta_flip_bg_posx (max)")
    sp(r"$L>0$ in 8 of 8 pairs, pooled $L=0.50$, sign test $p=0.0039$}", "S-X6 results: bold sentence")
    L.exact("S-X6 results: L > 0 in 8 of 8 pairs", (8, 8), (prim["k_L_positive"], prim["n_qualified"]), sx + ":primary")
    L.num("S-X6 results: pooled L = 0.50", "0.50", prim["L_pooled"], sx + ":primary.L_pooled")
    L.num("S-X6 results: sign test p = 0.0039", "0.0039", prim["sign_test_p"], sx + ":primary.sign_test_p")
    L.exact("S-X6 results: sign test p = 0.5^8 (exact one-sided, 8 of 8)", 0.5 ** 8, prim["sign_test_p"], sx + ":primary.sign_test_p")
    L.exact("S-X6 results: prediction met", True, bool(prim["prediction_met"]), sx + ":primary.prediction_met")
    sp(r"Per pair: 0.59, 0.54, 0.53, 0.23, 0.52, 0.60, 0.50, 0.48", "S-X6 per-pair L list")
    for pid, printed, v in zip(range(130, 138), ("0.59", "0.54", "0.53", "0.23", "0.52", "0.60", "0.50", "0.48"), prim["L_per_pair"]):
        L.num(f"S-X6 per-pair L, pair {pid}", printed, v, sx + f":primary.L_per_pair[{pid - 130}]")
    sp(r"near the subgroup 0.46--0.75 of witnesses change size most, far from it 0.13--0.23; witnesses were found for 0.95--1.00 of queries", "S-X6 near/far/found sentence")
    L.num("S-X6 near the subgroup: low end 0.46", "0.46", min(near), sx + ":pairs[*].dce_N.share_size_near")
    L.num("S-X6 near the subgroup: high end 0.75", "0.75", max(near), "same")
    L.num("S-X6 far: low end 0.13", "0.13", min(far), sx + ":pairs[*].dce_N.share_size_far")
    L.num("S-X6 far: high end 0.23", "0.23", max(far), "same")
    fnd = [p["dce_N"]["found"] for p in pr]
    L.num("S-X6 witnesses found: low end 0.95", "0.95", min(fnd), sx + ":pairs[*].dce_N.found")
    L.num("S-X6 witnesses found: high end 1.00", "1.00", max(fnd), "same")
    sp(r"Controls (pooled $L$): Grad-diff 0.09, CF-diff 0.13, Imp-diff 0 by construction (one vector per pair), and CF-new 0.51", "S-X6 controls sentence")
    L.num("S-X6 controls: Grad-diff 0.09", "0.09", pooled["b3"], sx + ":L_pooled_by_method.b3")
    L.num("S-X6 controls: CF-diff 0.13", "0.13", pooled["b1_det"], sx + ":L_pooled_by_method.b1_det")
    L.exact("S-X6 controls: Imp-diff 0 by construction (pooled L = 0 and L = 0 in every pair)", (0.0, 0.0), (pooled["b4"], max(abs(p["b4"]["L"]) for p in pr)), sx + ":L_pooled_by_method.b4; pairs[*].b4.L")
    L.num("S-X6 controls: CF-new 0.51", "0.51", pooled["a2"], sx + ":L_pooled_by_method.a2")
    sp(r"with the same top-1 shares and MRR as \DCE{} (Table~\ref{tab:x6})", "S-X6 'same top-1 and MRR' wording")
    mean = lambda m, k: float(np.mean([p[m][k] for p in pr]))
    L.exact("S-X6: CF-new has the same top-1 shares and MRR as Delta-CE (as printed in Table S-X6, 2 decimals)", True,
            all(round(mean("dce_N", k), 2) == round(mean("a2", k), 2) for k in ("mrr_colour", "mrr_size", "top1_colour", "top1_size")), sx + ":pairs[*].dce_N / a2 means")
    res17 = J("exp_0017/results.json")
    nulls = [p for p in res17["per_pair"].values() if p["condition"] in ("N0", "N1")]
    L.num("S-X6: CF-new returns a point for 0.996 of queries on nulls (main Table 2)", "0.996", float(np.mean([p["methods"]["a2"]["found_rate"] for p in nulls])), "exp_0017/results.json:per_pair[N0,N1].methods.a2.found_rate (mean of 8)")
    sp(r"Imp-diff ranks neither colour nor size first in 8 of 8 pairs (prediction met)", "S-X6 Imp-diff sentence")
    L.exact("S-X6: Imp-diff ranks neither colour nor size first in 8 of 8 pairs", (8, 8), (sum(r6["imp_diff_neither_first"].values()), len(r6["imp_diff_neither_first"])), sx + ":imp_diff_neither_first")
    # Table S-X6 (rows parsed from the tex)
    t = PAPER["supp"].read_text()
    blk = t[t.index(r"\label{tab:x6}"):]
    blk = blk[blk.index(r"\midrule"):blk.index(r"\bottomrule")]
    rows = {c[0]: c[1:] for c in (tex_cells(ln) for ln in blk.splitlines() if " & " in ln)}
    names = {r"\DCE": "dce_N", "CF-new": "a2", "Grad-diff": "b3", "CF-diff": "b1_det", "Imp-diff": "b4"}
    L.exact("S-X6 Table S-X6 has the 5 methods Delta-CE, CF-new, Grad-diff, CF-diff, Imp-diff", sorted(names), sorted(rows), "paper/supplement.tex Table S-X6")
    cols = ("set_recovery", "mrr_colour", "mrr_size", "top1_colour", "top1_size")
    for nm, cells in rows.items():
        for k, c in zip(cols, cells):
            L.num(f"S-X6 Table S-X6 {nm} {k}", c, mean(names[nm], k), sx + f":pairs[*].{names[nm]}.{k} (mean over the 8 pairs)")
    sp(r"X6, mean over the 8 pairs.", "S-X6 table caption")

    # ---- D-059: filtered CF-new and null-adjusted Recovery (exp_0061, descriptive, after seeing results)
    cf = J("exp_0061/results_cfnew_filter.json")
    sa, x6r = cf["split_a"]["summary"], cf["x6"]
    src61 = "exp_0061/results_cfnew_filter.json"
    L.text("main", r"\textbf{Search versus filtering} (post hoc, saved outputs). The counterfactual of $f_{new}$ kept only where $m_o\ge\kappa$ (filtered CF-new) is a witness; it is silent on the nulls and localises ($L=0.48$), but finds witnesses for only 0.42--0.46 of the main-test queries, against 0.47--0.84", "S4.1 checked-CF paragraph")
    L.exact("S4.1 checked CF-new silent on the 8 split-A nulls", 0.0, sa["Null"]["found_a2_filtered"], src61 + ":split_a.summary.Null.found_a2_filtered")
    L.num("S4.1 checked CF-new L in X6", "0.48", x6r["L_pooled"]["a2_filtered"], src61 + ":x6.L_pooled.a2_filtered")
    fr = [sa[c]["found_a2_filtered"] for c in ("T1", "T3", "T4")]
    fd = [sa[c]["found_dce"] for c in ("T1", "T3", "T4")]
    L.num("S4.1 checked CF-new found, low end", "0.42", min(fr), src61 + ":split_a.summary.*.found_a2_filtered")
    L.num("S4.1 checked CF-new found, high end", "0.46", max(fr), "same")
    L.num("S4.1 Delta-CE found, low end", "0.47", min(fd), src61 + ":split_a.summary.*.found_dce")
    L.num("S4.1 Delta-CE found, high end", "0.84", max(fd), "same")
    for c, nm, d, f in (("T1", "Shortcut", "0.38", "0.03"), ("T3", "Subgroup", "0.53", "0.30"), ("T4", "Threshold", "0.47", "0.45")):
        L.num(f"S4.1 {nm}: Delta-CE Recovery", d, sa[c]["recovery"]["dce_N"], src61 + f":split_a.summary.{c}.recovery.dce_N")
        L.num(f"S4.1 {nm}: checked CF-new Recovery", f, sa[c]["recovery"]["a2_filtered"], src61 + f":split_a.summary.{c}.recovery.a2_filtered")
    L.text("main", r"Shortcut (0.38 vs.\ 0.03) and part on Subgroup (0.53 vs.\ 0.30), higher in 3 of 3 pairs each, but not on Threshold (0.47 vs.\ 0.45). Null-adjusted (minus the rate of naming $k^\star$ on nulls), $\Delta$-CE leads on Subgroup and Threshold, Grad-diff on Shortcut (0.79); unfiltered CF-new falls to $-0.09$ on Threshold.", "S4.1 checked-CF numbers")
    L.num("S4.1 null-adjusted CF-new Recovery on Threshold", "-0.09", sa["T4"]["null_adjusted"]["a2"], src61 + ":split_a.summary.T4.null_adjusted.a2")
    L.text("main", r"the search adds witnesses a filtered ordinary counterfactual misses", "Abstract checked-CF sentence")
    L.exact("Abstract: the extra (case B) witnesses carry Shortcut Recovery (case-B share >= 0.9 of Delta-CE Recovery)", True,
            sa["T1"]["dce_recovery_case_b"] / sa["T1"]["recovery"]["dce_N"] >= 0.9, src61 + ":split_a.summary.T1.dce_recovery_case_b / recovery.dce_N")
    # D-059 additions (final round): per-pair advantage, null-adjusted leaders, null recipe and realised thresholds
    pa = cf["split_a"]["pairs"]
    for c, k, nm in (("T1", "2", "Shortcut"), ("T3", "1", "Subgroup")):
        g = [q for q in pa if q["condition"] == c]
        L.exact(f"S4.1 {nm}: Delta-CE Recovery above checked CF-new in 3 of 3 pairs", (3, 3), (sum(q["names"][k]["dce_N"] > q["names"][k]["a2_filtered"] for q in g), len(g)), src61 + ":split_a.pairs[*].names")
    na = {c: sa[c]["null_adjusted"] for c in ("T1", "T3", "T4")}
    meths = ["dce_N", "b1_det", "b3", "b4", "b5", "a1", "a2"]
    L.exact("S4.1 null-adjusted leaders: Shortcut Grad-diff, Subgroup and Threshold Delta-CE", ("b3", "dce_N", "dce_N"), tuple(max(meths, key=lambda m: na[c][m]) for c in ("T1", "T3", "T4")), src61 + ":split_a.summary.*.null_adjusted")
    L.num("S4.1 null-adjusted Grad-diff on Shortcut", "0.79", na["T1"]["b3"], src61)
    L.num("S4.1 null-adjusted Delta-CE on Subgroup", "0.53", na["T3"]["dce_N"], src61)
    L.num("S4.1 null-adjusted Delta-CE on Threshold", "0.47", na["T4"]["dce_N"], src61)
    nb = cf["null_boundaries"]
    L.text("main", r"the nulls are fine-tuned more gently than the offsets (fewer epochs, lower learning rate; supplement), though their threshold moves as far ($|\Delta\hat t|\le0.035$); the grid's null cell, fine-tuned like the offsets, has witnesses on 22\% of queries at $\kappa=0.5$ and none at $\kappa=1$, where small offsets keep few witnesses (0.04).", "S4.1 null recipe sentence")
    c49 = yaml.safe_load(open("configs/experiments/exp_0049.yaml"))
    L.exact("S4.1 null recipe (printed in the supplement): 3 epochs at 3e-4, offset recipe 10 at 1e-3 (registered split D config)", ((3, 3e-4), (10, 1e-3)),
            ((c49["recipe_null"]["epochs"], c49["recipe_null"]["lr"]), (c49["recipe_offset"]["epochs"], c49["recipe_offset"]["lr"])), "configs/experiments/exp_0049.yaml")
    L.num("S4.1 split-A nulls: max |t_hat_new - t_hat_old|", "0.035", nb["max_abs_dt"], src61 + ":null_boundaries.max_abs_dt")
    L.exact("S4.1 grid null cell |dt| at least as large as the nulls' (0.032-0.039)", True, min(abs(c["dt"]) for c in nb["grid_null_cell"]) >= 0.03, src61 + ":null_boundaries.grid_null_cell")
    sp(r"Grid, $\alpha=0$ offsets & 0.47 (0.47) & 0.46 (0.47) & 0.35 (0.38) & 0.02 (0.04)", "S4.1 small grid offsets keep few witnesses at kappa=1 (found-rate 0.04; supplement table)")
    L.text("main", r"$\Delta$-CE stays silent on gently fine-tuned null updates", "Abstract: gently fine-tuned nulls")
    sp(r"Grid null cell (pairs 20, 21) & 0.71 & 0.51 & 0.22 & 0.00", "S4.1 grid null cell 22% at kappa 0.5 (supplement table, from exp_0047 kappa analysis)")
    e1 = J("exp_0041/results_e1.json")
    L.text("main", r"type-O witnesses for the other class raise Recovery to 1.00 on new data", "S4.2 split C union")
    _u = [k for k in e1["methods"] if "union" in k or "N_or_O" in k or k.endswith("NO")]
    L.num("S4.2 split C: Recovery of the nearer of N and O", "1.00", e1["methods"][_u[0]]["recovery"]["mean"], "exp_0041/results_e1.json:methods." + _u[0] + ".recovery.mean")
    L.text("main", r"With each score cut on the nulls to flag none", "Abstract CF-diff at zero false positives")
    _x2b = [q for q in J("exp_0049/results_d.json")["x2"]["pairs"] if q["qualified"]]
    L.exact("Abstract: CF-diff flags no lowered-class offset query at zero null false positives (8 registered offsets)", 0.0, max(q["cfdiff_lowered_fpr0"] for q in _x2b), "exp_0049/results_d.json:x2.pairs.cfdiff_lowered_fpr0")

    # ---- supplement: protocol log, split G row, splits A--G
    sp(r"Test splits A--D are those of the main paper (Section~\ref{sec:log} lists them with E, F and G)", "S intro: splits E, F and G")
    sp(r"G & exp\_0060; data seeds $13000+p$, 300 queries per pair & X6, two independent local changes (D-057) & Registered 2026-11-05 at \url{https://osf.io/fp8uc}, before any X6 model was trained and before split G existed (commit 5b34205)\\",
       "S-log split G row")
    gitc = lambda *a: __import__("subprocess").run(["git", *a], capture_output=True, text=True)
    # Hashes printed in the paper are the original ones; the 2026-11-05 history rewrite (attribution lines only)
    # is resolved through research/COMMIT_REWRITE_2026-11-05.tsv (original -> current).
    def resolve(h):
        f = Path("research") / "COMMIT_REWRITE_2026-11-05.tsv"
        if f.exists():
            for ln in f.read_text().splitlines():
                parts = ln.split("\t")
                if not ln.startswith("#") and len(parts) >= 3 and parts[0].startswith(h):
                    return parts[2]
        return h
    g_hash = resolve("5b34205")
    c = gitc("show", "-s", "--format=%H %cI", g_hash)
    first = gitc("log", "--diff-filter=A", "--format=%H %cI", "--", "runs/experiments/exp_0060/models")
    add = first.stdout.strip().splitlines()[-1].split() if first.stdout.strip() else None
    anc = gitc("merge-base", "--is-ancestor", g_hash, add[0]).returncode == 0 if add else False
    L.exact("S-log split G: commit 5b34205 exists, is dated 2026-11-05 and precedes the commit that added the X6 models and data", (True, True, True),
            (c.returncode == 0, c.stdout.split()[-1].startswith("2026-11-05") if c.returncode == 0 else False, anc), "git: 5b34205 (resolved via research/COMMIT_REWRITE_2026-11-05.tsv) vs first commit touching runs/experiments/exp_0060/models")
    L.exact("S-log split G: 300 queries per pair, data seeds 13000 + p", (300, 13000), (cfg["n_points"], cfg["data_seed_base"]), "configs/experiments/exp_0060.yaml")
    L.exact("S-log: X6 is decision D-057 and the config says so", "D-057", cfg["decision"], "configs/experiments/exp_0060.yaml:decision")
    txt = PAPER["supp"].read_text()
    letters = re.findall(r"^([A-G]) & exp", txt, flags=re.M)
    L.exact("S-log / main Protocol: splits A--G (the log has rows for the letters A to G, once each)", list("ABCDEFG"), letters, "paper/supplement.tex Table tab:splits (first column)")
    L.exact("Main Protocol: follow-ups on splits D, F and G are registered on OSF (three registration ids in DECISIONS: D-052 vfjtk, D-056 fr7jv, D-057 fp8uc)", (True, True, True),
            ("osf.io/vfjtk" in dec, "osf.io/fr7jv" in dec, "osf.io/fp8uc" in dec), "research/DECISIONS.md")
    L.exact("Main Protocol: the three OSF links of the main bibliography are the three registrations of the protocol log (D, F, G)", True,
            all(x in txt for x in ("osf.io/vfjtk", "osf.io/fr7jv", "osf.io/fp8uc")), "paper/supplement.tex Table tab:splits")
    # main Limitations: "with two disclosed exceptions" against the enumerate of the supplement paragraph "Disclosed exceptions"
    blk = txt[txt.index(r"\paragraph{Disclosed exceptions.}"):]
    blk = blk[:blk.index(r"\end{enumerate}")]
    L.exact("Main Limitations (4): 'three disclosed exceptions' vs the items listed under 'Disclosed exceptions' in the supplement", 3, blk.count(r"\item"),
            "paper/supplement.tex \\paragraph{Disclosed exceptions.} (count of \\item)", note="the supplement lists D-041, D-055 and D-053; main.tex says 'three' (corrected 2026-11-05)")


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
    sec_x6_and_protocol()
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
               "Section E (fixed elements, results, Table X5), Section X6 (design, results, Table S-X6), the split-G row and the split letters of the protocol log, "
               "the Table 1 shift column and 'Both' row, and the Table 3 row 'Witness localises'. "
               "Sources: exp_0017, 0019, 0026, 0028, 0032, 0033, 0046-0051, 0054, 0056, 0057, 0058, 0060."), "",
              "Not machine-checkable from saved outputs: proofs and prose claims without a number; registered predictions are checked only where a run README records them."]
    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(cnt))
    for r in bad:
        print(f"{r['status']}: [{r['id']}] {r['location']} | printed {r['printed']} | computed {r['computed']} | {r['source']} | {r['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
