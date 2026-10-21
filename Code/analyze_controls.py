"""Paired analysis of the control experiments (exp_controls.py).

Same statistics as the paper for experiment A: per-model mean +/- std, paired
difference in percentage points with a 95% percentile bootstrap CI (5,000
resamples), Cohen's d_z, runs won, and the exact Wilcoxon signed-rank p-value
(enumeration of all sign patterns, so tied differences do not force a normal approximation).

It also checks that the controls are paired with A: SV150 must reproduce A's
SingleView run by run, and MVS_LR or MVS_RF must reproduce A's MultiViewStacking.

    python Code/analyze_controls.py
    python Code/analyze_controls.py --a-csv Analysis/comparing/multiview_vs_singleview_per_run.csv
Writes Analysis/controls/controls_comparisons.csv and prints a Markdown summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "Analysis" / "controls"
DEFAULT_A = ROOT / "Analysis" / "comparing" / "multiview_vs_singleview_per_run.csv"

# (model, reference, question it answers)
COMPARISONS = [
    # Random Forest meta-learner: the configuration that reproduces experiment A
    ("MVS_RF", "SV150", "main result (reproduces experiment A)"),
    ("MVS_RF", "SV600", "capacity: does MVS still win against a 600-tree SingleView?"),
    ("MVS_RF", "MVS_RF_RANDVIEWS", "semantics: do semantic views beat random views of equal size?"),
    ("MVS_RF_RANDVIEWS", "SV150", "does a random partition alone already beat SingleView?"),
    ("MVS_RF_FOLDSMOTE", "SV150", "leak-free stacking vs SingleView"),
    ("MVS_RF_FOLDSMOTE", "MVS_RF", "effect of moving SMOTE inside the folds"),
    # Logistic meta-learner: what the paper currently describes
    ("MVS_LR", "SV150", "logistic meta-learner vs SingleView"),
    ("MVS_LR", "MVS_RF", "logistic vs Random Forest meta-learner"),
    ("MVS_LR", "SV600", "capacity, logistic meta-learner"),
    ("MVS_LR", "MVS_LR_RANDVIEWS", "semantics, logistic meta-learner"),
    ("MVS_LR_FOLDSMOTE", "MVS_LR", "SMOTE inside the folds, logistic meta-learner"),
    ("MVS_LR_FOLDSMOTE", "SV150", "leak-free stacking vs SingleView, logistic meta-learner"),
]
METRICS = ["accuracy", "f1_macro"]
N_BOOT = 5000
BOOT_SEED = 2026


def md(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table if `tabulate` is installed, plain text otherwise."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return df.to_string(index=index)


def exact_wilcoxon_p(d: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value by enumerating every sign pattern (2^n), using
    mid-ranks for tied |d|. Exact even with ties (scipy falls back to the normal approximation
    when |d| has ties, which gave 8.8e-5 instead of 1.9e-6 for a 20/20 comparison).
    Zero differences are dropped, as in scipy's default. Falls back to scipy for n > 22."""
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    if n > 22:
        return float(wilcoxon(d).pvalue)
    r = rankdata(np.abs(d))
    obs = abs(float((r * np.sign(d)).sum()))
    signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    return float((np.abs(signs @ r) >= obs - 1e-9).mean())


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = (a - b) * 100.0                                   # percentage points
    rng = np.random.default_rng(BOOT_SEED)
    boots = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sd = d.std(ddof=1)
    p = exact_wilcoxon_p(a - b)
    return {"mean_diff_pp": d.mean(), "ci95_lo": lo, "ci95_hi": hi,
            "d_z": d.mean() / sd if sd > 0 else np.nan,
            "wins": int((d > 0).sum()), "n": len(d), "min_diff_pp": d.min(), "wilcoxon_p": p}


def reproduction_check(ctrl: pd.DataFrame, a_csv: Path) -> None:
    if not a_csv.exists():
        print(f"(A's CSV not found at {a_csv}; skipping the reproduction check)\n")
        return
    a = pd.read_csv(a_csv)
    print("## Pairing with experiment A\n")
    for ours, theirs in (("SV150", "SingleView"), ("MVS_LR", "MultiViewStacking"),
                         ("MVS_RF", "MultiViewStacking")):
        x = ctrl[ctrl.model == ours].set_index("run")
        y = a[a.model == theirs].set_index("run")
        runs = x.index.intersection(y.index)
        if len(runs) == 0:
            continue
        diffs = (x.loc[runs, "accuracy"] - y.loc[runs, "accuracy"]).abs()
        gap = diffs.max()
        same = int((diffs < 1e-12).sum())
        verdict = ("IDENTICAL" if gap < 1e-12 else "matches" if gap < 1e-4 else
                   "near (< 5e-4, numerical noise?)" if gap < 5e-4 else "DIFFERENT")
        print(f"- {ours:<8} vs A's {theirs:<18} over {len(runs)} runs: bit-identical in {same}/{len(runs)}, "
              f"max |diff| = {gap:.2e} -> {verdict}")
    print("\nSV150 must be IDENTICAL; otherwise the controls are not paired with A "
          "(check library versions and the dataset file).\n"
          "Whichever of MVS_LR / MVS_RF is IDENTICAL is the meta-learner A really used.\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--a-csv", type=Path, default=DEFAULT_A)
    args = ap.parse_args(argv)

    ctrl = pd.read_csv(args.dir / "controls_per_run.csv")
    reproduction_check(ctrl, args.a_csv)

    print("## Per model (mean +/- std over runs, %)\n")
    summ = ctrl.groupby("model").agg(
        runs=("run", "nunique"),
        acc=("accuracy", "mean"), acc_sd=("accuracy", "std"),
        f1=("f1_macro", "mean"), f1_sd=("f1_macro", "std"),
        fit_s=("fit_seconds", "mean"), pred_us_per_flow=("predict_seconds_per_flow", "mean"))
    summ["pred_us_per_flow"] *= 1e6
    for c in ("acc", "acc_sd", "f1", "f1_sd"):
        summ[c] *= 100
    print(md(summ.round(2)))

    rows = []
    for model, ref, question in COMPARISONS:
        x = ctrl[ctrl.model == model].set_index("run")
        y = ctrl[ctrl.model == ref].set_index("run")
        runs = x.index.intersection(y.index).sort_values()
        if len(runs) < 2:
            continue
        for m in METRICS:
            s = paired_stats(x.loc[runs, m].to_numpy(), y.loc[runs, m].to_numpy())
            rows.append({"model": model, "reference": ref, "metric": m, "question": question, **s})
    if not rows:
        print("\nNo complete pairs yet.")
        return 1
    out = pd.DataFrame(rows)
    out.to_csv(args.dir / "controls_comparisons.csv", index=False)

    print("\n## Paired comparisons (model minus reference, pp)\n")
    show = out.copy()
    show["CI95"] = show.apply(lambda r: f"[{r.ci95_lo:.2f}, {r.ci95_hi:.2f}]", axis=1)
    show["wins"] = show.apply(lambda r: f"{r.wins}/{r.n}", axis=1)
    print(show[["model", "reference", "metric", "mean_diff_pp", "CI95", "d_z", "wins", "wilcoxon_p"]]
          .round({"mean_diff_pp": 3, "d_z": 3}).pipe(md, index=False))
    print(f"\nSaved {args.dir / 'controls_comparisons.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
