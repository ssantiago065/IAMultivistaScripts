#!/usr/bin/env python3
"""Main results of the paper under the fold-level oversampling protocol.

Since the IEEE version, the paper's main configuration is MultiView Stacking with SMOTE applied
inside each internal stacking fold (model MVS_RF_FOLDSMOTE of Code/exp_controls.py), compared with a
150-tree SingleView (SV150) on the same 20 partitions and seeds. The original protocol (SMOTE before
the folds, MVS_RF, which reproduces experiment A) is kept as a sensitivity analysis and is the one used
by the capacity and random-view controls.

Everything is computed from paper/data/controls_*.csv; nothing is retrained.

Statistics per paired comparison (all differences in percentage points, MVS minus reference):
  * mean +/- std of each model, mean paired difference, smallest margin, runs won
  * 95% percentile bootstrap CI of the mean difference (5,000 resamples, seed 2026)
  * Cohen's d_z
  * exact Wilcoxon signed-rank p (enumeration of all 2^n sign patterns, mid-ranks for ties)
  * corrected resampled t-test (Nadeau and Bengio, 2003): the 20 partitions share data, so the
    variance of the mean difference is inflated by (1/n + n_test/n_train) with n_test/n_train = 0.3/0.7

It also reports per-class F1/precision/recall (with Holm-corrected exact Wilcoxon), metrics at the
natural class prevalence of the flow-based subset, and the confusion cells discussed in the paper.

    python paper/scripts/main_results.py            # prints Markdown tables
    python paper/scripts/main_results.py --out paper/data/main_results_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t, wilcoxon

DATA = Path(__file__).resolve().parent.parent / "data"
MAIN, BASE = "MVS_RF_FOLDSMOTE", "SV150"
TEST_TRAIN_RATIO = 0.30 / 0.70
N_BOOT, BOOT_SEED = 5000, 2026
COUNTS = {"DoS": 23_143_924, "DDoS": 3_478_814, "Recon": 442_158, "Benign": 398_330,
          "Mirai": 174_588, "Spoofing": 157_238, "Web-Based": 11_328, "BruteForce": 3_619}
PI = {k: v / sum(COUNTS.values()) for k, v in COUNTS.items()}
COMPARISONS = [  # (model, reference, role in the paper)
    (MAIN, BASE, "main result"),
    ("MVS_RF", BASE, "original protocol (SMOTE before folds)"),
    (MAIN, "MVS_RF", "effect of fold-level SMOTE"),
    ("MVS_RF", "SV600", "capacity control (original protocol)"),
    ("MVS_RF", "MVS_RF_RANDVIEWS", "semantic vs random views (original protocol)"),
    ("MVS_RF_RANDVIEWS", BASE, "random views vs SingleView (original protocol)"),
    ("MVS_LR", BASE, "logistic meta-learner"),
    ("MVS_LR_FOLDSMOTE", BASE, "logistic meta-learner, fold-level SMOTE"),
]


def exact_wilcoxon_p(d: np.ndarray) -> float:
    """Two-sided exact Wilcoxon signed-rank p by enumerating all sign patterns (same as
    Code/analyze_controls.py)."""
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


def nadeau_bengio(d: np.ndarray) -> tuple[float, float]:
    """Corrected resampled t-test (Nadeau and Bengio, 2003) for repeated random train/test splits."""
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return np.inf, 0.0
    t = d.mean() / np.sqrt((1.0 / n + TEST_TRAIN_RATIO) * var)
    return float(t), float(2 * student_t.sf(abs(t), n - 1))


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).to_numpy() * 100.0
    boots = np.random.default_rng(BOOT_SEED).choice(d, size=(N_BOOT, len(d))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    t, p_nb = nadeau_bengio(d)
    return {"model_mean": a.mean() * 100, "model_sd": a.std() * 100,
            "ref_mean": b.mean() * 100, "ref_sd": b.std() * 100,
            "mean_diff_pp": d.mean(), "ci95_lo": lo, "ci95_hi": hi, "min_diff_pp": d.min(),
            "d_z": d.mean() / d.std(ddof=1), "wins": int((d > 0).sum()), "n": len(d),
            "wilcoxon_p": exact_wilcoxon_p(d), "nb_t": t, "nb_p": p_nb}


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m, running, out = len(p), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        out[k] = running
    return pd.Series(out)[p.index]


def at_natural_prevalence(cm: pd.DataFrame) -> tuple[float, float]:
    classes = list(cm.index)
    m = cm.loc[classes, classes].to_numpy(float)
    w = m / m.sum(axis=1, keepdims=True) * np.array([PI[c] for c in classes])[:, None]
    tp = np.diag(w)
    prec = np.divide(tp, w.sum(axis=0), out=np.zeros_like(tp), where=w.sum(axis=0) > 0)
    rec = tp / w.sum(axis=1)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return tp.sum() * 100, f1.mean() * 100


def md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:
        return df.to_string()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="write the paired comparisons to this CSV")
    args = ap.parse_args(argv)

    runs = pd.read_csv(DATA / "controls_per_run.csv")
    per_class = pd.read_csv(DATA / "controls_per_class.csv")
    conf = pd.read_csv(DATA / "controls_confusion.csv")

    rows = []
    for model, ref, role in COMPARISONS:
        x = runs[runs.model == model].set_index("run").sort_index()
        y = runs[runs.model == ref].set_index("run").sort_index()
        for metric in ("accuracy", "f1_macro"):
            rows.append({"model": model, "reference": ref, "role": role, "metric": metric,
                         **paired(x[metric], y[metric])})
    comp = pd.DataFrame(rows)
    print("## Paired comparisons (pp; p_nb = Nadeau-Bengio corrected t-test)\n")
    print(md(comp.set_index(["model", "reference", "metric"]).drop(columns="role").round(4)))
    if args.out:
        comp.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    print(f"\n## Per class, {MAIN} vs {BASE} (mean over runs, %)\n")
    pc = per_class[per_class.model.isin([MAIN, BASE])]
    rows = []
    for cls, g in pc.groupby("class"):
        a = g[g.model == MAIN].set_index("run").sort_index()
        b = g[g.model == BASE].set_index("run").sort_index()
        d = (a["f1-score"] - b["f1-score"]).to_numpy()
        rows.append({"class": cls, "support": a["support"].mean(),
                     "f1_mvs": a["f1-score"].mean() * 100, "f1_sv": b["f1-score"].mean() * 100,
                     "delta_pp": d.mean() * 100, "wins": int((d > 0).sum()),
                     "p": exact_wilcoxon_p(d),
                     "prec_mvs": a.precision.mean() * 100, "prec_mvs_sd": a.precision.std() * 100,
                     "prec_sv": b.precision.mean() * 100, "prec_sv_sd": b.precision.std() * 100,
                     "rec_mvs": a.recall.mean() * 100, "rec_mvs_sd": a.recall.std() * 100,
                     "rec_sv": b.recall.mean() * 100, "rec_sv_sd": b.recall.std() * 100})
    cls_df = pd.DataFrame(rows).set_index("class").sort_values("delta_pp", ascending=False)
    cls_df["p_holm"] = holm(cls_df["p"])
    print(md(cls_df.round(2)))

    print("\n## Natural class prevalence (confusion rows re-weighted to the full-subset shares)\n")
    nat = {}
    for model in (MAIN, BASE, "MVS_RF"):
        vals = []
        for _, g in conf[conf.model == model].groupby("run"):
            cm = g.pivot_table(index="true", columns="pred", values="count", fill_value=0)
            vals.append(at_natural_prevalence(cm))
        nat[model] = np.array(vals)
        print(f"- {model:<17} accuracy {nat[model][:, 0].mean():.2f} +/- {nat[model][:, 0].std(ddof=1):.2f}, "
              f"F1-Macro {nat[model][:, 1].mean():.2f} +/- {nat[model][:, 1].std(ddof=1):.2f}")
    for k, name in ((0, "accuracy"), (1, "F1-Macro")):
        d = nat[MAIN][:, k] - nat[BASE][:, k]
        print(f"  {MAIN} - {BASE} {name}: mean paired difference {d.mean():+.2f} pp, won {(d > 0).sum()}/{len(d)}")

    print("\n## Confusion cells discussed in the paper (mean flows per run)\n")
    for model in (BASE, "MVS_RF", MAIN):
        cm = conf[conf.model == model].groupby(["true", "pred"])["count"].sum().unstack(fill_value=0)
        cm = cm / conf[conf.model == model].run.nunique()
        bf_fp = cm["BruteForce"].sum() - cm.loc["BruteForce", "BruteForce"]
        print(f"- {model:<17} Spoofing->BruteForce {cm.loc['Spoofing', 'BruteForce']:.0f}, "
              f"BruteForce false alarms {bf_fp:.0f}, BruteForce->Spoofing {cm.loc['BruteForce', 'Spoofing']:.0f}, "
              f"Spoofing correct {cm.loc['Spoofing', 'Spoofing']:.0f}, "
              f"Recon->Benign {cm.loc['Recon', 'Benign']:.0f}, Benign->Recon {cm.loc['Benign', 'Recon']:.0f}, "
              f"BruteForce correct {cm.loc['BruteForce', 'BruteForce']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
