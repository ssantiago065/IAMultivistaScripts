"""Accuracy and F1-Macro re-weighted to the natural class prevalence of the flow-based subset.

Under a change of class prevalence (label shift) each row of the confusion matrix keeps its
conditional distribution P(pred | true); the matrix is re-weighted so that row c sums to pi_c, and
accuracy and F1-Macro are computed from it. This assumes that the errors measured on the sampled
flows of a class hold for the whole class. Class shares are those of Table 1 (data/class_count.txt,
27,809,999 flows).

- Accuracy from exp. A: data/multiview_vs_singleview_per_class.csv (accuracy = sum_c pi_c recall_c).
- Accuracy and F1-Macro from the confusion matrices of the control runs, which share A's 20
  partitions: data/controls_confusion.csv (SV150 = SingleView, MVS_RF = MultiView Stacking).

    python scripts/prevalence_accuracy.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
COUNTS = {"DoS": 23_143_924, "DDoS": 3_478_814, "Recon": 442_158, "Benign": 398_330,
          "Mirai": 174_588, "Spoofing": 157_238, "Web-Based": 11_328, "BruteForce": 3_619}
TOTAL = sum(COUNTS.values())
PI = {k: v / TOTAL for k, v in COUNTS.items()}


def reweighted_metrics(cm: pd.DataFrame) -> tuple[float, float]:
    """cm: square confusion matrix (rows true, columns predicted) indexed by class name."""
    classes = list(cm.index)
    m = cm.loc[classes, classes].to_numpy(dtype=float)
    rows = m / m.sum(axis=1, keepdims=True)                 # P(pred | true)
    w = rows * np.array([PI[c] for c in classes])[:, None]  # joint P(true, pred) at natural prevalence
    tp = np.diag(w)
    prec = np.divide(tp, w.sum(axis=0), out=np.zeros_like(tp), where=w.sum(axis=0) > 0)
    rec = tp / w.sum(axis=1)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return tp.sum() * 100, f1.mean() * 100


def summary(name: str, vals: np.ndarray) -> str:
    return f"{name:<34} {vals.mean():.2f} +/- {vals.std(ddof=1):.2f}"


def main() -> None:
    d = pd.read_csv(DATA / "multiview_vs_singleview_per_class.csv")
    d["pi"] = d["class"].map(PI)
    acc_a = {m: d[d.model == m].groupby("run").apply(lambda g: (g.pi * g.recall).sum() * 100,
                                                     include_groups=False).to_numpy()
             for m in ("MultiViewStacking", "SingleView")}
    print("## Experiment A (per-class recall)")
    for m, a in acc_a.items():
        print(summary(f"{m} accuracy", a))
    diff = acc_a["MultiViewStacking"] - acc_a["SingleView"]
    print(f"paired accuracy difference {diff.mean():.3f} pp, MVS ahead in {(diff > 0).sum()}/{len(diff)}")

    cm = pd.read_csv(DATA / "controls_confusion.csv")
    print("\n## Control runs (confusion matrices, same partitions as A)")
    res = {}
    for model in ("SV150", "MVS_RF"):
        acc, f1 = [], []
        for _, g in cm[cm.model == model].groupby("run"):
            a, f = reweighted_metrics(g.pivot(index="true", columns="pred", values="count").fillna(0))
            acc.append(a)
            f1.append(f)
        res[model] = (np.array(acc), np.array(f1))
        print(summary(f"{model} accuracy", res[model][0]))
        print(summary(f"{model} F1-Macro", res[model][1]))
    for i, name in ((0, "accuracy"), (1, "F1-Macro")):
        dd = res["MVS_RF"][i] - res["SV150"][i]
        print(f"paired {name} difference {dd.mean():.3f} pp, MVS ahead in {(dd > 0).sum()}/{len(dd)}")


if __name__ == "__main__":
    main()
