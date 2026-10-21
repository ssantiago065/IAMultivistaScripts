#!/usr/bin/env python3
"""Regenerate every results figure of the paper from the CSVs in data/.

Style rules enforced here (see CLAUDE.md and plan_estilo.md section 6.2):
  * English labels, no title inside the image -- the LaTeX caption replaces it
  * figures are drawn at their final printed size for the LNCS text width (122 mm):
    FULL = 4.8 in wide, HALF = 2.35 in wide, so fonts are not shrunk by LaTeX
  * 8-9 pt text, one font family, light grid on the value axis only
  * colour-blind-safe Okabe-Ito palette with a fixed meaning across the paper
    (SingleView blue, MVS orange -- the main configuration, Random Forest meta-learner --,
    MVS with a logistic meta-learner vermillion, one colour per view),
    plus hatches/markers so every figure also reads in grayscale
  * vector PDF output into figures/
  * Spanish view names from the CSVs are renamed to English

Usage:
    python scripts/make_figures.py            # build everything that has data
    python scripts/make_figures.py --list     # report which inputs are present

Missing inputs are skipped with a message instead of aborting, so the script is
usable while the team is still uploading CSVs.

Metrics are stored as fractions in the CSVs and reported as percentages in the
paper, so every plotted value goes through pct().
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    PLOTTING_OK = True
except ImportError as _exc:  # pragma: no cover
    PLOTTING_OK = False
    MISSING = _exc.name

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"

# --- shared style -----------------------------------------------------------

FULL = 4.8   # inches, full LNCS text width (122 mm)
HALF = 2.35  # inches, one of two side-by-side panels

# Okabe-Ito palette; a colour never means two different things across the paper
COLOR_SV = "#0072B2"      # SingleView -- blue
COLOR_MVS = "#E69F00"     # MultiView Stacking, main configuration (Random Forest meta) -- orange
COLOR_MVS_LR = "#D55E00"  # MultiView Stacking variant with a logistic meta-learner -- vermillion
COLOR_NEUTRAL = "#999999"  # plots with no model identity (class distribution)
VIEW_COLORS = {"Volume": "#009E73", "Time": "#CC79A7",
               "Topology": "#56B4E9", "Flags": "#F0E442"}
VIEW_HATCHES = {"Volume": "", "Time": "////", "Topology": "....", "Flags": "xxxx"}


def pct(values):
    """Metrics are stored as fractions in the CSVs; the paper reports percent."""
    arr = np.asarray(values, dtype=float)
    return arr * 100 if np.nanmax(arr) <= 1.5 else arr


if PLOTTING_OK:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,
    })

# CSV label -> label used in the paper
MODEL_NAMES = {
    "SingleView": "SingleView",
    "singleview": "SingleView",
    "SV": "SingleView",
    "MultiView": "MultiView Stacking",
    "MultiViewStacking": "MultiView Stacking",
    "MVS": "MultiView Stacking",
    "Vista_Tiempo": "Time",
    "Vista_Volumen": "Volume",
    "Vista_Banderas": "Flags",
    "Vista_Topologia": "Topology",
}

# CSV class label -> label used in the paper's tables
CLASS_NAMES = {"BruteForce": "Brute Force", "Brute_Force": "Brute Force",
               "WebBased": "Web-Based", "Web_Based": "Web-Based"}

MODEL_CANDIDATES = ["model", "modelo", "architecture", "arquitectura", "method"]
ACC_CANDIDATES = ["accuracy", "acc", "exactitud"]
# macro F1 of a whole run
F1_CANDIDATES = ["f1_macro", "f1macro", "macro_f1", "f1_macro_score"]
# F1 of a single class, as sklearn's classification_report writes it
F1_CLASS_CANDIDATES = ["f1-score", "f1_score", "f1score", "f1"]
CLASS_CANDIDATES = ["class", "clase", "label", "etiqueta"]
SIZE_CANDIDATES = ["train_size_pct", "pct_train", "porcentaje", "train_size",
                   "size", "n_train"]


def pick(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    raise KeyError(
        f"no usable {what} column; looked for {candidates}, "
        f"file has {list(df.columns)}"
    )


def load(name: str) -> pd.DataFrame | None:
    path = DATA / name
    if not path.exists():
        print(f"  skip: data/{name} not found")
        return None
    df = pd.read_csv(path)
    lowered = {c.lower() for c in df.columns}
    if "model" in lowered:
        col = pick(df, MODEL_CANDIDATES, "model")
        df[col] = df[col].replace(MODEL_NAMES)
    if "class" in lowered:
        col = pick(df, CLASS_CANDIDATES, "class")
        df[col] = df[col].replace(CLASS_NAMES)
    return df


def value_grid(ax, axis: str) -> None:
    """Light grid on the value axis only."""
    ax.grid(False)
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    FIGS.mkdir(exist_ok=True)
    out = FIGS / name
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def boxplot_with_points(ax, series, colors, labels) -> None:
    """Boxes coloured by model, with every run drawn as a dot on top."""
    bp = ax.boxplot([pct(s) for s in series], patch_artist=True, widths=0.55,
                    medianprops={"color": "black", "linewidth": 0.9},
                    boxprops={"linewidth": 0.7}, whiskerprops={"linewidth": 0.7},
                    capprops={"linewidth": 0.7}, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    rng = np.random.default_rng(0)  # fixed jitter so the PDF is reproducible
    for i, s in enumerate(series, start=1):
        x = i + rng.uniform(-0.12, 0.12, len(s))
        ax.scatter(x, pct(s), s=6, color="black", alpha=0.6, zorder=3, linewidths=0)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    value_grid(ax, "y")


# --- fig:global-boxplot (Section 7.1, experiment A) -------------------------

def global_boxplot() -> None:
    print("fig:global-boxplot")
    df = load("multiview_vs_singleview_per_run.csv")
    if df is None:
        return
    mcol = pick(df, MODEL_CANDIDATES, "model")
    pair = ["SingleView", "MultiView Stacking"]
    sub = df[df[mcol].isin(pair)]
    if sub.empty:
        print("  skip: no SingleView/MVS rows")
        return
    for metric_cands, fname, ylabel in (
        (ACC_CANDIDATES, "global_boxplot_accuracy.pdf", "Accuracy (%)"),
        (F1_CANDIDATES, "global_boxplot_f1.pdf", "F1-Macro (%)"),
    ):
        col = pick(sub, metric_cands, "metric")
        data = [sub.loc[sub[mcol] == m, col].to_numpy() for m in pair]
        fig, ax = plt.subplots(figsize=(HALF, 2.1))
        boxplot_with_points(ax, data, (COLOR_SV, COLOR_MVS), ["SingleView", "MVS"])
        ax.set_ylabel(ylabel)
        save(fig, fname)


# --- fig:view-ablation (Section 7.3, experiment A) --------------------------

def view_ablation() -> None:
    """Per-class F1 of each view on its own (the per-model accuracies are in Table 4)."""
    print("fig:view-ablation")
    dfc = load("multiview_vs_singleview_per_class.csv")
    if dfc is None:
        return
    mcol = pick(dfc, MODEL_CANDIDATES, "model")
    ccol = pick(dfc, CLASS_CANDIDATES, "class")
    fcol = pick(dfc, F1_CLASS_CANDIDATES, "F1")
    views = ["Volume", "Time", "Topology", "Flags"]
    sub = dfc[dfc[mcol].isin(views)]
    if sub.empty:
        print("  skip: no per-view rows in the per-class file")
        return
    wide = sub.groupby([ccol, mcol])[fcol].mean().unstack()
    wide = wide * (100 if wide.to_numpy().max() <= 1.5 else 1)
    cols = [v for v in views if v in wide.columns]
    fig, ax = plt.subplots(figsize=(FULL, 2.5))
    x = np.arange(len(wide.index))
    width = 0.8 / len(cols)
    for i, v in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * width, wide[v], width=width,
               color=VIEW_COLORS[v], hatch=VIEW_HATCHES[v], edgecolor="black",
               linewidth=0.4, label=v)
    # two-word class names on two lines so the labels stay horizontal
    ax.set_xticks(x, [str(c).replace(" ", "\n").replace("Web-Based", "Web-\nBased")
                      for c in wide.index])
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    value_grid(ax, "y")
    ax.legend(frameon=False, ncol=4, loc="upper center", handlelength=1.6,
              columnspacing=1.2)
    save(fig, "view_ablation_per_class.pdf")


# --- fig:learning-curve (Section 7.4, experiment C) -------------------------

def learning_curve() -> None:
    print("fig:learning-curve")
    df = load("learning_curve_per_run.csv")
    if df is None:
        return
    mcol = pick(df, MODEL_CANDIDATES, "model")
    scol = pick(df, SIZE_CANDIDATES, "training size")
    fcol = pick(df, F1_CANDIDATES, "F1")
    fig, ax = plt.subplots(figsize=(FULL, 2.5))
    for model, color, marker, ls in (("SingleView", COLOR_SV, "s", "--"),
                                     ("MultiView Stacking", COLOR_MVS, "o", "-")):
        sub = df[df[mcol] == model]
        if sub.empty:
            continue
        grp = sub.groupby(scol)[fcol]
        mean, std = grp.mean(), grp.std()
        # one scale factor for both, decided by the mean (fractions -> percent)
        scale = 100.0 if mean.max() <= 1.5 else 1.0
        x = mean.index.to_numpy()
        m, s = mean.to_numpy() * scale, std.to_numpy() * scale
        ax.plot(x, m, marker=marker, markersize=3.5, linestyle=ls, linewidth=1.1,
                color=color, label=model)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, linewidth=0)
    ax.set_xlabel("Training pool used (%)")
    ax.set_ylabel("F1-Macro (%)")
    value_grid(ax, "y")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "learning_curve.pdf")


# --- fig:class-dist (Section 3.2) -------------------------------------------

def class_distribution() -> None:
    print("fig:class-dist")
    path = DATA / "class_count.txt"
    if not path.exists():
        print("  skip: data/class_count.txt not found")
        return
    # class_count.txt is the console dump of the auditing script. Read only the
    # "RESUMEN POR CATEGORIAS" block: lines of "<name> <count> <pct>%".
    text = path.read_text(encoding="utf-8", errors="replace")
    body = text.split("CATEGOR", 1)[-1].split("SUBCATEGOR", 1)[0]
    row_re = re.compile(r"^\s*(\S.*?)\s+(\d[\d,]*)\s+([\d.]+)\s*%\s*$")
    rows = []
    for line in body.splitlines():
        m = row_re.match(line)
        if m:
            rows.append((m.group(1).strip(), int(m.group(2).replace(",", ""))))
    if not rows:
        print("  skip: could not parse the category block of class_count.txt")
        return
    total = sum(c for _, c in rows)
    print(f"  parsed {len(rows)} categories, total {total:,}")
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(FULL, 2.3))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=COLOR_NEUTRAL)
    ax.set_xscale("log")
    ax.set_xlabel("Instances (log scale)")
    value_grid(ax, "x")
    save(fig, "class_distribution.pdf")


# --- fig:balanced-only (Section 7.5, balanced six-class pool) ---------------

def balanced_only() -> None:
    """Accuracy per run on the balanced six-class pool (no oversampling).

    balanced_lr_per_run.csv: branch exp/undersampling-100k-6classes (logistic meta)
    balanced_rf_per_run.csv: branch exp/base-rf-meta-rf (Random Forest meta)
    Both runs share the same 20 partitions, so their SingleView rows must be
    identical run by run; if not, the comparison is invalid and we stop.
    """
    print("fig:balanced-only")
    lr = load("balanced_lr_per_run.csv")
    rf = load("balanced_rf_per_run.csv")
    if lr is None or rf is None:
        return

    def series(df, model):
        mcol = pick(df, MODEL_CANDIDATES, "model")
        acol = pick(df, ACC_CANDIDATES, "accuracy")
        return df[df[mcol] == model].set_index("run")[acol].sort_index()

    sv_lr, sv_rf = series(lr, "SingleView"), series(rf, "SingleView")
    if not sv_lr.index.equals(sv_rf.index) or not (sv_lr == sv_rf).all():
        raise RuntimeError(
            "SingleView differs between balanced_lr_per_run.csv and "
            "balanced_rf_per_run.csv: the runs are not paired, aborting")
    print(f"  SingleView identical in both files over {len(sv_lr)} runs")

    data = [sv_lr.to_numpy(), series(rf, "MultiView Stacking").to_numpy(),
            series(lr, "MultiView Stacking").to_numpy()]
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    boxplot_with_points(ax, data, (COLOR_SV, COLOR_MVS, COLOR_MVS_LR),
                        ["SingleView", "MVS, RF\nmeta-learner",
                         "MVS, logistic\nmeta-learner"])
    ax.set_ylabel("Accuracy (%)")
    save(fig, "balanced_only.pdf")


# class_distribution is kept but not built: Table 1 replaced that figure (plan_estilo.md, cut 2)
BUILDERS = [global_boxplot, view_ablation, learning_curve, balanced_only]

EXPECTED = [
    "multiview_vs_singleview_per_run.csv",
    "multiview_vs_singleview_per_class.csv",
    "learning_curve_per_run.csv",
    "class_count.txt",
    "balanced_lr_per_run.csv",
    "balanced_rf_per_run.csv",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                    help="report which expected inputs are present and exit")
    args = ap.parse_args()

    if args.list:
        for name in EXPECTED:
            mark = "present" if (DATA / name).exists() else "MISSING"
            print(f"{mark:>8}  data/{name}")
        return 0

    if not PLOTTING_OK:
        print(f"missing dependency: {MISSING}", file=sys.stderr)
        print("install the plotting stack first:  pip install pandas matplotlib",
              file=sys.stderr)
        return 1

    if not DATA.exists() or not any(DATA.iterdir()):
        print("data/ is empty -- nothing to plot.")
        print("Upload the experiment CSVs first; run with --list to check.")
        return 1

    for build in BUILDERS:
        try:
            build()
        except KeyError as exc:
            print(f"  skip: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
