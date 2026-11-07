"""Round-3 experiments for the MultiView Stacking paper (plan of 2026-11-03).

Same 20 seeds, partitions and preprocessing as experiment A and the controls
(Code/exp_controls.py), so every number is paired with paper/data/controls_*.csv.
The main model, MVS_RF_FOLDSMOTE, is refitted here and must reproduce the controls
file run by run; that check guards the whole round.

Models (plan step in brackets):
  MAIN          MVS_RF_FOLDSMOTE, the paper's main configuration (reference)
  SV69          SingleView, 150 trees, on the 69 view features only        [1]
  RANDV_FS      random views of the semantic sizes, SMOTE inside the folds  [2]
  LOVO_<VIEW>   main protocol without the time / volume / flags / topology view [3]
  LGBM          LightGBM on all 70 features, SMOTE as for SingleView      [4]
  STACK_HET     classic stacking: RF, ExtraTrees, LightGBM and logistic regression,
                each on all 70 features, RF meta-learner, SMOTE inside the folds [4]
  META_BAL      main protocol, meta-learner with class_weight="balanced"   [5]
  HYB_FULL      hybrid: the four semantic views plus a fifth view with all 70
                features, RF learners, RF meta-learner, SMOTE inside the folds (v3 follow-up)
  HYB_RAND      control for HYB_FULL: random views of the semantic sizes plus the
                full-feature view

Diagnostics written for MAIN and RANDV_FS:
  round3_diversity.csv    pairwise disagreement and Q-statistic of the base learners
                          on the test partition                              [6]
  round3_view_importance.csv  per-class recall drop when one view's base-learner
                          output is permuted before the meta-learner         [7]
  scores/run_XX_<MODEL>.npz  Brute Force scores of MAIN and META_BAL (PR curve) [5]
fit_seconds is measured with one process on the machine                   [11]

    python Code/exp_round3.py --dataset H:/Dataset/dataset_expA_8clases.csv
    python Code/exp_round3.py --runs 1 --models MAIN SV69   # smoke test
Results are appended run by run; an interrupted job resumes where it stopped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

import exp_controls as ec

ec.DETERMINISTIC_PROBA = True

OUTPUT_DIR = ec.PROJECT_ROOT / "Analysis" / "round3"
VIEW_NAMES = ["time", "volume", "flags", "topology"]
LOVO = [f"LOVO_{n.upper()}" for n in VIEW_NAMES]
ALL_MODELS = ["MAIN", "SV69", "RANDV_FS", *LOVO, "LGBM", "STACK_HET", "META_BAL", "HYB_FULL", "HYB_RAND"]
DIAGNOSED = ("MAIN", "RANDV_FS")
N_PERMUTATIONS = 3


def lgbm(seed):
    """min_child_samples=100 and reg_lambda=1 keep the multiclass boosting from diverging on the
    SMOTE-balanced data: with the defaults (20, 0) training log-loss blew up to about 16 in 5 of 20
    seeds (test F1-macro 0.29-0.60); with these values all 7 seeds checked stay near 0.25 and
    F1-macro 0.814-0.820 (2026-11-04)."""
    return LGBMClassifier(n_estimators=300, random_state=seed, n_jobs=ec.N_JOBS, verbose=-1,
                          deterministic=True, force_col_wise=True, min_child_samples=100,
                          reg_lambda=1.0)


class OrderedExtraTrees(ec.OrderedProbaMixin, ExtraTreesClassifier):
    pass


def fold_mvs(views, learners, meta, seed, n_real):
    return ec.FoldResampledMVS(views_indices=[list(v) for v in views], first_level_learners=learners,
                               meta_learner=meta, k=ec.KFOLD, random_state=seed,
                               resampler=ec.ProportionalSMOTE(n_real, seed))


def build(model_name, data, seed, views):
    """Returns (model, X_train, y_train)."""
    n_real = len(data["y_real"])
    all_cols = list(range(data["X_real"].shape[1]))
    rfs = lambda vs: [ec.rf(ec.RF_ESTIMATORS, seed) for _ in vs]
    if model_name == "MAIN":
        return fold_mvs(views, rfs(views), ec.make_meta("RF", seed), seed, n_real), "real"
    if model_name == "META_BAL":
        meta = ec.OrderedRandomForest(n_estimators=ec.RF_ESTIMATORS, random_state=seed,
                                      n_jobs=ec.N_JOBS, class_weight="balanced")
        return fold_mvs(views, rfs(views), meta, seed, n_real), "real"
    if model_name == "RANDV_FS":
        rv = ec.random_views(views, seed)
        return fold_mvs(rv, rfs(rv), ec.make_meta("RF", seed), seed, n_real), "real"
    if model_name in LOVO:
        kept = [v for v, n in zip(views, VIEW_NAMES) if f"LOVO_{n.upper()}" != model_name]
        return fold_mvs(kept, rfs(kept), ec.make_meta("RF", seed), seed, n_real), "real"
    if model_name == "SV69":
        return ColumnSubset(ec.rf(150, seed), sorted(c for v in views for c in v)), "bal"
    if model_name == "LGBM":
        return lgbm(seed), "bal"
    if model_name == "HYB_FULL":
        hv = [*views, all_cols]
        return fold_mvs(hv, rfs(hv), ec.make_meta("RF", seed), seed, n_real), "real"
    if model_name == "HYB_RAND":
        hv = [*ec.random_views(views, seed), all_cols]
        return fold_mvs(hv, rfs(hv), ec.make_meta("RF", seed), seed, n_real), "real"
    if model_name == "STACK_HET":
        learners = [ec.rf(ec.RF_ESTIMATORS, seed),
                    OrderedExtraTrees(n_estimators=ec.RF_ESTIMATORS, random_state=seed,
                                         n_jobs=ec.N_JOBS),
                    lgbm(seed), LogisticRegression(max_iter=1000, random_state=seed)]
        return fold_mvs([all_cols] * 4, learners, ec.make_meta("RF", seed), seed, n_real), "real"
    raise ValueError(model_name)


class ColumnSubset:
    """Fits and predicts an estimator on a fixed subset of columns."""

    def __init__(self, est, cols):
        self.est, self.cols = est, cols

    def fit(self, X, y):
        self.est.fit(X[:, self.cols], y)
        return self

    def predict(self, X):
        return self.est.predict(X[:, self.cols])


# =====================================================================
# Diagnostics on a fitted MultiViewStacking
# =====================================================================
def view_outputs(model, X):
    """Per-view base-learner (predictions, probabilities) on X."""
    out = []
    for j, cols in enumerate(model.views_indices):
        learner = model.fitted_first_level_learners_[j]
        out.append((learner.predict(X[:, cols]), learner.predict_proba(X[:, cols])))
    return out


def meta_predict(model, outputs):
    """Rebuilds the library's meta-features from per-view outputs and predicts."""
    avg = sum(p for _, p in outputs) / len(outputs)
    df = pd.DataFrame(avg)
    df.columns = df.columns.astype(str)
    names = [f"preds_v{v + 1}" for v in range(len(outputs))]
    df = df.assign(**{n: o[0] for n, o in zip(names, outputs)})
    metaX = pd.concat([df, model.enc_.fit_transform(df[names])], axis=1).drop(columns=names)
    return model.meta_learner.predict(metaX)


def diversity_rows(outputs, y, names):
    """Pairwise disagreement and Yule's Q on correct/incorrect (Kuncheva and Whitaker, 2003)."""
    correct = [o[0] == y for o in outputs]
    rows = []
    for a in range(len(outputs)):
        for b in range(a + 1, len(outputs)):
            ca, cb = correct[a], correct[b]
            n11, n00 = np.sum(ca & cb), np.sum(~ca & ~cb)
            n10, n01 = np.sum(ca & ~cb), np.sum(~ca & cb)
            den = n11 * n00 + n01 * n10
            rows.append({"view_a": names[a], "view_b": names[b],
                         "disagreement": np.mean(outputs[a][0] != outputs[b][0]),
                         "q_statistic": (n11 * n00 - n01 * n10) / den if den else np.nan,
                         "acc_a": ca.mean(), "acc_b": cb.mean()})
    return rows


def view_importance_rows(model, outputs, y, classes, seed):
    """Per-class recall drop when one view's output rows are shuffled before the meta-learner."""
    base = classification_report(y, meta_predict(model, outputs), output_dict=True, zero_division=0)
    rng = np.random.RandomState(seed)
    rows = []
    for v in range(len(outputs)):
        drops = []
        for _ in range(N_PERMUTATIONS):
            perm = rng.permutation(len(y))
            shuffled = [(p[perm], s[perm]) if i == v else (p, s) for i, (p, s) in enumerate(outputs)]
            rep = classification_report(y, meta_predict(model, shuffled), output_dict=True,
                                        zero_division=0)
            drops.append({k: base[k]["recall"] - rep[k]["recall"] for k in base if k.isdigit()})
        for k in drops[0]:
            rows.append({"view": v, "class": classes[int(k)],
                         "recall_drop": float(np.mean([d[k] for d in drops]))})
    return rows


# =====================================================================
# Main loop
# =====================================================================
def append_rows(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--runs", type=int, default=ec.N_RUNS)
    ap.add_argument("--dataset", type=Path, default=ec.INPUT_DATASET)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args(argv)

    (args.out / "scores").mkdir(parents=True, exist_ok=True)
    f_run, f_cls = args.out / "round3_per_run.csv", args.out / "round3_per_class.csv"
    f_cm, f_div = args.out / "round3_confusion.csv", args.out / "round3_diversity.csv"
    f_imp = args.out / "round3_view_importance.csv"

    df = pd.read_csv(args.dataset)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    X = df.drop(columns=["Label"])
    cols = list(X.columns)
    views = [[cols.index(c) for c in v if c in cols]
             for v in (ec.vista_tiempo, ec.vista_volumen, ec.vista_banderas, ec.vista_topologia)]
    classes = list(le.classes_)
    bf = classes.index("BruteForce")
    print(f"Dataset {X.shape}, view sizes {[len(v) for v in views]}")

    seeds = np.random.RandomState(ec.SEED_GENERATOR_SEED).randint(0, 10000, size=ec.N_RUNS)[: args.runs]
    finished = ec.done_pairs(f_run)
    for run_idx, seed in enumerate(seeds, start=1):
        seed = int(seed)
        todo = [m for m in args.models if (run_idx, m) not in finished]
        if not todo:
            continue
        data = ec.prepare_run(X, y, seed)
        yt, Xte = data["y_test"], data["X_test"]
        for name in todo:
            model, which = build(name, data, seed, views)
            Xtr, ytr = (data["X_real"], data["y_real"]) if which == "real" else (data["X_bal"], data["y_bal"])
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            t1 = time.perf_counter()
            preds = np.asarray(model.predict(Xte))
            t2 = time.perf_counter()
            tag = {"run": run_idx, "seed": seed, "model": name}
            rep = classification_report(yt, preds, output_dict=True, zero_division=0)
            append_rows(f_cls, [{**tag, "class": classes[int(k)], "precision": v["precision"],
                                 "recall": v["recall"], "f1-score": v["f1-score"], "support": v["support"]}
                                for k, v in rep.items() if k.isdigit()])
            cm = confusion_matrix(yt, preds, labels=range(len(classes)))
            append_rows(f_cm, [{**tag, "true": classes[i], "pred": classes[j], "count": int(cm[i, j])}
                               for i in range(len(classes)) for j in range(len(classes))])
            if name in DIAGNOSED:
                outs = view_outputs(model, Xte)
                assert np.array_equal(meta_predict(model, outs), preds), "meta rebuild mismatch"
                names = VIEW_NAMES if name == "MAIN" else [f"rand{i + 1}" for i in range(4)]
                append_rows(f_div, [{**tag, **r} for r in diversity_rows(outs, yt, names)])
                append_rows(f_imp, [{**tag, **r, "view": names[r["view"]]}
                                    for r in view_importance_rows(model, outs, yt, classes, seed)])
            if name in ("MAIN", "META_BAL"):
                np.savez_compressed(args.out / "scores" / f"run_{run_idx:02d}_{name}.npz", y_bf=(yt == bf),
                                    score=model.predict_proba(Xte)[:, bf].astype(np.float32))
            # per_run is written last: it marks the (run, model) pair as finished for --resume
            append_rows(f_run, [{**tag, "accuracy": accuracy_score(yt, preds),
                                 "f1_macro": f1_score(yt, preds, average="macro"),
                                 "fit_seconds": t1 - t0, "predict_seconds_per_flow": (t2 - t1) / len(yt)}])
            print(f"run {run_idx:02d} seed={seed} {name:<10} acc={accuracy_score(yt, preds):.4f} "
                  f"f1={f1_score(yt, preds, average='macro'):.4f} fit={t1 - t0:.0f}s", flush=True)
    print(f"Done. Results in {args.out}")


if __name__ == "__main__":
    main()
