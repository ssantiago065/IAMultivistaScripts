"""Control experiments for the MultiView Stacking paper (review round, 2026-10-15).

Three threats to validity raised in the review of experiment A are tested here,
on EXACTLY the same 20 seeds, train/test partitions and preprocessing as A
(Code/multiview_test.py at commit 1363933, branch exp/smote-downsampling-50k):

  1. Capacity confound: MVS trains 4 x 150 trees, SingleView only 150.
     -> SV600: SingleView with 600 trees.
  2. Do the *semantic* views matter, or would any partition do (random-subspace effect)?
     -> MVS_LR_RANDVIEWS: the same 69 features split at random into groups of the
        same sizes as the semantic views (a new random partition per run).
  3. SMOTE is applied before the internal k folds, so synthetic flows leak into the
     out-of-fold meta-features.
     -> MVS_LR_FOLDSMOTE: SMOTE applied inside each internal fold (training part only)
        and again before the final refit; the meta-learner sees only real flows.

Controls 2 and 3 are run with both meta-learners (MVS_LR_* and MVS_RF_*): the pilot
run (seed 2305) showed that MVS_RF, not MVS_LR, reproduces A's MultiViewStacking.

Two reference models are recomputed so the new numbers are paired with A:
  SV150    -- must reproduce A's SingleView run by run (pipeline check)
  MVS_LR   -- logistic meta-learner, as described in the paper
  MVS_RF   -- Random Forest meta-learner (the code at commit 1363933 has this line
              active); whichever of MVS_LR / MVS_RF reproduces A's MultiViewStacking
              column settles which meta-learner A actually used.

Run from the IAMultivistaScripts repo (same layout as Code/multiview_test.py):
    python Code/exp_controls.py                       # all models, 20 runs
    python Code/exp_controls.py --models SV150 MVS_LR # a subset
    python Code/exp_controls.py --runs 2              # quick smoke test
Results are appended run by run, so an interrupted job resumes where it stopped.
Then:  python Code/analyze_controls.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from multiviewstacking import MultiViewStacking
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

# =====================================================================
# Configuration -- identical to experiment A
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DATASET = PROJECT_ROOT / "dataset_poc_multivista.csv"

N_RUNS = 20
SEED_GENERATOR_SEED = 2026
TEST_SIZE = 0.30
RF_ESTIMATORS = 150
KFOLD = 3
SMOTE_LIMIT = 30000          # minority classes are raised to this many training flows

OUTPUT_DIR = PROJECT_ROOT / "Analysis" / "controls"

ALL_MODELS = ["SV150", "SV600", "MVS_LR", "MVS_RF", "MVS_LR_FOLDSMOTE", "MVS_LR_RANDVIEWS",
              "MVS_RF_FOLDSMOTE", "MVS_RF_RANDVIEWS"]
N_JOBS = -1                  # threads per forest; --n-jobs 1 makes every forest single-threaded

# Views exactly as in experiment A (names absent from the CSV are skipped,
# e.g. CWR/ECE Flag Count, removed as constant columns during cleaning).
vista_tiempo = [
    "Flow IAT Min", "Bwd IAT Max", "Bwd IAT Std", "Flow Packets/s",
    "Bwd IAT Min", "Active Min", "Fwd IAT Max", "Active Max",
    "Flow Duration", "Idle Mean", "Flow IAT Max", "Fwd Packets/s",
    "Fwd IAT Min", "Flow IAT Mean", "Bwd Packets/s", "Bwd IAT Total",
    "Fwd IAT Std", "Idle Min", "Active Mean", "Bwd IAT Mean",
    "Fwd IAT Mean", "Active Std", "Idle Max", "Fwd IAT Total",
    "Flow Bytes/s", "Idle Std", "Flow IAT Std",
]
vista_volumen = [
    "Total Bwd packets", "Packet Length Std", "Packet Length Min",
    "Packet Length Mean", "Bwd Header Length", "Subflow Fwd Packets",
    "Bwd Packet Length Max", "Bwd Bulk Rate Avg", "Bwd Packet Length Std",
    "Packet Length Max", "Total Length of Bwd Packet", "Packet Length Variance",
    "Subflow Bwd Bytes", "Fwd Packet Length Std", "Total Length of Fwd Packet",
    "Fwd Act Data Pkts", "Bwd Segment Size Avg", "Fwd Header Length",
    "Bwd Packet Length Min", "Bwd Bytes/Bulk Avg", "FWD Init Win Bytes",
    "Average Packet Size", "Total Fwd Packet", "Fwd Packet Length Mean",
    "Bwd Packet Length Mean", "Fwd Packet Length Min", "Bwd Init Win Bytes",
    "Fwd Segment Size Avg", "Bwd Packet/Bulk Avg", "Fwd Seg Size Min",
    "Subflow Bwd Packets", "Fwd Packet Length Max",
]
vista_banderas = [
    "ACK Flag Count", "Fwd PSH Flags", "PSH Flag Count",
    "CWR Flag Count", "SYN Flag Count", "RST Flag Count",
    "FIN Flag Count", "Down/Up Ratio", "ECE Flag Count",
]
vista_topologia = ["Protocol", "Src Port", "Dst Port"]


# =====================================================================
# SMOTE helpers
# =====================================================================
def smote_strategy(y, limit):
    """A's rule: every class below `limit` is raised to `limit`; the rest stay as they are."""
    counts = pd.Series(y).value_counts().to_dict()
    return {c: (limit if n < limit else n) for c, n in counts.items()}


def apply_smote_like_A(X, y, seed):
    smote = SMOTE(sampling_strategy=smote_strategy(y, SMOTE_LIMIT), random_state=seed)
    return smote.fit_resample(X, y)


class ProportionalSMOTE:
    """SMOTE for one internal fold: the target scales with the size of the part being
    resampled, so a fold holding 2/3 of the training data is raised to 2/3 of 30,000,
    and the final refit on the whole training partition is raised to 30,000, as in A."""

    def __init__(self, n_full_train, seed):
        self.n_full = n_full_train
        self.seed = seed

    def __call__(self, X, y, call_idx):
        limit = int(round(SMOTE_LIMIT * len(y) / self.n_full))
        smote = SMOTE(sampling_strategy=smote_strategy(y, limit),
                      random_state=self.seed + call_idx)
        return smote.fit_resample(X, y)


# =====================================================================
# MultiView Stacking with resampling inside the internal folds
# =====================================================================
class FoldResampledMVS(MultiViewStacking):
    """multiviewstacking 0.7.5 MultiViewStacking.fit, with one change: an optional
    `resampler` is applied to the TRAINING part of every internal fold and to the
    full training set before the final refit. Out-of-fold predictions, and therefore
    the meta-learner's training set, are computed on real instances only.
    With resampler=None this class reproduces the library exactly (see test_controls.py).
    Prediction is inherited unchanged from the library."""

    def __init__(self, views_indices=None, first_level_learners=None,
                 meta_learner=None, k=10, random_state=123, resampler=None):
        super().__init__(views_indices=views_indices,
                         first_level_learners=first_level_learners,
                         meta_learner=meta_learner, k=k, random_state=random_state)
        self.resampler = resampler

    def _resample(self, X, y, call_idx):
        if self.resampler is None:
            return X, y
        return self.resampler(X, y, call_idx)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.num_views_ = len(self.views_indices)
        self.uniquelabels_ = [str(x) for x in np.unique(y)]
        self.classes_ = np.unique(y)
        self.num_rows = len(y)

        truelabels = []
        preds = [[] for _ in range(self.num_views_)]
        scores = [None] * self.num_views_
        kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.random_state)

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            xtrain, ytrain = self._resample(X[train_index], y[train_index], call_idx=i)
            xtest, ytest = X[test_index], y[test_index]          # always real flows
            for j in range(self.num_views_):
                m_v = self.first_level_learners[j].fit(xtrain[:, self.views_indices[j]], ytrain)
                v_test = xtest[:, self.views_indices[j]]
                preds[j].append(m_v.predict(v_test))
                raw_v = m_v.predict_proba(v_test)
                scores[j] = raw_v if scores[j] is None else np.vstack((scores[j], raw_v))
            truelabels.append(ytest)

        truelabels = np.concatenate(truelabels).ravel()
        for v in range(self.num_views_):
            preds[v] = np.concatenate(preds[v]).ravel()

        Xr, yr = self._resample(X, y, call_idx=self.k)            # final refit
        for v in range(self.num_views_):
            self.first_level_learners[v].fit(Xr[:, self.views_indices[v]], yr)
        self.fitted_first_level_learners_ = self.first_level_learners

        # Meta-features: identical construction to the library.
        avgscores = sum(scores) / self.num_views_
        df = pd.DataFrame(avgscores)
        df.columns = df.columns.astype(str)
        df = df.assign(**{f"preds_v{v + 1}": preds[v] for v in range(self.num_views_)})
        self.enc_ = OneHotEncoder(categories=[self.uniquelabels_] * self.num_views_,
                                  drop="first", sparse_output=False).set_output(transform="pandas")
        predcolnames = [f"preds_v{v + 1}" for v in range(self.num_views_)]
        ohe = self.enc_.fit_transform(df[predcolnames])
        metaX = pd.concat([df, ohe], axis=1).drop(columns=predcolnames)
        self.meta_learner.fit(metaX, truelabels)
        return self


# =====================================================================
# Models
# =====================================================================
def rf(n, seed):
    return RandomForestClassifier(n_estimators=n, random_state=seed, n_jobs=N_JOBS)


def make_meta(kind, seed):
    if kind == "LR":
        return LogisticRegression(random_state=seed, max_iter=1000)
    return rf(RF_ESTIMATORS, seed)


def random_views(ind_vistas, seed):
    """Shuffle the union of the view columns and cut it into groups with the same sizes
    as the semantic views (27/32/7/3 on the real data)."""
    pool = np.array([c for view in ind_vistas for c in view])
    np.random.RandomState(seed + 7919).shuffle(pool)
    out, start = [], 0
    for view in ind_vistas:
        out.append(sorted(pool[start:start + len(view)].tolist()))
        start += len(view)
    return out


def fit_predict(model_name, data, seed, ind_vistas):
    """Returns (predictions, fit_seconds, predict_seconds)."""
    Xb, yb, Xr, yr, Xte = data["X_bal"], data["y_bal"], data["X_real"], data["y_real"], data["X_test"]
    if model_name in ("SV150", "SV600"):
        model, X, y = rf(150 if model_name == "SV150" else 600, seed), Xb, yb
    elif model_name in ("MVS_LR", "MVS_RF"):
        model = MultiViewStacking(views_indices=[list(v) for v in ind_vistas],
                                  first_level_learners=[rf(RF_ESTIMATORS, seed) for _ in ind_vistas],
                                  meta_learner=make_meta(model_name[-2:], seed),
                                  k=KFOLD, random_state=seed)
        X, y = Xb, yb
    elif model_name in ("MVS_LR_RANDVIEWS", "MVS_RF_RANDVIEWS"):
        views = random_views(ind_vistas, seed)
        model = MultiViewStacking(views_indices=views,
                                  first_level_learners=[rf(RF_ESTIMATORS, seed) for _ in views],
                                  meta_learner=make_meta(model_name[4:6], seed), k=KFOLD,
                                  random_state=seed)
        X, y = Xb, yb
    elif model_name in ("MVS_LR_FOLDSMOTE", "MVS_RF_FOLDSMOTE"):
        model = FoldResampledMVS(views_indices=[list(v) for v in ind_vistas],
                                 first_level_learners=[rf(RF_ESTIMATORS, seed) for _ in ind_vistas],
                                 meta_learner=make_meta(model_name[4:6], seed), k=KFOLD,
                                 random_state=seed, resampler=ProportionalSMOTE(len(yr), seed))
        X, y = Xr, yr                                             # real flows; SMOTE happens inside
    else:
        raise ValueError(model_name)
    t0 = time.perf_counter()
    model.fit(X, y)
    t1 = time.perf_counter()
    preds = model.predict(Xte)
    t2 = time.perf_counter()
    return np.asarray(preds), t1 - t0, t2 - t1


# =====================================================================
# Main loop
# =====================================================================
def prepare_run(X, y, seed):
    """Experiment A's per-run preprocessing, step for step."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    imputer = SimpleImputer(strategy="median")
    X_train_clean = imputer.fit_transform(X_train)
    X_test_clean = imputer.transform(X_test)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    X_bal, y_bal = apply_smote_like_A(X_train_scaled, y_train, seed)
    return {"X_bal": X_bal, "y_bal": y_bal, "X_real": X_train_scaled, "y_real": np.asarray(y_train),
            "X_test": X_test_scaled, "y_test": np.asarray(y_test)}


def done_pairs(path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["run"], df["model"]))


def append_rows(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def main(argv=None):
    global N_JOBS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--runs", type=int, default=N_RUNS, help="first N of A's 20 seeds")
    ap.add_argument("--dataset", type=Path, default=INPUT_DATASET)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--n-jobs", type=int, default=N_JOBS,
                    help="threads per forest (default -1 = all cores, as in experiment A)")
    args = ap.parse_args(argv)
    N_JOBS = args.n_jobs

    args.out.mkdir(parents=True, exist_ok=True)
    f_run = args.out / "controls_per_run.csv"
    f_cls = args.out / "controls_per_class.csv"
    f_cm = args.out / "controls_confusion.csv"

    df = pd.read_csv(args.dataset)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    X = df.drop(columns=["Label"])
    colnames = list(X.columns)
    ind_vistas = [[colnames.index(c) for c in v if c in colnames]
                  for v in (vista_tiempo, vista_volumen, vista_banderas, vista_topologia)]
    print(f"Dataset {X.shape}, view sizes {[len(v) for v in ind_vistas]}, classes {list(le.classes_)}")

    seeds = np.random.RandomState(SEED_GENERATOR_SEED).randint(0, 10000, size=N_RUNS)[: args.runs]
    finished = done_pairs(f_run)

    for run_idx, seed in enumerate(seeds, start=1):
        todo = [m for m in args.models if (run_idx, m) not in finished]
        if not todo:
            print(f"run {run_idx:02d}: already done")
            continue
        data = prepare_run(X, y, int(seed))
        for model_name in todo:
            preds, t_fit, t_pred = fit_predict(model_name, data, int(seed), ind_vistas)
            yt = data["y_test"]
            append_rows(f_run, [{
                "run": run_idx, "seed": int(seed), "model": model_name,
                "accuracy": accuracy_score(yt, preds), "f1_macro": f1_score(yt, preds, average="macro"),
                "fit_seconds": t_fit, "predict_seconds_per_flow": t_pred / len(yt),
            }])
            rep = classification_report(yt, preds, output_dict=True, zero_division=0)
            append_rows(f_cls, [{
                "run": run_idx, "seed": int(seed), "model": model_name,
                "class": le.inverse_transform([int(k)])[0], "precision": v["precision"],
                "recall": v["recall"], "f1-score": v["f1-score"], "support": v["support"],
            } for k, v in rep.items() if k.isdigit()])
            cm = confusion_matrix(yt, preds, labels=range(len(le.classes_)))
            append_rows(f_cm, [{
                "run": run_idx, "model": model_name, "true": le.classes_[i], "pred": le.classes_[j],
                "count": int(cm[i, j]),
            } for i in range(len(le.classes_)) for j in range(len(le.classes_))])
            print(f"run {run_idx:02d} seed={seed} {model_name:<17} "
                  f"acc={accuracy_score(yt, preds):.4f} f1={f1_score(yt, preds, average='macro'):.4f} "
                  f"fit={t_fit:.0f}s")
    print(f"\nDone. Results in {args.out}. Next: python Code/analyze_controls.py")


if __name__ == "__main__":
    main()
