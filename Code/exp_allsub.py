"""Plan steps 12 and 9 (round 3): SingleView vs. the main MultiView Stacking on the
all-subtype dataset (Code/dataset_preparing_allsubtypes.py).

Protocols:
  random    stratified 70/30 flow-level split, as in experiment A            [12]
  disjoint  capture-disjoint where possible: in classes with at least
            MIN_CAPTURES captures about 30% of the captures (by seed) go to test;
            classes with fewer captures (Brute Force, Recon, Spoofing, Web-Based)
            are split in time inside each capture (last 30% of each capture's
            flows to test), so no class shares flows from the same time window   [9]
Models: SV150 (SMOTE like A) and MAIN (MVS_RF_FOLDSMOTE), same code as the controls; with
--models, also any exp_round3 model (HYB_FULL, HYB_RAND, STACK_HET, ...), built by exp_round3.build.

    python Code/exp_allsub.py --dataset H:/Dataset/dataset_allsub_8clases.csv --protocol random
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import exp_controls as ec

ec.DETERMINISTIC_PROBA = True

MIN_CAPTURES = 4
TEST_FRACTION = 0.30
MODELS = ["SV150", "MAIN"]
EXTRA_MODELS = ["HYB_FULL", "HYB_RAND", "STACK_HET", "LGBM", "SV69", "META_BAL"]


def disjoint_split(labels, captures, t_rank, seed):
    """Boolean test mask: whole captures for multi-capture classes, last 30% in time otherwise."""
    rng = np.random.RandomState(seed)
    test = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        in_cls = labels == cls
        caps = np.unique(captures[in_cls])
        if len(caps) >= MIN_CAPTURES:
            n_test = max(1, int(round(TEST_FRACTION * len(caps))))
            test |= in_cls & np.isin(captures, rng.choice(caps, size=n_test, replace=False))
        else:
            test |= in_cls & (t_rank >= 1 - TEST_FRACTION)
    return test


def prepare(X, y, train_idx, test_idx, seed):
    """Experiment A's preprocessing (median impute, min-max, SMOTE to 30,000) on a given split."""
    imputer, scaler = SimpleImputer(strategy="median"), MinMaxScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X.iloc[train_idx]))
    Xte = scaler.transform(imputer.transform(X.iloc[test_idx]))
    ytr = y[train_idx]
    Xb, yb = ec.apply_smote_like_A(Xtr, ytr, seed)
    return {"X_bal": Xb, "y_bal": yb, "X_real": Xtr, "y_real": ytr, "X_test": Xte, "y_test": y[test_idx]}


def build(name, data, seed, views):
    if name == "SV150":
        return ec.rf(150, seed), data["X_bal"], data["y_bal"]
    if name in EXTRA_MODELS:
        import exp_round3
        model, which = exp_round3.build(name, data, seed, views)
        return (model, data["X_real"], data["y_real"]) if which == "real" else             (model, data["X_bal"], data["y_bal"])
    model = ec.FoldResampledMVS(views_indices=[list(v) for v in views],
                                first_level_learners=[ec.rf(ec.RF_ESTIMATORS, seed) for _ in views],
                                meta_learner=ec.make_meta("RF", seed), k=ec.KFOLD, random_state=seed,
                                resampler=ec.ProportionalSMOTE(len(data["y_real"]), seed))
    return model, data["X_real"], data["y_real"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--protocol", choices=["random", "disjoint"], required=True)
    ap.add_argument("--runs", type=int, default=ec.N_RUNS)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS + EXTRA_MODELS)
    ap.add_argument("--out", type=Path, default=ec.PROJECT_ROOT / "Analysis" / "round3")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    f_run = args.out / f"allsub_{args.protocol}_per_run.csv"
    f_cls = args.out / f"allsub_{args.protocol}_per_class.csv"

    df = pd.read_csv(args.dataset, low_memory=False)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    captures, t_rank = df.pop("capture").to_numpy(), df.pop("t_rank").to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    X = df.drop(columns=["Label"])
    cols = list(X.columns)
    views = [[cols.index(c) for c in v if c in cols]
             for v in (ec.vista_tiempo, ec.vista_volumen, ec.vista_banderas, ec.vista_topologia)]
    print(f"Dataset {X.shape}, view sizes {[len(v) for v in views]}", flush=True)

    seeds = np.random.RandomState(ec.SEED_GENERATOR_SEED).randint(0, 10000, size=ec.N_RUNS)[: args.runs]
    finished = ec.done_pairs(f_run)
    idx = np.arange(len(y))
    for run_idx, seed in enumerate(seeds, start=1):
        seed = int(seed)
        todo = [m for m in args.models if (run_idx, m) not in finished]
        if not todo:
            continue
        if args.protocol == "random":
            tr, te = train_test_split(idx, test_size=TEST_FRACTION, random_state=seed, stratify=y)
        else:
            mask = disjoint_split(y, captures, t_rank, seed)
            tr, te = idx[~mask], idx[mask]
        data = prepare(X, y, tr, te, seed)
        for name in todo:
            model, Xtr, ytr = build(name, data, seed, views)
            t0 = time.perf_counter()
            preds = np.asarray(model.fit(Xtr, ytr).predict(data["X_test"]))
            yt = data["y_test"]
            tag = {"run": run_idx, "seed": seed, "model": name}
            rep = classification_report(yt, preds, output_dict=True, zero_division=0,
                                        labels=range(len(le.classes_)))
            ec.append_rows(f_cls, [{**tag, "class": le.classes_[int(k)], "precision": v["precision"],
                                    "recall": v["recall"], "f1-score": v["f1-score"], "support": v["support"]}
                                   for k, v in rep.items() if k.isdigit()])
            # per_run is written last: it marks the (run, model) pair as finished
            ec.append_rows(f_run, [{**tag, "accuracy": accuracy_score(yt, preds),
                                    "f1_macro": f1_score(yt, preds, average="macro"),
                                    "n_test": len(yt), "seconds": time.perf_counter() - t0}])
            print(f"{args.protocol} run {run_idx:02d} {name:<6} acc={accuracy_score(yt, preds):.4f} "
                  f"f1={f1_score(yt, preds, average='macro'):.4f}", flush=True)


if __name__ == "__main__":
    main()
