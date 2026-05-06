from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from multiviewstacking import MultiViewStacking


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DATASET = PROJECT_ROOT / "dataset_poc_multivista.csv"

N_RUNS = 20
SEED_GENERATOR_SEED = 2026
TEST_SIZE = 0.30
RF_ESTIMATORS = 20
KFOLD = 3

OUTPUT_DIR = PROJECT_ROOT / "Analysis" / "comparing"
OUTPUT_PER_RUN = OUTPUT_DIR / "multiview_vs_singleview_per_run.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "multiview_vs_singleview_summary.csv"
OUTPUT_PER_CLASS = OUTPUT_DIR / "multiview_vs_singleview_per_class.csv"  # <-- nuevo

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
    "ACK Flag Count", "Fwd PSH Flags", "PSH Flag Count", "CWR Flag Count",
    "SYN Flag Count", "RST Flag Count", "FIN Flag Count", "Down/Up Ratio",
    "ECE Flag Count",
]

vista_topologia = ["Protocol", "Src Port", "Dst Port"]


def build_view_indices(columns: list[str]):
    ind_tiempo    = [columns.index(c) for c in vista_tiempo    if c in columns]
    ind_volumen   = [columns.index(c) for c in vista_volumen   if c in columns]
    ind_banderas  = [columns.index(c) for c in vista_banderas  if c in columns]
    ind_topologia = [columns.index(c) for c in vista_topologia if c in columns]

    if not all([ind_tiempo, ind_volumen, ind_banderas, ind_topologia]):
        raise ValueError("Una o mas vistas quedaron vacias. Revisa nombres de columnas.")

    return [ind_tiempo, ind_volumen, ind_banderas, ind_topologia]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0)
    return x


def metrics_per_class(y_test, preds, label_encoder, model_name: str, run: int, seed: int) -> list[dict]:
    """Genera una fila por clase con precision, recall y f1."""
    report = classification_report(
        y_test, preds,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for class_name in label_encoder.classes_:
        metrics = report[class_name]
        rows.append({
            "model":     model_name,
            "run":       run,
            "seed":      seed,
            "class":     class_name,
            "precision": round(metrics["precision"], 6),
            "recall":    round(metrics["recall"],    6),
            "f1":        round(metrics["f1-score"],  6),
            "support":   int(metrics["support"]),
        })
    return rows


def evaluate_multiview(views_indices, x_train, y_train, x_test, y_test, run_seed: int) -> tuple[dict, np.ndarray]:
    learner_factory = lambda: RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        random_state=run_seed,
        n_jobs=-1,
    )
    first_level_models = [learner_factory() for _ in views_indices]
    meta_learner = learner_factory()

    model = MultiViewStacking(
        views_indices=views_indices,
        first_level_learners=first_level_models,
        meta_learner=meta_learner,
        k=KFOLD,
        random_state=run_seed,
    )

    model.fit(x_train.values, y_train.values)
    preds = model.predict(x_test.values)

    summary = {
        "model":       "MultiViewStacking",
        "seed":        run_seed,
        "accuracy":    accuracy_score(y_test, preds),
        "f1_macro":    f1_score(y_test, preds, average="macro",     zero_division=0),
        "f1_weighted": f1_score(y_test, preds, average="weighted",  zero_division=0),
    }
    return summary, preds


def evaluate_singleview(x_train, y_train, x_test, y_test, run_seed: int) -> tuple[dict, np.ndarray]:
    model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        random_state=run_seed,
        n_jobs=-1,
    )
    model.fit(x_train.values, y_train.values)
    preds = model.predict(x_test.values)

    summary = {
        "model":       "SingleView",
        "seed":        run_seed,
        "accuracy":    accuracy_score(y_test, preds),
        "f1_macro":    f1_score(y_test, preds, average="macro",     zero_division=0),
        "f1_weighted": f1_score(y_test, preds, average="weighted",  zero_division=0),
    }
    return summary, preds


def summarize_results(df_runs: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_runs.groupby("model")
        .agg(
            runs=("accuracy", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            f1_weighted_mean=("f1_weighted", "mean"),
            f1_weighted_std=("f1_weighted", "std"),
        )
        .reset_index()
    )

    for col in summary.columns:
        if col not in {"model", "runs"}:
            summary[col] = summary[col].round(6)

    return summary


def main():
    print("=" * 80)
    print("MULTIVIEW VS SINGLEVIEW - 20 CORRIDAS")
    print("=" * 80)
    print(f"Dataset: {INPUT_DATASET}")
    print(f"Corridas: {N_RUNS}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"No existe el dataset de entrada: {INPUT_DATASET}")

    df = pd.read_csv(INPUT_DATASET, low_memory=False)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("El dataset no contiene la columna 'Label'.")

    print("\nDistribucion del dataset (conteo):")
    print(df["Label"].value_counts())
    print("\nDistribucion del dataset (%):")
    print((df["Label"].value_counts(normalize=True) * 100).round(4))

    df = df.sample(frac=1.0, random_state=SEED_GENERATOR_SEED).reset_index(drop=True)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df["Label"]))

    x = df.drop(columns=["Label"])
    x = prepare_features(x)

    columns = list(x.columns)
    views_indices = build_view_indices(columns)

    rng = np.random.default_rng(SEED_GENERATOR_SEED)
    run_seeds = rng.choice(1_000_000, size=N_RUNS, replace=False)

    all_results = []
    all_per_class = []  # <-- acumula filas por clase

    for run_idx, run_seed_raw in enumerate(run_seeds):
        run_seed = int(run_seed_raw)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=run_seed,
        )

        scaler = MinMaxScaler()
        x_train_scaled = pd.DataFrame(
            scaler.fit_transform(x_train), columns=columns, index=x_train.index,
        )
        x_test_scaled = pd.DataFrame(
            scaler.transform(x_test), columns=columns, index=x_test.index,
        )

        mv_result, mv_preds = evaluate_multiview(
            views_indices=views_indices,
            x_train=x_train_scaled, y_train=y_train,
            x_test=x_test_scaled,   y_test=y_test,
            run_seed=run_seed,
        )
        sv_result, sv_preds = evaluate_singleview(
            x_train=x_train_scaled, y_train=y_train,
            x_test=x_test_scaled,   y_test=y_test,
            run_seed=run_seed,
        )

        mv_result["run"] = run_idx + 1
        sv_result["run"] = run_idx + 1
        all_results.extend([mv_result, sv_result])

        # Metricas por clase para esta corrida
        all_per_class.extend(
            metrics_per_class(y_test, mv_preds, le, "MultiViewStacking", run_idx + 1, run_seed)
        )
        all_per_class.extend(
            metrics_per_class(y_test, sv_preds, le, "SingleView", run_idx + 1, run_seed)
        )

        print(
            f"Run {run_idx + 1:02d}/{N_RUNS} | seed={run_seed} | "
            f"SingleView acc={sv_result['accuracy']:.4f} | "
            f"MultiView acc={mv_result['accuracy']:.4f}"
        )

    df_runs = pd.DataFrame(all_results)
    df_summary = summarize_results(df_runs)
    df_per_class = pd.DataFrame(all_per_class)

    df_runs.to_csv(OUTPUT_PER_RUN, index=False)
    df_summary.to_csv(OUTPUT_SUMMARY, index=False)
    df_per_class.to_csv(OUTPUT_PER_CLASS, index=False)

    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print(f"\nResultados por corrida:    {OUTPUT_PER_RUN}")
    print(f"Resumen agregado:          {OUTPUT_SUMMARY}")
    print(f"Metricas por clase:        {OUTPUT_PER_CLASS}")


if __name__ == "__main__":
    main()