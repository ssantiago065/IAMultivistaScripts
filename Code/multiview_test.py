from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from multiviewstacking import MultiViewStacking

# =====================================================================
# CONFIGURACIÓN Y CONSTANTES
# =====================================================================
SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parent
INPUT_DATASET = PROJECT_ROOT / "dataset_poc_multivista.csv"

N_RUNS              = 15
SEED_GENERATOR_SEED = 2026
TEST_SIZE           = 0.30
RF_ESTIMATORS       = 150
KFOLD               = 3

# Pasos del tamaño de train a evaluar (fracción del pool de train disponible).
# Ajusta N_STEPS si quieres más o menos granularidad.
# Con 100k datos totales → ~70k en pool de train → pasos de ~7k cada uno.
N_STEPS = 10   # 10%, 20%, ..., 100% del pool de train

# Seed fija para el split inicial (test set queda congelado para todo el experimento)
INITIAL_SPLIT_SEED = 42

OUTPUT_DIR = PROJECT_ROOT / "Analysis" / "learning_curve"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PER_RUN   = OUTPUT_DIR / "learning_curve_per_run.csv"
OUTPUT_SUMMARY   = OUTPUT_DIR / "learning_curve_summary.csv"
OUTPUT_PER_CLASS = OUTPUT_DIR / "learning_curve_per_class.csv"

# =====================================================================
# DEFINICIÓN DE VISTAS (Las 4 Originales Ganadoras)
# =====================================================================
vista_tiempo = [
    "Flow IAT Min", "Bwd IAT Max", "Bwd IAT Std", "Flow Packets/s",
    "Bwd IAT Min", "Active Min", "Fwd IAT Max", "Active Max",
    "Flow Duration", "Idle Mean", "Flow IAT Max", "Fwd Packets/s",
    "Fwd IAT Min", "Flow IAT Mean", "Bwd Packets/s", "Bwd IAT Total",
    "Fwd IAT Std", "Idle Min", "Active Mean", "Bwd IAT Mean",
    "Fwd IAT Mean", "Active Std", "Idle Max", "Fwd IAT Total",
    "Flow Bytes/s", "Idle Std", "Flow IAT Std"
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
    "Subflow Bwd Packets", "Fwd Packet Length Max"
]

vista_banderas = [
    "ACK Flag Count", "Fwd PSH Flags", "PSH Flag Count",
    "CWR Flag Count", "SYN Flag Count", "RST Flag Count",
    "FIN Flag Count", "Down/Up Ratio", "ECE Flag Count"
]

vista_topologia = ["Protocol", "Src Port", "Dst Port"]

# =====================================================================
# FUNCIONES MODULARES
# =====================================================================

def metrics_per_class(y_true, y_pred, le, model_name, train_size, train_size_pct, run_idx, seed):
    """Métricas detalladas por clase incluyendo el tamaño de train para trazabilidad."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics_list = []
    for class_index_str, metrics in report.items():
        if class_index_str.isdigit():
            class_name = le.inverse_transform([int(class_index_str)])[0]
            metrics_list.append({
                "train_size":     train_size,
                "train_size_pct": train_size_pct,
                "run":            run_idx,
                "seed":           seed,
                "model":          model_name,
                "class":          class_name,
                "precision":      metrics["precision"],
                "recall":         metrics["recall"],
                "f1-score":       metrics["f1-score"],
                "support":        metrics["support"],
            })
    return metrics_list


def evaluate_singleview(X_train, y_train, X_test, y_test, seed):
    """RF sobre todas las features — baseline absoluto."""
    rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    return {
        "model":    "SingleView",
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
    }, preds


def evaluate_multiview(X_train, y_train, X_test, y_test, seed, ind_vistas):
    """MultiViewStacking con configuración RF Base + RF Meta (Ganadora)."""
    base_learners = [
        RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=seed, n_jobs=-1)
        for _ in range(len(ind_vistas))
    ]
    meta_learner = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=seed, n_jobs=-1)

    mv_model = MultiViewStacking(
        views_indices=ind_vistas,
        first_level_learners=base_learners,
        meta_learner=meta_learner,
        k=KFOLD,
        random_state=seed,
    )
    mv_model.fit(X_train, y_train)
    preds = mv_model.predict(X_test)
    return {
        "model":    "MultiViewStacking",
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
    }, preds


def apply_smote(X_train_scaled, y_train, run_seed):
    """SMOTE dinámico: eleva todas las clases al máximo de la clase mayoritaria."""
    conteo_clases_train = pd.Series(y_train).value_counts()
    target_max = int(conteo_clases_train.max())
    estrategia_smote = {
        clase: target_max if cantidad < target_max else cantidad
        for clase, cantidad in conteo_clases_train.items()
    }
    smote = SMOTE(sampling_strategy=estrategia_smote, random_state=run_seed)
    return smote.fit_resample(X_train_scaled, y_train), target_max


def summarize_results(df_runs):
    return (
        df_runs
        .groupby(["train_size", "train_size_pct", "model"])[["accuracy", "f1_macro"]]
        .agg(["mean", "std"])
        .reset_index()
    )


# =====================================================================
# FLUJO PRINCIPAL
# =====================================================================

def main():
    print("=" * 70)
    print("EXPERIMENTO: LEARNING CURVE — TRAIN SIZE vs ACCURACY")
    print(f"   RF_ESTIMATORS={RF_ESTIMATORS} | KFOLD={KFOLD} | N_RUNS={N_RUNS} | N_STEPS={N_STEPS}")
    print(f"   Test set: {int(TEST_SIZE*100)}% fijo (seed={INITIAL_SPLIT_SEED})")
    print("=" * 70)

    # --- CARGA Y PREPROCESAMIENTO BASE ---
    df = pd.read_csv(INPUT_DATASET)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    le = LabelEncoder()
    y  = le.fit_transform(df["Label"])
    X  = df.drop(columns=["Label"])
    colnames = list(X.columns)

    nombres_vistas = ["Tiempo", "Volumen", "Banderas", "Topologia"]
    ind_vistas = [
        [colnames.index(c) for c in vista_tiempo    if c in colnames],
        [colnames.index(c) for c in vista_volumen   if c in colnames],
        [colnames.index(c) for c in vista_banderas  if c in colnames],
        [colnames.index(c) for c in vista_topologia if c in colnames],
    ]

    # ─────────────────────────────────────────────────────────────────
    # SPLIT INICIAL ÚNICO: el test set queda congelado para todo
    # el experimento. X_pool / y_pool son el 70% que usaremos
    # para muestrear los diferentes tamaños de train.
    # ─────────────────────────────────────────────────────────────────
    X_pool, X_test_raw, y_pool, y_test_raw = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=INITIAL_SPLIT_SEED, stratify=y
    )
    pool_size = len(X_pool)
    print(f"\nPool de train disponible: {pool_size:,} muestras")
    print(f"Test set fijo:            {len(X_test_raw):,} muestras")
    print(f"Clases: {list(le.classes_)}\n")

    # Pasos: 10%, 20%, ..., 100% del pool
    step_fractions = [round((i + 1) / N_STEPS, 2) for i in range(N_STEPS)]
    step_sizes     = [max(int(f * pool_size), 1) for f in step_fractions]

    # Seeds distintas para las 20 corridas de cada paso
    seeds = np.random.RandomState(SEED_GENERATOR_SEED).randint(0, 10000, size=N_RUNS)

    all_results   = []
    all_per_class = []

    for step_idx, (frac, train_size) in enumerate(zip(step_fractions, step_sizes)):
        pct_label = int(round(frac * 100))
        print(f"\n{'='*70}")
        print(f"PASO {step_idx+1}/{N_STEPS}: train_size={train_size:,}  ({pct_label}% del pool)")
        print(f"{'='*70}")

        for run_idx, run_seed in enumerate(seeds):
            # Subsampleo aleatorio del pool para este paso/corrida
            rng = np.random.RandomState(run_seed)
            idx_subsample = rng.choice(pool_size, size=train_size, replace=False)

            X_train_raw = X_pool.iloc[idx_subsample]
            y_train_raw = y_pool[idx_subsample]

            # Imputación (fit solo sobre train del paso actual)
            imputer = SimpleImputer(strategy="median")
            X_train_clean = imputer.fit_transform(X_train_raw)
            X_test_clean  = imputer.transform(X_test_raw)

            # Escalado (fit solo sobre train del paso actual)
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_clean)
            X_test_scaled  = scaler.transform(X_test_clean)

            # SMOTE dinámico
            (X_train_bal, y_train_bal), smote_target = apply_smote(
                X_train_scaled, y_train_raw, run_seed
            )

            if run_idx == 0:
                print(f"   SMOTE: eleva clases a {smote_target:,} muestras → "
                      f"train balanceado={len(X_train_bal):,}")

            # ── SingleView ──
            sv_result, sv_preds = evaluate_singleview(
                X_train_bal, y_train_bal, X_test_scaled, y_test_raw, run_seed
            )
            sv_result.update({"run": run_idx + 1, "train_size": train_size, "train_size_pct": pct_label})
            all_results.append(sv_result)
            all_per_class.extend(metrics_per_class(
                y_test_raw, sv_preds, le, "SingleView",
                train_size, pct_label, run_idx + 1, run_seed
            ))

            # ── MultiViewStacking ──
            mv_result, mv_preds = evaluate_multiview(
                X_train_bal, y_train_bal, X_test_scaled, y_test_raw, run_seed, ind_vistas
            )
            mv_result.update({"run": run_idx + 1, "train_size": train_size, "train_size_pct": pct_label})
            all_results.append(mv_result)
            all_per_class.extend(metrics_per_class(
                y_test_raw, mv_preds, le, "MultiViewStacking",
                train_size, pct_label, run_idx + 1, run_seed
            ))

            print(
                f"  Run {run_idx+1:02d}/{N_RUNS} seed={run_seed} | "
                f"SV={sv_result['accuracy']:.4f} MV={mv_result['accuracy']:.4f}"
            )

    # =====================================================================
    # GUARDADO Y RESUMEN FINAL
    # =====================================================================
    print("\nGuardando reportes en disco...")
    df_runs      = pd.DataFrame(all_results)
    df_summary   = summarize_results(df_runs)
    df_per_class = pd.DataFrame(all_per_class)

    df_runs.to_csv(OUTPUT_PER_RUN,    index=False)
    df_summary.to_csv(OUTPUT_SUMMARY,  index=False)
    df_per_class.to_csv(OUTPUT_PER_CLASS, index=False)

    # ── Tabla resumen en consola ──
    print("\n" + "=" * 70)
    print("RESUMEN: accuracy media por modelo y tamaño de train")
    print("=" * 70)
    mv_sv = df_runs[df_runs["model"].isin(["SingleView", "MultiViewStacking"])]
    pivot = (
        mv_sv.groupby(["train_size_pct", "model"])["accuracy"]
        .mean()
        .unstack("model")
        .reset_index()
    )
    pivot["diff_MV_vs_SV"] = pivot["MultiViewStacking"] - pivot["SingleView"]
    print(pivot.to_string(index=False))

    print("\n--- F1-macro ---")
    pivot_f1 = (
        mv_sv.groupby(["train_size_pct", "model"])["f1_macro"]
        .mean()
        .unstack("model")
        .reset_index()
    )
    pivot_f1["diff_MV_vs_SV"] = pivot_f1["MultiViewStacking"] - pivot_f1["SingleView"]
    print(pivot_f1.to_string(index=False))

    print(f"\nArchivos CSV guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()