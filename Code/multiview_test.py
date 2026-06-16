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

N_RUNS              = 20
SEED_GENERATOR_SEED = 2026
TEST_SIZE           = 0.30
RF_ESTIMATORS       = 150
KFOLD               = 3

OUTPUT_DIR = PROJECT_ROOT / "Analysis" / "comparing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PER_RUN   = OUTPUT_DIR / "multiview_vs_singleview_per_run.csv"
OUTPUT_SUMMARY   = OUTPUT_DIR / "multiview_vs_singleview_summary.csv"
OUTPUT_PER_CLASS = OUTPUT_DIR / "multiview_vs_singleview_per_class.csv"

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
# FUNCIONES MODULARES (Aprovechando la refactorización del compañero)
# =====================================================================

def metrics_per_class(y_true, y_pred, le, model_name, run_idx, seed):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics_list = []
    for class_index_str, metrics in report.items():
        if class_index_str.isdigit():
            class_name = le.inverse_transform([int(class_index_str)])[0]
            metrics_list.append({
                "run":        run_idx,
                "seed":       seed,
                "model":      model_name,
                "class":      class_name,
                "precision":  metrics["precision"],
                "recall":     metrics["recall"],
                "f1-score":   metrics["f1-score"],
                "support":    metrics["support"],
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
        "f1_macro": f1_score(y_test, preds, average='macro'),
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
        "f1_macro": f1_score(y_test, preds, average='macro'),
    }, preds

def evaluate_individual_views(X_train_scaled, y_train, X_test_scaled, y_test, seed, ind_vistas, nombres_vistas):
    """Extraído del diff: Diagnóstico limpio de contribución individual."""
    results = []
    for i, indices_vista in enumerate(ind_vistas):
        X_tr = X_train_scaled[:, indices_vista]
        X_te = X_test_scaled[:, indices_vista]
        v_result, v_preds = evaluate_singleview(X_tr, y_train, X_te, y_test, seed)
        v_result["model"] = f"Vista_{nombres_vistas[i]}"
        results.append((v_result, v_preds))
    return results

def summarize_results(df_runs):
    return df_runs.groupby("model")[["accuracy", "f1_macro"]].agg(["mean", "std"]).reset_index()

# =====================================================================
# FLUJO PRINCIPAL
# =====================================================================

def main():
    print("=" * 70)
    print("🚀 EXPERIMENTO UNIFICADO: SMOTE MAXIMIZADO + RF-META")
    print(f"   RF_ESTIMATORS={RF_ESTIMATORS} | KFOLD={KFOLD} | N_RUNS={N_RUNS}")
    print("=" * 70)

    df = pd.read_csv(INPUT_DATASET)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    le = LabelEncoder()
    y  = le.fit_transform(df["Label"])
    X  = df.drop(columns=["Label"])
    colnames = list(X.columns)

    nombres_vistas = ["Tiempo", "Volumen", "Banderas", "Topologia"]
    ind_vistas = [
        [colnames.index(c) for c in vista_tiempo if c in colnames],
        [colnames.index(c) for c in vista_volumen if c in colnames],
        [colnames.index(c) for c in vista_banderas if c in colnames],
        [colnames.index(c) for c in vista_topologia if c in colnames]
    ]

    all_results   = []
    all_per_class = []
    seeds = np.random.RandomState(SEED_GENERATOR_SEED).randint(0, 10000, size=N_RUNS)

    for run_idx, run_seed in enumerate(seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=run_seed, stratify=y
        )

        imputer = SimpleImputer(strategy="median")
        X_train_clean = imputer.fit_transform(X_train)
        X_test_clean  = imputer.transform(X_test)

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_test_scaled  = scaler.transform(X_test_clean)

        # --- LÓGICA DE SMOTE DINÁMICO (El aporte de tu rama) ---
        conteo_clases_train = pd.Series(y_train).value_counts()
        TARGET_MAX_SAMPLES = int(conteo_clases_train.max()) # Busca automáticamente la clase más grande
        
        estrategia_smote = {}
        for clase, cantidad in conteo_clases_train.items():
            if cantidad < TARGET_MAX_SAMPLES:
                estrategia_smote[clase] = TARGET_MAX_SAMPLES
            else:
                estrategia_smote[clase] = cantidad

        if run_idx == 0:
            print(f"\n🧹 SMOTE configurado para elevar clases a {TARGET_MAX_SAMPLES} muestras en Train.")

        smote = SMOTE(sampling_strategy=estrategia_smote, random_state=run_seed)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

        # --- ENTRENAMIENTO ---
        sv_result, sv_preds = evaluate_singleview(X_train_bal, y_train_bal, X_test_scaled, y_test, run_seed)
        sv_result["run"] = run_idx + 1
        all_results.append(sv_result)
        all_per_class.extend(metrics_per_class(y_test, sv_preds, le, "SingleView", run_idx + 1, run_seed))

        mv_result, mv_preds = evaluate_multiview(X_train_bal, y_train_bal, X_test_scaled, y_test, run_seed, ind_vistas)
        mv_result["run"] = run_idx + 1
        all_results.append(mv_result)
        all_per_class.extend(metrics_per_class(y_test, mv_preds, le, "MultiViewStacking", run_idx + 1, run_seed))

        vista_results = evaluate_individual_views(X_train_bal, y_train_bal, X_test_scaled, y_test, run_seed, ind_vistas, nombres_vistas)
        for v_result, v_preds in vista_results:
            v_result["run"] = run_idx + 1
            all_results.append(v_result)
            all_per_class.extend(metrics_per_class(y_test, v_preds, le, v_result["model"], run_idx + 1, run_seed))

        # --- IMPRESIÓN LIMPIA POR CORRIDA ---
        vista_accs = " | ".join(f"{vr['model'].replace('Vista_', '')}={vr['accuracy']:.4f}" for vr, _ in vista_results)
        print(f"  Run {run_idx+1:02d}/{N_RUNS} seed={run_seed} | SV={sv_result['accuracy']:.4f} MV={mv_result['accuracy']:.4f} | {vista_accs}")

    # =====================================================================
    # GUARDADO Y RESUMEN FINAL (El aporte MLOps del compañero)
    # =====================================================================
    print("\n💾 Guardando reportes en disco...")
    df_runs      = pd.DataFrame(all_results)
    df_summary   = summarize_results(df_runs)
    df_per_class = pd.DataFrame(all_per_class)

    df_runs.to_csv(OUTPUT_PER_RUN, index=False)
    df_summary.to_csv(OUTPUT_SUMMARY, index=False)
    df_per_class.to_csv(OUTPUT_PER_CLASS, index=False)

    print("\n" + "=" * 70)
    print("RESUMEN FINAL: MultiViewStacking vs SingleView")
    print("=" * 70)
    mv_sv = df_runs[df_runs["model"].isin(["SingleView", "MultiViewStacking"])]
    pivot = mv_sv.groupby("model")["accuracy"].mean().to_frame().T
    pivot["diff_MV_vs_SV"] = pivot["MultiViewStacking"] - pivot["SingleView"]
    print(pivot.to_string(index=False))

    print(f"\n✅ Archivos CSV guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()