from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from multiviewstacking import MultiViewStacking

# =====================================================================
# CONFIGURACIÓN Y CONSTANTES
# =====================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DATASET = PROJECT_ROOT / "dataset_poc_multivista.csv"

N_RUNS              = 20
SEED_GENERATOR_SEED = 2026
TEST_SIZE           = 0.30
RF_ESTIMATORS       = 150
KFOLD               = 3

OUTPUT_DIR = PROJECT_ROOT / "Analysis" / "comparing3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PER_RUN   = OUTPUT_DIR / "multiview_vs_singleview_per_run.csv"
OUTPUT_SUMMARY   = OUTPUT_DIR / "multiview_vs_singleview_summary.csv"
OUTPUT_PER_CLASS = OUTPUT_DIR / "multiview_vs_singleview_per_class.csv"

# =====================================================================
# DEFINICIÓN DE FEATURES POR DOMINIO
# =====================================================================
_tiempo = [
    "Flow IAT Min", "Bwd IAT Max", "Bwd IAT Std", "Flow Packets/s",
    "Bwd IAT Min", "Active Min", "Fwd IAT Max", "Active Max",
    "Flow Duration", "Idle Mean", "Flow IAT Max", "Fwd Packets/s",
    "Fwd IAT Min", "Flow IAT Mean", "Bwd Packets/s", "Bwd IAT Total",
    "Fwd IAT Std", "Idle Min", "Active Mean", "Bwd IAT Mean",
    "Fwd IAT Mean", "Active Std", "Idle Max", "Fwd IAT Total",
    "Flow Bytes/s", "Idle Std", "Flow IAT Std"
]
_volumen = [
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
_banderas = [
    "ACK Flag Count", "Fwd PSH Flags", "PSH Flag Count",
    "CWR Flag Count", "SYN Flag Count", "RST Flag Count",
    "FIN Flag Count", "Down/Up Ratio", "ECE Flag Count"
]
_topologia = ["Protocol", "Src Port", "Dst Port"]

# =====================================================================
# DIAGNÓSTICO ACTUALIZADO (runs con RF como meta-learner, 20k muestras/clase)
# =====================================================================
# Resultados mostraron que el meta-learner RF supera a LR en todos los casos.
# El meta-dataset (avgscores + OHE de predicciones) tiene estructura no lineal:
#   "si Vista_Volumen vota clase X con alta confianza y Vista_Banderas discrepa,
#    confía en Volumen" → reglas tipo if/then que RF aprende y LR no.
# Por eso el baseline RF base + RF meta es la configuración correcta hoy.

EXPERIMENTS = [
    {
        # RF-150 base + RF-150 meta (baseline ganador)
        "name": "4v_baseline_RF_RF",
        "vistas": [
            ("Tiempo",    _tiempo),
            ("Volumen",   _volumen),
            ("Banderas",  _banderas),
            ("Topologia", _topologia),
        ],
        "meta_fn": lambda seed: RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, random_state=seed, n_jobs=-1
        ),
        "base_fn": lambda seed, n: [
            RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=seed, n_jobs=-1)
            for _ in range(n)
        ],
    },
]

# =====================================================================
# FUNCIONES MODULARES
# =====================================================================

def metrics_per_class(y_true, y_pred, le, model_name, experiment_name, run_idx, seed):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics_list = []
    for class_index_str, metrics in report.items():
        if class_index_str.isdigit():
            class_name = le.inverse_transform([int(class_index_str)])[0]
            metrics_list.append({
                "experiment": experiment_name,
                "run":        run_idx,
                "seed":       seed,
                "model":      model_name,
                "class":      class_name,
                "precision":  metrics["precision"],
                "recall":     metrics["recall"],
                "f1-score":   metrics["f1-score"],
                "support":    metrics["support"]
            })
    return metrics_list


def evaluate_singleview(X_train, y_train, X_test, y_test, seed):
    """RF sobre todas las features — baseline absoluto que no cambia entre experimentos."""
    rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    return {
        "model":    "SingleView",
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average='macro'),
    }, preds


def evaluate_multiview(X_train, y_train, X_test, y_test, seed, ind_vistas, base_fn, meta_fn):
    """MultiViewStacking parametrizado: base_fn y meta_fn son factories que reciben seed."""
    n = len(ind_vistas)
    base_learners = base_fn(seed, n)
    meta_learner  = meta_fn(seed)

    mv_model = MultiViewStacking(
        views_indices=ind_vistas,
        first_level_learners=base_learners,
        meta_learner=meta_learner,
        k=KFOLD,
        random_state=seed
    )
    mv_model.fit(X_train, y_train)
    preds = mv_model.predict(X_test)
    return {
        "model":    "MultiViewStacking",
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average='macro'),
    }, preds


def evaluate_individual_views(X_train_scaled, y_train, X_test_scaled, y_test,
                               seed, ind_vistas, nombres_vistas):
    """RF puro por cada vista — diagnóstico de contribución individual."""
    results = []
    for i, indices_vista in enumerate(ind_vistas):
        X_tr = X_train_scaled[:, indices_vista]
        X_te = X_test_scaled[:, indices_vista]
        v_result, v_preds = evaluate_singleview(X_tr, y_train, X_te, y_test, seed)
        v_result["model"] = f"Vista_{nombres_vistas[i]}"
        results.append((v_result, v_preds))
    return results


def summarize_results(df_runs):
    return (df_runs
            .groupby(["experiment", "model"])[["accuracy", "f1_macro"]]
            .agg(["mean", "std"])
            .reset_index())


# =====================================================================
# FLUJO PRINCIPAL
# =====================================================================

def main():
    print("=" * 70)
    print("EXPERIMENTO: ABLACION DE META-LEARNER EN MULTIVIEW STACKING")
    print(f"   RF_ESTIMATORS={RF_ESTIMATORS} | KFOLD={KFOLD} | N_RUNS={N_RUNS}")
    print(f"   Experimentos: {[e['name'] for e in EXPERIMENTS]}")
    print("=" * 70)

    df = pd.read_csv(INPUT_DATASET)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    le = LabelEncoder()
    y  = le.fit_transform(df['Label'])
    X  = df.drop(columns=['Label'])
    colnames = list(X.columns)

    seeds = np.random.RandomState(SEED_GENERATOR_SEED).randint(0, 10000, size=N_RUNS)

    all_results   = []
    all_per_class = []

    for exp in EXPERIMENTS:
        exp_name   = exp["name"]
        vistas_def = exp["vistas"]
        nombres_v  = [v[0] for v in vistas_def]
        features_v = [v[1] for v in vistas_def]
        base_fn    = exp["base_fn"]
        meta_fn    = exp["meta_fn"]

        ind_vistas = [
            [colnames.index(c) for c in feat_list if c in colnames]
            for feat_list in features_v
        ]

        print(f"\n{'='*70}")
        print(f"EXPERIMENTO: {exp_name}")
        print(f"   Meta-learner: {meta_fn(0).__class__.__name__}")
        for n, idx in zip(nombres_v, ind_vistas):
            print(f"   Vista '{n}': {len(idx)} features")
        print(f"{'='*70}")

        for run_idx, run_seed in enumerate(seeds):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=run_seed, stratify=y
            )

            imputer = SimpleImputer(strategy='median')
            X_train_clean = imputer.fit_transform(X_train)
            X_test_clean  = imputer.transform(X_test)

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_clean)
            X_test_scaled  = scaler.transform(X_test_clean)

            # SingleView — referencia estable
            sv_result, sv_preds = evaluate_singleview(
                X_train_scaled, y_train, X_test_scaled, y_test, run_seed
            )
            sv_result["run"] = run_idx + 1
            sv_result["experiment"] = exp_name
            all_results.append(sv_result)
            all_per_class.extend(metrics_per_class(
                y_test, sv_preds, le, "SingleView", exp_name, run_idx + 1, run_seed
            ))

            # MultiViewStacking con configuración del experimento
            mv_result, mv_preds = evaluate_multiview(
                X_train_scaled, y_train, X_test_scaled, y_test,
                run_seed, ind_vistas, base_fn, meta_fn
            )
            mv_result["run"] = run_idx + 1
            mv_result["experiment"] = exp_name
            all_results.append(mv_result)
            all_per_class.extend(metrics_per_class(
                y_test, mv_preds, le, "MultiViewStacking", exp_name, run_idx + 1, run_seed
            ))

            # Vistas individuales (diagnóstico)
            vista_results = evaluate_individual_views(
                X_train_scaled, y_train, X_test_scaled, y_test,
                run_seed, ind_vistas, nombres_v
            )
            for v_result, v_preds in vista_results:
                v_result["run"] = run_idx + 1
                v_result["experiment"] = exp_name
                all_results.append(v_result)
                all_per_class.extend(metrics_per_class(
                    y_test, v_preds, le, v_result["model"], exp_name, run_idx + 1, run_seed
                ))

            vista_accs = " | ".join(
                f"{vr['model'].replace('Vista_','')}={vr['accuracy']:.4f}"
                for vr, _ in vista_results
            )
            print(
                f"  Run {run_idx+1:02d}/{N_RUNS} seed={run_seed} | "
                f"SV={sv_result['accuracy']:.4f} MV={mv_result['accuracy']:.4f} | {vista_accs}"
            )

    # Guardar
    print("\nGuardando reportes en disco...")
    df_runs      = pd.DataFrame(all_results)
    df_summary   = summarize_results(df_runs)
    df_per_class = pd.DataFrame(all_per_class)

    df_runs.to_csv(OUTPUT_PER_RUN,     index=False)
    df_summary.to_csv(OUTPUT_SUMMARY,   index=False)
    df_per_class.to_csv(OUTPUT_PER_CLASS, index=False)

    # Resumen comparativo
    print("\n" + "=" * 70)
    print("RESUMEN: MultiViewStacking vs SingleView  (diff = MV - SV)")
    print("=" * 70)
    mv_sv = df_runs[df_runs["model"].isin(["SingleView", "MultiViewStacking"])]
    pivot = mv_sv.groupby(["experiment", "model"])["accuracy"].mean().unstack("model")
    pivot["diff_MV_vs_SV"] = pivot["MultiViewStacking"] - pivot["SingleView"]
    print(pivot.sort_values("diff_MV_vs_SV", ascending=False).to_string())

    print("\n--- F1-macro ---")
    pivot_f1 = mv_sv.groupby(["experiment", "model"])["f1_macro"].mean().unstack("model")
    pivot_f1["diff_MV_vs_SV"] = pivot_f1["MultiViewStacking"] - pivot_f1["SingleView"]
    print(pivot_f1.sort_values("diff_MV_vs_SV", ascending=False).to_string())

    print(f"\nArchivos CSV guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()