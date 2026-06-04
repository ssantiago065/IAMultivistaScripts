from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

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
# FACTORIES DE CLASIFICADORES
# =====================================================================
# Todos los experimentos usan RF base + RF meta (configuración ganadora).
# Definidas aquí como funciones para ser explícitas y reutilizables.

def rf_base(seed, n):
    """RF-150 para cada vista base. n = número de vistas del experimento."""
    return [
        RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=seed, n_jobs=-1)
        for _ in range(n)
    ]

def rf_meta(seed):
    """RF-150 como meta-learner."""
    return RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=seed, n_jobs=-1)

# =====================================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# =====================================================================
# Historial de decisiones:
#
#   Experimento base RF+RF con 4 vistas (100k muestras/clase):
#     SingleView        acc=88.68%  f1=88.74%
#     MultiViewStacking acc=88.50%  f1=88.57%   ← MV pierde −0.18%
#     Vista_Volumen     acc=83.19%  (más fuerte)
#     Vista_Tiempo      acc=76.36%
#     Vista_Topologia   acc=69.77%  (3 features, débil)
#     Vista_Banderas    acc=65.63%  (más débil)
#
#   Ablación de meta-learner (mismo setup):
#     RF+RF  diff=−0.176%  ← menor pérdida, mejor opción
#     LR     diff=−0.354%  ← peor: LR no captura interacciones no lineales
#     RF-small diff=−1.305% ← peor aún: underfitting en meta-nivel
#
#   Conclusión: RF base + RF meta es la configuración correcta.
#   Siguiente palanca: configuración de vistas.
#
# Hipótesis a probar:
#   BASELINE: 4 vistas originales separadas
#   EXP A:    Topologia (3 cols, 69%) se funde en Volumen → 3 vistas
#   EXP B:    Fundir Topologia + eliminar Banderas → 2 vistas fuertes
#   EXP C:    Quitar solo Topologia, mantener Banderas → 3 vistas
#   EXP D:    Solo Tiempo + Volumen → mínimo absoluto

_rf = {"base_fn": rf_base, "meta_fn": rf_meta}   # shorthand, mismo para todos

EXPERIMENTS = [
    {
        "name": "4vistas_original",
        **_rf,
        "vistas": [
            ("Tiempo",    _tiempo),
            ("Volumen",   _volumen),
            ("Banderas",  _banderas),
            ("Topologia", _topologia),
        ],
    },
    {
        "name": "3vistas_volumen+topo",
        **_rf,
        "vistas": [
            ("Tiempo",       _tiempo),
            ("Volumen+Topo", _volumen + _topologia),
            ("Banderas",     _banderas),
        ],
    },
    {
        "name": "2vistas_tiempo+volumen+topo",
        **_rf,
        "vistas": [
            ("Tiempo",       _tiempo),
            ("Volumen+Topo", _volumen + _topologia),
        ],
    },
    {
        "name": "3vistas_sin_topologia",
        **_rf,
        "vistas": [
            ("Tiempo",   _tiempo),
            ("Volumen",  _volumen),
            ("Banderas", _banderas),
        ],
    },
    {
        "name": "2vistas_tiempo+volumen",
        **_rf,
        "vistas": [
            ("Tiempo",  _tiempo),
            ("Volumen", _volumen),
        ],
    },
]

# =====================================================================
# FUNCIONES MODULARES
# =====================================================================

def metrics_per_class(y_true, y_pred, le, model_name, experiment_name, run_idx, seed):
    """Métricas detalladas por clase para un modelo y experimento dado."""
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
                "support":    metrics["support"],
            })
    return metrics_list


def evaluate_singleview(X_train, y_train, X_test, y_test, seed):
    """RF sobre todas las features — baseline absoluto, no cambia entre experimentos."""
    rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    return {
        "model":    "SingleView",
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average='macro'),
    }, preds


def evaluate_multiview(X_train, y_train, X_test, y_test, seed, ind_vistas, base_fn, meta_fn):
    """MultiViewStacking parametrizado. base_fn y meta_fn reciben seed y devuelven estimators."""
    mv_model = MultiViewStacking(
        views_indices=ind_vistas,
        first_level_learners=base_fn(seed, len(ind_vistas)),
        meta_learner=meta_fn(seed),
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
    print("EXPERIMENTO: CONFIGURACIONES DE VISTAS (base RF + meta RF)")
    print(f"   RF_ESTIMATORS={RF_ESTIMATORS} | KFOLD={KFOLD} | N_RUNS={N_RUNS}")
    print(f"   Experimentos: {[e['name'] for e in EXPERIMENTS]}")
    print("=" * 70)

    df = pd.read_csv(INPUT_DATASET)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    le = LabelEncoder()
    y  = le.fit_transform(df["Label"])
    X  = df.drop(columns=["Label"])
    colnames = list(X.columns)

    seeds = np.random.RandomState(SEED_GENERATOR_SEED).randint(0, 10000, size=N_RUNS)

    all_results   = []
    all_per_class = []

    for exp in EXPERIMENTS:
        exp_name  = exp["name"]
        nombres_v = [v[0] for v in exp["vistas"]]
        base_fn   = exp["base_fn"]
        meta_fn   = exp["meta_fn"]

        ind_vistas = [
            [colnames.index(c) for c in feat_list if c in colnames]
            for _, feat_list in exp["vistas"]
        ]

        print(f"\n{'='*70}")
        print(f"EXPERIMENTO: {exp_name}  |  base={base_fn(0,1)[0].__class__.__name__}  meta={meta_fn(0).__class__.__name__}")
        for nombre, idx in zip(nombres_v, ind_vistas):
            print(f"   Vista '{nombre}': {len(idx)} features")
        print(f"{'='*70}")

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

            # SingleView — referencia estable (igual en todos los experimentos)
            sv_result, sv_preds = evaluate_singleview(
                X_train_scaled, y_train, X_test_scaled, y_test, run_seed
            )
            sv_result["run"] = run_idx + 1
            sv_result["experiment"] = exp_name
            all_results.append(sv_result)
            all_per_class.extend(metrics_per_class(
                y_test, sv_preds, le, "SingleView", exp_name, run_idx + 1, run_seed
            ))

            # MultiViewStacking con vistas y clasificadores del experimento
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
                f"{vr['model'].replace('Vista_', '')}={vr['accuracy']:.4f}"
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

    # Resumen comparativo en consola
    print("\n" + "=" * 70)
    print("RESUMEN: MultiViewStacking vs SingleView  (diff = MV − SV)")
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