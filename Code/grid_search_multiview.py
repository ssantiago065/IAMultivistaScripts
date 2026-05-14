import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from multiviewstacking import MultiViewStacking

# --- RUTAS Y CONFIGURACIÓN ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DATASET = PROJECT_ROOT / "dataset_poc_multivista.csv"

# Muestra del 10% (90,000 filas) para velocidad extrema
SAMPLE_FRACTION = 0.10 
SEED = 42

# --- LISTAS DE VISTAS ---
vista_tiempo = ["Flow IAT Min", "Bwd IAT Max", "Bwd IAT Std", "Flow Packets/s", "Bwd IAT Min", "Active Min", "Fwd IAT Max", "Active Max", "Flow Duration", "Idle Mean", "Flow IAT Max", "Fwd Packets/s", "Fwd IAT Min", "Flow IAT Mean", "Bwd Packets/s", "Bwd IAT Total", "Fwd IAT Std", "Idle Min", "Active Mean", "Bwd IAT Mean", "Fwd IAT Mean", "Active Std", "Idle Max", "Fwd IAT Total", "Flow Bytes/s", "Idle Std", "Flow IAT Std"]
vista_volumen = ["Total Bwd packets", "Packet Length Std", "Packet Length Min", "Packet Length Mean", "Bwd Header Length", "Subflow Fwd Packets", "Bwd Packet Length Max", "Bwd Bulk Rate Avg", "Bwd Packet Length Std", "Packet Length Max", "Total Length of Bwd Packet", "Packet Length Variance", "Subflow Bwd Bytes", "Fwd Packet Length Std", "Total Length of Fwd Packet", "Fwd Act Data Pkts", "Bwd Segment Size Avg", "Fwd Header Length", "Bwd Packet Length Min", "Bwd Bytes/Bulk Avg", "FWD Init Win Bytes", "Average Packet Size", "Total Fwd Packet", "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Fwd Packet Length Min", "Bwd Init Win Bytes", "Fwd Segment Size Avg", "Bwd Packet/Bulk Avg", "Fwd Seg Size Min", "Subflow Bwd Packets", "Fwd Packet Length Max"]
vista_banderas = ["ACK Flag Count", "Fwd PSH Flags", "PSH Flag Count", "CWR Flag Count", "SYN Flag Count", "RST Flag Count", "FIN Flag Count", "Down/Up Ratio", "ECE Flag Count"]
vista_topologia = ["Protocol", "Src Port", "Dst Port"]

def run_fast_grid_search():
    print("="*60)
    print("⚡ INICIANDO BÚSQUEDA RÁPIDA DE ARQUITECTURAS MAESTRAS")
    print("="*60)

    # 1. Carga y preprocesamiento
    df = pd.read_csv(INPUT_DATASET)
    df = df.sample(frac=SAMPLE_FRACTION, random_state=SEED).reset_index(drop=True)
    print(f"✅ Utilizando una muestra de {len(df)} filas.")

    le = LabelEncoder()
    y = le.fit_transform(df['Label'])
    X = df.drop(columns=['Label'])
    colnames = list(X.columns)

    ind_vistas = [
        [colnames.index(c) for c in vista_tiempo if c in colnames],
        [colnames.index(c) for c in vista_volumen if c in colnames],
        [colnames.index(c) for c in vista_banderas if c in colnames],
        [colnames.index(c) for c in vista_topologia if c in colnames]
    ]

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 2. Catálogo de Algoritmos (Configurados para ser rápidos)
    rf_model = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
    dt_model = DecisionTreeClassifier(random_state=SEED)
    nb_model = GaussianNB()
    lr_meta = LogisticRegression(max_iter=1000, random_state=SEED)

    # 3. Inicialización del modelo con "Maniquíes"
    base_mvs = MultiViewStacking(
        views_indices=ind_vistas,
        first_level_learners=[rf_model, rf_model, rf_model, rf_model],
        meta_learner=lr_meta,
        k=3,
        random_state=SEED
    )

    # 4. Las 4 Combinaciones Estratégicas
    grid_params = [
        {
            # 1. Baseline Fuerte (Todo RF)
            'first_level_learners': [[rf_model, rf_model, rf_model, rf_model]],
            'meta_learner': [lr_meta]
        },
        {
            # 2. El Experto Funcional (Máxima diversidad)
            'first_level_learners': [[rf_model, dt_model, nb_model, dt_model]],
            'meta_learner': [lr_meta]
        },
        {
            # 3. Ultra-Ligero (Solo árboles simples y Naive Bayes)
            'first_level_learners': [[dt_model, dt_model, nb_model, dt_model]],
            'meta_learner': [lr_meta]
        },
        {
            # 4. Híbrido Pesado (RF en Tiempo y Topología)
            'first_level_learners': [[rf_model, dt_model, nb_model, rf_model]],
            'meta_learner': [lr_meta]
        }
    ]

    # 5. Ejecutar GridSearchCV con Verbose=3 para ver el progreso real
    print("⏳ Iniciando evaluación de las 4 arquitecturas candidatas...")
    grid_search = GridSearchCV(
        base_mvs, 
        grid_params, 
        cv=3, 
        scoring='accuracy',
        n_jobs=1,
        verbose=3 
    )

    grid_search.fit(X_scaled, y)

    # 6. Resultados
    print("\n" + "="*60)
    print("🏆 RESULTADOS DE LA PRUEBA RÁPIDA")
    print("="*60)
    print(f"🎯 Mejor Accuracy en validación cruzada: {grid_search.best_score_:.4f}")
    
    best_params = grid_search.best_params_
    print("\n🥇 Arquitectura Ganadora:")
    for i, model in enumerate(best_params['first_level_learners']):
        print(f"  - Vista {i+1} ({['Tiempo', 'Volumen', 'Banderas', 'Topología'][i]}): {model.__class__.__name__}")
    print(f"\n🧠 Meta-Modelo: {best_params['meta_learner'].__class__.__name__}")

if __name__ == "__main__":
    run_fast_grid_search()