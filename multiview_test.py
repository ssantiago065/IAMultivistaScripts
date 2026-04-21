import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# Clasificadores para los modelos base y meta-modelo
from sklearn.ensemble import RandomForestClassifier

# Importar la librería
from multiviewstacking import MultiViewStacking

# ==========================================
# DEFINICIÓN DE VISTAS
# ==========================================
vista_tiempo = [
    'Flow IAT Min', 'Bwd IAT Max', 'Bwd IAT Std',
    'Flow Packets/s', 'Bwd IAT Min', 'Active Min',
    'Fwd IAT Max', 'Active Max', 'Flow Duration',
    'Idle Mean', 'Flow IAT Max', 'Fwd Packets/s',
    'Fwd IAT Min', 'Flow IAT Mean', 'Bwd Packets/s',
    'Bwd IAT Total', 'Fwd IAT Std', 'Idle Min',
    'Active Mean', 'Bwd IAT Mean', 'Fwd IAT Mean',
    'Active Std', 'Idle Max', 'Fwd IAT Total',
    'Flow Bytes/s', 'Idle Std', 'Flow IAT Std'
]

vista_volumen = [
    'Total Bwd packets', 'Packet Length Std', 'Packet Length Min',
    'Packet Length Mean', 'Bwd Header Length', 'Subflow Fwd Packets',
    'Bwd Packet Length Max', 'Bwd Bulk Rate Avg', 'Bwd Packet Length Std',
    'Packet Length Max', 'Total Length of Bwd Packet', 'Packet Length Variance',
    'Subflow Bwd Bytes', 'Fwd Packet Length Std', 'Total Length of Fwd Packet',
    'Fwd Act Data Pkts', 'Bwd Segment Size Avg', 'Fwd Header Length',
    'Bwd Packet Length Min', 'Bwd Bytes/Bulk Avg', 'FWD Init Win Bytes', 
    'Average Packet Size', 'Total Fwd Packet', 'Fwd Packet Length Mean', 
    'Bwd Packet Length Mean', 'Fwd Packet Length Min', 'Bwd Init Win Bytes', 
    'Fwd Segment Size Avg', 'Bwd Packet/Bulk Avg', 'Fwd Seg Size Min', 
    'Subflow Bwd Packets', 'Fwd Packet Length Max'
]

vista_banderas = [
    'ACK Flag Count', 'Fwd PSH Flags', 'PSH Flag Count',
    'CWR Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'FIN Flag Count', 'Down/Up Ratio', 'ECE Flag Count'
]

vista_topologia = [
    'Protocol', 'Src Port', 'Dst Port'
]

def entrenar_poc():
    print("="*60)
    print("🚀 INICIANDO ENTRENAMIENTO MULTIVISTA (PoC)")
    print("="*60)

    # 1. CARGAR EL DATASET DE PRUEBA
    try:
        df = pd.read_csv('../dataset_poc_multivista.csv')
        print(f"✅ Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
    except FileNotFoundError:
        print("❌ Error: No se encontró '../dataset_poc_multivista.csv'. Verifica la ruta.")
        return

    # 2. CODIFICAR LAS ETIQUETAS
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['Label'])
    
    y = df['Label_Encoded']
    X = df.drop(columns=['Label', 'Label_Encoded'])

    # 3. LIMPIEZA DE DATOS
    print("🧹 Limpiando valores Nulos e Infinitos...")
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    colnames = list(X.columns)

    # 4. OBTENER LOS ÍNDICES DE CADA VISTA
    ind_tiempo = [colnames.index(c) for c in vista_tiempo if c in colnames]
    ind_volumen = [colnames.index(c) for c in vista_volumen if c in colnames]
    ind_banderas = [colnames.index(c) for c in vista_banderas if c in colnames]
    ind_topologia = [colnames.index(c) for c in vista_topologia if c in colnames]

    if not all([ind_tiempo, ind_volumen, ind_banderas, ind_topologia]):
        print("⚠️ Advertencia: Una o más vistas están vacías.")
        return
    
    # 5. ESCALADO
    print("📏 Escalando los datos...")
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X = X_scaled

    # 6. SEPARACIÓN EN TRAIN / TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"✂️  Datos separados: {len(X_train)} Train | {len(X_test)} Test")

    # ==========================================
    # 7. ENTRENAMIENTO DE MODELOS
    # ==========================================
    
    # 7.1 MODELO MULTIVISTA
    print("\n⚙️  Entrenando modelo MultiViewStacking...")
    modelo_tiempo = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_volumen = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_banderas = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_topologia = RandomForestClassifier(n_estimators=50, random_state=42)
    meta_learner = RandomForestClassifier(n_estimators=50, random_state=42)

    modelo_multivista = MultiViewStacking(
        views_indices=[ind_tiempo, ind_volumen, ind_banderas, ind_topologia],
        first_level_learners=[modelo_tiempo, modelo_volumen, modelo_banderas, modelo_topologia],
        meta_learner=meta_learner,
        k=5, 
        random_state=42
    )
    modelo_multivista.fit(X_train.values, y_train.values)
    print("   ✅ MultiViewStacking entrenado.")

    # 7.2 MODELO BASELINE (Todas las características juntas)
    print("\n⚙️  Entrenando modelo Baseline (Todas las características)...")
    modelo_baseline = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_baseline.fit(X_train.values, y_train.values)
    print("   ✅ Baseline entrenado.")

    # 7.3 MODELOS INDIVIDUALES POR VISTA
    print("\n⚙️  Entrenando modelos individuales por vista...")
    modelos_individuales = []
    vistas_indices = [ind_tiempo, ind_volumen, ind_banderas, ind_topologia]
    nombres_vistas = ['Tiempo y Comportamiento', 'Volumen y Tamaño', 'Banderas y Control', 'Topología Local']
    
    for modelo, indices, nombre in zip([modelo_tiempo, modelo_volumen, modelo_banderas, modelo_topologia], vistas_indices, nombres_vistas):
        # Entrenamos clonando un nuevo modelo para no interferir con el estado interno del MultiView
        rf_individual = RandomForestClassifier(n_estimators=50, random_state=42)
        X_train_view = X_train.iloc[:, indices]
        rf_individual.fit(X_train_view.values, y_train.values)
        modelos_individuales.append(rf_individual)
        print(f"   ✅ Modelo '{nombre}' entrenado.")

    # ==========================================
    # 8. EVALUACIÓN Y COMPARATIVA
    # ==========================================
    print("\n" + "="*80)
    print("📊 COMPARATIVA DE RESULTADOS EN EL CONJUNTO DE PRUEBA")
    print("="*80)

    diccionario_resultados = {}

    # --- Evaluar Multi-View ---
    pred_mv = modelo_multivista.predict(X_test.values)
    diccionario_resultados['Multi-View Stacking'] = {
        'Accuracy': accuracy_score(y_test, pred_mv),
        'F1-Macro': f1_score(y_test, pred_mv, average='macro')
    }

    # --- Evaluar Baseline ---
    pred_base = modelo_baseline.predict(X_test.values)
    diccionario_resultados['Baseline (Todas cols)'] = {
        'Accuracy': accuracy_score(y_test, pred_base),
        'F1-Macro': f1_score(y_test, pred_base, average='macro')
    }

    # --- Evaluar Individuales ---
    for rf_individual, indices, nombre in zip(modelos_individuales, vistas_indices, nombres_vistas):
        X_test_view = X_test.iloc[:, indices]
        pred_ind = rf_individual.predict(X_test_view.values)
        diccionario_resultados[f'Vista: {nombre}'] = {
            'Accuracy': accuracy_score(y_test, pred_ind),
            'F1-Macro': f1_score(y_test, pred_ind, average='macro')
        }

    # --- Imprimir Tabla Comparativa ---
    print(f"{'Modelo / Enfoque':<30} | {'Accuracy':<10} | {'F1-Macro':<10}")
    print("-" * 56)
    for nombre, metricas in diccionario_resultados.items():
        print(f"{nombre:<30} | {metricas['Accuracy']:.4f}     | {metricas['F1-Macro']:.4f}")

    # --- Reporte Completo del Mejor Enfoque (Opcional, enfocado al Multi-View) ---
    print("\n" + "="*80)
    print("📋 REPORTE DETALLADO: MULTI-VIEW STACKING")
    print("="*80)
    nombres_reales_test = le.inverse_transform(y_test)
    nombres_reales_pred = le.inverse_transform(pred_mv)
    print(classification_report(nombres_reales_test, nombres_reales_pred))

if __name__ == "__main__":
    entrenar_poc()