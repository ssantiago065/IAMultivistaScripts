import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Clasificadores para los modelos base y meta-modelo
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Importar la librería
from multiviewstacking import MultiViewStacking

# ==========================================
# 1. DEFINICIÓN DE VISTAS (Pega tus columnas aquí)
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

    # 2. CARGAR EL DATASET DE PRUEBA
    try:
        df = pd.read_csv('../dataset_poc_multivista.csv')
        print(f"✅ Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
    except FileNotFoundError:
        print("❌ Error: No se encontró 'dataset_poc_multivista.csv'.")
        return

    # 3. CODIFICAR LAS ETIQUETAS (Requisito de la librería)
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['Label'])
    
    # Separar características (X) y variable objetivo (y)
    y = df['Label_Encoded']
    # Eliminamos la etiqueta original y la codificada para dejar solo características
    X = df.drop(columns=['Label', 'Label_Encoded'])

    # ==========================================
    # NUEVO: LIMPIEZA DE DATOS (NaNs e Infinitos)
    # ==========================================
    print("🧹 Limpiando valores Nulos e Infinitos...")
    
    # 1. Convertir todo a numérico por si algo se leyó como texto
    # (errors='coerce' forzará cualquier texto raro a NaN)
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # 2. Reemplazar valores Infinitos por NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # 3. Rellenar los NaN con la media de su respectiva columna (Imputación)
    X = X.fillna(X.mean())

    colnames = list(X.columns)

    # 4. OBTENER LOS ÍNDICES DE CADA VISTA
    # List comprehension que busca el índice numérico de cada columna, 
    # verificando que exista en X para evitar errores si alguna se eliminó.
    ind_tiempo = [colnames.index(c) for c in vista_tiempo if c in colnames]
    ind_volumen = [colnames.index(c) for c in vista_volumen if c in colnames]
    ind_banderas = [colnames.index(c) for c in vista_banderas if c in colnames]
    ind_topologia = [colnames.index(c) for c in vista_topologia if c in colnames]

    # Validar que las vistas no estén vacías
    if not all([ind_tiempo, ind_volumen, ind_banderas, ind_topologia]):
        print("⚠️ Advertencia: Una o más vistas están vacías. Asegúrate de pegar los nombres de las columnas.")
        return
    
    print("📏 Escalando los datos...")
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X = X_scaled

    # 5. SEPARACIÓN EN TRAIN / TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"✂️  Datos separados: {len(X_train)} Train | {len(X_test)} Test")

    # 6. DEFINICIÓN DE LOS MODELOS (First-Level Learners)
    # Puedes ajustar estos algoritmos según veas qué funciona mejor para cada vista
    """"
    Default
    modelo_tiempo = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_volumen = DecisionTreeClassifier(random_state=42)
    modelo_banderas = KNeighborsClassifier(n_neighbors=5)
    modelo_topologia = GaussianNB()
    """
    modelo_tiempo = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_volumen = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_banderas = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_topologia = RandomForestClassifier(n_estimators=50, random_state=42)

    # Meta-Learner (El modelo que decide basándose en las predicciones de los 4 anteriores)
    meta_learner = RandomForestClassifier(n_estimators=50, random_state=42)

    # 7. CREACIÓN Y ENTRENAMIENTO DEL ENSAMBLE MULTIVISTA
    print("\n⚙️  Entrenando modelo MultiViewStacking... (Esto puede tomar unos minutos)")
    modelo_multivista = MultiViewStacking(
        views_indices=[ind_tiempo, ind_volumen, ind_banderas, ind_topologia],
        first_level_learners=[modelo_tiempo, modelo_volumen, modelo_banderas, modelo_topologia],
        meta_learner=meta_learner,
        k=5, # 5-fold cross-validation interno (suficiente para la PoC)
        random_state=42
    )

    modelo_multivista.fit(X_train.values, y_train.values)
    print("✅ Entrenamiento completado.")

    # 8. EVALUACIÓN DEL MODELO
    print("\n" + "="*60)
    print("📊 RESULTADOS EN EL CONJUNTO DE PRUEBA")
    print("="*60)
    
    # Se pasa .values porque a veces las librerías construidas sobre numpy
    # prefieren arrays crudos en lugar de DataFrames de pandas
    predicciones = modelo_multivista.predict(X_test.values)
    
    accuracy = accuracy_score(y_test, predicciones)
    print(f"🎯 Exactitud Global (Accuracy): {accuracy:.4f}\n")
    
    # Convertir las predicciones numéricas de vuelta a sus nombres reales (Benign, DDoS, etc.)
    nombres_reales_test = le.inverse_transform(y_test)
    nombres_reales_pred = le.inverse_transform(predicciones)
    
    print(classification_report(nombres_reales_test, nombres_reales_pred))

if __name__ == "__main__":
    entrenar_poc()