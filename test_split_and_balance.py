import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

def preparar_y_balancear(df, target_column='Label', metodo='hibrido', punto_medio=2000):
    print("="*60)
    print(f"🛠️  INICIANDO BALANCEO: MÉTODO '{metodo.upper()}'")
    print("="*60)
    
    print("\n📊 1. DISTRIBUCIÓN ORIGINAL (MOCK):")
    print(df[target_column].value_counts())

    # --- REGLA DE ORO: SEPARAR ANTES DE BALANCEAR ---
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # stratify=y asegura que la proporción de clases se mantenga en la división
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y 
    )
    
    print(f"\n✂️  2. SEPARACIÓN TRAIN/TEST (80/20):")
    print(f"   Tamaño Train (A balancear): {len(y_train)} muestras")
    print(f"   Tamaño Test  (Intacto):     {len(y_test)} muestras")

    X_train_bal, y_train_bal = X_train, y_train

    # --- APLICAR TÉCNICAS SOLO AL TRAIN ---
    if metodo == 'hibrido':
        print(f"\n⚖️  3. APLICANDO HÍBRIDO (Objetivo por clase: ~{punto_medio})")
        
        # Estrategia: Bajar los que superan el punto medio, subir los que no llegan
        conteo_actual = y_train.value_counts()
        under_strategy = {clase: punto_medio for clase, count in conteo_actual.items() if count > punto_medio}
        # Nota para SMOTE: Requiere al menos k_neighbors (por defecto 5) muestras originales para funcionar
        # Filtramos clases que tengan muy poquitas muestras para evitar errores en esta simulación pequeña
        over_strategy = {clase: punto_medio for clase, count in conteo_actual.items() if count < punto_medio and count > 5}

        pipeline = Pipeline([
            ('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)),
            ('smote', SMOTE(sampling_strategy=over_strategy, random_state=42, k_neighbors=2)) 
        ])
        X_train_bal, y_train_bal = pipeline.fit_resample(X_train, y_train)

    elif metodo == 'class_weights':
        print("\n⚖️  3. CALCULANDO PESOS DE CLASE (Sin alterar datos)")
        clases_unicas = np.unique(y_train)
        pesos = compute_class_weight('balanced', classes=clases_unicas, y=y_train)
        diccionario_pesos = dict(zip(clases_unicas, pesos))
        print("\n   Pesos generados:")
        for clase, peso in diccionario_pesos.items():
            print(f"   - {clase}: {peso:.4f}")
        return X_train, X_test, y_train, y_test, diccionario_pesos

    print(f"\n✅ 4. RESULTADO FINAL DEL CONJUNTO DE ENTRENAMIENTO:")
    print(pd.Series(y_train_bal).value_counts())
    
    return X_train_bal, X_test, y_train_bal, y_test, None

if __name__ == '__main__':
    # ==========================================
    # CREACIÓN DEL DATASET SIMULADO (MOCK)
    # Proporciones reales divididas entre 1,000
    # ==========================================
    distribucion_mock = {
        'DoS': 14853,
        'DDoS': 3478,
        'Recon': 442,
        'Benign': 398,
        'Mirai': 174,
        'Spoofing': 157,
        'WebBased': 11,
        'BruteForce': 6 # Subido a 6 para que SMOTE (k_neighbors) no falle en el mock
    }
    
    # Generar datos falsos numéricos (ej. 5 características)
    datos = []
    etiquetas = []
    for clase, cantidad in distribucion_mock.items():
        # Creamos filas con números aleatorios para simular las características
        filas = np.random.rand(cantidad, 5) 
        datos.append(filas)
        etiquetas.extend([clase] * cantidad)
        
    df_mock = pd.DataFrame(np.vstack(datos), columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
    df_mock['Label'] = etiquetas
    
    # Barajar el dataset para simular un caso real
    df_mock = df_mock.sample(frac=1, random_state=42).reset_index(drop=True)

    # ==========================================
    # EJECUCIÓN DE PRUEBAS
    # ==========================================
    # Prueba 1: Método Híbrido (Punto medio 1000 para esta simulación a escala)
    X_tr, X_te, y_tr, y_te, _ = preparar_y_balancear(df_mock, metodo='hibrido', punto_medio=1000)
    
    print("\n" + "*"*60 + "\n")
    
    # Prueba 2: Class Weights
    X_tr, X_te, y_tr, y_te, pesos = preparar_y_balancear(df_mock, metodo='class_weights')