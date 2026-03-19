import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Simularemos la carga de datos. En tu caso, cargarás tu CSV unificado.
# df = pd.read_csv('tu_dataset_50GB_unificado.csv') 
# NOTA: Para 50GB, considera procesar en chunks o usar Dask si pandas se queda sin RAM.

def preparar_y_balancear(df, target_column='Label', metodo='hibrido'):
    print(f"Distribución original:\n{df[target_column].value_counts()}\n")

    # 1. SEPARAR PRIMERO (Regla de Oro) - Guardamos el 20% para Test real
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y 
    )
    
    print(f"Tamaño Train: {len(y_train)} | Tamaño Test: {len(y_test)}")

    X_train_bal, y_train_bal = X_train, y_train

    # 2. APLICAR TÉCNICAS (Solo al conjunto de entrenamiento)
    
    if metodo == 'undersampling':
        # Reduce TODAS las clases a la cantidad de la clase minoritaria
        print("Aplicando Undersampling puro...")
        rus = RandomUnderSampler(random_state=42)
        X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

    elif metodo == 'smote':
        # Aumenta TODAS las clases a la cantidad de la clase mayoritaria (CUIDADO: RAM)
        print("Aplicando SMOTE puro...")
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    elif metodo == 'hibrido':
        # Tu idea: Punto medio. 
        # Definimos una estrategia manual: limitamos las mayorías y subimos las minorías a un tope (ej. 200,000)
        # Ajusta este número según tu capacidad de RAM y tiempo de entrenamiento
        PUNTO_MEDIO = 200000 
        
        # Diccionarios de estrategia
        # under_strategy: solo toca las clases que tienen MÁS de 200k
        under_strategy = {clase: PUNTO_MEDIO for clase, count in y_train.value_counts().items() if count > PUNTO_MEDIO}
        # over_strategy: solo toca las clases que tienen MENOS de 200k
        over_strategy = {clase: PUNTO_MEDIO for clase, count in y_train.value_counts().items() if count < PUNTO_MEDIO}

        print(f"Aplicando Híbrido (Punto medio: {PUNTO_MEDIO})...")
        # Pipeline: Primero recorta los gigantes, luego hace SMOTE a los pequeños
        pipeline = Pipeline([
            ('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)),
            ('smote', SMOTE(sampling_strategy=over_strategy, random_state=42))
        ])
        X_train_bal, y_train_bal = pipeline.fit_resample(X_train, y_train)

    elif metodo == 'class_weights':
        # No modificamos los datos, calculamos los pesos para dárselos al modelo (PyTorch/TensorFlow)
        print("Calculando Pesos de Clase (Data intacta)...")
        clases_unicas = np.unique(y_train)
        pesos = compute_class_weight('balanced', classes=clases_unicas, y=y_train)
        diccionario_pesos = dict(zip(clases_unicas, pesos))
        print("Pesos generados para el modelo:", diccionario_pesos)
        # Retornamos los datos tal cual, el balanceo ocurrirá en el cálculo de pérdida del modelo
        return X_train, X_test, y_train, y_test, diccionario_pesos

    print(f"\nDistribución Train Balanceado ({metodo}):\n{pd.Series(y_train_bal).value_counts()}")
    
    return X_train_bal, X_test, y_train_bal, y_test, None

# Ejemplo de uso:
# X_train, X_test, y_train, y_test, pesos = preparar_y_balancear(df, metodo='hibrido')