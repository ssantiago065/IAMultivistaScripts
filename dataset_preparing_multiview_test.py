import os
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FLOW_DIR = PROJECT_ROOT / "cic-iot-diad-2024-dataset" / "FlowBased"
OUTPUT_FILE = PROJECT_ROOT / "dataset_poc_multivista.csv"

# Parámetros para la Prueba de Concepto (PoC)
MUESTRAS_POR_CLASE = 5000  # Ajusta este número según lo que necesites para probar

# Columnas estrictamente prohibidas (identificadores y fugas de tiempo)
COLUMNAS_A_ELIMINAR = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']

# Categorías esperadas basadas en tus carpetas
TARGET_CLASSES = ['Benign', 'DDoS', 'Brute Force', 'Spoofing', 'DoS', 'Recon', 'Web-based', 'Mirai']

def generar_dataset_poc():
    print("="*80)
    print("🚀 GENERANDO DATASET DE PRUEBA (PoC) BALANCEADO")
    print("="*80)
    
    df_final = pd.DataFrame()
    
    for clase in TARGET_CLASSES:
        print(f"\nProcesando clase: {clase}...")
        muestras_recolectadas = 0
        dfs_clase = []
        
        # Buscar archivos de esta clase (ignorando mayúsculas/minúsculas o espacios)
        # Adaptación simple para encontrar las carpetas
        archivos = list(FLOW_DIR.rglob(f"*{clase.replace(' ', '*')}*.csv"))
        if not archivos:
            archivos = list(FLOW_DIR.rglob(f"*{clase.split('-')[0]}*.csv"))
            
        for archivo in archivos:
            if muestras_recolectadas >= MUESTRAS_POR_CLASE:
                break
                
            try:
                # Leemos solo un chunk pequeño para ahorrar RAM
                df_chunk = pd.read_csv(archivo, nrows=MUESTRAS_POR_CLASE)
                
                # Limpieza de nombres de columnas
                df_chunk.columns = df_chunk.columns.str.strip()
                
                # Eliminar columnas irrelevantes
                cols_drop = [c for c in COLUMNAS_A_ELIMINAR if c in df_chunk.columns]
                df_chunk = df_chunk.drop(columns=cols_drop)
                
                # Etiquetado manual
                if 'Label' in df_chunk.columns:
                    df_chunk['Label'] = clase
                    
                dfs_clase.append(df_chunk)
                muestras_recolectadas += len(df_chunk)
                
            except Exception as e:
                print(f"  ❌ Error leyendo {archivo.name}: {e}")
                
        # Concatenar lo recolectado para esta clase y recortar exactamente al límite
        if dfs_clase:
            df_unido = pd.concat(dfs_clase, ignore_index=True)
            df_recortado = df_unido.head(MUESTRAS_POR_CLASE)
            df_final = pd.concat([df_final, df_recortado], ignore_index=True)
            print(f"  ✅ {len(df_recortado)} muestras recolectadas.")
        else:
            print(f"  ⚠️ No se encontraron datos para la clase {clase}.")

    # Guardar el resultado final
    print("\n" + "="*80)
    print("💾 GUARDANDO DATASET...")
    # Barajar (shuffle) el dataset para que las clases no estén en orden secuencial
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Proceso completado. Archivo guardado en: {OUTPUT_FILE}")
    print(f"📊 Dimensiones finales: {df_final.shape}")
    print("Distribución de clases:")
    print(df_final['Label'].value_counts())

if __name__ == "__main__":
    generar_dataset_poc()