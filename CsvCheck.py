import os
import pandas as pd
import sys

# --- CONFIGURACIÓN ---
ROOT_DIR = '../cic-iot-diad-2024-dataset'  # Directorio raíz
OUTPUT_FILE = 'CIC_IoT_2024_Master_Dataset.csv'
TARGET_CLASSES = [
    'Benign', 'DDoS', 'Brute Force', 'Spoofing', 
    'DoS', 'Recon', 'Web-based', 'Mirai'
]

def obtener_archivos_csv(root_path):
    """Busca recursivamente y ordena alfabéticamente los archivos."""
    archivos = []
    print(f"🔍 Buscando archivos CSV en: {os.path.abspath(root_path)}...")
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".csv") and file != OUTPUT_FILE:
                full_path = os.path.join(root, file)
                # Detectar etiqueta basada en la ruta
                label_detectada = "Desconocida"
                for clase in TARGET_CLASSES:
                    # Buscamos la clase como parte de la ruta (carpeta)
                    if clase in full_path.split(os.sep):
                        label_detectada = clase
                        break
                archivos.append((full_path, label_detectada))
    
    # ORDENAR: Esto asegura que el orden de unión sea siempre el mismo (alfabético)
    archivos.sort(key=lambda x: x[0])
    return archivos

def procesar_dataset():
    # 1. Limpieza inicial
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"🗑️  Archivo previo '{OUTPUT_FILE}' eliminado para empezar de cero.\n")

    # 2. Obtener lista de archivos
    lista_archivos = obtener_archivos_csv(ROOT_DIR)
    
    if not lista_archivos:
        print("❌ No se encontraron archivos CSV.")
        return

    print(f"📋 Se encontraron {len(lista_archivos)} archivos. Iniciando proceso...\n")
    print("-" * 80)
    print(f"{'ARCHIVO':<50} | {'COLS':<5} | {'CLASE':<12} | {'ESTADO'}")
    print("-" * 80)

    columnas_maestras = None
    archivos_procesados = 0
    archivos_error = 0

    for filepath, clase in lista_archivos:
        nombre_archivo = os.path.basename(filepath)
        estado = "OK"
        n_cols = 0
        
        try:
            # Leemos el archivo
            df = pd.read_csv(filepath)
            
            # --- VALIDACIÓN 1: Limpieza de nombres de columnas ---
            # Eliminamos espacios en blanco: " Label" -> "Label"
            df.columns = df.columns.str.strip()
            cols_actuales = list(df.columns)
            n_cols = len(cols_actuales)

            # --- VALIDACIÓN 2: Comparación con el Maestro ---
            if columnas_maestras is None:
                # Este es el primer archivo, define el estándar para todos los demás
                columnas_maestras = cols_actuales
                print(f"ℹ️  REFERENCIA ESTABLECIDA con: {nombre_archivo}")
                # Verificamos que tenga la columna Label original
                if 'Label' not in columnas_maestras:
                     print("⚠️  ADVERTENCIA CRÍTICA: El primer archivo no tiene columna 'Label'.")
            else:
                # Verificar si las columnas son idénticas
                if cols_actuales != columnas_maestras:
                    # Intento de corrección: ¿Son las mismas columnas pero en desorden?
                    if set(cols_actuales) == set(columnas_maestras):
                        # Reordenamos el DF actual para que coincida con el maestro
                        df = df[columnas_maestras]
                        estado = "⚠️ Reordenado"
                    else:
                        # Si faltan columnas o sobran, esto es un error grave
                        diferencia = set(columnas_maestras).symmetric_difference(set(cols_actuales))
                        print(f"{nombre_archivo:<50} | {n_cols:<5} | {clase:<12} | ❌ ERROR DE COLUMNAS")
                        print(f"   Diferencia: {diferencia}")
                        archivos_error += 1
                        continue # Saltamos este archivo, no lo agregamos

            # --- PASO 3: Inyección de Etiqueta ---
            # Si la clase es desconocida, avisamos pero procesamos (o podrías saltarlo)
            if clase == "Desconocida":
                estado = "⚠️ Clase?"
            
            # Sobreescribimos la columna Label con la carpeta padre
            df['Label'] = clase

            # --- PASO 4: Guardado (Append) ---
            es_primero = (archivos_procesados == 0)
            df.to_csv(OUTPUT_FILE, mode='a', index=False, header=es_primero)
            
            # Imprimir reporte de fila
            print(f"{nombre_archivo[-50:]:<50} | {n_cols:<5} | {clase:<12} | ✅ {estado}")
            archivos_procesados += 1

        except Exception as e:
            print(f"{nombre_archivo:<50} | {0:<5} | {clase:<12} | ❌ ERROR LECTURA: {str(e)}")
            archivos_error += 1

    print("-" * 80)
    print("\n📊 RESUMEN FINAL")
    print(f"   Archivos procesados correctamente: {archivos_procesados}")
    print(f"   Archivos con error/omitidos:     {archivos_error}")
    if archivos_procesados > 0:
        print(f"   Archivo final generado:          {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("   No se generó el archivo final debido a errores.")

if __name__ == "__main__":
    procesar_dataset()