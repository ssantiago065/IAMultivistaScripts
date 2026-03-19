import os
import pandas as pd
from pathlib import Path
import random

# Rutas dinámicas
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FLOW_DIR = PROJECT_ROOT / "cic-iot-diad-2024-dataset" / "FlowBased"

def obtener_muestra_datos(n_archivos=5, filas_por_archivo=20000):
    """Lee una muestra de múltiples archivos para un análisis rápido de varianza."""
    archivos_csv = list(FLOW_DIR.rglob("*.csv"))
    if not archivos_csv:
        raise FileNotFoundError(f"No se encontraron CSVs en {FLOW_DIR}")
    
    archivos_muestra = random.sample(archivos_csv, min(n_archivos, len(archivos_csv)))
    dfs = []
    
    print("Lectura de muestra en progreso...")
    for archivo in archivos_muestra:
        try:
            df_temp = pd.read_csv(archivo, nrows=filas_por_archivo)
            df_temp.columns = df_temp.columns.str.strip() # Limpieza vital
            dfs.append(df_temp)
        except Exception as e:
            print(f"Error leyendo {archivo.name}: {e}")
            
    df_maestro = pd.concat(dfs, ignore_index=True)
    print(f"Muestra cargada: {df_maestro.shape[0]} filas, {df_maestro.shape[1]} columnas.\n")
    return df_maestro

def analizar_y_agrupar_caracteristicas(df):
    print("="*80)
    print("📊 ANÁLISIS DE CARACTERÍSTICAS Y FEATURE SPLITTING")
    print("="*80)

    todas_las_columnas = set(df.columns)
    
    # 1. CARACTERÍSTICAS IRRELEVANTES (Metadatos e Identificadores)
    # Excluidos explícitamente para evitar memorización y fugas de datos
    identificadores = {'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label'}
    columnas_identificadoras = todas_las_columnas.intersection(identificadores)
    
    # 2. CARACTERÍSTICAS DE VARIANZA CERO
    # Si una columna tiene el mismo valor en todas las filas (ej. todos los paquetes son UDP), se descarta
    columnas_constantes = set(df.columns[df.nunique() <= 1])
    
    irrelevantes_totales = columnas_identificadoras.union(columnas_constantes)
    
    # Filtrar el dataframe para quedarnos con las útiles
    columnas_utiles = todas_las_columnas - irrelevantes_totales
    
    # 3. DEFINICIÓN DE LAS VISTAS (Basado en la taxonomía de CICFlowMeter)
    
    vistas = {
        "Vista_Tiempo_y_Comportamiento": [], # IAT, Active/Idle, Duración, Tasas
        "Vista_Volumen_y_Tamano": [],        # Longitudes, Segmentos, Bulk, Bytes
        "Vista_Banderas_y_Control": [],      # Flags TCP, Down/Up
        "Vista_Topologia_Local": []          # Puertos, Protocolo
    }
    
    # Palabras clave para asignar columnas a sus vistas
    kw_tiempo = ['Duration', 'IAT', 'Active', 'Idle', 'Packets/s', 'Bytes/s']
    kw_banderas = ['Flag', 'PSH', 'URG', 'Down/Up']
    kw_topologia = ['Port', 'Protocol']
    # El resto que no encaje, generalmente son de Volumen (Length, Size, Bulk, Bytes)
    
    for col in columnas_utiles:
        asignada = False
        
        # Revisar Topología
        if any(kw in col for kw in kw_topologia):
            vistas["Vista_Topologia_Local"].append(col)
            asignada = True
            continue
            
        # Revisar Banderas
        if any(kw in col for kw in kw_banderas):
            vistas["Vista_Banderas_y_Control"].append(col)
            asignada = True
            continue
            
        # Revisar Tiempo
        if any(kw in col for kw in kw_tiempo):
            vistas["Vista_Tiempo_y_Comportamiento"].append(col)
            asignada = True
            continue
            
        # Si no es ninguna de las anteriores, es Volumen/Tamaño
        if not asignada:
            vistas["Vista_Volumen_y_Tamano"].append(col)

    # REPORTE
    print(f"🗑️  COLUMNAS DESCARTADAS ({len(irrelevantes_totales)}):")
    print(f"   - Identificadores y Labels: {list(columnas_identificadoras)}")
    if columnas_constantes:
        print(f"   - Constantes (Varianza 0): {list(columnas_constantes)}")
    else:
        print("   - Constantes (Varianza 0): Ninguna detectada en esta muestra.")
        
    print("\n" + "-"*80)
    print("👁️  VISTAS GENERADAS PARA EL MODELO MULTIVISTA")
    print("-"*80)
    
    for nombre_vista, columnas in vistas.items():
        print(f"\n📌 {nombre_vista.replace('_', ' ').upper()} ({len(columnas)} características):")
        # Imprimir en grupos de 3 para que sea legible
        for i in range(0, len(columnas), 3):
            print("   " + ", ".join(columnas[i:i+3]))

if __name__ == "__main__":
    try:
        df_muestra = obtener_muestra_datos()
        analizar_y_agrupar_caracteristicas(df_muestra)
    except Exception as e:
        print(f"Error en la ejecución: {e}")