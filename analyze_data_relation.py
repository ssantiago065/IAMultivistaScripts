import os
import pandas as pd
from pathlib import Path

# ==============================================================================
# 🛠️ SOLUCIÓN A RUTAS: Rutas relativas y dinámicas
# Esto asegura que funcione en tu Linux y en el Windows de tus compañeros
# Asume que el script está en: ai-multivista/Scripts/analisis_multivista.py
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "cic-iot-diad-2024-dataset"

FLOW_DIR = DATASET_ROOT / "FlowBased"
PACKET_DIR = DATASET_ROOT / "PacketBased"

def encontrar_archivo_comun(categoria="Brute Force"):
    """Busca un archivo que exista tanto en FlowBased como en PacketBased para comparar"""
    # Buscamos el primer CSV en FlowBased para la categoría dada
    archivos_flow = list(FLOW_DIR.rglob(f"*{categoria.replace(' ', '')}*.csv"))
    if not archivos_flow:
        archivos_flow = list(FLOW_DIR.rglob("*.csv")) # Fallback a cualquiera
    
    if not archivos_flow:
        raise FileNotFoundError(f"No se encontraron CSVs en {FLOW_DIR}")

    archivo_flow = archivos_flow[0]
    nombre_base = archivo_flow.name
    
    # Buscamos el equivalente en PacketBased
    archivo_packet = list(PACKET_DIR.rglob(nombre_base))
    
    if not archivo_packet:
        print(f"⚠️ No se encontró el equivalente de {nombre_base} en PacketBased.")
        # Fallback: tomamos el primer archivo de PacketBased para comparar al menos la estructura
        archivo_packet = list(PACKET_DIR.rglob("*.csv"))[0]
    else:
        archivo_packet = archivo_packet[0]

    return archivo_flow, archivo_packet

def analizar_relacion_1_a_1(flow_path, packet_path):
    print("="*80)
    print("🔍 ANÁLISIS DE RELACIÓN: FLOW vs PACKET")
    print("="*80)
    print(f"📁 Flow File:   {flow_path.relative_to(PROJECT_ROOT)}")
    print(f"📁 Packet File: {packet_path.relative_to(PROJECT_ROOT)}\n")

    # Leer solo un chunk para no saturar memoria en pruebas
    try:
        df_flow = pd.read_csv(flow_path, nrows=100000)
        df_packet = pd.read_csv(packet_path, nrows=100000)
    except Exception as e:
        print(f"Error leyendo archivos: {e}")
        return

    # Limpiar columnas
    df_flow.columns = df_flow.columns.str.strip()
    df_packet.columns = df_packet.columns.str.strip()

    print(f"📊 Dimensiones (Filas, Columnas):")
    print(f"   Flow:   {df_flow.shape}")
    print(f"   Packet: {df_packet.shape}\n")

    # 1. Analizar Columnas Comunes e Identificadores
    cols_flow = set(df_flow.columns)
    cols_packet = set(df_packet.columns)
    cols_comunes = cols_flow.intersection(cols_packet)
    
    print(f"🔗 Columnas en común ({len(cols_comunes)}):")
    
    # Identificadores clave de red (la 5-tupla)
    claves_red = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol', 'Timestamp']
    claves_comunes = [c for c in claves_red if c in cols_comunes]
    
    print(f"   Claves de red encontradas en ambos: {claves_comunes}\n")

    # 2. Prueba de Relación 1 a 1 (Merge)
    if claves_comunes:
        print("🔄 Intentando cruzar (merge) datos para verificar relación 1 a 1...")
        # Intentamos hacer un inner join usando las claves comunes
        merged_df = pd.merge(df_flow, df_packet, on=claves_comunes, how='inner')
        
        print(f"   Filas cruzadas con éxito: {len(merged_df)}")
        if len(merged_df) == len(df_flow) and len(df_flow) == len(df_packet):
            print("   ✅ CONCLUSIÓN: EXISTE una relación 1 a 1 perfecta en esta muestra. IDEAL para multivista.")
        elif len(merged_df) > 0:
            print(f"   ⚠️ CONCLUSIÓN: Relación parcial. Flow tiene {len(df_flow)} filas, Packet tiene {len(df_packet)}, cruzaron {len(merged_df)}.")
            print("   Probablemente un Flujo representa múltiples Paquetes (1 a N).")
        else:
            print("   ❌ CONCLUSIÓN: No cruzó ninguna fila. Los timestamps o IDs no coinciden exactamente.")
    else:
        print("❌ No hay identificadores de red comunes para intentar un cruce.")

    # 3. Limpieza de Datos (Detección de columnas inútiles)
    print("\n🧹 ANÁLISIS DE LIMPIEZA (Columnas candidatas a eliminar):")
    for nombre, df in [("Flow", df_flow), ("Packet", df_packet)]:
        # Columnas con un solo valor constante (Varianza = 0)
        constantes = [col for col in df.columns if df[col].nunique() <= 1]
        print(f"   [{nombre}] Columnas con 1 solo valor (no aportan al modelo): {len(constantes)}")
        if constantes:
            print(f"      Ejemplos: {constantes[:5]}")

if __name__ == "__main__":
    try:
        f_path, p_path = encontrar_archivo_comun()
        analizar_relacion_1_a_1(f_path, p_path)
    except Exception as e:
        print(f"Error fatal: {e}")