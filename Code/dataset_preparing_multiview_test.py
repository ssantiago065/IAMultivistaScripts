# =============================================================================
# DATASET PREPARATION FOR MULTIVIEW TEST
# -----------------------------------------------------------------------------
# Este script genera un dataset balanceado para experimentos multivista,
# asegurando que no haya fuga de datos y eliminando identificadores.

import pandas as pd
from pathlib import Path

# --- CONFIGURACION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FLOW_DIR = PROJECT_ROOT / "cleaned_dataset_flowbased"
OUTPUT_FILE = PROJECT_ROOT / "dataset_poc_multivista.csv"

TARGET_SAMPLES_PER_CLASS = 100_000
CHUNK_SIZE = 100_000
RANDOM_STATE = 42
VALID_CLASSES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing"]

# Columnas estrictamente prohibidas (identificadores y fugas de tiempo)
COLUMNAS_A_ELIMINAR = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]


def get_class_dirs(flow_dir: Path) -> list[Path]:
    if not flow_dir.exists():
        raise FileNotFoundError(f"No existe directorio de entrada: {flow_dir}")
    return sorted([p for p in flow_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def read_class_data(class_dir: Path, target_samples: int) -> pd.DataFrame:
    class_label = class_dir.name
    
    csv_files = sorted(class_dir.rglob("*.csv"), key=lambda p: str(p).lower())

    if not csv_files:
        print(f"  ADVERTENCIA: No se encontraron CSVs para la clase {class_label}")
        return pd.DataFrame()

    parts = []
    total_rows = 0

    for csv_file in csv_files:
        try:
            for chunk in pd.read_csv(csv_file, chunksize=CHUNK_SIZE):
                chunk.columns = chunk.columns.str.strip()
                cols_drop = [c for c in COLUMNAS_A_ELIMINAR if c in chunk.columns]
                chunk = chunk.drop(columns=cols_drop)

                if "Label" in chunk.columns:
                    chunk["Label"] = class_label

                parts.append(chunk)
                total_rows += len(chunk)
                
                # Optimización: dejar de leer si ya superamos el objetivo holgadamente
                if total_rows >= target_samples * 1.5:
                    break
        except Exception as e:
            print(f"  ERROR leyendo {csv_file.name}: {e}")
            
        if total_rows >= target_samples * 1.5:
            break

    if not parts:
        return pd.DataFrame()

    df_class = pd.concat(parts, ignore_index=True)
    original_rows = len(df_class)

    # LÓGICA CORREGIDA: Solo Undersampling. Nunca Oversampling.
    if original_rows > target_samples:
        df_class = df_class.sample(n=target_samples, replace=False, random_state=RANDOM_STATE)
        print(f"  OK: {original_rows} filas leidas, submuestreadas a {target_samples}")
    else:
        # Si tiene 100,000 exactas o menos (clases minoritarias), se queda igual
        print(f"  OK: {original_rows} filas leidas, se mantienen intactas (sin duplicar)")

    return df_class.reset_index(drop=True)


def generar_dataset_balanceado():
    print("=" * 80)
    print("GENERANDO DATASET ESTRUCTURADO PARA MULTIVISTA (SIN FUGA DE DATOS)")
    print("=" * 80)
    print(f"Entrada: {FLOW_DIR}")
    print(f"Salida:  {OUTPUT_FILE}")
    print(f"Límite máximo por clase: {TARGET_SAMPLES_PER_CLASS}")

    class_dirs = get_class_dirs(FLOW_DIR)
    print("\nClases detectadas:", ", ".join([d.name for d in class_dirs]))

    df_parts = []
    for class_dir in class_dirs:

        if class_dir.name in VALID_CLASSES:
            print(f"\nProcesando clase: {class_dir.name}")
            df_class = read_class_data(class_dir, TARGET_SAMPLES_PER_CLASS)
            if not df_class.empty:
                df_parts.append(df_class)
        else:
            print(f"\nClase {class_dir.name} no procesada por no estas en clases validas")

    if not df_parts:
        raise RuntimeError("No se genero ningun dataframe de clase. Revisa los archivos de entrada.")

    df_final = pd.concat(df_parts, ignore_index=True)
    
    # Shuffle final para mezclar todas las clases
    df_final = df_final.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Guardar CSV
    df_final.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO")
    print("=" * 80)
    print(f"Archivo guardado en: {OUTPUT_FILE}")
    print(f"Dimensiones finales: {df_final.shape}")
    print("Distribucion real de clases (Lista para Train/Test split):")
    print(df_final["Label"].value_counts())


if __name__ == "__main__":
    generar_dataset_balanceado()
