"""
Análisis Exploratorio del Dataset CIC IoT-DIAD 2024 usando YData Profiling
Este script genera reportes HTML detallados con estadísticas y visualizaciones
"""

import os
import pandas as pd
import glob
from pathlib import Path
from ydata_profiling import ProfileReport
import warnings

warnings.filterwarnings('ignore')

# Directorio raíz del dataset
DATASET_ROOT = "/home/sergio/Codigo/ai-multivista/cic-iot-diad-2024-dataset/FlowBased"  # Cambia esto a tu ruta local

def get_category_from_path(file_path):
    """Determina la categoría del ataque basándose en la ruta del archivo"""
    path_parts = Path(file_path).parts
    
    for part in path_parts:
        part_upper = part.upper()
        
        if 'BENIGN' in part_upper:
            return 'Benign'
        elif 'BRUTE' in part_upper:
            return 'Brute Force'
        elif 'DDOS' in part_upper:
            return 'DDoS'
        elif 'DOS' in part_upper and 'DDOS' not in part_upper:
            return 'DoS'
        elif 'MIRAI' in part_upper:
            return 'Mirai'
        elif 'RECON' in part_upper:
            return 'Recon'
        elif 'SPOOF' in part_upper:
            return 'Spoofing'
        elif 'WEB' in part_upper or 'SQL' in part_upper or 'XSS' in part_upper or 'UPLOAD' in part_upper:
            return 'Web-Based'
        elif 'MQTT' in part_upper:
            return 'MQTT'
    
    return 'Other'

def create_sample_dataset(sample_size_per_category=5000):
    """
    Crea un dataset de muestra balanceado tomando muestras de cada categoría
    """
    print("="*80)
    print("CREANDO DATASET DE MUESTRA PARA ANÁLISIS EXPLORATORIO")
    print("="*80)
    
    # Encontrar todos los archivos CSV
    csv_files = glob.glob(os.path.join(DATASET_ROOT, '**', '*.csv'), recursive=True)
    csv_files = [f for f in csv_files if 'Analysis' not in f]
    
    # Agrupar archivos por categoría
    files_by_category = {}
    for csv_file in csv_files:
        category = get_category_from_path(csv_file)
        if category not in files_by_category:
            files_by_category[category] = []
        files_by_category[category].append(csv_file)
    
    print(f"Categorías encontradas: {list(files_by_category.keys())}\n")
    
    # Crear dataframe de muestra
    sample_dfs = []
    
    for category, files in files_by_category.items():
        print(f"Procesando categoría: {category}")
        print(f"  - Archivos: {len(files)}")
        
        category_samples = []
        samples_per_file = max(1, sample_size_per_category // len(files))
        
        for file in files:
            try:
                # Leer una muestra del archivo
                # Primero obtener el número total de líneas
                df_temp = pd.read_csv(file, nrows=1)
                total_rows = sum(1 for _ in open(file, encoding='utf-8', errors='ignore')) - 1
                
                if total_rows > 0:
                    # Tomar muestra aleatoria
                    skip_rows = sorted(pd.np.random.choice(range(1, total_rows + 1), 
                                                           size=min(samples_per_file, total_rows), 
                                                           replace=False))
                    df_sample = pd.read_csv(file, skiprows=lambda x: x not in skip_rows and x != 0, 
                                           low_memory=False)
                    
                    # Agregar columna de categoría
                    df_sample['Attack_Category'] = category
                    category_samples.append(df_sample)
                    
                    print(f"    ✓ {Path(file).name}: {len(df_sample)} muestras")
                    
            except Exception as e:
                print(f"    ✗ Error en {Path(file).name}: {str(e)}")
                continue
        
        if category_samples:
            # Combinar todas las muestras de la categoría
            category_df = pd.concat(category_samples, ignore_index=True)
            
            # Tomar muestra final si excede el límite
            if len(category_df) > sample_size_per_category:
                category_df = category_df.sample(n=sample_size_per_category, random_state=42)
            
            sample_dfs.append(category_df)
            print(f"  Total muestras de {category}: {len(category_df)}\n")
    
    # Combinar todas las categorías
    if sample_dfs:
        final_df = pd.concat(sample_dfs, ignore_index=True)
        print(f"\n{'='*80}")
        print(f"Dataset de muestra creado: {len(final_df)} instancias")
        print(f"{'='*80}\n")
        
        # Distribución por categoría
        print("Distribución de muestras por categoría:")
        print(final_df['Attack_Category'].value_counts().to_string())
        print()
        
        return final_df
    else:
        raise ValueError("No se pudieron cargar muestras del dataset")

def generate_profiling_report(df, output_filename='profiling_report.html', title='CIC IoT-DIAD 2024 Dataset'):
    """
    Genera un reporte de profiling usando YData Profiling
    """
    print("\n" + "="*80)
    print("GENERANDO REPORTE DE PROFILING")
    print("="*80)
    
    # Configuración del reporte
    profile = ProfileReport(
        df,
        title=title,
        explorative=True,
        dark_mode=False,
        minimal=False,
        # Configuraciones adicionales para datasets grandes
        correlations={
            "auto": {"calculate": True},
            "pearson": {"calculate": True},
            "spearman": {"calculate": False},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": False},
            "cramers": {"calculate": False},
        },
        interactions=None,  # Desactivar para datasets grandes
        missing_diagrams={
            "bar": True,
            "matrix": True,
            "heatmap": True,
        },
    )
    
    output_dir = os.path.join(DATASET_ROOT, 'Analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Generando reporte... (esto puede tomar varios minutos)")
    profile.to_file(output_path)
    
    print(f"\n✓ Reporte guardado en: {output_path}")
    return output_path

def generate_analysis_by_category(sample_df):
    """
    Genera reportes separados para cada categoría de ataque
    """
    print("\n" + "="*80)
    print("GENERANDO REPORTES POR CATEGORÍA")
    print("="*80)
    
    categories = sample_df['Attack_Category'].unique()
    
    for category in categories:
        print(f"\nProcesando categoría: {category}")
        
        # Filtrar datos de la categoría
        category_df = sample_df[sample_df['Attack_Category'] == category].copy()
        
        # Eliminar la columna de categoría para el análisis
        category_df = category_df.drop('Attack_Category', axis=1)
        
        # Generar reporte
        filename = f'profiling_report_{category.replace(" ", "_").replace("/", "_")}.html'
        
        try:
            generate_profiling_report(
                category_df, 
                output_filename=filename,
                title=f'CIC IoT-DIAD 2024 - {category}'
            )
        except Exception as e:
            print(f"  ✗ Error generando reporte para {category}: {str(e)}")

def main():
    """
    Función principal
    """
    print("\n" + "="*80)
    print("ANÁLISIS EXPLORATORIO - CIC IoT-DIAD 2024")
    print("="*80 + "\n")
    
    # Crear dataset de muestra
    sample_df = create_sample_dataset(sample_size_per_category=5000)
    
    # Guardar dataset de muestra
    output_dir = os.path.join(DATASET_ROOT, 'Analysis')
    sample_path = os.path.join(output_dir, 'sample_dataset.csv')
    sample_df.to_csv(sample_path, index=False)
    print(f"Dataset de muestra guardado en: {sample_path}\n")
    
    # Generar reporte general
    print("\n" + "="*80)
    print("GENERANDO REPORTE GENERAL")
    print("="*80)
    generate_profiling_report(sample_df, output_filename='profiling_report_general.html')
    
    # Preguntar si generar reportes por categoría
    print("\n" + "="*80)
    print("¿Deseas generar reportes individuales por categoría?")
    print("Esto generará un reporte HTML para cada tipo de ataque.")
    print("="*80)
    
    generate_by_category = input("Generar reportes por categoría? (s/n): ").lower().strip()
    
    if generate_by_category == 's':
        generate_analysis_by_category(sample_df)
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print(f"\nTodos los archivos han sido guardados en:")
    print(f"  {output_dir}")

if __name__ == "__main__":
    main()
