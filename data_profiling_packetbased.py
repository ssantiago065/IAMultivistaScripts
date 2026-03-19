import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class PacketBasedDataProfiler:
    def __init__(self, base_path):
        self.base_path = base_path
        self.packet_based_path = os.path.join(base_path, 
            "CIC IoT-IDAD Dataset 2024", "Dataset", 
            "Device Identification_Anomaly Detection - Packet Based Features")
        self.profiling_results = {}
        
    def analyze_file(self, file_path, attack_type):
        """Realiza análisis detallado de un archivo CSV"""
        print(f"\n{'='*80}")
        print(f"Analizando: {os.path.basename(file_path)}")
        print(f"Tipo de ataque: {attack_type}")
        print(f"{'='*80}")
        
        try:
            # Leer CSV
            df = pd.read_csv(file_path)
            
            results = {
                'file': os.path.basename(file_path),
                'attack_type': attack_type,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': {},
                'duplicate_rows': 0,
                'constant_columns': [],
                'near_constant_columns': [],
                'all_zero_rows': 0,
                'infinite_values': {},
                'negative_values': {},
                'rows_to_remove': set()
            }
            
            print(f"\nDimensiones originales: {df.shape}")
            
            # 1. Valores faltantes
            print("\n1. ANALISIS DE VALORES FALTANTES:")
            missing = df.isnull().sum()
            missing_percent = (missing / len(df)) * 100
            for col in df.columns:
                if missing[col] > 0:
                    results['missing_values'][col] = {
                        'count': int(missing[col]),
                        'percentage': float(missing_percent[col])
                    }
                    print(f"   - {col}: {missing[col]} ({missing_percent[col]:.2f}%)")
                    results['rows_to_remove'].update(df[df[col].isnull()].index.tolist())
            
            if not results['missing_values']:
                print("   [OK] No se encontraron valores faltantes")
            
            # 2. Filas duplicadas
            print("\n2. ANALISIS DE FILAS DUPLICADAS:")
            duplicates = df.duplicated()
            results['duplicate_rows'] = int(duplicates.sum())
            print(f"   - Filas duplicadas: {results['duplicate_rows']}")
            if results['duplicate_rows'] > 0:
                results['rows_to_remove'].update(df[duplicates].index.tolist())
            
            # 3. Columnas constantes
            print("\n3. ANALISIS DE COLUMNAS CONSTANTES:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in df.columns:  # Revisar todas las columnas
                if df[col].nunique() == 1:
                    results['constant_columns'].append(col)
                    print(f"   - {col}: valor constante = {df[col].iloc[0]}")
            
            if not results['constant_columns']:
                print("   [OK] No se encontraron columnas constantes")
            
            # 4. Columnas casi constantes (>99%)
            print("\n4. ANALISIS DE COLUMNAS CASI CONSTANTES (>99%):")
            for col in df.columns:
                if col not in results['constant_columns']:
                    value_counts = df[col].value_counts()
                    if len(value_counts) > 0:
                        max_freq = value_counts.iloc[0]
                        max_percent = (max_freq / len(df)) * 100
                        if max_percent > 99:
                            results['near_constant_columns'].append({
                                'column': col,
                                'dominant_value': str(value_counts.index[0])[:50],  # Truncar valores largos
                                'percentage': float(max_percent)
                            })
                            print(f"   - {col}: {max_percent:.2f}%")
            
            if not results['near_constant_columns']:
                print("   [OK] No se encontraron columnas casi constantes")
            
            # 5. Valores infinitos (solo columnas numéricas)
            print("\n5. ANALISIS DE VALORES INFINITOS:")
            for col in numeric_cols:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    results['infinite_values'][col] = int(inf_count)
                    print(f"   - {col}: {inf_count} valores infinitos")
                    results['rows_to_remove'].update(df[np.isinf(df[col])].index.tolist())
            
            if not results['infinite_values']:
                print("   [OK] No se encontraron valores infinitos")
            
            # 6. Valores negativos en columnas que deberían ser positivas
            print("\n6. ANALISIS DE VALORES NEGATIVOS ANOMALOS:")
            positive_only_cols = [col for col in numeric_cols 
                                if any(keyword in col.lower() for keyword in 
                                      ['count', 'length', 'size', 'time', 'entropy', 'sum'])]
            
            for col in positive_only_cols:
                if col in df.columns:
                    neg_count = (df[col] < 0).sum()
                    if neg_count > 0:
                        results['negative_values'][col] = int(neg_count)
                        print(f"   - {col}: {neg_count} valores negativos")
                        results['rows_to_remove'].update(df[df[col] < 0].index.tolist())
            
            if not results['negative_values']:
                print("   [OK] No se encontraron valores negativos anomalos")
            
            # Resumen
            total_rows_to_remove = len(results['rows_to_remove'])
            results['total_rows_to_remove'] = total_rows_to_remove
            percentage_to_remove = (total_rows_to_remove / len(df)) * 100
            results['percentage_to_remove'] = float(percentage_to_remove)
            
            print(f"\n{'='*80}")
            print(f"RESUMEN DE LIMPIEZA:")
            print(f"  Total de filas a eliminar: {total_rows_to_remove} ({percentage_to_remove:.2f}%)")
            print(f"  Filas que permaneceran: {len(df) - total_rows_to_remove}")
            print(f"{'='*80}")
            
            return results
            
        except Exception as e:
            print(f"ERROR al analizar {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_all_files(self, sample_size=3):
        """Analiza una muestra de archivos de cada categoría"""
        categories = ['BenignTraffic', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-Based']
        
        all_results = []
        
        for category in categories:
            category_path = os.path.join(self.packet_based_path, category)
            
            if not os.path.exists(category_path):
                print(f"\nCategoria no encontrada: {category}")
                continue
            
            # Obtener archivos CSV
            csv_files = []
            for root, dirs, files in os.walk(category_path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                print(f"\nNo se encontraron archivos CSV en: {category}")
                continue
            
            # Analizar sample_size archivos
            files_to_analyze = csv_files[:sample_size] if len(csv_files) > sample_size else csv_files
            
            print(f"\n\n{'#'*80}")
            print(f"# CATEGORIA: {category} ({len(files_to_analyze)} archivos a analizar)")
            print(f"{'#'*80}")
            
            for csv_file in files_to_analyze:
                result = self.analyze_file(csv_file, category)
                if result:
                    all_results.append(result)
        
        return all_results
    
    def generate_summary_report(self, results):
        """Genera reporte resumen"""
        print("\n\n" + "="*100)
        print("REPORTE RESUMEN DE DATA PROFILING - PACKET BASED FEATURES")
        print("="*100)
        
        if not results:
            print("No hay resultados para reportar")
            return
        
        # Resumen por categoría
        print("\n1. RESUMEN POR CATEGORIA DE ATAQUE:")
        print("-" * 100)
        categories = {}
        for result in results:
            cat = result['attack_type']
            if cat not in categories:
                categories[cat] = {
                    'files': 0,
                    'total_rows': 0,
                    'rows_to_remove': 0,
                    'files_with_issues': 0
                }
            categories[cat]['files'] += 1
            categories[cat]['total_rows'] += result['total_rows']
            categories[cat]['rows_to_remove'] += result['total_rows_to_remove']
            if result['total_rows_to_remove'] > 0:
                categories[cat]['files_with_issues'] += 1
        
        for cat, stats in categories.items():
            cleanup_percent = (stats['rows_to_remove'] / stats['total_rows'] * 100) if stats['total_rows'] > 0 else 0
            print(f"\n{cat}:")
            print(f"  - Archivos analizados: {stats['files']}")
            print(f"  - Total de filas: {stats['total_rows']:,}")
            print(f"  - Filas a eliminar: {stats['rows_to_remove']:,} ({cleanup_percent:.2f}%)")
            print(f"  - Archivos con problemas: {stats['files_with_issues']}")
        
        # Columnas constantes
        print("\n\n2. COLUMNAS CONSTANTES ENCONTRADAS:")
        print("-" * 100)
        constant_cols = set()
        for result in results:
            constant_cols.update(result['constant_columns'])
        
        if constant_cols:
            print(f"  Total: {len(constant_cols)} columnas constantes")
            for col in sorted(constant_cols):
                print(f"    - {col}")
        else:
            print("  [OK] No se encontraron columnas constantes")
        
        print("\n" + "="*100)


def main():
    """Función principal"""
    base_path = r"f:\IOTDataset"
    
    print("="*100)
    print("ANALISIS DE DATA PROFILING - PACKET BASED FEATURES")
    print("="*100)
    print(f"Ruta base: {base_path}")
    
    profiler = PacketBasedDataProfiler(base_path)
    
    # Analizar archivos (3 por categoría como muestra)
    results = profiler.analyze_all_files(sample_size=3)
    
    # Guardar resultados
    import json
    results_clean = []
    for r in results:
        r_copy = r.copy()
        r_copy['rows_to_remove'] = list(r_copy['rows_to_remove'])
        
        # Limpiar valores NaN e infinitos para JSON
        for key, value in r_copy.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (float, np.floating)):
                        if np.isnan(v) or np.isinf(v):
                            value[k] = None
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if isinstance(v, (float, np.floating)):
                                if np.isnan(v) or np.isinf(v):
                                    item[k] = None
        
        results_clean.append(r_copy)
    
    with open(os.path.join(base_path, 'profiling_results_packetbased.json'), 'w') as f:
        json.dump(results_clean, f, indent=2, default=str)
    
    print(f"\n[OK] Resultados guardados en: {os.path.join(base_path, 'profiling_results_packetbased.json')}")
    
    # Generar reporte resumen
    profiler.generate_summary_report(results)


if __name__ == "__main__":
    main()
