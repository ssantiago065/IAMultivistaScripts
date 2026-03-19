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

class DataCleaner:
    def __init__(self, base_path, output_path=None):
        self.base_path = base_path
        self.output_path = output_path or os.path.join(base_path, "cleaned_dataset_flowbased")
        self.flow_based_path = os.path.join(base_path, "FlowBased")
        
        # Columnas constantes que no aportan información (basadas en el profiling)
        self.constant_columns_to_remove = [
            'Bwd PSH Flags',
            'Fwd URG Flags',
            'Bwd URG Flags',
            'URG Flag Count',
            'Fwd Bytes/Bulk Avg',
            'Fwd Packet/Bulk Avg',
            'Fwd Bulk Rate Avg',
            'CWR Flag Count',
            'ECE Flag Count'
        ]
        
        self.cleaning_stats = []
    
    def _save_progress(self):
        """Guarda el progreso actual en un archivo temporal"""
        import json
        progress_file = os.path.join(self.output_path, 'cleaning_progress.json')
        with open(progress_file, 'w') as f:
            json.dump(self.cleaning_stats, f, indent=2)
        print(f"\n[PROGRESO GUARDADO: {len(self.cleaning_stats)} archivos procesados]")
    
    def clean_file(self, file_path, attack_type, output_dir):
        """Limpia un archivo CSV individual"""
        print(f"\n{'='*80}")
        print(f"Limpiando: {os.path.basename(file_path)}")
        print(f"Tipo de ataque: {attack_type}")
        
        try:
            # Leer CSV en chunks si es muy grande
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"Tamaño: {file_size_mb:.2f} MB")
            print(f"{'='*80}")
            
            # Si el archivo es muy grande (>100MB), procesarlo en chunks
            if file_size_mb > 100:
                print("Archivo grande detectado, procesando en chunks...")
                chunk_size = 100000
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                # Leer CSV
                df = pd.read_csv(file_path)
            original_rows = len(df)
            original_cols = len(df.columns)
            
            print(f"\nDimensiones originales: {df.shape}")
            
            # Estadísticas de limpieza
            stats = {
                'file': os.path.basename(file_path),
                'attack_type': attack_type,
                'original_rows': original_rows,
                'original_columns': original_cols,
                'rows_removed': {
                    'missing_values': 0,
                    'duplicates': 0,
                    'infinite_values': 0,
                    'negative_values': 0,
                    'all_zeros': 0
                },
                'columns_removed': 0
            }
            
            # 1. Eliminar filas con valores faltantes
            print("\n1. Eliminando filas con valores faltantes...")
            missing_mask = df.isnull().any(axis=1)
            missing_count = missing_mask.sum()
            if missing_count > 0:
                df = df[~missing_mask]
                stats['rows_removed']['missing_values'] = int(missing_count)
                print(f"   - Eliminadas: {missing_count} filas")
            else:
                print("   - No se encontraron valores faltantes")
            
            # 2. Eliminar filas duplicadas
            print("\n2. Eliminando filas duplicadas...")
            duplicates_count = df.duplicated().sum()
            if duplicates_count > 0:
                df = df.drop_duplicates(keep='first')
                stats['rows_removed']['duplicates'] = int(duplicates_count)
                print(f"   - Eliminadas: {duplicates_count} filas duplicadas")
            else:
                print("   - No se encontraron duplicados")
            
            # 3. Eliminar filas con valores infinitos
            print("\n3. Eliminando filas con valores infinitos...")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_mask = np.isinf(df[numeric_cols]).any(axis=1)
            inf_count = inf_mask.sum()
            if inf_count > 0:
                df = df[~inf_mask]
                stats['rows_removed']['infinite_values'] = int(inf_count)
                print(f"   - Eliminadas: {inf_count} filas con valores infinitos")
            else:
                print("   - No se encontraron valores infinitos")
            
            # 4. Eliminar filas con valores negativos anómalos
            print("\n4. Eliminando filas con valores negativos anómalos...")
            positive_only_cols = [col for col in numeric_cols 
                                if 'Length' in col or 'Bytes' in col or 'Packet' in col
                                or 'Duration' in col or 'Count' in col]
            
            negative_mask = pd.Series([False] * len(df), index=df.index)
            for col in positive_only_cols:
                if col in df.columns:
                    negative_mask |= (df[col] < 0)
            
            negative_count = negative_mask.sum()
            if negative_count > 0:
                df = df[~negative_mask]
                stats['rows_removed']['negative_values'] = int(negative_count)
                print(f"   - Eliminadas: {negative_count} filas con valores negativos")
            else:
                print("   - No se encontraron valores negativos anómalos")
            
            # 5. Eliminar filas con todos los valores numéricos en cero
            print("\n5. Eliminando filas con todos los valores en cero...")
            exclude_cols = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 
                          'Protocol', 'Timestamp', 'Label']
            numeric_cols_to_check = [col for col in numeric_cols if col not in exclude_cols]
            
            if numeric_cols_to_check:
                all_zero_mask = (df[numeric_cols_to_check] == 0).all(axis=1)
                all_zero_count = all_zero_mask.sum()
                if all_zero_count > 0:
                    df = df[~all_zero_mask]
                    stats['rows_removed']['all_zeros'] = int(all_zero_count)
                    print(f"   - Eliminadas: {all_zero_count} filas con todos los valores en 0")
                else:
                    print("   - No se encontraron filas con todos los valores en 0")
            
            # 6. Eliminar columnas constantes
            print("\n6. Eliminando columnas constantes...")
            cols_removed = []
            for col in self.constant_columns_to_remove:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    cols_removed.append(col)
            
            if cols_removed:
                stats['columns_removed'] = len(cols_removed)
                print(f"   - Eliminadas {len(cols_removed)} columnas:")
                for col in cols_removed:
                    print(f"     * {col}")
            else:
                print("   - No se eliminaron columnas constantes")
            
            # 7. Agregar columna de etiqueta si no existe
            if 'Label' not in df.columns:
                df['Label'] = attack_type
                print(f"\n7. Agregada columna 'Label' con valor: {attack_type}")
            
            # Guardar archivo limpio
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, os.path.basename(file_path))
            df.to_csv(output_file, index=False)
            
            final_rows = len(df)
            total_rows_removed = original_rows - final_rows
            percentage_removed = (total_rows_removed / original_rows) * 100 if original_rows > 0 else 0
            
            stats['final_rows'] = final_rows
            stats['total_rows_removed'] = total_rows_removed
            stats['percentage_removed'] = percentage_removed
            stats['final_columns'] = len(df.columns)
            
            print(f"\n{'='*80}")
            print(f"RESUMEN DE LIMPIEZA:")
            print(f"  Filas originales: {original_rows:,}")
            print(f"  Filas eliminadas: {total_rows_removed:,} ({percentage_removed:.2f}%)")
            print(f"  Filas finales: {final_rows:,}")
            print(f"  Columnas originales: {original_cols}")
            print(f"  Columnas finales: {len(df.columns)}")
            print(f"  Archivo guardado en: {output_file}")
            print(f"{'='*80}")
            
            return stats
            
        except Exception as e:
            print(f"ERROR al limpiar {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def clean_all_files(self):
        """Limpia todos los archivos del dataset"""
        categories = ['Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-Based']
        
        print("="*100)
        print("LIMPIEZA COMPLETA DEL DATASET IOT")
        print("="*100)
        print(f"Ruta de entrada: {self.flow_based_path}")
        print(f"Ruta de salida: {self.output_path}")
        print("="*100)
        
        for category in categories:
            category_path = os.path.join(self.flow_based_path, category)
            
            if not os.path.exists(category_path):
                print(f"\nCategoría no encontrada: {category}")
                continue
            
            # Crear directorio de salida para esta categoría
            output_category_dir = os.path.join(self.output_path, category)
            
            # Obtener todos los archivos CSV recursivamente
            csv_files = []
            for root, dirs, files in os.walk(category_path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                print(f"\nNo se encontraron archivos CSV en: {category}")
                continue
            
            print(f"\n\n{'#'*80}")
            print(f"# CATEGORÍA: {category} ({len(csv_files)} archivos)")
            print(f"{'#'*80}")
            
            file_counter = 0
            for csv_file in csv_files:
                file_counter += 1
                print(f"\n[Progreso: {file_counter}/{len(csv_files)}]")
                
                # Mantener estructura de subdirectorios
                rel_path = os.path.relpath(os.path.dirname(csv_file), category_path)
                if rel_path == '.':
                    output_dir = output_category_dir
                else:
                    output_dir = os.path.join(output_category_dir, rel_path)
                
                # Verificar si el archivo ya fue procesado (skip si ya existe)
                output_file = os.path.join(output_dir, os.path.basename(csv_file))
                if os.path.exists(output_file):
                    print(f"\nSaltando (ya procesado): {os.path.basename(csv_file)}")
                    continue
                
                stats = self.clean_file(csv_file, category, output_dir)
                if stats:
                    self.cleaning_stats.append(stats)
                    
                    # Guardar progreso cada 10 archivos
                    if len(self.cleaning_stats) % 10 == 0:
                        self._save_progress()
        
        return self.cleaning_stats
    
    def generate_cleaning_report(self):
        """Genera reporte final de limpieza"""
        if not self.cleaning_stats:
            print("No hay estadísticas de limpieza para reportar")
            return
        
        print("\n\n" + "="*100)
        print("REPORTE FINAL DE LIMPIEZA DEL DATASET")
        print("="*100)
        
        # Crear DataFrame con estadísticas
        report_data = []
        for stats in self.cleaning_stats:
            report_data.append({
                'Archivo': stats['file'],
                'Categoría': stats['attack_type'],
                'Filas Originales': stats['original_rows'],
                'Filas Eliminadas': stats['total_rows_removed'],
                '% Eliminado': f"{stats['percentage_removed']:.2f}%",
                'Filas Finales': stats['final_rows'],
                'Columnas Eliminadas': stats['columns_removed']
            })
        
        df_report = pd.DataFrame(report_data)
        
        # Estadísticas generales
        total_original_rows = df_report['Filas Originales'].sum()
        total_rows_removed = df_report['Filas Eliminadas'].sum()
        total_final_rows = df_report['Filas Finales'].sum()
        overall_percentage = (total_rows_removed / total_original_rows * 100) if total_original_rows > 0 else 0
        
        print(f"\n1. ESTADÍSTICAS GENERALES:")
        print("="*100)
        print(f"  Total de archivos procesados: {len(self.cleaning_stats)}")
        print(f"  Total de filas originales: {total_original_rows:,}")
        print(f"  Total de filas eliminadas: {total_rows_removed:,} ({overall_percentage:.2f}%)")
        print(f"  Total de filas finales: {total_final_rows:,}")
        
        # Resumen por categoría
        print(f"\n\n2. RESUMEN POR CATEGORÍA:")
        print("="*100)
        category_summary = df_report.groupby('Categoría').agg({
            'Filas Originales': 'sum',
            'Filas Eliminadas': 'sum',
            'Filas Finales': 'sum'
        }).reset_index()
        
        category_summary['% Eliminado'] = (
            category_summary['Filas Eliminadas'] / category_summary['Filas Originales'] * 100
        ).round(2)
        
        print(category_summary.to_string(index=False))
        
        # Detalle por tipo de problema
        print(f"\n\n3. DETALLE POR TIPO DE PROBLEMA:")
        print("="*100)
        
        total_missing = sum(s['rows_removed']['missing_values'] for s in self.cleaning_stats)
        total_duplicates = sum(s['rows_removed']['duplicates'] for s in self.cleaning_stats)
        total_infinite = sum(s['rows_removed']['infinite_values'] for s in self.cleaning_stats)
        total_negative = sum(s['rows_removed']['negative_values'] for s in self.cleaning_stats)
        total_zeros = sum(s['rows_removed']['all_zeros'] for s in self.cleaning_stats)
        
        print(f"  Filas con valores faltantes: {total_missing:,}")
        print(f"  Filas duplicadas: {total_duplicates:,}")
        print(f"  Filas con valores infinitos: {total_infinite:,}")
        print(f"  Filas con valores negativos: {total_negative:,}")
        print(f"  Filas con todos los valores en 0: {total_zeros:,}")
        
        print(f"\n\n4. COLUMNAS ELIMINADAS:")
        print("="*100)
        for col in self.constant_columns_to_remove:
            print(f"  - {col}")
        
        # Guardar reporte en CSV
        report_file = os.path.join(self.output_path, 'cleaning_report_flowbased.csv')
        df_report.to_csv(report_file, index=False)
        print(f"\n\n[OK] Reporte completo guardado en: {report_file}")
        
        print("\n" + "="*100)
        print("LIMPIEZA COMPLETADA EXITOSAMENTE")
        print("="*100)


def main():
    """Función principal"""
    base_path = os.path.join(
        Path(__file__).resolve().parents[1],
        "cic-iot-diad-2024-dataset"
    )
    output_path = os.path.join(base_path, "cleaned_dataset_flowbased")
    
    cleaner = DataCleaner(base_path, output_path)
    
    # Limpiar todos los archivos
    cleaner.clean_all_files()
    
    # Generar reporte final
    cleaner.generate_cleaning_report()


if __name__ == "__main__":
    main()
