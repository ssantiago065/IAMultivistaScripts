import json
import sys
import warnings
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURACIÓN ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FLOW_DIR = PROJECT_ROOT / 'Anomaly Detection - Flow Based features'
OUTPUT_FILE = PROJECT_ROOT / 'profiling_results.json'

class DataProfiler:
    def __init__(self, flow_based_path, output_file):
        self.flow_based_path = Path(flow_based_path)
        self.output_file = Path(output_file)
        self.profiling_results = {}

    @staticmethod
    def _to_serializable_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value
        
    def analyze_file(self, file_path, attack_type, chunk_size=200000, duplicate_mode='global'):
        """Realiza análisis detallado de un CSV por chunks."""
        file_path = Path(file_path)

        print(f"\n{'='*80}")
        print(f"Analizando: {file_path.name}")
        print(f"Tipo de ataque: {attack_type}")
        print(f"{'='*80}")
        
        try:
            total_rows = 0
            total_columns = 0
            processed_rows = 0

            missing_counts = defaultdict(int)
            infinite_counts = defaultdict(int)
            negative_counts = defaultdict(int)

            duplicate_rows = 0
            all_zero_rows = 0
            rows_to_remove_count = 0
            rows_to_remove_sample = []
            sample_limit = 1000

            numeric_cols = None
            numeric_cols_to_check = []
            positive_only_cols = []
            constant_candidates = {}
            constant_flags = {}

            dominant_candidates = defaultdict(lambda: defaultdict(int))
            seen_hashes = set() if duplicate_mode == 'global' else None

            reader = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)

            for chunk_idx, chunk in enumerate(reader, 1):
                chunk.columns = chunk.columns.str.strip()
                chunk_rows = len(chunk)
                if chunk_rows == 0:
                    continue

                total_rows += chunk_rows
                if total_columns == 0:
                    total_columns = len(chunk.columns)

                if numeric_cols is None:
                    numeric_cols = list(chunk.select_dtypes(include=[np.number]).columns)
                    exclude_cols = [
                        'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
                        'Protocol', 'Timestamp', 'Label'
                    ]
                    numeric_cols_to_check = [col for col in numeric_cols if col not in exclude_cols]
                    positive_only_cols = [
                        col for col in numeric_cols
                        if ('Length' in col or 'Bytes' in col or 'Packet' in col or 'Duration' in col or 'Count' in col)
                    ]
                    for col in numeric_cols:
                        constant_candidates[col] = None
                        constant_flags[col] = True

                # 1) Missing values por columna
                miss_series = chunk.isnull().sum()
                for col, count in miss_series.items():
                    if count:
                        missing_counts[col] += int(count)

                # 2) Duplicados por modo seleccionado
                if duplicate_mode == 'global':
                    row_hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype=np.uint64)
                    duplicate_mask = np.zeros(chunk_rows, dtype=bool)
                    for i, h in enumerate(row_hashes):
                        h_int = int(h)
                        if h_int in seen_hashes:
                            duplicate_mask[i] = True
                        else:
                            seen_hashes.add(h_int)
                else:
                    # Más eficiente en memoria: detecta duplicados dentro de cada chunk.
                    duplicate_mask = chunk.duplicated().to_numpy()
                duplicate_rows += int(duplicate_mask.sum())

                # 3) Constantes y candidatos para casi-constantes
                for col in numeric_cols:
                    series = chunk[col]
                    non_null = series.dropna()

                    if not non_null.empty:
                        if constant_candidates[col] is None:
                            first_val = non_null.iloc[0]
                            constant_candidates[col] = first_val
                            if (non_null != first_val).any():
                                constant_flags[col] = False
                        elif constant_flags[col] and (non_null != constant_candidates[col]).any():
                            constant_flags[col] = False

                    top_counts = series.value_counts(dropna=False)
                    if not top_counts.empty:
                        top_value = top_counts.index[0]
                        top_count = int(top_counts.iloc[0])
                        key = '__NaN__' if pd.isna(top_value) else top_value
                        dominant_candidates[col][key] += top_count

                # 4) Filas todo cero en numéricas (sin IDs)
                if numeric_cols_to_check:
                    zero_mask = (chunk[numeric_cols_to_check] == 0).all(axis=1).to_numpy()
                    all_zero_rows += int(zero_mask.sum())
                else:
                    zero_mask = np.zeros(chunk_rows, dtype=bool)

                # 5) Infinitos y máscara de filas
                inf_row_mask = np.zeros(chunk_rows, dtype=bool)
                for col in numeric_cols:
                    inf_mask = np.isinf(chunk[col].to_numpy())
                    inf_count = int(inf_mask.sum())
                    if inf_count > 0:
                        infinite_counts[col] += inf_count
                        inf_row_mask |= inf_mask

                # 6) Negativos en columnas que deberían ser positivas
                neg_row_mask = np.zeros(chunk_rows, dtype=bool)
                for col in positive_only_cols:
                    neg_mask = (chunk[col].to_numpy() < 0)
                    neg_count = int(neg_mask.sum())
                    if neg_count > 0:
                        negative_counts[col] += neg_count
                        neg_row_mask |= neg_mask

                # Unir reglas para estimar filas a remover (sin duplicarlas)
                missing_row_mask = chunk.isnull().any(axis=1).to_numpy()
                remove_mask = missing_row_mask | duplicate_mask | zero_mask | inf_row_mask | neg_row_mask
                local_bad_positions = np.flatnonzero(remove_mask)
                rows_to_remove_count += int(local_bad_positions.size)

                if len(rows_to_remove_sample) < sample_limit and local_bad_positions.size > 0:
                    remaining = sample_limit - len(rows_to_remove_sample)
                    absolute_rows = local_bad_positions[:remaining] + processed_rows
                    rows_to_remove_sample.extend(absolute_rows.astype(int).tolist())

                processed_rows += chunk_rows

                if chunk_idx % 10 == 0:
                    print(f"   Progreso: chunk {chunk_idx} | filas procesadas: {processed_rows:,}")

            if total_rows == 0:
                print("   [WARN] Archivo sin filas (solo cabecera)")
                return {
                    'file': file_path.name,
                    'attack_type': attack_type,
                    'total_rows': 0,
                    'total_columns': int(total_columns),
                    'missing_values': {},
                    'duplicate_rows': 0,
                    'constant_columns': [],
                    'near_constant_columns': [],
                    'all_zero_rows': 0,
                    'infinite_values': {},
                    'negative_values': {},
                    'rows_to_remove': [],
                    'rows_to_remove_sample_limit': sample_limit,
                    'total_rows_to_remove': 0,
                    'percentage_to_remove': 0.0,
                    'empty_file': True,
                }

            print(f"\nDimensiones originales: ({total_rows}, {total_columns})")

            # Resolver columnas constantes
            constant_columns = [
                col for col in (numeric_cols or [])
                if constant_flags.get(col) and constant_candidates.get(col) is not None
            ]

            # Segunda pasada para porcentaje exacto de casi-constantes
            near_constant_columns = []
            candidate_by_col = {}
            for col in (numeric_cols or []):
                if col in constant_columns:
                    continue
                if not dominant_candidates[col]:
                    continue
                candidate, candidate_count = max(dominant_candidates[col].items(), key=lambda x: x[1])
                approx_pct = (candidate_count / total_rows) * 100
                if approx_pct >= 95:
                    candidate_by_col[col] = candidate

            if candidate_by_col:
                exact_counts = {col: 0 for col in candidate_by_col}
                verify_reader = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
                for chunk in verify_reader:
                    chunk.columns = chunk.columns.str.strip()
                    for col, candidate in candidate_by_col.items():
                        if candidate == '__NaN__':
                            exact_counts[col] += int(chunk[col].isna().sum())
                        else:
                            exact_counts[col] += int((chunk[col] == candidate).sum())

                for col, count in exact_counts.items():
                    pct = (count / total_rows) * 100
                    if pct > 99:
                        near_constant_columns.append(
                            {
                                'column': col,
                                'dominant_value': self._to_serializable_value(candidate_by_col[col] if candidate_by_col[col] != '__NaN__' else np.nan),
                                'percentage': float(pct)
                            }
                        )

            results = {
                'file': file_path.name,
                'attack_type': attack_type,
                'total_rows': int(total_rows),
                'total_columns': int(total_columns),
                'missing_values': {},
                'duplicate_rows': int(duplicate_rows),
                'constant_columns': sorted(constant_columns),
                'near_constant_columns': near_constant_columns,
                'all_zero_rows': int(all_zero_rows),
                'infinite_values': {k: int(v) for k, v in infinite_counts.items() if v > 0},
                'negative_values': {k: int(v) for k, v in negative_counts.items() if v > 0},
                'rows_to_remove': rows_to_remove_sample,
                'rows_to_remove_sample_limit': sample_limit,
                'total_rows_to_remove': int(rows_to_remove_count),
                'percentage_to_remove': float((rows_to_remove_count / total_rows) * 100)
            }

            for col, count in missing_counts.items():
                if count > 0:
                    results['missing_values'][col] = {
                        'count': int(count),
                        'percentage': float((count / total_rows) * 100)
                    }

            print("\n1. ANÁLISIS DE VALORES FALTANTES:")
            if results['missing_values']:
                for col, info in results['missing_values'].items():
                    print(f"   - {col}: {info['count']} ({info['percentage']:.2f}%)")
            else:
                print("   [OK] No se encontraron valores faltantes")

            print("\n2. ANÁLISIS DE FILAS DUPLICADAS:")
            print(f"   - Filas duplicadas: {results['duplicate_rows']}")

            print("\n3. ANÁLISIS DE COLUMNAS CONSTANTES:")
            if results['constant_columns']:
                for col in results['constant_columns']:
                    print(f"   - {col}")
            else:
                print("   [OK] No se encontraron columnas constantes")

            print("\n4. ANÁLISIS DE COLUMNAS CASI CONSTANTES (>99%):")
            if results['near_constant_columns']:
                for item in results['near_constant_columns']:
                    print(f"   - {item['column']}: {item['percentage']:.2f}%")
            else:
                print("   [OK] No se encontraron columnas casi constantes")

            print("\n5. ANÁLISIS DE FILAS CON TODOS LOS VALORES EN CERO:")
            print(f"   - Filas con todos los valores en 0: {results['all_zero_rows']}")

            print("\n6. ANÁLISIS DE VALORES INFINITOS:")
            if results['infinite_values']:
                for col, count in results['infinite_values'].items():
                    print(f"   - {col}: {count} valores infinitos")
            else:
                print("   [OK] No se encontraron valores infinitos")

            print("\n7. ANÁLISIS DE VALORES NEGATIVOS ANÓMALOS:")
            if results['negative_values']:
                for col, count in results['negative_values'].items():
                    print(f"   - {col}: {count} valores negativos")
            else:
                print("   [OK] No se encontraron valores negativos anomalos")

            print(f"\n{'='*80}")
            print("RESUMEN DE LIMPIEZA:")
            print(
                f"  Total de filas a eliminar: {results['total_rows_to_remove']} "
                f"({results['percentage_to_remove']:.2f}%)"
            )
            print(f"  Filas que permanecerán: {total_rows - results['total_rows_to_remove']}")
            print(f"  Muestra de índices marcados: {len(results['rows_to_remove'])}")
            print(f"{'='*80}")

            return results
            
        except Exception as e:
            print(f"ERROR al analizar {file_path}: {str(e)}")
            return None
    
    def analyze_all_files(self, sample_size=0, chunk_size=200000, duplicate_mode='global', categories=None):
        """Analiza archivos por categoría; sample_size=0 procesa todos."""
        default_categories = ['Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-Based']
        categories = categories or default_categories
        
        all_results = []

        if not self.flow_based_path.exists():
            raise FileNotFoundError(f"No existe el dataset: {self.flow_based_path}")
        
        for category in categories:
            category_path = self.flow_based_path / category
            
            if not category_path.exists():
                print(f"\nCategoría no encontrada: {category}")
                continue
            
            # Obtener archivos CSV
            if category_path.is_file():
                continue

            csv_files = sorted(category_path.rglob('*.csv'))
            
            if not csv_files:
                print(f"\nNo se encontraron archivos CSV en: {category}")
                continue
            
            # Analizar muestra o totalidad de archivos
            files_to_analyze = csv_files if sample_size <= 0 else csv_files[:sample_size]
            
            print(f"\n\n{'#'*80}")
            print(f"# CATEGORÍA: {category} ({len(files_to_analyze)} archivos a analizar)")
            print(f"{'#'*80}")
            
            for csv_file in files_to_analyze:
                result = self.analyze_file(
                    csv_file,
                    category,
                    chunk_size=chunk_size,
                    duplicate_mode=duplicate_mode,
                )
                if result:
                    all_results.append(result)
        
        return all_results
    
    def generate_summary_report(self, results):
        """Genera un reporte resumen del análisis"""
        print("\n\n" + "="*100)
        print("REPORTE RESUMEN DE DATA PROFILING")
        print("="*100)
        
        if not results:
            print("No hay resultados para reportar")
            return
        
        # Resumen por categoría
        print("\n1. RESUMEN POR CATEGORÍA DE ATAQUE:")
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
        
        # Problemas más comunes
        print("\n\n2. PROBLEMAS MÁS COMUNES ENCONTRADOS:")
        print("-" * 100)
        
        total_missing = sum(len(r['missing_values']) for r in results if r['missing_values'])
        total_duplicates = sum(r['duplicate_rows'] for r in results)
        total_all_zeros = sum(r['all_zero_rows'] for r in results)
        total_infinite = sum(len(r['infinite_values']) for r in results if r['infinite_values'])
        
        print(f"  - Archivos con valores faltantes: {total_missing}")
        print(f"  - Total de filas duplicadas: {total_duplicates:,}")
        print(f"  - Total de filas con todos los valores en 0: {total_all_zeros:,}")
        print(f"  - Archivos con valores infinitos: {total_infinite}")
        
        # Columnas constantes comunes
        print("\n\n3. COLUMNAS CONSTANTES ENCONTRADAS:")
        print("-" * 100)
        constant_cols = set()
        for result in results:
            constant_cols.update(result['constant_columns'])
        
        if constant_cols:
            print("  Estas columnas tienen un valor constante en uno o más archivos:")
            for col in sorted(constant_cols):
                print(f"    - {col}")
        else:
            print("  [OK] No se encontraron columnas constantes")
        
        print("\n" + "="*100)
        print("FIN DEL REPORTE")
        print("="*100)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Data profiling para dataset Flow Based')
    parser.add_argument('--dataset-dir', type=Path, default=FLOW_DIR, help='Ruta al dataset Flow Based')
    parser.add_argument('--output-file', type=Path, default=OUTPUT_FILE, help='Archivo JSON de salida')
    parser.add_argument('--sample-size', type=int, default=0, help='Cantidad de archivos por categoría (0 = todos)')
    parser.add_argument('--chunk-size', type=int, default=200000, help='Tamaño de chunk para lectura incremental')
    parser.add_argument('--duplicate-mode', choices=['global', 'chunk'], default='global', help='Modo de detección de duplicados')
    parser.add_argument('--categories', type=str, default='', help='Categorías separadas por coma. Vacío = todas')
    args = parser.parse_args()

    dataset_path = args.dataset_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    
    print("="*100)
    print("ANÁLISIS DE DATA PROFILING - DATASET IOT")
    print("="*100)
    print(f"Dataset: {dataset_path}")
    print(f"Salida: {output_file}")
    print(f"Modo duplicados: {args.duplicate_mode}")

    selected_categories = [c.strip() for c in args.categories.split(',') if c.strip()]
    if selected_categories:
        print(f"Categorías seleccionadas: {selected_categories}")
    else:
        selected_categories = None
    
    profiler = DataProfiler(dataset_path, output_file)
    
    # Analizar por chunks (todo el dataset por defecto)
    results = profiler.analyze_all_files(
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        duplicate_mode=args.duplicate_mode,
        categories=selected_categories,
    )
    
    # Guardar resultados
    results_clean = []
    for r in results:
        r_copy = r.copy()
        
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
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, default=str)
    
    print(f"\n[OK] Resultados guardados en: {output_file}")
    
    # Generar reporte resumen
    profiler.generate_summary_report(results)


if __name__ == "__main__":
    main()
