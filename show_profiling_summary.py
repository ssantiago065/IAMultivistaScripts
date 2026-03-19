import json
import pandas as pd

# Cargar resultados del profiling
with open(r'f:\IOTDataset\profiling_results.json', 'r') as f:
    results = json.load(f)

print("="*100)
print("RESUMEN EJECUTIVO DEL DATA PROFILING")
print("="*100)

# Crear tabla resumen
summary_data = []
for result in results:
    summary_data.append({
        'Archivo': result['file'][:50],  # Truncar nombre
        'Tipo': result['attack_type'],
        'Filas Totales': result['total_rows'],
        'Filas a Eliminar': result['total_rows_to_remove'],
        '% a Eliminar': f"{result['percentage_to_remove']:.2f}%",
        'Duplicados': result['duplicate_rows'],
        'Valores Infinitos': len(result['infinite_values']),
        'Valores Faltantes': len(result['missing_values'])
    })

df_summary = pd.DataFrame(summary_data)
print("\n1. TABLA RESUMEN POR ARCHIVO:")
print(df_summary.to_string(index=False))

# Resumen por categoría
print("\n\n2. RESUMEN POR CATEGORÍA DE ATAQUE:")
print("="*100)
category_summary = df_summary.groupby('Tipo').agg({
    'Filas Totales': 'sum',
    'Filas a Eliminar': 'sum',
    'Duplicados': 'sum',
    'Valores Infinitos': 'sum',
    'Valores Faltantes': 'sum'
}).reset_index()

category_summary['% a Eliminar'] = (category_summary['Filas a Eliminar'] / category_summary['Filas Totales'] * 100).round(2)
print(category_summary.to_string(index=False))

# Problemas comunes
print("\n\n3. PROBLEMAS COMUNES ENCONTRADOS:")
print("="*100)

# Columnas constantes
all_constant_cols = set()
for result in results:
    all_constant_cols.update(result['constant_columns'])

print(f"\nCOLUMNAS CONSTANTES ENCONTRADAS ({len(all_constant_cols)}):")
for col in sorted(all_constant_cols):
    print(f"  - {col}")

# Totales generales
total_rows = sum(r['total_rows'] for r in results)
total_to_remove = sum(r['total_rows_to_remove'] for r in results)
total_duplicates = sum(r['duplicate_rows'] for r in results)

print(f"\n\n4. ESTADÍSTICAS GENERALES:")
print("="*100)
print(f"  Total de archivos analizados: {len(results)}")
print(f"  Total de filas en el dataset: {total_rows:,}")
print(f"  Total de filas a eliminar: {total_to_remove:,} ({total_to_remove/total_rows*100:.2f}%)")
print(f"  Total de filas que permanecerán: {total_rows - total_to_remove:,}")
print(f"  Total de filas duplicadas: {total_duplicates:,}")

print("\n\n5. RECOMENDACIONES DE LIMPIEZA:")
print("="*100)
print("  ✓ Eliminar filas con valores faltantes")
print("  ✓ Eliminar filas duplicadas (mantener solo la primera ocurrencia)")
print("  ✓ Eliminar filas con valores infinitos")
print("  ✓ Considerar eliminar columnas constantes (no aportan información)")
print("  ✓ Evaluar columnas casi constantes (>99% del mismo valor)")

print("\n" + "="*100)
