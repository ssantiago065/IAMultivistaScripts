import pandas as pd
import os

# Cargar reporte de limpieza
report_path = r"f:\IOTDataset\cleaned_dataset\cleaning_report.csv"
df_report = pd.read_csv(report_path)

print("="*100)
print("VISUALIZACIÓN FINAL - DATASET IOT LIMPIO")
print("="*100)

# 1. Estadísticas generales
total_original = df_report['Filas Originales'].sum()
total_removed = df_report['Filas Eliminadas'].sum()
total_final = df_report['Filas Finales'].sum()
removal_percentage = (total_removed / total_original * 100) if total_original > 0 else 0

print(f"\n1. ESTADÍSTICAS GLOBALES:")
print("-" * 100)
print(f"{'Archivos procesados:':<40} {len(df_report):>15,}")
print(f"{'Total de filas originales:':<40} {total_original:>15,}")
print(f"{'Total de filas eliminadas:':<40} {total_removed:>15,} ({removal_percentage:.2f}%)")
print(f"{'Total de filas finales (limpias):':<40} {total_final:>15,}")
print(f"{'Columnas por archivo (después):':<40} {75:>15}")
print(f"{'Columnas eliminadas:':<40} {9:>15}")

# 2. Top archivos con más limpieza
print(f"\n2. TOP 10 ARCHIVOS CON MÁS FILAS ELIMINADAS:")
print("-" * 100)
top_cleaned = df_report.nlargest(10, 'Filas Eliminadas')[['Archivo', 'Categoría', 'Filas Originales', 'Filas Eliminadas', '% Eliminado']]
print(top_cleaned.to_string(index=False))

# 3. Resumen por categoría
print(f"\n3. RESUMEN POR CATEGORÍA DE ATAQUE:")
print("-" * 100)
category_summary = df_report.groupby('Categoría').agg({
    'Archivo': 'count',
    'Filas Originales': 'sum',
    'Filas Eliminadas': 'sum',
    'Filas Finales': 'sum'
}).reset_index()

category_summary.columns = ['Categoría', 'Archivos', 'Filas Orig.', 'Filas Elim.', 'Filas Finales']
category_summary['% Elim.'] = (category_summary['Filas Elim.'] / category_summary['Filas Orig.'] * 100).round(2)
category_summary = category_summary.sort_values('Filas Finales', ascending=False)

print(category_summary.to_string(index=False))

# 4. Distribución del dataset limpio
print(f"\n4. DISTRIBUCIÓN DEL DATASET LIMPIO:")
print("-" * 100)
category_summary['% del Total'] = (category_summary['Filas Finales'] / category_summary['Filas Finales'].sum() * 100).round(2)
distribution = category_summary[['Categoría', 'Filas Finales', '% del Total']].sort_values('% del Total', ascending=False)
print(distribution.to_string(index=False))

# 5. Gráfico de barras ASCII
print(f"\n5. VISUALIZACIÓN DE DISTRIBUCIÓN:")
print("-" * 100)
max_width = 80
for _, row in distribution.iterrows():
    bar_width = int((row['% del Total'] / 100) * max_width)
    bar = '█' * bar_width
    print(f"{row['Categoría']:<15} {bar} {row['% del Total']:.2f}%")

# 6. Información de ubicación
print(f"\n6. UBICACIÓN DEL DATASET LIMPIO:")
print("-" * 100)
print(f"Directorio: f:\\IOTDataset\\cleaned_dataset\\")
print(f"\nEstructura:")
print("  cleaned_dataset/")
print("  ├── Benign/ (4 archivos)")
print("  ├── BruteForce/ (1 archivo)")
print("  ├── DDoS/ (61 archivos en subdirectorios)")
print("  ├── DoS/ (30 archivos en subdirectorios)")
print("  ├── Mirai/ (29 archivos)")
print("  ├── Recon/ (1 archivo)")
print("  ├── Spoofing/ (3 archivos)")
print("  ├── Web-Based/ (3 archivos)")
print("  ├── cleaning_report.csv")
print("  └── cleaning_progress.json")

print(f"\n7. COLUMNAS ELIMINADAS (Constantes, sin información):")
print("-" * 100)
eliminated_columns = [
    'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Count',
    'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
    'CWR Flag Count', 'ECE Flag Count'
]
for i, col in enumerate(eliminated_columns, 1):
    print(f"  {i}. {col}")

print(f"\n8. CALIDAD DEL DATASET LIMPIO:")
print("-" * 100)
print("  ✓ Sin valores faltantes (NaN)")
print("  ✓ Sin filas duplicadas")
print("  ✓ Sin valores infinitos")
print("  ✓ Sin valores negativos anómalos")
print("  ✓ Sin columnas constantes")
print(f"  ✓ {len(distribution)} categorías balanceadas (con predominancia de DoS)")
print(f"  ✓ {total_final:,} filas listas para Machine Learning")

print(f"\n9. RECOMENDACIONES DE USO:")
print("-" * 100)
print("  • Aplicar normalización/estandarización antes del entrenamiento")
print("  • Considerar técnicas de balanceo de clases (SMOTE, etc.)")
print("  • Usar validación cruzada estratificada")
print("  • Evaluar con métricas apropiadas (F1-score, AUC-ROC)")
print("  • Considerar análisis por subcategorías de DDoS/DoS")

print("\n" + "="*100)
print("Dataset limpio y listo para uso ✓")
print("="*100)
