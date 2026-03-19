"""
Script de Análisis Rápido para CIC IoT-DIAD 2024
Combina conteo de clases y visualizaciones básicas
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Directorio raíz del dataset
DATASET_ROOT = r"e:\beca\IoT device identification and anomaly detection dataset (CIC IoT-DIAD 2024)"

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

def count_and_visualize():
    """
    Cuenta las clases y genera visualizaciones
    """
    print("="*80)
    print("ANÁLISIS RÁPIDO - CIC IoT-DIAD 2024")
    print("="*80 + "\n")
    
    # Encontrar todos los archivos CSV
    csv_files = glob.glob(os.path.join(DATASET_ROOT, '**', '*.csv'), recursive=True)
    csv_files = [f for f in csv_files if 'Analysis' not in f]
    
    print(f"Se encontraron {len(csv_files)} archivos CSV\n")
    
    # Contadores
    category_counts = defaultdict(int)
    file_info = []
    
    # Procesar cada archivo
    total_files = len(csv_files)
    for idx, csv_file in enumerate(csv_files, 1):
        try:
            print(f"Procesando [{idx}/{total_files}]: {Path(csv_file).name[:50]}...", end='\r')
            
            # Contar líneas usando chunks
            chunk_size = 100000
            total_rows = 0
            
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
                total_rows += len(chunk)
            
            category = get_category_from_path(csv_file)
            category_counts[category] += total_rows
            
            file_info.append({
                'File': Path(csv_file).name,
                'Category': category,
                'Instances': total_rows,
                'Path': csv_file
            })
            
        except Exception as e:
            print(f"\nError procesando {csv_file}: {str(e)}")
            continue
    
    print("\n")
    
    # Crear DataFrame
    df_results = pd.DataFrame([
        {
            'Categoría': cat, 
            'Cantidad': count, 
            'Porcentaje': f"{(count/sum(category_counts.values()))*100:.2f}%"
        }
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    ])
    
    # Mostrar tabla
    print("\n" + "="*80)
    print("DISTRIBUCIÓN DE CLASES")
    print("="*80 + "\n")
    print(df_results.to_string(index=False))
    print(f"\nTotal de instancias: {sum(category_counts.values()):,}\n")
    
    # Guardar resultados
    output_dir = os.path.join(DATASET_ROOT, 'Analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    df_results.to_csv(os.path.join(output_dir, 'class_distribution.csv'), index=False)
    
    # Crear visualizaciones
    print("="*80)
    print("GENERANDO VISUALIZACIONES")
    print("="*80 + "\n")
    
    # Preparar datos para gráficos
    categories = list(category_counts.keys())
    counts = [category_counts[cat] for cat in categories]
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribución de Clases - CIC IoT-DIAD 2024', fontsize=16, fontweight='bold')
    
    # 1. Gráfico de barras
    ax1 = axes[0, 0]
    colors = sns.color_palette("husl", len(categories))
    bars = ax1.barh(categories, counts, color=colors)
    ax1.set_xlabel('Número de Instancias', fontweight='bold')
    ax1.set_title('Distribución por Categoría', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Añadir valores en las barras
    for bar in bars:
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}', 
                ha='left', va='center', fontweight='bold', fontsize=9)
    
    # 2. Gráfico de pastel
    ax2 = axes[0, 1]
    explode = [0.05 if i == 0 else 0 for i in range(len(categories))]
    wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.1f%%',
                                        startangle=90, colors=colors, explode=explode)
    ax2.set_title('Proporción por Categoría', fontweight='bold')
    
    # Mejorar visibilidad del texto
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 3. Gráfico de barras con escala logarítmica
    ax3 = axes[1, 0]
    ax3.bar(range(len(categories)), counts, color=colors)
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(categories, rotation=45, ha='right')
    ax3.set_yscale('log')
    ax3.set_ylabel('Número de Instancias (escala log)', fontweight='bold')
    ax3.set_title('Distribución (Escala Logarítmica)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Tabla resumen
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count/sum(category_counts.values()))*100
        table_data.append([cat, f'{count:,}', f'{pct:.2f}%'])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Categoría', 'Instancias', 'Porcentaje'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Estilo de la tabla
    for i in range(len(table_data) + 1):
        if i == 0:
            for j in range(3):
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            for j in range(3):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax4.set_title('Resumen Estadístico', fontweight='bold', pad=20)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    output_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {output_path}")
    
    # Mostrar gráfico
    plt.show()
    
    # Crear gráfico adicional: distribución por número de archivos
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    files_per_category = defaultdict(int)
    for info in file_info:
        files_per_category[info['Category']] += 1
    
    cats = list(files_per_category.keys())
    file_counts = [files_per_category[cat] for cat in cats]
    
    bars = ax.bar(cats, file_counts, color=colors[:len(cats)])
    ax.set_ylabel('Número de Archivos', fontweight='bold')
    ax.set_xlabel('Categoría', fontweight='bold')
    ax.set_title('Número de Archivos por Categoría', fontweight='bold', fontsize=14)
    ax.set_xticklabels(cats, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Añadir valores
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path2 = os.path.join(output_dir, 'files_per_category.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {output_path2}")
    plt.show()
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print(f"\nArchivos generados en: {output_dir}")
    print("  - class_distribution.csv")
    print("  - class_distribution.png")
    print("  - files_per_category.png")
    
    return df_results

if __name__ == "__main__":
    results = count_and_visualize()
