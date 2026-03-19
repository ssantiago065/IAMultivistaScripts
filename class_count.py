"""
Script para contar clases del dataset CIC IoT-DIAD 2024
Agrupa los archivos en categorías principales: Benign, DoS, DDoS, Mirai, Recon, Spoofing, Web-Based, Brute Force
"""

import os
import pandas as pd
import glob
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Directorio raíz del dataset
DATASET_ROOT = r"e:\beca\IoT device identification and anomaly detection dataset (CIC IoT-DIAD 2024)"

def get_category_from_path(file_path):
    """
    Determina la categoría del ataque basándose en la ruta del archivo
    """
    path_parts = Path(file_path).parts
    
    # Encontrar la categoría principal
    for part in path_parts:
        part_upper = part.upper()
        
        if 'BENIGN' in part_upper:
            return 'Benign'
        elif 'BRUTE' in part_upper:
            return 'Brute Force'
        elif 'DDOS' in part_upper or 'DDOS' in part_upper:
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

def get_subcategory_from_path(file_path):
    """
    Obtiene la subcategoría específica del ataque
    """
    path_parts = Path(file_path).parts
    filename = Path(file_path).stem
    
    # Para DDoS, obtener el tipo específico
    if 'DDOS' in str(file_path).upper():
        for part in path_parts:
            if 'ACK' in part.upper() and 'FRAG' in part.upper():
                return 'DDoS-ACK_Fragmentation'
            elif 'HTTP' in part.upper() and 'FLOOD' in part.upper():
                return 'DDoS-HTTP_Flood'
            elif 'ICMP' in part.upper() and 'FLOOD' in part.upper():
                return 'DDoS-ICMP_Flood'
            elif 'ICMP' in part.upper() and 'FRAG' in part.upper():
                return 'DDoS-ICMP_Fragmentation'
            elif 'UDP' in part.upper() and 'FLOOD' in part.upper():
                return 'DDoS-UDP_Flood'
            elif 'SYN' in part.upper() and 'FLOOD' in part.upper():
                return 'DDoS-SYN_Flood'
            elif 'TCP' in part.upper() and 'FLOOD' in part.upper():
                return 'DDoS-TCP_Flood'
    
    # Para DoS
    elif 'DOS' in str(file_path).upper() and 'DDOS' not in str(file_path).upper():
        for part in path_parts:
            if 'HTTP' in part.upper():
                return 'DoS-HTTP_Flood'
            elif 'SYN' in part.upper():
                return 'DoS-SYN_Flood'
            elif 'TCP' in part.upper():
                return 'DoS-TCP_Flood'
            elif 'UDP' in part.upper():
                return 'DoS-UDP_Flood'
    
    # Para Mirai
    elif 'MIRAI' in str(file_path).upper():
        if 'greeth' in filename.lower():
            return 'Mirai-greeth_flood'
        elif 'greip' in filename.lower():
            return 'Mirai-greip_flood'
        elif 'udpplain' in filename.lower():
            return 'Mirai-udpplain'
    
    # Para Spoofing
    elif 'SPOOF' in str(file_path).upper():
        if 'ARP' in str(file_path).upper():
            return 'ARP_Spoofing'
        elif 'DNS' in str(file_path).upper():
            return 'DNS_Spoofing'
    
    # Para Web-Based
    elif 'WEB' in str(file_path).upper() or 'SQL' in str(file_path).upper() or 'XSS' in str(file_path).upper():
        if 'SQL' in str(file_path).upper():
            return 'SQL_Injection'
        elif 'XSS' in str(file_path).upper():
            return 'XSS'
        elif 'UPLOAD' in str(file_path).upper():
            return 'Uploading_Attack'
    
    # Para Recon
    elif 'RECON' in str(file_path).upper():
        return 'Vulnerability_Scan'
    
    # Para Brute Force
    elif 'BRUTE' in str(file_path).upper():
        return 'Dictionary_BruteForce'
    
    # Para Benign
    elif 'BENIGN' in str(file_path).upper():
        return 'Benign'
    
    return get_category_from_path(file_path)

def count_classes():
    """
    Cuenta las instancias de cada clase en el dataset
    """
    # Encontrar todos los archivos CSV
    csv_files = glob.glob(os.path.join(DATASET_ROOT, '**', '*.csv'), recursive=True)
    
    # Filtrar archivos en la carpeta Analysis
    csv_files = [f for f in csv_files if 'Analysis' not in f]
    
    print(f"Se encontraron {len(csv_files)} archivos CSV\n")
    
    # Contadores
    category_counts = defaultdict(int)
    subcategory_counts = defaultdict(int)
    file_info = []
    
    # Procesar cada archivo
    for idx, csv_file in enumerate(csv_files, 1):
        try:
            print(f"Procesando [{idx}/{len(csv_files)}]: {Path(csv_file).name}...", end='\r')
            
            # Contar líneas (excluyendo header)
            # Usar chunks para archivos grandes
            chunk_size = 100000
            total_rows = 0
            
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
                total_rows += len(chunk)
            
            category = get_category_from_path(csv_file)
            subcategory = get_subcategory_from_path(csv_file)
            
            category_counts[category] += total_rows
            subcategory_counts[subcategory] += total_rows
            
            file_info.append({
                'File': Path(csv_file).name,
                'Category': category,
                'Subcategory': subcategory,
                'Instances': total_rows
            })
            
        except Exception as e:
            print(f"\nError procesando {csv_file}: {str(e)}")
            continue
    
    print("\n\n" + "="*80)
    print("RESUMEN POR CATEGORÍAS PRINCIPALES")
    print("="*80)
    
    # Crear DataFrame de categorías
    category_df = pd.DataFrame([
        {'Category': cat, 'Count': count, 'Percentage': f"{(count/sum(category_counts.values()))*100:.2f}%"}
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    ])
    
    print(category_df.to_string(index=False))
    print(f"\nTotal de instancias: {sum(category_counts.values()):,}")
    
    print("\n" + "="*80)
    print("RESUMEN POR SUBCATEGORÍAS")
    print("="*80)
    
    # Crear DataFrame de subcategorías
    subcategory_df = pd.DataFrame([
        {'Subcategory': subcat, 'Count': count, 'Percentage': f"{(count/sum(subcategory_counts.values()))*100:.2f}%"}
        for subcat, count in sorted(subcategory_counts.items(), key=lambda x: x[1], reverse=True)
    ])
    
    print(subcategory_df.to_string(index=False))
    
    # Guardar resultados
    output_dir = os.path.join(DATASET_ROOT, 'Analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    category_df.to_csv(os.path.join(output_dir, 'category_counts.csv'), index=False)
    subcategory_df.to_csv(os.path.join(output_dir, 'subcategory_counts.csv'), index=False)
    
    # Guardar información detallada por archivo
    file_df = pd.DataFrame(file_info)
    file_df = file_df.sort_values('Instances', ascending=False)
    file_df.to_csv(os.path.join(output_dir, 'file_details.csv'), index=False)
    
    print(f"\n\nResultados guardados en: {output_dir}")
    print(f"  - category_counts.csv")
    print(f"  - subcategory_counts.csv")
    print(f"  - file_details.csv")
    
    return category_df, subcategory_df, file_df

if __name__ == "__main__":
    category_df, subcategory_df, file_df = count_classes()
