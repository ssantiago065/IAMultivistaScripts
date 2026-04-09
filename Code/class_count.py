"""
Script para contar clases del dataset CIC IoT-DIAD 2024.
Agrupa los archivos en categorías principales y subcategorías.
"""

from pathlib import Path
from collections import defaultdict
import argparse
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FLOW_DIR = PROJECT_ROOT / "cic-iot-diad-2024-dataset" / "FlowBased"
OUTPUT_FILE = PROJECT_ROOT / "dataset_poc_multivista.csv"

# Fallback para el nombre real de carpeta usado en este workspace.
FLOW_DIR_FALLBACK = PROJECT_ROOT / "Anomaly Detection - Flow Based features"


def resolve_dataset_dir(user_dataset_dir: Path | None = None) -> Path:
    """Resuelve el directorio del dataset usando rutas relativas al script."""
    candidates = []

    if user_dataset_dir is not None:
        candidates.append(user_dataset_dir.expanduser().resolve())

    candidates.extend([FLOW_DIR, FLOW_DIR_FALLBACK])

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    candidates_text = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "No se encontró el directorio del dataset. Se intentó en:\n"
        f"{candidates_text}\n"
        "Puedes pasarlo con --dataset-dir <ruta>."
    )


def get_category_from_path(file_path: Path) -> str:
    """Determina la categoría del ataque basándose en la ruta del archivo."""
    for part in file_path.parts:
        part_upper = part.upper()

        if "BENIGN" in part_upper:
            return "Benign"
        if "BRUTE" in part_upper:
            return "Brute Force"
        if "DDOS" in part_upper:
            return "DDoS"
        if "DOS" in part_upper and "DDOS" not in part_upper:
            return "DoS"
        if "MIRAI" in part_upper:
            return "Mirai"
        if "RECON" in part_upper:
            return "Recon"
        if "SPOOF" in part_upper:
            return "Spoofing"
        if any(k in part_upper for k in ("WEB", "SQL", "XSS", "UPLOAD")):
            return "Web-Based"
        if "MQTT" in part_upper:
            return "MQTT"

    return "Other"


def get_subcategory_from_path(file_path: Path) -> str:
    """Obtiene la subcategoría específica del ataque."""
    path_upper = str(file_path).upper()
    filename = file_path.stem.lower()

    if "DDOS" in path_upper:
        if "ACK" in path_upper and "FRAG" in path_upper:
            return "DDoS-ACK_Fragmentation"
        if "HTTP" in path_upper and "FLOOD" in path_upper:
            return "DDoS-HTTP_Flood"
        if "ICMP" in path_upper and "FLOOD" in path_upper:
            return "DDoS-ICMP_Flood"
        if "ICMP" in path_upper and "FRAG" in path_upper:
            return "DDoS-ICMP_Fragmentation"
        if "PSHACK" in path_upper:
            return "DDoS-PSHACK_Flood"
        if "RSTFIN" in path_upper:
            return "DDoS-RSTFINFlood"
        if "SLOWLORIS" in path_upper:
            return "DDoS-SlowLoris"
        if "SYNONYMOUSIP" in path_upper:
            return "DDoS-SynonymousIP_Flood"
        if "UDP" in path_upper and "FRAG" in path_upper:
            return "DDoS-UDP_Fragmentation"
        if "UDP" in path_upper and "FLOOD" in path_upper:
            return "DDoS-UDP_Flood"
        if "SYN" in path_upper and "FLOOD" in path_upper:
            return "DDoS-SYN_Flood"
        if "TCP" in path_upper and "FLOOD" in path_upper:
            return "DDoS-TCP_Flood"

    if "DOS" in path_upper and "DDOS" not in path_upper:
        if "HTTP" in path_upper:
            return "DoS-HTTP_Flood"
        if "SYN" in path_upper:
            return "DoS-SYN_Flood"
        if "TCP" in path_upper:
            return "DoS-TCP_Flood"
        if "UDP" in path_upper:
            return "DoS-UDP_Flood"

    if "MIRAI" in path_upper:
        if "GREETH" in filename.upper():
            return "Mirai-greeth_flood"
        if "GREIP" in filename.upper():
            return "Mirai-greip_flood"
        if "UDPPLAIN" in filename.upper():
            return "Mirai-udpplain"

    if "SPOOF" in path_upper:
        if "ARP" in path_upper:
            return "ARP_Spoofing"
        if "DNS" in path_upper:
            return "DNS_Spoofing"

    if any(k in path_upper for k in ("WEB", "SQL", "XSS", "UPLOAD")):
        if "SQL" in path_upper:
            return "SQL_Injection"
        if "XSS" in path_upper:
            return "XSS"
        if "UPLOAD" in path_upper:
            return "Uploading_Attack"

    if "RECON" in path_upper:
        return "Vulnerability_Scan"

    if "BRUTE" in path_upper:
        return "Dictionary_BruteForce"

    if "BENIGN" in path_upper:
        return "Benign"

    return get_category_from_path(file_path)


def get_output_dir(user_output_dir: Path | None = None) -> Path:
    """Resuelve la carpeta de salida para los reportes."""
    if user_output_dir is not None:
        return user_output_dir.expanduser().resolve()

    # OUTPUT_FILE se conserva para mantener consistencia con tus scripts recientes.
    return OUTPUT_FILE.parent / "Analysis"


def count_classes(dataset_dir: Path, output_dir: Path):
    """Cuenta las instancias de cada clase en el dataset."""
    csv_files = sorted(
        p
        for p in dataset_dir.rglob("*.csv")
        if not any(part.lower() == "analysis" for part in p.parts)
    )

    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en: {dataset_dir}")

    print(f"Dataset: {dataset_dir}")
    print(f"Se encontraron {len(csv_files)} archivos CSV\n")

    category_counts = defaultdict(int)
    subcategory_counts = defaultdict(int)
    file_info = []

    chunk_size = 100000

    for idx, csv_file in enumerate(csv_files, 1):
        try:
            print(f"Procesando [{idx}/{len(csv_files)}]: {csv_file.name}...", end="\r")

            total_rows = 0
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
                total_rows += len(chunk)

            category = get_category_from_path(csv_file)
            subcategory = get_subcategory_from_path(csv_file)

            category_counts[category] += total_rows
            subcategory_counts[subcategory] += total_rows

            file_info.append(
                {
                    "File": csv_file.name,
                    "Category": category,
                    "Subcategory": subcategory,
                    "Instances": total_rows,
                }
            )

        except Exception as exc:
            print(f"\nError procesando {csv_file}: {exc}")
            continue

    category_total = sum(category_counts.values())
    subcategory_total = sum(subcategory_counts.values())

    print("\n\n" + "=" * 80)
    print("RESUMEN POR CATEGORÍAS PRINCIPALES")
    print("=" * 80)

    category_df = pd.DataFrame(
        [
            {
                "Category": cat,
                "Count": count,
                "Percentage": f"{(count / category_total) * 100:.2f}%" if category_total else "0.00%",
            }
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        ]
    )

    print(category_df.to_string(index=False))
    print(f"\nTotal de instancias: {category_total:,}")

    print("\n" + "=" * 80)
    print("RESUMEN POR SUBCATEGORÍAS")
    print("=" * 80)

    subcategory_df = pd.DataFrame(
        [
            {
                "Subcategory": subcat,
                "Count": count,
                "Percentage": f"{(count / subcategory_total) * 100:.2f}%" if subcategory_total else "0.00%",
            }
            for subcat, count in sorted(subcategory_counts.items(), key=lambda x: x[1], reverse=True)
        ]
    )

    print(subcategory_df.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)

    category_df.to_csv(output_dir / "category_counts.csv", index=False)
    subcategory_df.to_csv(output_dir / "subcategory_counts.csv", index=False)

    file_df = pd.DataFrame(file_info).sort_values("Instances", ascending=False)
    file_df.to_csv(output_dir / "file_details.csv", index=False)

    print(f"\n\nResultados guardados en: {output_dir}")
    print("  - category_counts.csv")
    print("  - subcategory_counts.csv")
    print("  - file_details.csv")

    return category_df, subcategory_df, file_df


def parse_args():
    parser = argparse.ArgumentParser(description="Contar clases del dataset CIC IoT-DIAD 2024")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Ruta al directorio raíz con CSVs de flujo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Ruta de salida para reportes CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    output_dir = get_output_dir(args.output_dir)
    category_df, subcategory_df, file_df = count_classes(dataset_dir, output_dir)
