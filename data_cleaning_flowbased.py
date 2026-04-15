import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Fix encoding for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --- CONFIGURACION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FLOW_DIR = PROJECT_ROOT / "Anomaly Detection - Flow Based features"
OUTPUT_DIR = PROJECT_ROOT / "cleaned_dataset_flowbased"
PROFILING_SUMMARY = PROJECT_ROOT / "Analysis" / "data_profiling" / "profiling_summary_by_category.csv"


class DataCleaner:
    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Path,
        chunk_size: int = 200000,
        duplicate_mode: str = "chunk",
        skip_existing: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.duplicate_mode = duplicate_mode
        self.skip_existing = skip_existing

        # Columnas constantes detectadas en el profiling.
        self.constant_columns_to_remove = [
            "Bwd PSH Flags",
            "Fwd URG Flags",
            "Bwd URG Flags",
            "URG Flag Count",
            "Fwd Bytes/Bulk Avg",
            "Fwd Packet/Bulk Avg",
            "Fwd Bulk Rate Avg",
            "CWR Flag Count",
            "ECE Flag Count",
        ]

        self.cleaning_stats = []

    def _save_progress(self):
        progress_file = self.output_dir / "cleaning_progress.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(self.cleaning_stats, f, indent=2)
        print(f"\n[PROGRESO GUARDADO: {len(self.cleaning_stats)} archivos procesados]")

    def _clean_chunk(self, df: pd.DataFrame, attack_type: str, seen_hashes: set | None):
        df = df.copy()
        df.columns = df.columns.str.strip()

        chunk_stats = {
            "missing_values": 0,
            "duplicates": 0,
            "infinite_values": 0,
            "negative_values": 0,
            "all_zeros": 0,
        }

        # 1) Filas con missing
        missing_mask = df.isnull().any(axis=1)
        chunk_stats["missing_values"] = int(missing_mask.sum())
        if chunk_stats["missing_values"]:
            df = df.loc[~missing_mask].copy()

        # 2) Filas duplicadas
        if self.duplicate_mode == "global":
            row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64)
            duplicate_mask = np.zeros(len(df), dtype=bool)
            for i, h in enumerate(row_hashes):
                h_int = int(h)
                if h_int in seen_hashes:
                    duplicate_mask[i] = True
                else:
                    seen_hashes.add(h_int)
            chunk_stats["duplicates"] = int(duplicate_mask.sum())
            if chunk_stats["duplicates"]:
                df = df.loc[~duplicate_mask].copy()
        else:
            duplicates_mask = df.duplicated(keep="first")
            chunk_stats["duplicates"] = int(duplicates_mask.sum())
            if chunk_stats["duplicates"]:
                df = df.loc[~duplicates_mask].copy()

        # 3) Filas con infinitos
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if numeric_cols:
            inf_mask = np.isinf(df[numeric_cols]).any(axis=1)
            chunk_stats["infinite_values"] = int(inf_mask.sum())
            if chunk_stats["infinite_values"]:
                df = df.loc[~inf_mask].copy()

        # 4) Filas con negativos anomales
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        positive_only_cols = [
            col
            for col in numeric_cols
            if ("Length" in col or "Bytes" in col or "Packet" in col or "Duration" in col or "Count" in col)
        ]

        if positive_only_cols:
            negative_mask = np.zeros(len(df), dtype=bool)
            for col in positive_only_cols:
                negative_mask |= (df[col].to_numpy() < 0)
            chunk_stats["negative_values"] = int(negative_mask.sum())
            if chunk_stats["negative_values"]:
                df = df.loc[~negative_mask].copy()

        # 5) Filas con todas las numericas en cero
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        exclude_cols = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Label"]
        numeric_cols_to_check = [col for col in numeric_cols if col not in exclude_cols]
        if numeric_cols_to_check:
            all_zero_mask = (df[numeric_cols_to_check] == 0).all(axis=1)
            chunk_stats["all_zeros"] = int(all_zero_mask.sum())
            if chunk_stats["all_zeros"]:
                df = df.loc[~all_zero_mask].copy()

        # 6) Eliminar columnas constantes detectadas en profiling
        cols_removed = 0
        for col in self.constant_columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
                cols_removed += 1

        # 7) Asegurar columna Label
        if "Label" not in df.columns:
            df["Label"] = attack_type

        return df, chunk_stats, cols_removed

    def _build_empty_output(self, file_path: Path, attack_type: str, output_file: Path):
        header_df = pd.read_csv(file_path, nrows=0)
        cols = [c.strip() for c in header_df.columns]
        cols = [c for c in cols if c not in self.constant_columns_to_remove]
        if "Label" not in cols:
            cols.append("Label")

        df_empty = pd.DataFrame(columns=cols)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_empty.to_csv(output_file, index=False)

    def clean_file(self, file_path: Path, attack_type: str, output_dir: Path):
        """Limpia un CSV por chunks y guarda resultado incremental."""
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        output_file = output_dir / file_path.name

        print(f"\n{'=' * 80}")
        print(f"Limpiando: {file_path.name}")
        print(f"Tipo de ataque: {attack_type}")

        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"Tamano: {file_size_mb:.2f} MB")
            print(f"{'=' * 80}")

            stats = {
                "file": file_path.name,
                "attack_type": attack_type,
                "original_rows": 0,
                "original_columns": 0,
                "rows_removed": {
                    "missing_values": 0,
                    "duplicates": 0,
                    "infinite_values": 0,
                    "negative_values": 0,
                    "all_zeros": 0,
                },
                "columns_removed": 0,
            }

            output_dir.mkdir(parents=True, exist_ok=True)
            if output_file.exists():
                output_file.unlink()

            seen_hashes = set() if self.duplicate_mode == "global" else None
            wrote_any_row = False
            wrote_header = False

            reader = pd.read_csv(file_path, chunksize=self.chunk_size, low_memory=False)
            for chunk_idx, chunk in enumerate(reader, 1):
                stats["original_rows"] += len(chunk)
                if stats["original_columns"] == 0:
                    stats["original_columns"] = len(chunk.columns)

                cleaned_chunk, chunk_stats, cols_removed_chunk = self._clean_chunk(chunk, attack_type, seen_hashes)
                for key, value in chunk_stats.items():
                    stats["rows_removed"][key] += int(value)
                stats["columns_removed"] = max(stats["columns_removed"], cols_removed_chunk)

                if not cleaned_chunk.empty:
                    cleaned_chunk.to_csv(
                        output_file,
                        index=False,
                        mode="a",
                        header=not wrote_header,
                    )
                    wrote_header = True
                    wrote_any_row = True

                if chunk_idx % 10 == 0:
                    print(f"   Progreso chunk {chunk_idx} | filas leidas: {stats['original_rows']:,}")

            # Caso archivo vacio (solo cabecera)
            if stats["original_rows"] == 0:
                stats["original_columns"] = len(pd.read_csv(file_path, nrows=0).columns)
                self._build_empty_output(file_path, attack_type, output_file)
                stats["empty_file"] = True
                stats["final_rows"] = 0
                stats["total_rows_removed"] = 0
                stats["percentage_removed"] = 0.0
                stats["final_columns"] = len(pd.read_csv(output_file, nrows=0).columns)
                print("[WARN] Archivo sin filas de datos, se guardo limpio con cabecera.")
                return stats

            # Si no quedo ninguna fila tras limpieza, conservar cabecera limpia.
            if not wrote_any_row:
                self._build_empty_output(file_path, attack_type, output_file)
                wrote_header = True

            final_rows = 0
            if output_file.exists():
                for chunk in pd.read_csv(output_file, chunksize=self.chunk_size, low_memory=False):
                    final_rows += len(chunk)

            total_rows_removed = stats["original_rows"] - final_rows
            percentage_removed = (
                (total_rows_removed / stats["original_rows"]) * 100 if stats["original_rows"] > 0 else 0.0
            )

            stats["final_rows"] = int(final_rows)
            stats["total_rows_removed"] = int(total_rows_removed)
            stats["percentage_removed"] = float(percentage_removed)
            stats["final_columns"] = len(pd.read_csv(output_file, nrows=0).columns)
            stats["empty_file"] = False

            print(f"\n{'=' * 80}")
            print("RESUMEN DE LIMPIEZA:")
            print(f"  Filas originales: {stats['original_rows']:,}")
            print(f"  Filas eliminadas: {stats['total_rows_removed']:,} ({stats['percentage_removed']:.2f}%)")
            print(f"  Filas finales: {stats['final_rows']:,}")
            print(f"  Columnas originales: {stats['original_columns']}")
            print(f"  Columnas finales: {stats['final_columns']}")
            print(f"  Archivo guardado en: {output_file}")
            print(f"{'=' * 80}")

            return stats

        except Exception as exc:
            print(f"ERROR al limpiar {file_path}: {exc}")
            return None

    def clean_all_files(self, categories=None):
        """Limpia todos los CSV del dataset manteniendo subdirectorios."""
        default_categories = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-Based"]
        categories = categories or default_categories

        print("=" * 100)
        print("LIMPIEZA COMPLETA DEL DATASET IOT (FLOW-BASED)")
        print("=" * 100)
        print(f"Ruta de entrada: {self.dataset_dir}")
        print(f"Ruta de salida: {self.output_dir}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Modo duplicados: {self.duplicate_mode}")
        print("=" * 100)

        for category in categories:
            category_path = self.dataset_dir / category

            if not category_path.exists():
                print(f"\nCategoria no encontrada: {category}")
                continue

            output_category_dir = self.output_dir / category
            csv_files = sorted(category_path.rglob("*.csv"))

            if not csv_files:
                print(f"\nNo se encontraron archivos CSV en: {category}")
                continue

            print(f"\n\n{'#' * 80}")
            print(f"# CATEGORIA: {category} ({len(csv_files)} archivos)")
            print(f"{'#' * 80}")

            for idx, csv_file in enumerate(csv_files, 1):
                print(f"\n[Progreso: {idx}/{len(csv_files)}]")

                rel_path = csv_file.parent.relative_to(category_path)
                output_dir = output_category_dir / rel_path
                output_file = output_dir / csv_file.name

                if self.skip_existing and output_file.exists():
                    print(f"Saltando (ya procesado): {csv_file.name}")
                    continue

                stats = self.clean_file(csv_file, category, output_dir)
                if stats is not None:
                    self.cleaning_stats.append(stats)
                    if len(self.cleaning_stats) % 10 == 0:
                        self._save_progress()

        return self.cleaning_stats

    def generate_cleaning_report(self):
        """Genera reportes CSV con estadisticas de limpieza."""
        if not self.cleaning_stats:
            print("No hay estadisticas de limpieza para reportar")
            return

        report_data = []
        for s in self.cleaning_stats:
            report_data.append(
                {
                    "Archivo": s["file"],
                    "Categoria": s["attack_type"],
                    "Filas Originales": s["original_rows"],
                    "Filas Eliminadas": s["total_rows_removed"],
                    "% Eliminado": round(s["percentage_removed"], 6),
                    "Filas Finales": s["final_rows"],
                    "Columnas Eliminadas": s["columns_removed"],
                    "Archivo Vacio": s.get("empty_file", False),
                }
            )

        df_report = pd.DataFrame(report_data)
        report_file = self.output_dir / "cleaning_report_flowbased.csv"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        df_report.to_csv(report_file, index=False)

        category_summary = (
            df_report.groupby("Categoria", as_index=False)
            .agg(
                {
                    "Archivo": "count",
                    "Filas Originales": "sum",
                    "Filas Eliminadas": "sum",
                    "Filas Finales": "sum",
                    "Archivo Vacio": "sum",
                }
            )
            .rename(columns={"Archivo": "Archivos"})
        )
        category_summary["% Eliminado"] = (
            category_summary["Filas Eliminadas"] / category_summary["Filas Originales"].replace(0, np.nan) * 100
        ).fillna(0.0)

        summary_file = self.output_dir / "cleaning_summary_by_category.csv"
        category_summary.to_csv(summary_file, index=False)

        total_original_rows = int(df_report["Filas Originales"].sum())
        total_removed = int(df_report["Filas Eliminadas"].sum())
        total_final_rows = int(df_report["Filas Finales"].sum())
        overall_pct = (total_removed / total_original_rows * 100) if total_original_rows > 0 else 0.0

        problem_totals = defaultdict(int)
        for s in self.cleaning_stats:
            for key, value in s["rows_removed"].items():
                problem_totals[key] += int(value)

        print("\n\n" + "=" * 100)
        print("REPORTE FINAL DE LIMPIEZA DEL DATASET")
        print("=" * 100)
        print(f"Total de archivos procesados: {len(self.cleaning_stats)}")
        print(f"Total de filas originales: {total_original_rows:,}")
        print(f"Total de filas eliminadas: {total_removed:,} ({overall_pct:.2f}%)")
        print(f"Total de filas finales: {total_final_rows:,}")
        print("\nDetalle por problema:")
        print(f"  Missing values: {problem_totals['missing_values']:,}")
        print(f"  Duplicados: {problem_totals['duplicates']:,}")
        print(f"  Infinitos: {problem_totals['infinite_values']:,}")
        print(f"  Negativos: {problem_totals['negative_values']:,}")
        print(f"  All zeros: {problem_totals['all_zeros']:,}")
        print(f"\nReporte detallado: {report_file}")
        print(f"Resumen por categoria: {summary_file}")
        print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(description="Limpieza Flow-Based usando reglas detectadas en profiling")
    parser.add_argument("--dataset-dir", type=Path, default=FLOW_DIR, help="Ruta al dataset Flow-Based")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Ruta de salida para dataset limpio")
    parser.add_argument("--chunk-size", type=int, default=200000, help="Tamano de chunk para procesamiento")
    parser.add_argument(
        "--duplicate-mode",
        choices=["chunk", "global"],
        default="chunk",
        help="Deteccion de duplicados por chunk (estable) o global (mas estricto)",
    )
    parser.add_argument("--categories", type=str, default="", help="Categorias separadas por coma (vacio = todas)")
    parser.add_argument("--skip-existing", action="store_true", help="Saltar archivos ya limpios")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_dir}")

    if PROFILING_SUMMARY.exists():
        print(f"Se detecto profiling summary: {PROFILING_SUMMARY}")
    else:
        print(f"[WARN] No se encontro profiling summary en: {PROFILING_SUMMARY}")

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not categories:
        categories = None

    cleaner = DataCleaner(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        duplicate_mode=args.duplicate_mode,
        skip_existing=args.skip_existing,
    )

    cleaner.clean_all_files(categories=categories)
    cleaner.generate_cleaning_report()


if __name__ == "__main__":
    main()
