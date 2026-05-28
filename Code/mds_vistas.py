import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DATASET = PROJECT_ROOT / "dataset_poc_multivista.csv"
OUTPUT_DIR = PROJECT_ROOT / "Analysis" / "comparing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 2000

VISTAS = {
    "Vista_Tiempo": [
        "Flow IAT Min", "Bwd IAT Max", "Bwd IAT Std", "Flow Packets/s",
        "Bwd IAT Min", "Active Min", "Fwd IAT Max", "Active Max",
        "Flow Duration", "Idle Mean", "Flow IAT Max", "Fwd Packets/s",
        "Fwd IAT Min", "Flow IAT Mean", "Bwd Packets/s", "Bwd IAT Total",
        "Fwd IAT Std", "Idle Min", "Active Mean", "Bwd IAT Mean",
        "Fwd IAT Mean", "Active Std", "Idle Max", "Fwd IAT Total",
        "Flow Bytes/s", "Idle Std", "Flow IAT Std"
    ],
    "Vista_Volumen": [
        "Total Bwd packets", "Packet Length Std", "Packet Length Min",
        "Packet Length Mean", "Bwd Header Length", "Subflow Fwd Packets",
        "Bwd Packet Length Max", "Bwd Bulk Rate Avg", "Bwd Packet Length Std",
        "Packet Length Max", "Total Length of Bwd Packet", "Packet Length Variance",
        "Subflow Bwd Packets", "Fwd Packet Length Std", "Total Length of Fwd Packet",
        "Fwd Act Data Pkts", "Bwd Segment Size Avg", "Fwd Header Length",
        "Bwd Packet Length Min", "Bwd Bytes/Bulk Avg", "FWD Init Win Bytes", 
        "Average Packet Size", "Total Fwd Packet", "Fwd Packet Length Mean", 
        "Bwd Packet Length Mean", "Fwd Packet Length Min", "Bwd Init Win Bytes", 
        "Fwd Segment Size Avg", "Bwd Packet/Bulk Avg", "Fwd Seg Size Min", 
        "Subflow Bwd Bytes", "Fwd Packet Length Max"
    ],
    "Vista_Banderas": [
        "ACK Flag Count", "Fwd PSH Flags", "PSH Flag Count",
        "CWR Flag Count", "SYN Flag Count", "RST Flag Count",
        "FIN Flag Count", "Down/Up Ratio", "ECE Flag Count"
    ],
    "Vista_Topologia": [
        "Protocol", "Src Port", "Dst Port"
    ]
}

def plot_mds_per_vista(df, vistas, output_dir, n_samples=N_SAMPLES):
    df_sample = df.sample(n=n_samples, random_state=42)
    labels = df_sample["Label"].values
    unique_labels = sorted(df_sample["Label"].unique())
    palette = plt.cm.get_cmap("tab10", len(unique_labels))
    color_map = {lbl: palette(i) for i, lbl in enumerate(unique_labels)}
    colors = [color_map[l] for l in labels]

    for vista_name, cols in vistas.items():
        cols_disponibles = [c for c in cols if c in df_sample.columns]
        X_vista = df_sample[cols_disponibles].values
        X_scaled = MinMaxScaler().fit_transform(X_vista)
        
        # Imputar NaN/inf si los hay
        X_scaled = np.nan_to_num(X_scaled)

        print(f"Calculando MDS para {vista_name} ({len(cols_disponibles)} features)...")
        mds = MDS(n_components=2, random_state=42, normalized_stress='auto')
        X_2d = mds.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(10, 7))
        for lbl in unique_labels:
            mask = labels == lbl
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[color_map[lbl]], label=lbl, alpha=0.6, s=15, edgecolors='none')

        ax.set_title(f"MDS — {vista_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Clase", fontsize=8)
        plt.tight_layout()
        
        out_path = output_dir / f"mds_{vista_name.lower()}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  ✅ Guardado: {out_path.name}")

if __name__ == "__main__":
    df = pd.read_csv(INPUT_DATASET)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    plot_mds_per_vista(df, VISTAS, OUTPUT_DIR)
