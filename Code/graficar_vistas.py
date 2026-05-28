import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent 
ANALYSIS_DIR = PROJECT_ROOT / "Analysis" / "comparing"

INPUT_RUNS = ANALYSIS_DIR / "multiview_vs_singleview_per_run.csv"
INPUT_CLASS = ANALYSIS_DIR / "multiview_vs_singleview_per_class.csv"

OUTPUT_BOXPLOT = ANALYSIS_DIR / "01_boxplot_global.png"
OUTPUT_BAR_ENSAMBLES = ANALYSIS_DIR / "02_barplot_ensambles_por_clase.png"
OUTPUT_BAR_VISTAS = ANALYSIS_DIR / "03_barplot_vistas_por_clase.png"

def generar_graficas():
    print("=" * 60)
    print("📊 GENERANDO REPORTES VISUALES (ENSAMBLES Y VISTAS)")
    print("=" * 60)

    if not INPUT_RUNS.exists() or not INPUT_CLASS.exists():
        print("❌ Error: Faltan archivos CSV en el directorio de Análisis.")
        return

    # =========================================================
    # GRÁFICA 1: BOXPLOT CON ESCALA CORREGIDA
    # =========================================================
    df_runs = pd.read_csv(INPUT_RUNS)
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    orden_modelos = ["SingleView", "MultiViewStacking", "Voting_Hard", "Voting_Soft", "Vista_Tiempo", "Vista_Volumen", "Vista_Banderas", "Vista_Topologia"]
    colores_box = {"SingleView": "#7f8c8d", "MultiViewStacking": "#2c3e50", "Voting_Hard": "#e67e22", "Voting_Soft": "#9b59b6", "Vista_Tiempo": "#e74c3c", "Vista_Volumen": "#3498db", "Vista_Banderas": "#f1c40f", "Vista_Topologia": "#2ecc71"}

    sns.boxplot(data=df_runs, x="model", y="accuracy", order=orden_modelos, palette=colores_box, width=0.5, showfliers=False)
    sns.swarmplot(data=df_runs, x="model", y="accuracy", order=orden_modelos, color=".25", alpha=0.7, size=4)
    
    y_min = df_runs["accuracy"].min() - 0.02
    y_max = df_runs["accuracy"].max() + 0.02
    plt.ylim(y_min, y_max)
    
    plt.title("Estabilidad del Modelo Global (20 Iteraciones)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_BOXPLOT, dpi=300)
    plt.close()
    print(f"✅ Boxplot ajustado guardado como: {OUTPUT_BOXPLOT.name}")

    # =========================================================
    # PREPARACIÓN PARA GRÁFICAS POR CLASE
    # =========================================================
    df_class = pd.read_csv(INPUT_CLASS)
    
    # Separar los datos
    df_ensambles = df_class[df_class["model"].isin(["SingleView", "MultiViewStacking", "Voting_Hard", "Voting_Soft"])]
    df_vistas = df_class[df_class["model"].isin(["Vista_Tiempo", "Vista_Volumen", "Vista_Banderas", "Vista_Topologia"])]

    # =========================================================
    # GRÁFICA 2: ENSAMBLES POR CLASE (Single vs Multi)
    # =========================================================
    plt.figure(figsize=(14, 7))
    colores_ensambles = {"SingleView": "#bdc3c7", "MultiViewStacking": "#2980b9", "Voting_Hard": "#e67e22", "Voting_Soft": "#9b59b6"}
    
    sns.barplot(
        data=df_ensambles, x="class", y="f1-score", hue="model", 
        palette=colores_ensambles, capsize=.05, err_kws={'linewidth': 1.5}
    )
    
    plt.title("Comparativa de Ensambles por Tipo de Ataque (F1-Score)", fontsize=15, fontweight='bold', pad=15)
    plt.ylabel("F1-Score Promedio", fontsize=12)
    plt.xlabel("Tipo de Tráfico / Ataque", fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., title="Ensambles")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_BAR_ENSAMBLES, dpi=300)
    plt.close()
    print(f"✅ Gráfico Ensambles vs Clase guardado como: {OUTPUT_BAR_ENSAMBLES.name}")

    # =========================================================
    # GRÁFICA 3: VISTAS INDIVIDUALES POR CLASE
    # =========================================================
    plt.figure(figsize=(14, 7))
    
    colores_vistas = {
        "Vista_Tiempo": "#e74c3c", 
        "Vista_Volumen": "#3498db", 
        "Vista_Banderas": "#f1c40f", 
        "Vista_Topologia": "#2ecc71"
    }
    
    sns.barplot(
        data=df_vistas, x="class", y="f1-score", hue="model", 
        palette=colores_vistas, capsize=.05, err_kws={'linewidth': 1.5}
    )
    
    plt.title("Diagnóstico: Desempeño de Vistas Individuales por Tipo de Ataque (F1-Score)", fontsize=15, fontweight='bold', pad=15)
    plt.ylabel("F1-Score Promedio", fontsize=12)
    plt.xlabel("Tipo de Tráfico / Ataque", fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., title="Vistas Base")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_BAR_VISTAS, dpi=300)
    plt.close()
    print(f"✅ Gráfico Vistas vs Clase guardado como: {OUTPUT_BAR_VISTAS.name}")

if __name__ == "__main__":
    generar_graficas()