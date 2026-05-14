import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANALYSIS_DIR = PROJECT_ROOT / "Analysis" / "comparing"

INPUT_CSV = ANALYSIS_DIR / "multiview_vs_singleview_per_run.csv"
OUTPUT_PLOT_ACC = ANALYSIS_DIR / "boxplot_accuracy.png"
OUTPUT_PLOT_F1 = ANALYSIS_DIR / "boxplot_f1_macro.png"

def generar_graficos():
    print("=" * 60)
    print("📊 GENERANDO BOXPLOTS DE MÉTRICAS")
    print("=" * 60)

    # 1. Verificar si el archivo existe
    if not INPUT_CSV.exists():
        print(f"❌ Error: No se encontró el archivo de resultados en: {INPUT_CSV}")
        return

    # 2. Cargar los datos
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Datos cargados correctamente ({len(df)} filas encontradas).")

    # Configuración de estilo global para que se vean profesionales
    sns.set_theme(style="whitegrid")
    
    # Colores personalizados (Azul para MultiView, Naranja/Rojo para SingleView)
    colores = {"MultiViewStacking": "#1f77b4", "SingleView": "#ff7f0e"}

    # ==========================================
    # GRÁFICO 1: ACCURACY (Exactitud Global)
    # ==========================================
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="model", y="accuracy", palette=colores, width=0.5)
    
    # Añadir los puntos reales encima de la caja para ver la distribución exacta (Swarmplot)
    sns.swarmplot(data=df, x="model", y="accuracy", color=".25", alpha=0.6)

    plt.title("Comparación de Accuracy en 20 Corridas\n(MultiView vs SingleView)", fontsize=14, pad=15)
    plt.ylabel("Accuracy (Exactitud)", fontsize=12)
    plt.xlabel("Arquitectura del Modelo", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_ACC, dpi=300) # Guardar en alta resolución
    print(f"✅ Boxplot de Accuracy guardado en: {OUTPUT_PLOT_ACC.name}")
    plt.close()

    # ==========================================
    # GRÁFICO 2: F1-MACRO (Equilibrio entre clases)
    # ==========================================
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="model", y="f1_macro", palette=colores, width=0.5)
    sns.swarmplot(data=df, x="model", y="f1_macro", color=".25", alpha=0.6)

    plt.title("Comparación de F1-Macro en 20 Corridas\n(Sensibilidad a clases minoritarias)", fontsize=14, pad=15)
    plt.ylabel("F1-Macro Score", fontsize=12)
    plt.xlabel("Arquitectura del Modelo", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_F1, dpi=300)
    print(f"✅ Boxplot de F1-Macro guardado en: {OUTPUT_PLOT_F1.name}")
    plt.close()

    print("=" * 60)
    print("🎉 Gráficos generados con éxito.")

if __name__ == "__main__":
    generar_graficos()