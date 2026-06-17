import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent 
ANALYSIS_DIR = PROJECT_ROOT / "Analysis" / "learning_curve"

INPUT_CSV = ANALYSIS_DIR / "learning_curve_per_run.csv"
OUTPUT_PLOT = ANALYSIS_DIR / "boxplot_learning_curve.png"

def generar_curva_aprendizaje():
    print("=" * 60)
    print("📈 GENERANDO GRÁFICA DE CURVA DE APRENDIZAJE")
    print("=" * 60)

    if not INPUT_CSV.exists():
        print(f"❌ Error: No se encontró el archivo: {INPUT_CSV}")
        return

    # 1. Cargar Datos
    df = pd.read_csv(INPUT_CSV)
    
    # Transformar la fracción a porcentaje (ej. 0.1 -> "10%") para mayor legibilidad en el eje X
    df['train_percentage'] = (df['train_size_pct']).astype(int).astype(str) + "%"

    # 2. Configurar la Figura
    plt.figure(figsize=(16, 8)) # Ancho extendido para acomodar 20 cajas (10 pares)
    sns.set_theme(style="whitegrid")
    
    colores_modelos = {"SingleView": "#7f8c8d", "MultiViewStacking": "#2c3e50"}

    # 3. Dibujar el Boxplot Agrupado
    # El parámetro 'hue' separa las cajas de SingleView y MultiView en cada punto del eje X
    sns.boxplot(
        data=df, 
        x="train_percentage", 
        y="accuracy", 
        hue="model", 
        palette=colores_modelos,
        width=0.6,
        showfliers=False # Ocultamos outliers del boxplot para que el stripplot los maneje
    )

    # 4. Trazar la Línea de Tendencia (Une las medianas de cada caja)
    # Esto da el efecto visual clásico de la "Curva de Aprendizaje"
    sns.pointplot(
        data=df, 
        x="train_percentage", 
        y="accuracy", 
        hue="model", 
        palette={"SingleView": "#34495e", "MultiViewStacking": "#1a252f"}, # Tonos ligeramente más oscuros para las líneas
        dodge=0.3, # Alinea los puntos con las cajas agrupadas
        markers=["o", "s"],
        linestyles=["-", "--"],
        errorbar=None, # Ya tenemos las cajas para mostrar la varianza
        legend=False # Ocultamos la leyenda duplicada del pointplot
    )

    # 5. Estilización de Títulos y Ejes
    plt.title("Curva de Aprendizaje: Impacto del Volumen de Datos de Entrenamiento", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Accuracy (Exactitud Global)", fontsize=13)
    plt.xlabel("Porcentaje de Datos de Entrenamiento Utilizados (Train Set)", fontsize=13)
    
    # Ajuste dinámico del eje Y para maximizar el zoom en las cajas
    y_min = df["accuracy"].min() - 0.01
    y_max = df["accuracy"].max() + 0.01
    plt.ylim(y_min, y_max)
    
    # Ajustar la leyenda (Moverla fuera o a una esquina que no estorbe)
    plt.legend(title="Arquitectura", loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()
    
    print(f"✅ Curva de aprendizaje guardada exitosamente en: {OUTPUT_PLOT.name}")

if __name__ == "__main__":
    generar_curva_aprendizaje()