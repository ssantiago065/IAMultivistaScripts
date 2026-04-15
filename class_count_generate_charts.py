"""
generate_charts.py
Genera 4 gráficas de distribución de clases para el dataset CIC IoT-DIAD 2024.

Estructura esperada del proyecto:
    .
    ├── Analysis/
    │   └── class_count/
    │       ├── category_counts.csv
    │       ├── subcategory_counts.csv
    │       └── file_details.csv
    ├── class_count.py
    └── generate_charts.py   ← este archivo

Salida: ./Analysis/charts/fig1_category_bar.png
                           fig2_category_pie.png
                           fig3_subcategory_bar.png
                           fig4_treemap.png

Uso:
    python generate_charts.py
"""

from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Rutas ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "Analysis" / "class_count"
OUTPUT_DIR = SCRIPT_DIR / "Analysis" / "charts"


# ── Utilidades ───────────────────────────────────────────────────────────────
def fmt(n: int) -> str:
    if n >= 1_000_000: return f"{n/1e6:.1f}M"
    if n >= 1_000:     return f"{n/1e3:.0f}k"
    return str(n)

def _write_meta(path: Path, caption: str, description: str):
    with open(str(path) + ".meta.json", "w") as f:
        json.dump({"caption": caption, "description": description}, f)

def load_data():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"\n❌ No se encontró la carpeta de datos: {DATA_DIR}\n"
            f"   Asegúrate de correr class_count.py antes de este script."
        )
    cat_df  = pd.read_csv(DATA_DIR / "category_counts.csv")
    sub_df  = pd.read_csv(DATA_DIR / "subcategory_counts.csv")
    file_df = pd.read_csv(DATA_DIR / "file_details.csv")
    return cat_df, sub_df, file_df


# ── Figura 1: Barras – Categorías principales ────────────────────────────────
def fig1_category_bar(cat_df: pd.DataFrame):
    df = cat_df.sort_values("Count", ascending=True)
    n  = len(df)
    palette = (["#c8dfe1"] * max(0, n - 4)
               + ["#6eadb5", "#4f98a3", "#0e5e67", "#01696f"])[-n:]

    fig = go.Figure(go.Bar(
        x=df["Count"], y=df["Category"],
        orientation="h",
        marker_color=palette,
        hovertemplate="<b>%{y}</b><br>Instancias: %{x:,}<extra></extra>",
    ))
    for count, cat in zip(df["Count"], df["Category"]):
        fig.add_annotation(
            x=count, y=cat, text=fmt(count),
            xanchor="left", xshift=6, showarrow=False,
            font=dict(size=12, color="#2a2a2a"),
        )
    fig.update_layout(
        title=dict(
            text="Distribución por Categoría Principal<br>"
                 "<span style='font-size:15px;font-weight:normal;color:#555'>"
                 "CIC IoT-DIAD 2024 · Flow-Based</span>",
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="Instancias", tickformat=".2s",
                   showgrid=True, gridcolor="#e8e8e8"),
        yaxis=dict(title="Categoría", tickfont=dict(size=13)),
        margin=dict(l=20, r=110, t=85, b=55),
        height=420, width=900,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=13),
    )
    out = OUTPUT_DIR / "fig1_category_bar.png"
    fig.write_image(str(out), scale=2)
    _write_meta(out, "Figura 1 – Distribución por categoría principal",
                "Barras horizontales ordenadas por instancias.")
    print(f"  ✅ {out.name}")


# ── Figura 2: Donut – Proporción de clases ───────────────────────────────────
def fig2_category_pie(cat_df: pd.DataFrame):
    df     = cat_df.sort_values("Count", ascending=False)
    total  = df["Count"].sum()
    colors = ["#01696f","#4f98a3","#6daa45","#d19900",
              "#da7101","#a12c7b","#006494","#a13544"]
    labels_pct = [f"{r['Category']}  ({r['Count']/total*100:.1f}%)"
                  for _, r in df.iterrows()]

    fig = go.Figure(go.Pie(
        labels=labels_pct,
        values=df["Count"],
        hole=0.42,
        marker_colors=colors[:len(df)],
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{fmt(total)}</b><br>total",
        x=0.5, y=0.5,
        font=dict(size=16, family="Inter, sans-serif"),
        showarrow=False,
    )
    fig.update_layout(
        title=dict(
            text="Proporción de Clases<br>"
                 "<span style='font-size:15px;font-weight:normal;color:#555'>"
                 "Desbalance severo – DoS representa el 83% del total</span>",
            x=0.5, xanchor="center",
        ),
        legend=dict(orientation="v", x=1.0, y=0.5,
                    xanchor="left", font=dict(size=13)),
        margin=dict(l=20, r=220, t=85, b=40),
        height=460, width=920,
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=13),
        uniformtext_minsize=12, uniformtext_mode="hide",
    )
    out = OUTPUT_DIR / "fig2_category_pie.png"
    fig.write_image(str(out), scale=2)
    _write_meta(out, "Figura 2 – Proporción de clases",
                "Donut chart. DoS+DDoS suman el 95.7% del dataset.")
    print(f"  ✅ {out.name}")


# ── Figura 3: Barras – Subcategorías (escala log) ────────────────────────────
def fig3_subcategory_bar(sub_df: pd.DataFrame):
    df = sub_df[sub_df["Count"] > 0].sort_values("Count", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["Count"], y=df["Subcategory"],
        orientation="h",
        marker=dict(
            color=df["Count"],
            colorscale=[[0,"#c8dfe1"],[0.5,"#4f98a3"],[1,"#01696f"]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Instancias: %{x:,}<extra></extra>",
    ))
    for count, sub in zip(df["Count"], df["Subcategory"]):
        fig.add_annotation(
            x=count, y=sub, text=fmt(count),
            xanchor="left", xshift=6, showarrow=False,
            font=dict(size=11, color="#2a2a2a"),
        )
    fig.update_layout(
        title=dict(
            text="Distribución por Subcategoría (escala logarítmica)<br>"
                 "<span style='font-size:15px;font-weight:normal;color:#555'>"
                 f"{len(df)} subcategorías activas · Flow-Based</span>",
            x=0.5, xanchor="center", y=0.98,
        ),
        xaxis=dict(title="Instancias (log)", type="log", tickformat=".1s",
                   showgrid=True, gridcolor="#e8e8e8"),
        yaxis=dict(title="Subcategoría", tickfont=dict(size=11)),
        margin=dict(l=20, r=100, t=90, b=55),
        height=600, width=960,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
    )
    out = OUTPUT_DIR / "fig3_subcategory_bar.png"
    fig.write_image(str(out), scale=2)
    _write_meta(out, "Figura 3 – Distribución por subcategoría (escala log)",
                "Rango desde 16.3M (DoS-SYN_Flood) hasta 1.3k (Uploading_Attack).")
    print(f"  ✅ {out.name}")


# ── Figura 4: Treemap jerárquico ─────────────────────────────────────────────
CAT_MAP = {
    "DoS-SYN_Flood":"DoS",   "DoS-UDP_Flood":"DoS",
    "DoS-HTTP_Flood":"DoS",  "DoS-TCP_Flood":"DoS",
    "DDoS-ACK_Fragmentation":"DDoS", "DDoS-HTTP_Flood":"DDoS",
    "DDoS-ICMP_Fragmentation":"DDoS","DDoS-ICMP_Flood":"DDoS",
    "DDoS-SYN_Flood":"DDoS", "DDoS-UDP_Flood":"DDoS",
    "DDoS-PSHACK_Flood":"DDoS","DDoS-RSTFINFlood":"DDoS",
    "DDoS-SlowLoris":"DDoS", "DDoS-SynonymousIP_Flood":"DDoS",
    "DDoS-TCP_Flood":"DDoS", "DDoS-UDP_Fragmentation":"DDoS",
    "Vulnerability_Scan":"Recon",
    "Benign":"Benign",
    "Mirai-greeth_flood":"Mirai","Mirai-greip_flood":"Mirai","Mirai-udpplain":"Mirai",
    "ARP_Spoofing":"Spoofing","DNS_Spoofing":"Spoofing",
    "SQL_Injection":"Web-Based","XSS":"Web-Based","Uploading_Attack":"Web-Based",
    "Dictionary_BruteForce":"Brute Force",
}

def fig4_treemap(sub_df: pd.DataFrame):
    df = sub_df[sub_df["Count"] > 0].copy()
    df["Category"] = df["Subcategory"].map(CAT_MAP).fillna("Other")

    fig = px.treemap(
        df, path=["Category","Subcategory"], values="Count",
        color="Count",
        color_continuous_scale=["#c8dfe1","#4f98a3","#01696f","#0c4e54"],
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Instancias: %{value:,}<br>"
                      "Del total: %{percentRoot:.2%}<extra></extra>",
        textinfo="label+percent root",
        textfont=dict(size=13),
    )
    fig.update_layout(
        title=dict(
            text="Jerarquía de Clases – CIC IoT-DIAD 2024<br>"
                 "<span style='font-size:15px;font-weight:normal;color:#555'>"
                 "Categoría → Subcategoría por volumen de instancias</span>",
            x=0.5, xanchor="center",
        ),
        margin=dict(l=10, r=10, t=85, b=10),
        height=530, width=960,
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=13),
        coloraxis_showscale=False,
    )
    out = OUTPUT_DIR / "fig4_treemap.png"
    fig.write_image(str(out), scale=2)
    _write_meta(out, "Figura 4 – Treemap jerarquía categoría/subcategoría",
                "Vista jerárquica: tamaño proporcional al número de instancias.")
    print(f"  ✅ {out.name}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDatos  : {DATA_DIR}")
    print(f"Salida : {OUTPUT_DIR}\n")

    cat_df, sub_df, file_df = load_data()

    fig1_category_bar(cat_df)
    fig2_category_pie(cat_df)
    fig3_subcategory_bar(sub_df)
    fig4_treemap(sub_df)

    print(f"\n🎉 4 gráficas guardadas en: {OUTPUT_DIR}")