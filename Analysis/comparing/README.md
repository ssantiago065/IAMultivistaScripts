# Multi-View Stacking para Clasificación de Tráfico de Red

Comparativa experimental de **MultiViewStacking vs. SingleView (Random Forest baseline)** sobre un dataset de tráfico de red con 6 clases de ataque. Se evalúan 5 configuraciones de vistas usando 20 runs con seeds fijas para reproducibilidad estadística.

---

## Estructura del repositorio

```
Dataset/
├── cleaned_dataset_flowbased/   # CSVs originales por clase
├── dataset_poc_multivista.csv   # Dataset balanceado generado
├── Code/
│   ├── dataset_preparing_multiview_test.py   # Genera el dataset balanceado
│   └── multiview_test.py                     # Experimentos comparativos
└── Analysis/
    └── comparing/
        ├── multiview_vs_singleview_per_run.csv    # Accuracy por run y experimento
        ├── multiview_vs_singleview_summary.csv    # Media y std por modelo
        └── multiview_vs_singleview_per_class.csv  # F1 por clase
```

---

## Dataset

Generado por `dataset_preparing_multiview_test.py` a partir de `cleaned_dataset_flowbased/`.

| Parámetro | Valor |
|---|---|
| Clases | Benign, DDoS, DoS, Mirai, Recon, Spoofing |
| Muestras por clase | 100,000 (undersampling, nunca oversampling) |
| Total filas | ~600,000 |
| Columnas eliminadas | Flow ID, Src IP, Dst IP, Timestamp |
| Split Train/Test | 70% / 30% estratificado |

---

## Definición de vistas

Las features del dataset se particionan en 4 grupos semánticos mutuamente excluyentes:

| Vista | Features | Descripción |
|---|---|---|
| **Tiempo** | 27 | IAT, duración, idle/active times, paquetes/s |
| **Volumen** | 32 | Longitudes de paquete, bytes, bulk stats |
| **Banderas** | 7 | TCP flags: SYN, ACK, PSH, RST, FIN, CWR, ECE |
| **Topología** | 3 | Protocol, Src Port, Dst Port |

---

## Arquitectura del modelo

### SingleView (baseline)
Random Forest de 150 árboles entrenado sobre **todas las features** concatenadas (69 columnas).

### MultiViewStacking
Ensamble de dos niveles basado en la librería `multiviewstacking`:

```
Nivel 1 (base learners):
  Vista_Tiempo    → RF-150
  Vista_Volumen   → RF-150
  Vista_Banderas  → RF-150
  Vista_Topologia → RF-150

         ↓  K-Fold (k=3) genera meta-features:
         ↓  avgscores (6 cols) + OHE de predicciones (20 cols)
         ↓  = 26 meta-features

Nivel 2 (meta-learner):
  RF-150 sobre las 26 meta-features
```

**Justificación de RF como meta-learner:** se probaron también LogisticRegression y RF restringido (max_depth=5). LR perdió −0.354% vs SV porque el meta-dataset tiene estructura no lineal — el meta-learner necesita aprender reglas condicionales del tipo *"si Vista_Volumen da alta confianza en clase X pero Vista_Banderas discrepa, confiar en Volumen"*. RF captura esas interacciones; LR no.

### Preprocesamiento (sin fugas de datos)
Por cada run, el `fit` del imputador (mediana) y el escalador (MinMax) se hace **solo sobre train** y se aplica al test:

```python
imputer = SimpleImputer(strategy='median')
X_train_clean = imputer.fit_transform(X_train)
X_test_clean  = imputer.transform(X_test)      # solo transform

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled  = scaler.transform(X_test_clean)  # solo transform
```

---

## Experimentos

Se prueban 5 configuraciones de vistas, todas con RF base + RF meta, para identificar la combinación óptima:

| Experimento | Vistas | Hipótesis |
|---|---|---|
| `4vistas_original` | Tiempo, Volumen, Banderas, Topología | Baseline de 4 vistas separadas |
| `3vistas_volumen+topo` | Tiempo, Volumen+Topo, Banderas | Topología (débil, 3 cols) se funde en Volumen |
| `2vistas_tiempo+volumen+topo` | Tiempo, Volumen+Topo | Eliminar Banderas + fusionar Topología |
| `3vistas_sin_topologia` | Tiempo, Volumen, Banderas | Quitar la vista más débil |
| `2vistas_tiempo+volumen` | Tiempo, Volumen | Mínimo absoluto, solo vistas fuertes |

Cada experimento corre **20 runs** con seeds reproducibles generadas desde `SEED_GENERATOR_SEED = 2026`.

---

## Resultados

### Accuracy media (20 runs) — MultiViewStacking vs SingleView

| Experimento | MultiViewStacking | SingleView | Diff (MV − SV) |
|---|---|---|---|
| `4vistas_original` | **93.164%** | 92.597% | **+0.567%** ✅ |
| `3vistas_volumen+topo` | 92.693% | 92.597% | +0.095% ✅ |
| `2vistas_tiempo+volumen+topo` | 92.448% | 92.597% | −0.150% ❌ |
| `3vistas_sin_topologia` | 90.620% | 92.597% | −1.977% ❌ |
| `2vistas_tiempo+volumen` | 90.391% | 92.597% | −2.206% ❌ |

### F1-macro media (20 runs)

| Experimento | MultiViewStacking | SingleView | Diff |
|---|---|---|---|
| `4vistas_original` | **93.195%** | 92.634% | **+0.561%** ✅ |
| `3vistas_volumen+topo` | 92.723% | 92.634% | +0.089% ✅ |
| `2vistas_tiempo+volumen+topo` | 92.477% | 92.634% | −0.157% ❌ |
| `3vistas_sin_topologia` | 90.670% | 92.634% | −1.964% ❌ |
| `2vistas_tiempo+volumen` | 90.438% | 92.634% | −2.196% ❌ |

### Accuracy de vistas individuales (diagnóstico)

| Vista | Accuracy media |
|---|---|
| Vista_Volumen | 88.13% |
| Vista_Volumen+Topo | 89.63% |
| Vista_Tiempo | 83.78% |
| Vista_Topologia | 75.33% |
| Vista_Banderas | 72.09% |

### Estabilidad (std de accuracy MV, 20 runs)

| Experimento | Std |
|---|---|
| `4vistas_original` | 0.00071 |
| `3vistas_volumen+topo` | 0.00090 |
| `2vistas_tiempo+volumen+topo` | 0.00075 |
| `3vistas_sin_topologia` | 0.00112 |
| `2vistas_tiempo+volumen` | 0.00122 |

`4vistas_original` es el experimento más estable (menor std) además de ser el más preciso.

---

## Análisis e interpretación

### ¿Por qué `4vistas_original` gana con las 4 vistas, incluso con las débiles?

El resultado contraintuitivo — que Banderas (72%) y Topología (75%) *ayuden* en lugar de perjudicar — se explica por la naturaleza del meta-dataset de MultiViewStacking.

Las vistas débiles aún aportan **señal complementaria** para clases específicas:
- **Banderas** discrimina bien DDoS (flood de SYN) y ataques de reconocimiento (RST/FIN anómalos)
- **Topología** detecta Spoofing mediante combinaciones anómalas de puertos

El meta-learner RF aprende a ponderar esas contribuciones condicionalmente: cuando Banderas es informativa (ataques de flood), se apoya en ella; cuando no lo es (tráfico Benign), la ignora via feature importance. Esa capacidad de ponderación contextual es exactamente lo que SingleView no puede hacer al tener todas las features mezcladas.

### ¿Por qué eliminar vistas empeora?

Al reducir de 4 a 2-3 vistas, el meta-dataset pierde columnas de OHE y avgscores. El meta-learner recibe menos señal diferenciada para tomar su decisión de segundo nivel, y el efecto neto es negativo.

---

## Configuración de ejecución

```python
N_RUNS              = 20
SEED_GENERATOR_SEED = 2026
TEST_SIZE           = 0.30      # 70/30 split estratificado
RF_ESTIMATORS       = 150       # árboles por RF
KFOLD               = 3         # K-Fold interno del stacking
```

### Reproducibilidad

Las seeds de cada run se generan deterministamente:
```python
seeds = np.random.RandomState(2026).randint(0, 10000, size=20)
# [2305, 8986, 2125, 6477, 5661, 9651, 7167, 1989, 8140, 28,
#   4534, 9243, 4857, 6919, 2656, 1159, 4719, 916, 4513, 5414]
```

---

## Dependencias

```
scikit-learn
numpy
pandas
xgboost
multiviewstacking
```

---

## Referencia

Arquitectura de stacking basada en:
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
- Breiman, L. (1996). Stacked regressions. *Machine Learning*, 24(1), 49–64.