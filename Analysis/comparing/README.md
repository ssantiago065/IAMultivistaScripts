# Ablación de Meta-Learner en MultiView Stacking

Experimento de ablación para determinar la configuración óptima del meta-learner y los base learners en un ensamble MultiViewStacking aplicado a clasificación de tráfico de red. Se comparan 5 variantes contra un baseline SingleView (Random Forest sobre todas las features), usando las mismas 4 vistas semánticas en todos los experimentos.

---

## Motivación

En experimentos previos con la configuración base (RF+RF, 4 vistas, 100k muestras/clase), MultiViewStacking perdía −0.18% contra SingleView. La hipótesis inicial era que el RF de 150 árboles como meta-learner sobreajustaba los 26 meta-features del segundo nivel. Este experimento prueba sistemáticamente 5 combinaciones de base/meta learners para identificar la causa raíz y la solución correcta.

---

## Estructura del experimento

### Dataset
- **Clases:** Benign, DDoS, DoS, Mirai, Recon, Spoofing
- **Muestras:** 20,000 por clase (subset de exploración)
- **Split:** 70% train / 30% test, estratificado
- **Runs:** 20 con seeds fijas desde `SEED_GENERATOR_SEED = 2026`

### Vistas (fijas en todos los experimentos)

| Vista | Features | Accuracy individual |
|---|---|---|
| Tiempo | 27 | 84.55% |
| Volumen | 32 | 88.44% |
| Banderas | 7 | 73.30% |
| Topología | 3 | 74.85% |

### Meta-dataset de MultiViewStacking

Con 4 vistas y 6 clases, el meta-learner recibe 26 columnas:
- **6 avgscores:** promedio de probabilidades de predicción entre las 4 vistas
- **20 OHE:** predicciones de cada vista codificadas (4 vistas × 5 columnas con `drop='first'`)

---

## Experimentos

| Experimento | Base learners | Meta-learner | Descripción |
|---|---|---|---|
| `4v_baseline_RF_RF` | RF-150 × 4 | RF-150 | Configuración estándar, referencia principal |
| `4v_meta_LR` | RF-150 × 4 | LogisticRegression (C=1, multinomial) | Meta-learner lineal regularizado |
| `4v_diverse_base_meta_LR` | RF, RF, ExtraTrees, ExtraTrees | LogisticRegression | Diversidad en bases + meta lineal |
| `4v_xgb_base_meta_LR` | XGBoost-100 × 4 | LogisticRegression | Bases boosting + meta lineal |
| `4v_meta_RF_restricted` | RF-150 × 4 | RF-30 (max_depth=5) | Meta RF restringido para reducir sobreajuste |

---

## Resultados

### Accuracy media (20 runs)

| Experimento | MultiViewStacking | SingleView | Diff (MV − SV) | MV > SV |
|---|---|---|---|---|
| `4v_baseline_RF_RF` | **91.685%** | 91.646% | **+0.039%** | 12 / 20 runs ✅ |
| `4v_meta_LR` | 91.291% | 91.646% | −0.354% | 0 / 20 runs ❌ |
| `4v_diverse_base_meta_LR` | 91.117% | 91.646% | −0.529% | 0 / 20 runs ❌ |
| `4v_xgb_base_meta_LR` | 90.980% | 91.646% | −0.666% | 0 / 20 runs ❌ |
| `4v_meta_RF_restricted` | 90.341% | 91.646% | −1.305% | 0 / 20 runs ❌ |

### F1-macro media (20 runs)

| Experimento | MultiViewStacking | SingleView | Diff |
|---|---|---|---|
| `4v_baseline_RF_RF` | **91.722%** | 91.691% | **+0.031%** ✅ |
| `4v_meta_LR` | 91.333% | 91.691% | −0.358% ❌ |
| `4v_diverse_base_meta_LR` | 91.161% | 91.691% | −0.530% ❌ |
| `4v_xgb_base_meta_LR` | 91.027% | 91.691% | −0.664% ❌ |
| `4v_meta_RF_restricted` | 90.434% | 91.691% | −1.257% ❌ |

### Significancia estadística (t-test pareado, df=19, umbral |t|>2.09)

| Experimento | t-statistic | Conclusión |
|---|---|---|
| `4v_baseline_RF_RF` | +0.869 | No significativo — diferencia marginal |
| `4v_meta_LR` | −9.166 | **Significativamente peor** que SV |
| `4v_diverse_base_meta_LR` | −12.840 | **Significativamente peor** que SV |
| `4v_xgb_base_meta_LR` | −20.496 | **Significativamente peor** que SV |
| `4v_meta_RF_restricted` | −25.097 | **Significativamente peor** que SV |

### Estabilidad (std de accuracy)

| Experimento | Std MV | Std SV |
|---|---|---|
| `4v_baseline_RF_RF` | 0.00258 | 0.00220 |
| `4v_meta_LR` | 0.00251 | 0.00220 |
| `4v_diverse_base_meta_LR` | 0.00245 | 0.00220 |
| `4v_xgb_base_meta_LR` | 0.00204 | 0.00220 ← única más estable |
| `4v_meta_RF_restricted` | 0.00273 | 0.00220 |

---

## Por qué se eligió `4v_baseline_RF_RF`

### La hipótesis del sobreajuste era incorrecta

La intuición de que RF-150 sobreajustaba el meta-dataset resultó falsa. El meta-dataset tiene **estructura no lineal**: la señal útil está en interacciones condicionales entre vistas, no en combinaciones lineales. Por ejemplo:

> *"Si Vista_Volumen predice Spoofing con alta confianza Y Vista_Banderas discrepa → confiar en Volumen"*

Esa regla es una condición `if/then` que RF captura mediante splits. LogisticRegression solo puede hacer combinación lineal de los 26 meta-features y no puede aprender esa regla. Por eso **LR pierde en los 20/20 runs con t=−9.166**, mientras que RF gana en 12/20.

### Jerarquía de degradación

El ranking de peor a mejor confirma la hipótesis:

```
RF_restricted (max_depth=5)  →  −1.305%   ← underfitting severo en meta-nivel
XGBoost + LR                 →  −0.666%   ← bases más fuertes no compensan meta débil
Diverse + LR                 →  −0.529%   ← diversidad no ayuda con meta lineal
RF + LR                      →  −0.354%   ← meta lineal es el cuello de botella
RF + RF-150                  →  +0.039%   ← único que supera SV ✅
```

Cuanto más se restringe el meta-learner (de RF-150 → RF-small → LR), peor es el resultado. Esto confirma que el meta-nivel necesita capacidad no lineal para integrar las señales de las 4 vistas.

### Por qué la ventaja es marginal (+0.039%) con 20k muestras

Con 20k muestras/clase el efecto de diversidad de vistas es pequeño porque los base learners de cada vista tienen menos datos para especializarse. El experimento posterior con 100k muestras/clase confirmó la decisión: la misma configuración `4v_baseline_RF_RF` logró **+0.567%** sobre SingleView, amplificando la ventaja al escalar el dataset.

---

## Configuración de ejecución

```python
N_RUNS              = 20
SEED_GENERATOR_SEED = 2026
TEST_SIZE           = 0.30
RF_ESTIMATORS       = 150
KFOLD               = 3
```

Seeds reproducibles:
```python
seeds = np.random.RandomState(2026).randint(0, 10000, size=20)
# [2305, 8986, 2125, 6477, 5661, 9651, 7167, 1989, 8140, 28,
#   4534, 9243, 4857, 6919, 2656, 1159, 4719, 916, 4513, 5414]
```

---

## Conclusión

**RF base + RF meta es la configuración correcta** para MultiViewStacking en este dominio. El meta-dataset de stacking tiene estructura no lineal inherente — el meta-learner debe ser capaz de modelar interacciones condicionales entre las predicciones de las vistas. Cualquier meta-learner más simple (lineal o con capacidad restringida) degrada el rendimiento de forma estadísticamente significativa.

La configuración elegida para experimentos posteriores es por lo tanto:
- Base learners: RF-150, uno por vista
- Meta-learner: RF-150
- Vistas: las 4 originales separadas (Tiempo, Volumen, Banderas, Topología)

---

## Limitaciones identificadas en `multi_view_stacking.py`

Durante el análisis del código fuente de la librería se identificaron dos problemas estructurales que explican parte de la brecha residual entre MultiViewStacking y SingleView, incluso con la configuración RF+RF óptima. Estos problemas no bloquean los experimentos actuales pero representan trabajo futuro para mejorar el ensamble.

### Bug 1 — `avgscores`: pérdida de señal por promedio simple

**Dónde:** método `fit()` al construir el meta-dataset.

**Problema:** las probabilidades de las 4 vistas se colapsan en un promedio simple antes de pasarlas al meta-learner:

```python
# Implementación actual (dentro de fit)
avgscores += scores[vista]
avgscores /= num_vistas   # → 6 columnas, una por clase
```

Esto produce un meta-dataset de **26 columnas** (6 avgscores + 20 OHE de predicciones). El problema es que Vista_Banderas (73% accuracy) y Vista_Volumen (88% accuracy) contribuyen con **el mismo peso** al promedio. El meta-learner RF recibe una señal ya degradada y no puede recuperar quién dijo qué — la información de origen se perdió en el promedio.

**Solución propuesta:** pasar las probabilidades de cada vista **por separado** en lugar del promedio:

```python
# Propuesto: concatenar probas individuales
meta_features = pd.concat([scores[v] for v in range(num_vistas)], axis=1)
# → 24 columnas (6 por vista × 4 vistas) en lugar de 6
```

Con 24 columnas de probas individuales + 20 OHE = **44 meta-features**, el RF del meta-nivel puede aprender feature importance diferenciada: asignar peso casi cero a las columnas de Banderas para clases donde esa vista es poco informativa, y peso alto a las de Volumen. Eso es exactamente la capacidad de ponderación contextual que MultiViewStacking necesita para superar consistentemente a SingleView.

**Impacto estimado:** moderado a alto. Este es el cambio arquitectónico más prometedor pendiente.

---

### Bug 2 — `fit_transform` en inferencia: OHE inconsistente

**Dónde:** método `__predict_labels_scores()`.

**Problema:** al predecir, el código recalcula el encoder OHE en cada llamada:

```python
# Bug: fit_transform recalcula el encoder en cada predicción
encoded = enc_.fit_transform(predictions)
```

Debería ser:

```python
# Correcto: usar el encoder ya ajustado durante el fit
encoded = enc_.transform(predictions)
```

**Consecuencia:** si durante la inferencia el test set no produce predicciones de todas las clases vistas en train (situación frecuente en clases minoritarias como Spoofing), `fit_transform` reordena silenciosamente las categorías OHE. El meta-learner recibe columnas con significado distinto al que aprendió durante el entrenamiento, produciendo **predicciones incorrectas sin ningún error visible**.

Este bug es especialmente crítico en producción o con datasets desbalanceados. En los experimentos actuales su impacto es parcialmente amortiguado por el uso de `stratify=y` en el split, que garantiza que todas las clases aparezcan en test, pero no elimina el riesgo completamente.

**Fix:** reemplazar `fit_transform` por `transform` en `__predict_labels_scores`. Es un cambio de una sola palabra.

---

### Resumen de cambios propuestos en `multi_view_stacking.py`

| # | Método | Cambio | Líneas afectadas |
|---|---|---|---|
| 1 | `fit()` | Reemplazar `avgscores` acumulado por `pd.concat` de probas individuales | ~3 líneas |
| 2 | `__predict_labels_scores()` | Reemplazar `enc_.fit_transform()` por `enc_.transform()` | 1 línea |

Ambos cambios son quirúrgicos y no alteran la interfaz pública de la clase (`fit`, `predict`). La corrección del Bug 2 es inmediata y sin riesgo; el Bug 1 requiere verificar que el tamaño del meta-dataset sea consistente entre `fit` y `predict`.

---

## Dependencias

```
scikit-learn
numpy
pandas
xgboost
multiviewstacking
```