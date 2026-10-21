# Experimentos de control (revisión 15-oct)

Tres amenazas a la validez del experimento A que un revisor va a señalar, más una duda
sobre qué meta-learner produjo A. Todo corre sobre **las mismas 20 seeds, particiones y
preprocesamiento que A** (código del commit `1363933`, rama `exp/smote-downsampling-50k`),
así que cada número nuevo queda **pareado** con los del paper.

| Modelo | Qué es | Pregunta que responde |
|---|---|---|
| `SV150` | SingleView, 150 árboles (igual que A) | Control de emparejamiento: debe reproducir A **idéntico** |
| `SV600` | SingleView, 600 árboles | **Capacidad**: MVS entrena 4 × 150 = 600 árboles; ¿sigue ganando contra un SV del mismo tamaño? |
| `MVS_LR` | MVS, meta-learner regresión logística | Variante: pierde contra SingleView (0/20 corridas) |
| `MVS_RF` | MVS, meta-learner Random Forest | El modelo del paper: reproduce A (idéntico en 10 de 20 corridas, el resto a <= 0.12 pp) |
| `MVS_LR_RANDVIEWS` | MVS con vistas **aleatorias** del mismo tamaño (27/32/7/3) | **Semántica**: ¿importa la partición semántica o cualquier partición sirve? |
| `MVS_LR_FOLDSMOTE` | MVS con SMOTE **dentro de cada fold interno** | **Fuga de SMOTE**: el meta-learner solo ve flujos reales |

## Cómo correrlo

Los scripts viven en `Code/`. El dataset de A (8 clases, tope de 50,000 flujos por clase) se genera con
`Code/dataset_preparing_expA.py` y se pasa con `--dataset dataset_expA_8clases.csv`.

```bash
pip install multiviewstacking==0.7.5 imbalanced-learn scipy tabulate pytest
python -m pytest Code/test_controls.py -q        # 5 pruebas (~1 min)
python Code/exp_controls.py --runs 1             # prueba rápida con la seed 1 de A
python Code/analyze_controls.py                  # revisar que SV150 salga IDENTICAL
python Code/exp_controls.py                      # las 20 corridas
python Code/analyze_controls.py
```

- **Primero revisar el emparejamiento.** Si `SV150` no sale `IDENTICAL` contra
  `Analysis/comparing/multiview_vs_singleview_per_run.csv`, el pipeline no reproduce A
  (versión de scikit-learn/imbalanced-learn o archivo de datos distinto). No seguir hasta resolverlo.
- **Meta-learner de A.** El análisis dice cuál de `MVS_LR` / `MVS_RF` reproduce la columna
  `MultiViewStacking` de A corrida por corrida. Si es `MVS_RF`, el paper (Sec. 5.5, 6, 7.5,
  Discussion y Conclusiones) describe mal el meta-learner y hay que corregirlo.
- Se puede interrumpir: los resultados se agregan corrida por corrida y el script retoma donde quedó.
  Para correr solo algunos modelos: `--models SV600 MVS_LR_RANDVIEWS`.
- Tiempo estimado: del orden de varias horas para las 20 corridas con los 6 modelos (cada MVS
  entrena 16 bosques por corrida). Conviene la prueba con `--runs 1` para medir.

## Salidas (`Analysis/controls/`)

| Archivo | Columnas |
|---|---|
| `controls_per_run.csv` | run, seed, model, accuracy, f1_macro, fit_seconds, predict_seconds_per_flow |
| `controls_per_class.csv` | run, seed, model, class, precision, recall, f1-score, support |
| `controls_confusion.csv` | run, model, true, pred, count |
| `controls_comparisons.csv` | model, reference, metric, mean_diff_pp, ci95_lo, ci95_hi, d_z, wins, n, min_diff_pp, wilcoxon_p |

Las estadísticas son las mismas del paper: diferencia pareada en pp, IC bootstrap 95 % (5,000
remuestreos), d_z, corridas ganadas y Wilcoxon exacto. Copiar los CSV a `data/` del repo del
paper (son la fuente de verdad) y reportar en la Sec. 7.5. Los tiempos y la matriz de confusión
quedan disponibles si el equipo decide sacarlos de Future Work.

## Detalles de diseño

- `FoldResampledMVS` es una copia del `fit` de `multiviewstacking` 0.7.5 con un solo cambio:
  aplica SMOTE a la parte de entrenamiento de cada fold y antes del re-entrenamiento final.
  Las predicciones out-of-fold (las que entrenan al meta-learner) son solo de flujos reales.
  Con `resampler=None` reproduce la librería **exactamente** (prueba
  `test_no_resampler_reproduces_library_exactly`).
- SMOTE proporcional: un fold con 2/3 del train sube las minoritarias a 2/3 de 30,000; el
  re-entrenamiento final, a 30,000, igual que A.
- Vistas aleatorias: nueva partición por corrida (determinista por seed), con los mismos
  tamaños que las vistas semánticas y las mismas 69 columnas; Subflow Fwd Bytes sigue fuera.
