# Reporte comparativo: `comparing` vs `comparing2`

## Objetivo
Comparar dos corridas del experimento multivista usando solo `RandomForest`, ejecutando el experimento 20 veces y cambiando el `seed` de `train_test_split` en cada corrida. El reporte resume promedio, desviación estándar, mínimo y máximo de las métricas principales.

## Configuración de los experimentos

### `comparing`
- Muestras por clase: 7,500
- Corridas: 20
- Modelo: `RandomForest`
- Semillas de split: 20 valores aleatorios distintos
- Salidas:
  - [multiview_comparing_per_run.csv](comparing/multiview_comparing_per_run.csv)
  - [multiview_comparing_summary.csv](comparing/multiview_comparing_summary.csv)

### `comparing2`
- Muestras por clase: 15,000
- Corridas: 20
- Modelo: `RandomForest`
- Semillas de split: 20 valores aleatorios distintos
- Salidas:
  - [multiview_comparing_per_run.csv](comparing2/multiview_comparing_per_run.csv)
  - [multiview_comparing_summary.csv](comparing2/multiview_comparing_summary.csv)

## Resultados

| Experimento | Runs | Accuracy media | Accuracy std | Accuracy min | Accuracy max | F1 macro media | F1 macro std | F1 weighted media | F1 weighted std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `comparing` | 20 | 0.840422 | 0.003213 | 0.833056 | 0.847778 | 0.840868 | 0.003144 | 0.840868 | 0.003144 |
| `comparing2` | 20 | 0.875922 | 0.001291 | 0.873083 | 0.877833 | 0.876427 | 0.001265 | 0.876427 | 0.001265 |

## Comparación directa

- Mejora absoluta en accuracy media: `+0.035500`
- Mejora relativa en accuracy media: `+4.22%`
- Reducción de la desviación estándar de accuracy: `-0.001922`
- Reducción relativa de la desviación estándar de accuracy: `-59.82%`
- Mejora absoluta en F1 macro medio: `+0.035559`
- Mejora absoluta en F1 weighted medio: `+0.035559`

## Conclusión
La configuración `comparing2` con 15,000 muestras por clase fue mejor que `comparing` con 7,500 muestras por clase. No solo incrementó el rendimiento medio, también redujo la variabilidad entre corridas, lo que sugiere un comportamiento más estable al cambiar el `seed` del `train_test_split`.

## Observación metodológica
Ambos experimentos usan el mismo enfoque de evaluación: 20 corridas con semillas distintas para el split, balance por clase y un único modelo `RandomForest`. Por eso la diferencia observada se atribuye principalmente al tamaño de muestra por clase.
