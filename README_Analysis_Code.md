# IAMultivistaScripts - Analysis y Code

Este README describe el contenido de las carpetas Analysis y Code, limitado a los 4 scripts y 4 salidas de analisis solicitadas.

## Objetivo

Documentar el flujo basico de trabajo para:
1) contar clases,
2) perfilar el dataset,
3) limpiar los CSVs, y
4) preparar un dataset balanceado para pruebas multivista.

## Estructura relevante

### Code (4 scripts)

- Code/class_count.py
  - Cuenta instancias por categoria y subcategoria usando los CSV del dataset flow-based.
  - Entrada por defecto: carpeta del dataset flow-based (configurable con --dataset-dir).
  - Salida: CSVs de resumen en Analysis/class_count.

- Code/data_profiling_analysis.py
  - Analiza calidad de datos por archivo (missing, duplicados, constantes, infinitos, negativos, todo-cero).
  - Entrada por defecto: carpeta del dataset flow-based (configurable con --dataset-dir).
  - Salida: Analysis/data_profiling/profiling_results.json y log en consola.

- Code/data_cleaning_flowbased.py
  - Limpia los CSVs con reglas derivadas del profiling y elimina columnas constantes.
  - Entrada por defecto: dataset flow-based (configurable con --dataset-dir).
  - Salida: dataset limpio en cleaned_dataset_flowbased y reportes en Analysis/clean_dataset.

- Code/dataset_preparing_multiview_test.py
  - Genera dataset balanceado sin fuga de datos y elimina identificadores.
  - Entrada por defecto: cleaned_dataset_flowbased.
  - Salida: dataset_poc_multivista.csv y log en Analysis/dataset_preparing.

## Salidas de analisis (4 carpetas)

- Analysis/class_count
  - category_counts.csv, subcategory_counts.csv, file_details.csv, class_count.txt

- Analysis/data_profiling
  - profiling_results.json, data_profiling.txt

- Analysis/clean_dataset
  - cleaning_report_flowbased.csv, cleaning_summary_by_category.csv, cleaning_progress.json, data_cleaning.txt

- Analysis/dataset_preparing
  - dataset_preparing.txt

## Flujo recomendado (4 pasos)

1) Ejecutar Code/class_count.py para entender la distribucion de clases.
2) Ejecutar Code/data_profiling_analysis.py para detectar problemas de calidad.
3) Ejecutar Code/data_cleaning_flowbased.py para limpiar los CSVs.
4) Ejecutar Code/dataset_preparing_multiview_test.py para crear el dataset balanceado.

## Ejecucion rapida

```bash
python Code/class_count.py
python Code/data_profiling_analysis.py
python Code/data_cleaning_flowbased.py
python Code/dataset_preparing_multiview_test.py
```

Si necesitas cambiar rutas, usa los argumentos --dataset-dir y --output-dir donde apliquen.
