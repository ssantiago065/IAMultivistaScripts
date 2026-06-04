# Experimento: Undersampling 100k Puro (6 Clases)

## 📋 Descripción del Enfoque

Este experimento establece una línea base de rendimiento aislando el ruido del desbalanceo extremo. Se eliminaron las clases ultra-minoritarias para garantizar que el modelo evaluara únicamente datos con representación estadística idéntica, permitiendo analizar la capacidad intrínseca de la arquitectura Multivista frente a la dependencia conjunta de variables.

- **Técnica de Balanceo:** \* _Undersampling estricto:_ Se limitaron las 6 clases principales (Benign, DDoS, DoS, Mirai, Recon, Spoofing) a exactamente **100,000 muestras** cada una.
  - _Exclusión:_ Las clases `BruteForce` y `Web-Based` fueron retiradas del dataset.
  - _Dataset Final:_ 600,000 filas perfectamente balanceadas.

## 🏗️ Arquitectura de los Modelos

- **SingleView:** RandomForestClassifier (n_estimators=100, n_jobs=-1)
- **MultiViewStacking:**
  - _Base Learners (4 vistas):_ RandomForestClassifier (n_estimators=100, n_jobs=-1)
  - _Meta Learner:_ LogisticRegression(max_iter=1000)

## 📊 Resultados Principales (Media de 20 Corridas)

- **Accuracy SingleView:** 0.88676 🏆
- **Accuracy MultiView:** 0.8850

### Desempeño de Vistas Individuales (Diagnóstico)

- Vista Volumen: 0.8732
- Vista Tiempo: 0.7635
- Vista Topología: 0.6976
- Vista Banderas: 0.8319

## 🧠 Conclusión Científica

**SingleView superó a MultiView bajo condiciones de balanceo perfecto.** Al tener 100,000 muestras por clase sin datos sintéticos, el modelo de vista única logró explotar la **correlación cruzada** (ej. evaluar simultáneamente el comportamiento del _Tiempo_ y el _Volumen_). La arquitectura MultiViewStacking degradó levemente el rendimiento global (0.8850 frente a 0.88676) debido a que la separación estricta de las vistas rompe las dependencias matemáticas conjuntas que definen las diferencias sutiles entre ataques similares (como DoS y DDoS). Adicionalmente, se diagnosticó que la _Vista de Topología_ (0.6976) es el componente más débil del ensamble en este dataset.
