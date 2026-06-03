# Experimento: SMOTE Híbrido con Downsampling a 50k

## 📋 Descripción del Enfoque

Este experimento busca evaluar el rendimiento de la arquitectura MultiView frente a SingleView aislando el ruido de las clases mayoritarias e introduciendo datos sintéticos para las clases minoritarias.

- **Técnica de Balanceo:** \* _Downsampling:_ Las clases dominantes (DDoS, DoS, Benign, etc.) se limitaron a un máximo de **50,000 muestras**.
  - _Oversampling (SMOTE):_ Las clases minoritarias (BruteForce, Web-Based) se amplificaron artificialmente hasta un límite de **30,000 muestras** (solo en el set de entrenamiento para evitar fuga de datos).
- **Dataset Final:** ~314,000 filas.

## 🏗️ Arquitectura de los Modelos

- **SingleView:** RandomForestClassifier (n_estimators=100, n_jobs=-1)
- **MultiViewStacking:**
  - _Base Learners (4 vistas):_ RandomForestClassifier (n_estimators=100, n_jobs=-1)
  - _Meta Learner:_ LogisticRegression(max_iter=1000)

## 📊 Resultados Principales (Media de 20 Corridas)

- **Accuracy SingleView:** ~0.8899
- **Accuracy MultiView:** ~0.8962 🏆

### Desempeño de Vistas Individuales (Diagnóstico)

- Vista Volumen: ~0.82
- Vista Tiempo: ~0.78
- Vista Topología: ~0.70
- Vista Banderas: ~0.68

## 🧠 Conclusión Científica

**El MultiView demostró ser superior en este escenario.** Al reducir el volumen de las clases dominantes y añadir ruido dimensional mediante SMOTE para las minoritarias, el modelo SingleView sufrió una leve degradación por la maldición de la dimensionalidad (procesar 71 variables sintéticas simultáneas). En contraste, las vistas aisladas del MultiView asimilaron el ruido de forma más robusta, permitiendo que el meta-learner (Regresión Logística) generara una predicción más precisa combinando las inferencias especializadas.
