# Ronda 3 (plan del 2026-11-03): estado al 2026-11-04

Rama `exp/round3` (desde `paper-ieee-smote-folds`), sin commit. Nada de esto es todavía resultado reportable.

## Código

| Archivo | Qué hace |
|---|---|
| `Code/exp_round3.py` | Pasos 1-7 y 11 del plan: MAIN (= `MVS_RF_FOLDSMOTE`), SV69, RANDV_FS, LOVO_TIME/VOLUME/FLAGS/TOPOLOGY, LGBM, STACK_HET, META_BAL; diversidad, importancia por vista y scores de Brute Force |
| `Code/dataset_preparing_allsubtypes.py` | Paso 12: tope de 50,000 por clase repartido entre todos los subtipos y capturas; guarda `capture` y `t_rank` |
| `Code/exp_allsub.py` | Pasos 12 y 9: SV150 vs MAIN con split aleatorio y con split disjunto por captura (por tiempo en clases con menos de 4 capturas) |
| `Code/run_round3.py` | Corre todo en orden y evita la suspensión por inactividad |
| `Code/test_round3.py` | 7 pruebas (11 con `test_controls.py`, todas pasan) |

Dataset nuevo: `H:\Dataset\dataset_allsub_8clases.csv` (314,944 x 73; 4/61/25/29 capturas de Benign/DDoS/DoS/Mirai).
LightGBM 4.7.0 instalado en Python 3.14.

## Problemas encontrados y corregidos

1. `LOVO_T` se usaba para tiempo y topología: el modelo sin topología nunca corría. Nombres completos y prueba de regresión.
2. Al reanudar se sobrescribía `scores/run_XX.npz`; ahora un archivo por modelo.
3. `per_run.csv` se escribía antes que los datos por clase; ahora al final (marca de terminado).
4. **No determinismo de MAIN**: con la misma seed, las probabilidades de los Random Forest cambian en el último bit
   (2.2e-16) porque scikit-learn suma los árboles en el orden en que terminan los hilos; el meta-learner RF corta
   sobre esos valores y MAIN se mueve 0.1-0.2 pp (corrida 1: 0.90238 paper, 0.90242 y 0.90263 aquí). Corregido con
   `OrderedRandomForest` (`predict_proba` suma en orden fijo; entrenamiento igual), activo en los scripts de la ronda 3
   (`ec.DETERMINISTIC_PROBA = True`); LightGBM con `deterministic=True`. La prueba unitaria pasa; **falta** confirmar
   con datos reales que tres ajustes de MAIN dan probabilidades idénticas (interrumpida).

## Corridas

- 01:30-02:01: lote completo, se detuvo sin error en el log después de 3 corridas (causa no confirmada; probablemente la
  app cerró el árbol de procesos). Esos resultados, de antes de las correcciones, están en `Analysis/round3_attempt1/`
  y deben descartarse.
- Prueba de humo (1 corrida, carpeta temporal, código antes de la corrección 4): todos los modelos terminan.
  Corrida 1, F1-macro: MAIN 0.827, SV69 0.808, RANDV_FS 0.815, LOVO_TIME 0.791, LOVO_VOLUME 0.775, LOVO_FLAGS 0.824,
  LOVO_TOPOLOGY 0.774, LGBM 0.819, STACK_HET 0.834, META_BAL 0.831.
  Dataset con todos los subtipos, split aleatorio: SV150 0.656, MAIN 0.682 (accuracy 0.714 y 0.724). El split
  disjunto no terminó.
- Tiempo medido: unos 11-18 min por corrida de `exp_round3` (STACK_HET es el más lento).

## Pendiente antes de lanzar el lote

1. Confirmar el determinismo de MAIN con datos reales (3 ajustes idénticos).
2. Prueba de humo del protocolo disjunto.
3. Lanzar `python Code/run_round3.py` con `Analysis/round3/` vacío; revisar `Analysis/round3/run_log.txt`.

## Actualización 2026-11-04 (tarde)

- Corrida 12:01-19:39 (lanzada a mano desde una terminal propia): `exp_round3.py` completo, 10 modelos x 20 corridas.
  Split aleatorio del dataset con todos los subtipos: 3 de 20 corridas, detenido a mano.
- 5. **LightGBM divergía** en 5 de 20 seeds (log-loss de entrenamiento ~16, F1-macro 0.29-0.60), y STACK_HET,
  que lo usa como base, en 2. Corregido con `min_child_samples=100, reg_lambda=1.0` (7 seeds probadas: F1
  0.814-0.820, log-loss ~0.25). Las filas viejas de LGBM y STACK_HET se quitaron de `Analysis/round3/` (respaldo
  completo en `Analysis/round3_lgbm_v1/`); se recalculan al relanzar.
- Pendiente: `python Code/run_round3.py` desde una terminal propia (LGBM y STACK_HET, resto del split aleatorio,
  split disjunto), unas 4-5 h. Los procesos lanzados desde la app de Claude se cerraron dos veces sin aviso.

## Actualización 2026-11-05: línea base híbrida

- El lote terminó a las 02:25 (ALL DONE). El análisis está en el plan del wiki y el paper v3 en `D:\tec\multiview`.
- Modelos nuevos en `exp_round3.py`:
  - `HYB_FULL`: las 4 vistas semánticas más una quinta vista con las 70 variables.
  - `HYB_RAND`: el mismo híbrido con vistas aleatorias, como control.
- Pruebas: 13 de `test_round3.py` y `test_controls.py`, todas pasan.
- Prueba de humo, corrida 1: `HYB_FULL` acc 0.9184 y F1-macro 0.8484, con fit de 189 s (MAIN 0.8273, STACK_HET 0.8346).
- Pendiente: las 20 corridas de los dos modelos (unas 2-2.5 h), lanzadas desde una terminal propia.
- `exp_allsub.py` acepta ahora `--models` con los modelos de `exp_round3` (HYB_FULL, HYB_RAND, STACK_HET, LGBM, SV69, META_BAL). Hay una prueba nueva y las 14 pasan.
- Prueba de humo del split disjunto, corrida 1 (F1-macro): HYB_FULL 0.5126, MAIN 0.5152, SV150 0.4957.
