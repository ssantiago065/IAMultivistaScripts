# IAMultivistaScripts

Code and results for the paper "MultiView Stacking: A Semantic Feature-View Ensemble for IoT Network
Intrusion Detection" (CIC IoT-DIAD 2024, flow-based features).

## Data

The raw CIC IoT-DIAD 2024 files are not included. Download the flow-based (anomaly detection) features
from https://www.unb.ca/cic/datasets/iot-diad-2024.html and place them in `FlowBased/`.

## Reproducing the experiments

| Step | Script |
|---|---|
| Class counts, profiling, cleaning | `Code/class_count.py`, `Code/data_profiling_analysis.py`, `Code/data_cleaning_flowbased.py` |
| Eight-class dataset (50,000 flows per class cap) | `Code/dataset_preparing_expA.py` (writes `dataset_expA_8clases.csv`; `Code/multiview_test.py` reads it as `dataset_poc_multivista.csv`, so rename it, or pass `--dataset` to `Code/exp_controls.py`) |
| MultiView Stacking vs. SingleView, SMOTE before the stacking folds (experiment A) | `Code/multiview_test.py` |
| Control experiments: capacity, random views, SMOTE inside the folds, logistic meta-learner | `Code/exp_controls.py`, `Code/analyze_controls.py`, `Code/test_controls.py` (see `README_CONTROLS.md`) |
| **Main results of the paper**: fold-level SMOTE vs. SingleView, corrected resampled t-test, per-class, natural-prevalence and confusion analysis | `paper/scripts/main_results.py` (reads `paper/data/controls_*.csv`, writes `paper/data/main_results_comparisons.csv`) |
| Tables and figures | `paper/scripts/make_figures.py` reads `paper/data/` and writes `paper/figures/`; `--ieee` draws the IEEE two-column sizes into `paper/figures/ieee/` |
| Accuracy and F1-Macro at natural class prevalence (original protocol) | `paper/scripts/prevalence_accuracy.py` |

`paper/data/` holds the per-run results behind every table and figure of the paper, so they can be
rebuilt without retraining.

## Protocol notes (checked against the code and the result files)

- **Main configuration of the paper.** MultiView Stacking with four Random Forest base learners and a
  Random Forest meta-learner (150 trees each, scikit-learn defaults otherwise), with SMOTE applied
  inside each internal stacking fold: model `MVS_RF_FOLDSMOTE` of `Code/exp_controls.py`. The baseline
  is `SV150`. Experiment A (`MVS_RF`, SMOTE applied to the whole training partition before the folds)
  is reported as the original protocol; the capacity (`SV600`) and random-view (`MVS_RF_RANDVIEWS`)
  controls were run under that protocol. The logistic meta-learner (`MVS_LR*`) is a variant.
- **Pairing.** All files in `paper/data/controls_*.csv` share experiment A's 20 partitions and seeds:
  `SV150` reproduces A's SingleView bit for bit in 19 of 20 runs, and `MVS_RF` reproduces A's
  MultiView Stacking in 10 of 20 runs (the rest within 0.12 pp).
- **SMOTE target.** Web-Based and Brute Force are raised to 30,000 training flows; the other classes are
  left as they are. This is the rule of commit `1363933`, which produced experiment A, and of
  `Code/exp_controls.py`. `Code/multiview_test.py` was restored to it; an intermediate version raised
  every class to the size of the largest one (35,000), which does not reproduce the reported results.
- **What the 50,000-flow cap contains.** `Code/dataset_preparing_expA.py` reads each class folder in
  path order and stops after 75,000 cleaned flows (or after the first 100,000-row chunk of a file),
  then draws 50,000 at random from what it read. Reconstructed from the per-file counts in
  `Analysis/clean_dataset/data_cleaning.txt`, the abundant classes therefore come from: DoS, the first
  100,000 flows of one SYN Flood capture; DDoS, the first 100,000 flows of one ACK Fragmentation
  capture; Benign, the first 100,000 flows of one of its four captures; Recon, the first 100,000 flows
  of its single capture; Spoofing, all 82,301 ARP spoofing flows (no DNS spoofing); Mirai, 14 of its 29
  captures. Web-Based and Brute Force keep every cleaned flow and subtype. The paper states this.
- **Logs.** `Analysis/dataset_preparing/dataset_preparing.txt` is the log of the earlier 100,000-flow
  cap (614,944 flows), not of the 50,000-flow dataset used in the paper.
- **Timing columns.** `fit_seconds` and `predict_seconds_per_flow` in `controls_per_run.csv` were
  measured while two copies of the control run were accidentally executing in parallel; they are not
  reported in the paper and should not be used as cost estimates.
- **Root-level scripts** (`multiview_test.py`, `data_cleaning_flowbased.py`, ...) are earlier versions
  kept for history; the paper uses the ones in `Code/` and `paper/scripts/`.

Further notes on the analysis scripts are in `README_Analysis_Code.md`.
