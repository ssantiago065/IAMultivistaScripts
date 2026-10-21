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
| Eight-class dataset (50,000 flows per class cap) | `Code/dataset_preparing_expA.py` |
| MultiView Stacking vs. SingleView (experiment A) | `Code/multiview_test.py` |
| Control experiments (capacity, random views, SMOTE inside folds) | `Code/exp_controls.py`, `Code/analyze_controls.py`, `Code/test_controls.py` (see `README_CONTROLS.md`) |
| Tables and figures | `paper/scripts/make_figures.py` reads `paper/data/` and writes `paper/figures/` |

`paper/data/` holds the per-run results behind every table and figure of the paper, so they can be
rebuilt without retraining. The meta-learner of the reported MultiView Stacking is a Random Forest of
150 trees; the logistic-regression meta-learner is evaluated as a variant.

Further notes on the analysis scripts are in `README_Analysis_Code.md`.
