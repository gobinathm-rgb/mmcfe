# MMCFE: Context-Aware Multimodal Activity Recognition for Older Adults

Code and data for the paper

> G. Murugan and S. Ayyasamy, "Context-Aware Multimodal Activity Recognition for Older Adults: A Leakage-Free Evaluation on Simulated Indian Households," *IEEE Access* (under review), 2026.

This repository contains:

- `mmcfe/simulator.py`: the parameterised simulator of Indian senior households (eight residents, 30 days, seed 42 in the main study). Every parameter is listed in `DEFAULT_CONFIG` and `PARAMS`.
- `mmcfe/features.py`, `mmcfe/pipeline.py`: 15-minute windowing, fold-wise preprocessing, the Temporal-Spatial-Semantic (TSS) stages, the CRF model and all baselines.
- `mmcfe/metrics.py`: macro-F1, the two-stage (resident, then day) hierarchical bootstrap, paired differences, calibration error and Brier score.
- `scripts/`: one script per experiment in the paper (see `reproduce.sh`).
- `results/tables/`: every table in the paper as CSV; `results/numbers.json` holds every number quoted in the text.
- `figures/`: every figure in the paper.
- `twin/`: the Digital Twin export, built only from out-of-fold (leave-one-resident-out) predictions: `digital_twin_windows.csv.gz` (one row per 15-minute window) and `digital_twin_days.json` (one record per resident-day).

## Installation

Python 3.11 or later. The paper's results were produced with Python 3.13, scikit-learn 1.9.1, sklearn-crfsuite 0.5.0, PyTorch 2.15 and SHAP 0.54.1 on one CPU core.

```bash
pip install -r requirements.txt
```

## Reproducing the paper

```bash
bash reproduce.sh
```

This regenerates the dataset from seed 42, runs every experiment and rebuilds all tables, figures and `results/numbers.json`. It takes about 6-8 hours on two CPU cores. Each step can also be run on its own, for example:

```bash
python3 scripts/make_data.py          # dataset only (CSV files in data/simulated_seed42/)
python3 scripts/run_main.py LORO      # main leave-one-resident-out results
python3 scripts/analysis.py           # tables and figures from saved results
```

### Leakage safeguards

- Every fitted component (median imputation, standardisation, the room-activity matrix M, all models) is fitted on training residents only, inside each fold.
- `features.leakage_audit` refuses any feature set that contains the label, the label purity or the true room.
- `scripts/run_inner.py` repeats a label-permutation test on every fold.

## Real-home check (CASAS Aruba)

The CASAS Aruba dataset may not be redistributed, so it is not included here. Download `new_labeled_data.zip` from the official CASAS release on Zenodo (https://zenodo.org/records/17180309), extract `aruba.txt` to `data/aruba/data`, and run

```bash
python3 scripts/run_aruba.py data/aruba/data
```

Please cite the CASAS project when you use these data:
D. J. Cook, A. S. Crandall, B. L. Thomas and N. C. Krishnan, "CASAS: A smart home in a box," *Computer*, 46(7):62-69, 2013.

## Real wearable check (HAR70+)

HAR70+ (18 older adults aged 70-95, two accelerometers) is downloaded automatically by `reproduce.sh` from the dataset authors' repository (https://github.com/ntnu-ai-lab/harth-ml-experiments, folder `har70plus`; also UCI ML Repository, doi:10.24432/C5CW3D). To run it alone:

```bash
python3 scripts/run_har70.py data/har70plus
```

Please cite: A. Ustad et al., "Validation of an activity type recognition model classifying daily physical behavior in older adults: The HAR70+ model," *Sensors*, 23(5):2368, 2023.

## Licence

Code: MIT (see `LICENSE`). Simulated data and the Digital Twin export: CC BY 4.0.

## Citation

If you use this code, please cite the paper above and the archived release (DOI on the Zenodo badge of this repository).
