#!/bin/bash
# Reproduces every table, figure and number in the paper. About 6-8 h on 2 CPU cores.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p results/sens figures data
python3 scripts/make_data.py                   # simulated dataset (seed 42)
python3 scripts/realism.py                     # plausibility checks against published values
python3 scripts/make_static_figures.py         # architecture and routine figures
python3 scripts/run_main.py LORO               # main leave-one-resident-out results (all models)
python3 scripts/run_main.py DAYS               # held-out-days protocol
python3 scripts/run_main.py MASK               # complete loss of a modality at test time
python3 scripts/run_robust.py                  # partial/systematic faults, modality-dropout training
python3 scripts/run_inner.py                   # training-only threshold, hybrid, permutation test
python3 scripts/run_anomaly.py                 # downstream anomaly-detection check
python3 scripts/run_dl.py                      # BiLSTM, BiLSTM-CRF, TCN, Transformer
python3 scripts/run_cost.py                    # latency, memory, size, explanation stability/faithfulness
python3 scripts/run_sensitivity.py             # 18 independently simulated configurations
if [ -f data/aruba/data ]; then python3 scripts/run_aruba.py data/aruba/data; else echo "CASAS Aruba not found at data/aruba/data - real-home check skipped (see README)"; fi
if [ ! -d data/har70plus ]; then git clone --depth 1 --filter=blob:none --sparse https://github.com/ntnu-ai-lab/harth-ml-experiments.git data/_harth && (cd data/_harth && git sparse-checkout set har70plus) && mv data/_harth/har70plus data/har70plus && rm -rf data/_harth; fi
python3 scripts/run_har70.py data/har70plus     # real wearable check on HAR70+ (18 adults aged 70-95)
python3 scripts/make_interp_figure.py          # interpretability figure (Fig. 10)
python3 scripts/analysis.py                    # tables, figures, results/numbers.json
python3 scripts/export_twin.py                 # Digital Twin export
