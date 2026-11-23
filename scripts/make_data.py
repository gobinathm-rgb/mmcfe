"""Generate the main simulated dataset (seed 42): minute-level CSVs and the 15-minute window table."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
from mmcfe.simulator import simulate, save_csv
from mmcfe.features import make_windows
os.makedirs('data', exist_ok=True); os.makedirs('results', exist_ok=True)
data, meta, residents, cfg = simulate()
save_csv(data, meta, residents, cfg, 'data/simulated_seed42')
make_windows(data).to_pickle('data/windows_default.pkl'); meta.to_csv('data/day_meta_default.csv', index=False)
print('data written to data/')
