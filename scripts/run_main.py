"""Main LORO + held-out-days results, modality removal. Saves out-of-fold predictions."""
import sys, pickle; import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import pandas as pd
from mmcfe.experiment import run_protocol, model_zoo
from mmcfe.pipeline import feature_sets
W = pd.read_pickle('data/windows_default.pkl')
ALL = list(model_zoo(feature_sets()).keys())
which = sys.argv[1]
if which == 'LORO':
    W2, out, tim = run_protocol(W, {m: m for m in ALL}, 'LORO')
    pickle.dump((W2, out, tim), open('results/oof_LORO.pkl', 'wb'))
elif which == 'DAYS':
    ms = [m for m in ALL if 'causal' not in m]
    W2, out, tim = run_protocol(W, {m: m for m in ms}, 'DAYS')
    pickle.dump((W2, out, tim), open('results/oof_DAYS.pkl', 'wb'))
elif which == 'MASK':
    res = {}
    for mask in ['wearable', 'location']:
        W2, out, tim = run_protocol(W, {m: m for m in ['MMCFE full (S1+S2+S3 compound)', 'Gradient boosting', 'HMM (supervised, Gaussian)']}, 'LORO', mask=mask, verbose=False)
        res[mask] = out
    pickle.dump((W2, res), open('results/oof_MASK.pkl', 'wb'))
print('DONE', which)
