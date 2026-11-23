"""Downstream validation of the confidence-gated Digital Twin (R2.10): injected behavioural anomalies."""
import sys, pickle, time; import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
from mmcfe.simulator import simulate, ANOMALY_TYPES
from mmcfe.features import make_windows
from mmcfe.pipeline import FoldPreprocessor, feature_sets, CRFModel, SKModel

rng = np.random.default_rng(77); AN = {}
for ri in range(8):
    days = rng.choice(np.arange(30, 50), 10, replace=False)
    for k, d in enumerate(sorted(days)):
        AN[(ri, int(d))] = ANOMALY_TYPES[(k + ri) % 4]
data, meta, res, cfg = simulate(N_DAYS=50, ANOMALIES=AN)
W = make_windows(data).sort_values(['resident_id', 'day', 'w']).reset_index(drop=True); del data
W0 = pd.read_pickle('data/windows_default.pkl').sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
same = np.allclose(W[W.day < 30][['hr_mean', 'light_mean']].fillna(-1).to_numpy(), W0[['hr_mean', 'light_mean']].fillna(-1).to_numpy())
print('days 0-29 identical to main dataset:', same, flush=True)
fs = feature_sets(); preds = {}
for r in sorted(W.resident_id.unique()):
    t0 = time.time()
    tr = ((W.resident_id != r) & (W.day < 30)).to_numpy(); te = (W.resident_id == r).to_numpy()
    pp = FoldPreprocessor().fit(W[tr]); Ptr, Pte = pp.transform(W[tr]).reset_index(drop=True), pp.transform(W[te]).reset_index(drop=True)
    for name, mdl in [('MMCFE', CRFModel(fs['S2'], compound=True)), ('Gradient boosting', SKModel('gb', fs['S2']))]:
        mdl.fit(Ptr, W.label.values[tr]); p, c, _ = mdl.predict(Pte)
        preds.setdefault(name, []).append(pd.DataFrame({'resident_id': r, 'day': W.day.values[te], 'w': W.w.values[te], 'pred': p, 'conf': c}))
    print(r, f'{time.time() - t0:.0f}s', flush=True)
preds = {k: pd.concat(v, ignore_index=True) for k, v in preds.items()}
pickle.dump(dict(W=W[['resident_id', 'day', 'w', 'label', 'purity']], meta=meta, preds=preds, anomalies=AN), open('results/anomaly.pkl', 'wb'))
print('DONE')
