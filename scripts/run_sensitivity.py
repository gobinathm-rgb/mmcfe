"""Sensitivity analysis (R1.3, R2.2, R3): independent populations, noise, household, localisation, activity frequency, size."""
import sys, pickle, os, time; import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
from mmcfe.simulator import simulate
from mmcfe.features import make_windows
from mmcfe.experiment import run_protocol
from mmcfe.metrics import confusions, acc_f1, bootstrap, ci
CORE = ['CRF: sensors only', 'CRF: + temporal (S1)', 'CRF: + temporal + spatial (S1+S2)', 'MMCFE full (S1+S2+S3 compound)', 'Gradient boosting']
CONFIGS = [(f'seed{s}', dict(SEED=s)) for s in [1, 2, 3, 4, 5]] + \
    [(f'noise{n}_seed{s}', dict(NOISE=n, SEED=s)) for n in [0.5, 1.5, 2.0] for s in [42, 1]] + \
    [('family_none', dict(FORCE_FAMILY=False)), ('family_all', dict(FORCE_FAMILY=True)),
     ('locerr10', dict(LOC_ERR=0.10)), ('locerr20', dict(LOC_ERR=0.20)),
     ('actfreq0.6', dict(ACT_FREQ=0.6)), ('actfreq1.4', dict(ACT_FREQ=1.4)),
     ('residents16', dict(N_RESIDENTS=16))]
only = sys.argv[1:] or [c for c, _ in CONFIGS]
for name, kw in CONFIGS:
    if name not in only or os.path.exists(f'results/sens/{name}.pkl'):
        continue
    t0 = time.time()
    data, meta, res, cfg = simulate(**kw)
    W = make_windows(data); del data
    models = CORE if name != 'residents16' else ['CRF: sensors only', 'CRF: + temporal + spatial (S1+S2)', 'MMCFE full (S1+S2+S3 compound)', 'Gradient boosting']
    W2, out, tim = run_protocol(W, {m: m for m in models}, 'LORO', verbose=False)
    g = (W2.resident_id + '#' + W2.day.astype(str)).to_numpy(); hier = W2.groupby(['resident_id', 'day']).size().reset_index().resident_id.to_numpy()
    summ = {}
    for m in models:
        C, gu = confusions(W2.label, out[m].pred, g)
        a, f, _ = acc_f1(C.sum(0)); bs = bootstrap(C, 1000, hier=[k.split('#')[0] for k in gu])
        perres = {r: acc_f1(C[[k.startswith(r + '#') for k in gu]].sum(0))[1] for r in sorted(W2.resident_id.unique())}
        summ[m] = dict(acc=a, f1=f, f1_ci=ci(bs[:, 1]), per_res=perres)
    info = dict(cfg=cfg, n_windows=len(W2), mixed=float((W2.purity < 1).mean()), n_family=sum(r['lives_with_family'] for r in res),
                day_types=meta.day_type.value_counts().to_dict(), label_dist=W2.label.value_counts(normalize=True).to_dict())
    pickle.dump(dict(summary=summ, info=info, oof={m: out[m][['pred', 'conf']] for m in models}, labels=W2[['resident_id', 'day', 'w', 'label', 'purity']]),
                open(f'results/sens/{name}.pkl', 'wb'))
    print(name, f'{time.time() - t0:.0f}s', {m[:20]: round(summ[m]['f1'], 3) for m in models}, flush=True)
