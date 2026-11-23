"""Nested LORO (R2.8 threshold selection, hybrid inputs) and label-permutation test on all folds (R2.10)."""
import sys, pickle, time; import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
from mmcfe.simulator import ACTIVITIES
from mmcfe.pipeline import FoldPreprocessor, feature_sets, CRFModel, SKModel
from mmcfe.features import TEMPORAL_FEATS, ROOM_FEATS, PRIOR_FEATS

W = pd.read_pickle('data/windows_default.pkl').sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
y = W.label.to_numpy(dtype=object)
RES = sorted(W.resident_id.unique())
fs = feature_sets()
GBP = ['gbp_' + a for a in ACTIVITIES]
HYB = GBP + TEMPORAL_FEATS + ROOM_FEATS + PRIOR_FEATS
out = dict(inner_conf={}, inner_correct={}, hybrid=[], perm=[])
for r in RES:
    t0 = time.time()
    T = [q for q in RES if q != r]
    trm = (W.resident_id != r).to_numpy(); tem = ~trm
    Wtr = W[trm].reset_index(drop=True)
    gbp_tr = np.zeros((len(Wtr), 9)); conf_tr = np.zeros(len(Wtr))
    for q in T:                                                # inner leave-one-resident-out on the 7 training residents
        itr = (Wtr.resident_id != q).to_numpy(); ite = ~itr
        pp = FoldPreprocessor().fit(Wtr[itr])
        Pi, Pq = pp.transform(Wtr[itr]).reset_index(drop=True), pp.transform(Wtr[ite]).reset_index(drop=True)
        crf = CRFModel(fs['S2'], compound=True).fit(Pi, Wtr.label.values[itr])
        pq, cq, _ = crf.predict(Pq)
        conf_tr[ite] = cq
        out['inner_correct'][(r, q)] = (pq == Wtr.label.values[ite])
        out['inner_conf'][(r, q)] = cq
        gb = SKModel('gb', fs['S2']).fit(Pi, Wtr.label.values[itr])
        gbp_tr[ite] = gb.predict(Pq)[2]
    pp = FoldPreprocessor().fit(Wtr)
    Ptr, Pte = pp.transform(Wtr).reset_index(drop=True), pp.transform(W[tem]).reset_index(drop=True)
    gb = SKModel('gb', fs['S2']).fit(Ptr, Wtr.label.values)
    gbp_te = gb.predict(Pte)[2]
    for i, a in enumerate(ACTIVITIES):
        Ptr['gbp_' + a] = gbp_tr[:, i]; Pte['gbp_' + a] = gbp_te[:, i]
    hyb = CRFModel(HYB, compound=True).fit(Ptr, Wtr.label.values)
    ph, ch, prh = hyb.predict(Pte)
    out['hybrid'].append(pd.DataFrame({'idx': np.where(tem)[0], 'pred': ph, 'conf': ch}))
    # label-permutation test: shuffle labels within each training resident-day, retrain full model
    rng = np.random.default_rng([2026, RES.index(r)])
    yp = Wtr.label.to_numpy(dtype=object).copy()
    for (_, _), ix in Wtr.groupby(['resident_id', 'day']).indices.items():
        yp[ix] = rng.permutation(yp[ix])
    perm = CRFModel(fs['S2'], compound=True).fit(Ptr, yp)
    pp_, _, _ = perm.predict(Pte)
    out['perm'].append(pd.DataFrame({'idx': np.where(tem)[0], 'pred': pp_}))
    print(r, f'{time.time() - t0:.0f}s', flush=True)
pickle.dump(out, open('results/inner.pkl', 'wb'))
print('DONE')
