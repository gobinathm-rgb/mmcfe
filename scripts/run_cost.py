"""R2.9: latency, memory, model size; explanation stability and faithfulness (CRF weights vs TreeSHAP for GB)."""
import sys, pickle, time, tracemalloc, io; import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd, shap
from collections import defaultdict
from scipy.stats import spearmanr
from mmcfe.pipeline import FoldPreprocessor, feature_sets, CRFModel, SKModel
from mmcfe.metrics import point
from mmcfe.simulator import ACTIVITIES
W = pd.read_pickle('data/windows_default.pkl').sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
fs = feature_sets(); COLS = fs['S2']; RES = sorted(W.resident_id.unique())
cost = defaultdict(list); imp_crf, imp_gb = {}, {}; faith = defaultdict(list)


def size_bytes(obj):
    b = io.BytesIO(); pickle.dump(obj, b); return b.tell()


def crf_importance(crf, cols):
    s = defaultdict(float)
    for (attr, lab), w in crf.state_features_.items():
        base = attr.split('=')[0] if attr.startswith('todroom') else attr
        s[base] += abs(w)
    return pd.Series({c: s.get(c, 0.0) for c in cols})


for k, r in enumerate(RES):
    tr = (W.resident_id != r).to_numpy(); te = ~tr
    pp = FoldPreprocessor().fit(W[tr]); Ptr, Pte = pp.transform(W[tr]).reset_index(drop=True), pp.transform(W[te]).reset_index(drop=True)
    ytr, yte = W.label.values[tr], W.label.values[te]
    crf = CRFModel(COLS, compound=True).fit(Ptr, ytr)
    gb = SKModel('gb', COLS).fit(Ptr, ytr)
    day = Pte[Pte.day == Pte.day.iloc[0]].reset_index(drop=True)
    # latency per resident-day (median of 30 repeats) and peak memory during inference
    for name, m in [('MMCFE', crf), ('Gradient boosting', gb)]:
        ts = []
        for _ in range(30):
            t0 = time.perf_counter(); m.predict(day); ts.append(time.perf_counter() - t0)
        tracemalloc.start(); m.predict(day); peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
        cost[name + '|lat_ms'].append(1000 * np.median(ts)); cost[name + '|peak_kb'].append(peak / 1024)
        cost[name + '|fit_s'].append(m.t_fit)
    cost['MMCFE|size_kb'].append(size_bytes(crf.crf) / 1024)
    cost['MMCFE|nonzero_weights'].append(int(sum(abs(w) > 1e-9 for w in crf.crf.state_features_.values()) + sum(abs(w) > 1e-9 for w in crf.crf.transition_features_.values())))
    cost['MMCFE|total_weights'].append(len(crf.crf.state_features_) + len(crf.crf.transition_features_))
    cost['Gradient boosting|size_kb'].append(size_bytes(gb.m) / 1024)
    cost['Gradient boosting|tree_nodes'].append(int(sum(len(p.nodes) for it in gb.m._predictors for p in it)))
    # explanations: CRF global |weight| per input vs mean |TreeSHAP| per input (500 test windows)
    imp_crf[r] = crf_importance(crf.crf, COLS)
    ex = shap.TreeExplainer(gb.m)
    sub = Pte[COLS].sample(500, random_state=k).to_numpy(float)
    t0 = time.perf_counter(); sv = ex.shap_values(sub); t_shap = (time.perf_counter() - t0) / 500 * 96
    sv = np.abs(np.asarray(sv)); sv = sv.mean(axis=tuple(i for i in range(sv.ndim) if sv.shape[i] != len(COLS)))
    imp_gb[r] = pd.Series(sv, index=COLS)
    cost['Gradient boosting|shap_ms_per_day'].append(1000 * t_shap)
    t0 = time.perf_counter(); _ = crf.crf.predict_marginals(crf._seqs(day)[0]); cost['MMCFE|explain_ms_per_day'].append(1000 * (time.perf_counter() - t0))
    # faithfulness: neutralise the top-k globally important inputs (set to training value 0 after standardisation) and measure the F1 drop
    base_c = point(yte, crf.predict(Pte)[0])[1]; base_g = point(yte, gb.predict(Pte)[0])[1]
    for kk in [3, 5, 10]:
        for name, m, imp, base in [('MMCFE', crf, imp_crf[r], base_c), ('Gradient boosting', gb, imp_gb[r], base_g)]:
            top = list(imp.sort_values(ascending=False).index[:kk]); Pm = Pte.copy()
            for c in top:
                Pm[c] = 0.0 if c.startswith('z_') else Ptr[c].mean()
            faith[(name, kk)].append(base - point(yte, m.predict(Pm)[0])[1])
    print(r, flush=True)


def stability(imp):
    keys = list(imp); rh = [spearmanr(imp[a], imp[b])[0] for i, a in enumerate(keys) for b in keys[i + 1:]]
    top10 = [len(set(imp[a].nlargest(10).index) & set(imp[b].nlargest(10).index)) / 10 for i, a in enumerate(keys) for b in keys[i + 1:]]
    return float(np.mean(rh)), float(np.mean(top10))


out = dict(cost={k: (float(np.mean(v)), float(np.std(v))) for k, v in cost.items()},
           stability={'MMCFE': stability(imp_crf), 'Gradient boosting': stability(imp_gb)},
           faith={f'{a}|top{b}': (float(np.mean(v)), float(np.std(v))) for (a, b), v in faith.items()},
           top_crf=pd.concat(imp_crf, axis=1).mean(1).sort_values(ascending=False).head(12).to_dict(),
           top_gb=pd.concat(imp_gb, axis=1).mean(1).sort_values(ascending=False).head(12).to_dict())
pickle.dump(out, open('results/cost.pkl', 'wb')); print(out); print('DONE')
