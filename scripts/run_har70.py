"""Real-data check of the wearable and sequence stages on HAR70+ (18 older adults aged 70-95, two accelerometers).

HAR70+ has no time-of-day, room or ambient context, so it tests the semantic stage (linear-chain CRF over wearable
window features) and the leakage-free evaluation protocol on real older adults: leave-one-subject-out, all
preprocessing fitted on training subjects, subject-level bootstrap. Source: github.com/ntnu-ai-lab/harth-ml-experiments
(folder har70plus), also UCI ML Repository, doi 10.24432/C5CW3D.
"""
import os, sys, glob, json, pickle, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
import sklearn_crfsuite
from sklearn.ensemble import HistGradientBoostingClassifier
from mmcfe.metrics import acc_f1, ci, ece

SRC = sys.argv[1] if len(sys.argv) > 1 else 'data/har70plus'
FS, WIN = 50, 5                     # 50 Hz, 5-s windows
LAB = {1: 'Walking', 3: 'Shuffling', 4: 'Walking', 5: 'Walking', 6: 'Standing', 7: 'Sitting', 8: 'Lying'}

rows = []
for f in sorted(glob.glob(f'{SRC}/*.csv')):
    sid = os.path.basename(f)[:-4]
    d = pd.read_csv(f, engine='c')
    n = len(d) // (FS * WIN) * FS * WIN; d = d.iloc[:n]
    X = {c: d[c].to_numpy(float).reshape(-1, FS * WIN) for c in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']}
    y = d['label'].to_numpy().reshape(-1, FS * WIN)
    lab = np.array([np.bincount(r).argmax() for r in y]); pur = np.array([(r == r_m).mean() for r, r_m in zip(y, lab)])
    W = pd.DataFrame({'sid': sid, 'w': np.arange(len(lab)), 'label': [LAB.get(int(v), 'Other') for v in lab], 'purity': pur})
    for s in ['back', 'thigh']:
        ax = np.stack([X[f'{s}_x'], X[f'{s}_y'], X[f'{s}_z']]); mag = np.sqrt((ax ** 2).sum(0))
        W[f'{s}_amag_mean'] = mag.mean(1); W[f'{s}_amag_sd'] = mag.std(1)
        for k, a in zip('xyz', ax):
            W[f'{s}_{k}_mean'] = a.mean(1); W[f'{s}_{k}_sd'] = a.std(1)
        # step-like peaks of the magnitude signal (proxy of the step count used by the simulator's wearable)
        dm = mag - mag.mean(1, keepdims=True)
        W[f'{s}_peaks'] = ((dm[:, 1:-1] > dm[:, :-2]) & (dm[:, 1:-1] > dm[:, 2:]) & (dm[:, 1:-1] > 0.1)).sum(1)
    rows.append(W)
W = pd.concat(rows, ignore_index=True)
W = W[W.label != 'Other'].reset_index(drop=True)
ACTS = sorted(W.label.unique()); A2I = {a: i for i, a in enumerate(ACTS)}
F = {'back': [c for c in W if c.startswith('back_')], 'both': [c for c in W if c.startswith(('back_', 'thigh_'))]}
print(len(W), 'windows,', W.sid.nunique(), 'subjects', W.label.value_counts().to_dict(), flush=True)


def prep(tr, te, cols):
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
    return ((tr[cols] - mu) / sd), ((te[cols] - mu) / sd)


def seq(Z, df, notr=False):
    X = Z.to_numpy(float); cols = list(Z.columns); feats = [dict(zip(cols, r), bias=1.0) for r in X]
    D = np.vstack([np.zeros((1, X.shape[1])), np.diff(X, axis=0)])        # previous-window differences (compound features)
    sid = df.sid.to_numpy(); D[np.r_[True, sid[1:] != sid[:-1]]] = 0
    for i, r in enumerate(D):
        feats[i].update({'d_' + c: v for c, v in zip(cols, r)})
    if notr:
        return [[f] for f in feats], [np.array([i]) for i in range(len(feats))]
    out, idx, st = [], [], 0
    for i in range(1, len(sid) + 1):
        if i == len(sid) or sid[i] != sid[st]:
            out.append(feats[st:i]); idx.append(np.arange(st, i)); st = i
    return out, idx


def crf_fit_predict(Ztr, dtr, Zte, dte, notr=False):
    s, ix = seq(Ztr, dtr, notr); m = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=200, all_possible_transitions=True)
    m.fit(s, [list(dtr.label.to_numpy()[i]) for i in ix])
    s2, ix2 = seq(Zte, dte, notr); yp = m.predict(s2); mg = m.predict_marginals(s2)
    pred = np.empty(len(dte), dtype=object); conf = np.zeros(len(dte))
    for q, x in enumerate(ix2):
        pred[x] = yp[q]; conf[x] = [mg[q][k][yp[q][k]] for k in range(len(x))]
    return pred, conf


def gb_fit_predict(Ztr, dtr, Zte, dte):
    m = HistGradientBoostingClassifier(random_state=0).fit(Ztr.to_numpy(float), dtr.label.to_numpy())
    p = m.predict_proba(Zte.to_numpy(float)); return m.classes_[p.argmax(1)], p.max(1)


MODELS = {'crf_back': ('back', 'crf'), 'notr_back': ('back', 'notr'), 'gb_back': ('back', 'gb'),
          'crf_both': ('both', 'crf'), 'notr_both': ('both', 'notr'), 'gb_both': ('both', 'gb')}
pred = {m: np.empty(len(W), dtype=object) for m in MODELS}; conf = {m: np.zeros(len(W)) for m in MODELS}
t0 = time.time()
for s in sorted(W.sid.unique()):
    te = (W.sid == s).to_numpy(); dtr, dte = W[~te].reset_index(drop=True), W[te].reset_index(drop=True)
    for m, (fs, kind) in MODELS.items():
        Ztr, Zte = prep(dtr, dte, F[fs])
        p, c = gb_fit_predict(Ztr, dtr, Zte, dte) if kind == 'gb' else crf_fit_predict(Ztr, dtr, Zte, dte, notr=(kind == 'notr'))
        pred[m][te] = p; conf[m][te] = c
    print(s, f'{time.time() - t0:.0f}s', flush=True)

y = W.label.to_numpy(object)
gi, gu = pd.factorize(W.sid)
def C(p):
    M = np.zeros((len(gu), len(ACTS), len(ACTS))); np.add.at(M, (gi, [A2I[v] for v in y], [A2I[v] for v in p]), 1); return M
def boot(M, B=2000, seed=0):
    r = np.random.default_rng(seed); return np.array([acc_f1(M[r.integers(0, len(M), len(M))].sum(0))[1] for _ in range(B)])
def pdelta(Ma, Mb, B=2000, seed=0):
    r = np.random.default_rng(seed); d = []
    for _ in range(B):
        s = r.integers(0, len(Ma), len(Ma)); d.append(acc_f1(Ma[s].sum(0))[1] - acc_f1(Mb[s].sum(0))[1])
    return acc_f1(Ma.sum(0))[1] - acc_f1(Mb.sum(0))[1], ci(np.array(d))

N = {}
def put(k, v, f='{:.3f}'):
    N[k] = f.format(v) if not isinstance(v, str) else v
Ms = {m: C(pred[m]) for m in MODELS}
for m in MODELS:
    a, f1, pc = acc_f1(Ms[m].sum(0)); lo, hi = ci(boot(Ms[m]))
    per = [acc_f1(Ms[m][i])[1] for i in range(len(gu))]
    put(f'h7-{m}-f1', f1); put(f'h7-{m}-f1lo', lo); put(f'h7-{m}-f1hi', hi); put(f'h7-{m}-acc', a)
    put(f'h7-{m}-ece', ece(conf[m], pred[m] == y, 15)[0]); put(f'h7-{m}-resmin', min(per)); put(f'h7-{m}-resmax', max(per)); put(f'h7-{m}-ressd', np.std(per, ddof=1))
    for i, act in enumerate(ACTS):
        put(f'h7-{m}-pc-{act}', pc[i])
    print(f'{m:10s} acc={a:.3f} macroF1={f1:.3f} ({lo:.3f}-{hi:.3f}) ECE={N[f"h7-{m}-ece"]} per-subject {min(per):.3f}-{max(per):.3f}', flush=True)
for a_, b_ in [('crf_both', 'gb_both'), ('crf_back', 'gb_back'), ('crf_both', 'notr_both'), ('crf_back', 'notr_back'), ('crf_both', 'crf_back')]:
    d, (lo, hi) = pdelta(Ms[a_], Ms[b_]); put(f'h7-d-{a_}-{b_}', d, '{:+.3f}'); put(f'h7-d-{a_}-{b_}-lo', lo, '{:+.3f}'); put(f'h7-d-{a_}-{b_}-hi', hi, '{:+.3f}')
    print(a_, b_, N[f'h7-d-{a_}-{b_}'], lo, hi)
put('h7-n', W.sid.nunique(), '{:.0f}'); put('h7-windows', f'{len(W):,}'); put('h7-nclass', len(ACTS), '{:.0f}'); put('h7-mixed', 100 * (W.purity < 1).mean(), '{:.1f}')
put('h7-minutes', len(W) * WIN / 60 / W.sid.nunique(), '{:.0f}')
for a in ACTS:
    put(f'h7-n-{a}', int((y == a).sum()), '{:,}')
json.dump(N, open('results/har70_numbers.json', 'w'), indent=0)
pickle.dump(dict(pred=pred, conf=conf, y=y, sid=W.sid.to_numpy(), acts=ACTS), open('results/har70.pkl', 'wb'))
print('DONE')
