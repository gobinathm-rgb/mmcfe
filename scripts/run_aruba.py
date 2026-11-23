"""Real-data check (R1.1, R2.1, R3): the ambient part of the TSS pipeline on the public CASAS Aruba home.

Aruba has one resident, motion/door/temperature sensors and annotated activities; it has no wearable, light,
humidity or plug sensors. Room-level location is derived from the motion sensors (room of the most recent
motion event), i.e. from real PIR data with real blind spots and delays. Because there is a single resident,
leave-one-resident-out is impossible; we use leave-one-period-out over five contiguous blocks of days.
All preprocessing (median imputation, standardisation, room-activity prior) is fitted on training blocks only.
"""
import sys, os, pickle, json, time
import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
import sklearn_crfsuite
from sklearn.ensemble import HistGradientBoostingClassifier
from mmcfe.metrics import acc_f1, bootstrap, paired_delta, ci, ece

SRC = sys.argv[1] if len(sys.argv) > 1 else 'data/aruba/data'
OUT = 'results'

# --------------------------------------------------------------------------- sensor -> room map (from aruba.jpg)
ROOM_OF = {}
for s in ['M001', 'M002', 'M003', 'M005', 'M006', 'M007']: ROOM_OF[s] = 'Bedroom'
for s in ['M004', 'M029', 'M031']: ROOM_OF[s] = 'Bathroom'
ROOM_OF['M024'] = 'Bedroom2'
for s in ['M025', 'M026', 'M027']: ROOM_OF[s] = 'Office'
for s in ['M015', 'M016', 'M017', 'M018', 'M019']: ROOM_OF[s] = 'Kitchen'
ROOM_OF['M014'] = 'Dining'
for s in ['M009', 'M010', 'M012', 'M013', 'M020']: ROOM_OF[s] = 'Living'
for s in ['M008', 'M011', 'M021', 'M022', 'M023', 'M028', 'M030']: ROOM_OF[s] = 'Hall'
ROOMS = ['Bedroom', 'Bathroom', 'Bedroom2', 'Office', 'Kitchen', 'Dining', 'Living', 'Hall']
DOORS = ['D001', 'D002', 'D004']
TEMPS = ['T001', 'T002', 'T003', 'T004', 'T005']
TOD = ['Night', 'Morning', 'Afternoon', 'Evening']

# --------------------------------------------------------------------------- parse
t0 = time.time()
ev = pd.read_csv(SRC, sep=r'\s+', header=None, names=['d', 't', 's', 'v', 'act', 'be'], dtype=str, engine='c', on_bad_lines='skip')
ev = ev[ev.s.str.match(r'^[MDT]\d{3}$', na=False)].copy()
ev['ts'] = pd.to_datetime(ev.d + ' ' + ev.t, format='mixed', errors='coerce')
ev = ev.dropna(subset=['ts']).sort_values('ts', kind='stable').reset_index(drop=True)
day0 = ev.ts.iloc[0].normalize(); ndays = int((ev.ts.iloc[-1].normalize() - day0).days) + 1
ev['m'] = ((ev.ts - day0).dt.total_seconds() // 60).astype(int)
NM = ndays * 1440
print('parsed', len(ev), 'events', ndays, 'days', f'{time.time() - t0:.0f}s', flush=True)

# --------------------------------------------------------------------------- minute labels
MAP = {'Sleeping': 'Sleeping', 'Bed_to_Toilet': 'Bed_to_Toilet', 'Meal_Preparation': 'Meal_Preparation', 'Eating': 'Eating',
       'Wash_Dishes': 'Wash_Dishes', 'Relax': 'Relax', 'Work': 'Work', 'Housekeeping': 'Housekeeping',
       'Enter_Home': 'Away', 'Leave_Home': 'Away', 'Resperate': 'Other'}
lab = np.array(['Other'] * NM, dtype=object)
L = ev[ev.act.notna() & ev.be.isin(['begin', 'end'])][['m', 'act', 'be']].to_numpy()
open_ = {}; intervals = []
for m, a, be in L:
    if be == 'begin':
        open_[a] = m
    elif a in open_:
        intervals.append((open_.pop(a), m, a))
# away = from end of Leave_Home to start of the next Enter_Home (gaps under 24 h)
lv = sorted(e for s, e, a in intervals if a == 'Leave_Home'); en = sorted(s for s, e, a in intervals if a == 'Enter_Home')
j = 0
for e in lv:
    while j < len(en) and en[j] < e:
        j += 1
    if j < len(en) and en[j] - e < 1440:
        lab[e:en[j]] = 'Away'
# long activities first, short ones last, so short activities nested in long ones keep their label
for s, e, a in sorted(intervals, key=lambda x: -(x[1] - x[0])):
    lab[s:e + 1] = MAP.get(a, 'Other')

# --------------------------------------------------------------------------- minute sensor streams
mot = ev[ev.s.str.startswith('M') & (ev.v == 'ON')]
motion_cnt = np.bincount(mot.m, minlength=NM)[:NM].astype(float)
room_seq = np.full(NM, -1)
rm = mot.s.map({s: ROOMS.index(r) for s, r in ROOM_OF.items()})
ok = rm.notna().to_numpy()
room_seq[mot.m.to_numpy()[ok]] = rm.to_numpy()[ok].astype(int)   # last event within the minute wins
room_obs = pd.Series(room_seq).replace(-1, np.nan).ffill().to_numpy()   # room of the most recent motion event
last_m = pd.Series(np.where(motion_cnt > 0, np.arange(NM), np.nan)).ffill().to_numpy()
since = np.minimum(np.arange(NM) - last_m, 240.0); since[np.isnan(since)] = 240.0
dist = np.zeros((NM, len(ROOMS)))
np.add.at(dist, (mot.m.to_numpy()[ok], rm.to_numpy()[ok].astype(int)), 1)
door = {d: np.bincount(ev[(ev.s == d) & ev.v.isin(['OPEN', 'CLOSE'])].m, minlength=NM)[:NM].astype(float) for d in DOORS}
temp = {}
for tsn in TEMPS:
    x = ev[ev.s == tsn]; v = pd.to_numeric(x.v, errors='coerce'); a = np.full(NM, np.nan); a[x.m.to_numpy()] = v.to_numpy()
    temp[tsn] = pd.Series(a).ffill(limit=180).to_numpy()   # hold a reading for up to 3 h

# --------------------------------------------------------------------------- 15-minute windows
NW = 96
def win(a):
    return a.reshape(ndays, NW, 15)
rows = {}
M = win(motion_cnt); rows['motion_total'] = M.sum(-1); rows['motion_minutes'] = (M > 0).sum(-1)
rows['sensors_active'] = np.stack([(win(dist[:, k]) > 0).any(-1) for k in range(len(ROOMS))], -1).sum(-1).astype(float)  # rooms with motion
rows['since_motion_end'] = win(since)[..., -1]
for d in DOORS:
    rows['door_' + d] = win(door[d]).sum(-1)
for tsn in TEMPS:
    T = win(temp[tsn])
    with np.errstate(all='ignore'):
        import warnings; warnings.simplefilter('ignore')
        rows['temp_' + tsn] = np.nanmean(T, -1); rows['dtemp_' + tsn] = T[..., -1] - T[..., 0]
SENSOR_FEATS = list(rows)
Wn = pd.DataFrame({k: v.ravel() for k, v in rows.items()})
Wn['day'] = np.repeat(np.arange(ndays), NW); Wn['w'] = np.tile(np.arange(NW), ndays); Wn['resident_id'] = 'Aruba'
R = win(room_obs)
for k, r in enumerate(ROOMS):
    Wn['loc_' + r] = (R == k).mean(-1).ravel()
Wn['loc_frac'] = (~np.isnan(R)).mean(-1).ravel()
dom = np.array([ROOMS[int(i)] if not np.isnan(i) else 'unknown' for i in
                pd.DataFrame(R.reshape(-1, 15)).mode(axis=1)[0].to_numpy()], dtype=object)
Wn['dom_room'] = dom
LW = lab.reshape(ndays * NW, 15)
lw = pd.DataFrame(LW).mode(axis=1)[0].to_numpy()
Wn['label'] = lw; Wn['purity'] = (LW == lw[:, None]).mean(1)
s = Wn.w * 15 + 7.5
Wn['sin_t'] = np.sin(2 * np.pi * s / 1440); Wn['cos_t'] = np.cos(2 * np.pi * s / 1440)
Wn['tod'] = np.array(TOD)[(s // 360).astype(int).to_numpy()]
for t in TOD:
    Wn['tod_' + t] = (Wn.tod == t).astype(float)
# minute room x activity counts (for the prior)
RA = {}
for k, r in enumerate(ROOMS):
    for a in np.unique(lab):
        RA[f'n|{r}|{a}'] = win(((room_obs == k) & (lab == a)).astype(float)).sum(-1).ravel()
Wn = pd.concat([Wn, pd.DataFrame(RA)], axis=1)

# exclude days without sensor data (gaps in the recording)
day_ev = Wn.groupby('day').motion_total.sum()
good = day_ev[day_ev >= 50].index
Wn = Wn[Wn.day.isin(good)].reset_index(drop=True)
# classes: keep those with at least 100 windows, fold the rest into Other
vc = Wn.label.value_counts(); rare = vc[vc < 100].index.tolist()
Wn.loc[Wn.label.isin(rare), 'label'] = 'Other'
ACTS = sorted(Wn.label.unique()); A2I = {a: i for i, a in enumerate(ACTS)}
print('days kept', Wn.day.nunique(), 'of', ndays, '| windows', len(Wn), '| folded into Other:', rare, flush=True)
print(Wn.label.value_counts().to_dict(), flush=True)
loc_agree = None

TEMPORAL = ['sin_t', 'cos_t'] + ['tod_' + t for t in TOD]
ROOMF = ['loc_' + r for r in ROOMS]
PRIOR = ['prior_' + a for a in ACTS]
NEIGH = ['motion_total', 'since_motion_end'] + ROOMF


class Prep:
    def fit(self, tr):
        X = tr[SENSOR_FEATS]; self.med = X.median(); Xi = X.fillna(self.med); self.mu, self.sd = Xi.mean(), Xi.std().replace(0, 1.0)
        n = np.array([[tr.get(f'n|{r}|{a}', pd.Series(0.0)).sum() for a in ACTS] for r in ROOMS])
        self.M = (n + 1) / (n + 1).sum(1, keepdims=True); self.marg = (n.sum(0) + 1) / (n.sum() + len(ACTS)); return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index); X = df[SENSOR_FEATS]; Z = (X.fillna(self.med) - self.mu) / self.sd
        for c in SENSOR_FEATS:
            out['z_' + c] = Z[c]; out['m_' + c] = X[c].isna().astype(float)
        for c in TEMPORAL + ROOMF + ['dom_room', 'tod', 'day', 'w', 'resident_id']:
            out[c] = df[c].values
        f = df[ROOMF].to_numpy(); num = f @ self.M; tot = num.sum(1, keepdims=True)
        pr = np.where(tot > 0, num / np.where(tot > 0, tot, 1), self.marg)
        for i, a in enumerate(ACTS):
            out['prior_' + a] = pr[:, i]
        return out


S0 = ['z_' + c for c in SENSOR_FEATS] + ['m_' + c for c in SENSOR_FEATS]
FS = {'S0': S0, 'S1': S0 + TEMPORAL, 'S2a': S0 + TEMPORAL + ROOMF, 'S2': S0 + TEMPORAL + ROOMF + PRIOR}


def neigh(P, causal=False):
    g = P.groupby('day', sort=False); c = {}
    for v in NEIGH:
        src = 'z_' + v if 'z_' + v in P else v
        c['dprev_' + v] = (P[src] - g[src].shift(1)).fillna(0.0)
        if not causal:
            c['dnext_' + v] = (g[src].shift(-1) - P[src]).fillna(0.0)
    return pd.DataFrame(c, index=P.index)


def seqs(P, cols, compound):
    X = P[cols].to_numpy(float); E = neigh(P).to_numpy(float) if compound else None; ec = list(neigh(P).columns) if compound else []
    tr = (P.tod + '|' + P.dom_room).to_numpy(); days = P.day.to_numpy(); out, idx, st = [], [], 0
    feats = []
    for i in range(len(P)):
        d = {'bias': 1.0}; d.update(zip(cols, X[i]))
        if compound:
            d.update(zip(ec, E[i])); d['todroom=' + tr[i]] = 1.0
        feats.append(d)
    for i in range(1, len(P) + 1):
        if i == len(P) or days[i] != days[st]:
            out.append(feats[st:i]); idx.append(np.arange(st, i)); st = i
    return out, idx


class CRF:
    def __init__(self, cols, compound=False, notr=False):
        self.cols, self.compound, self.notr = cols, compound, notr

    def _s(self, P):
        s, ix = seqs(P, self.cols, self.compound)
        if self.notr:
            return [q[i:i + 1] for q in s for i in range(len(q))], [np.array([j]) for x in ix for j in x]
        return s, ix

    def fit(self, P, y):
        s, ix = self._s(P); self.m = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=200, all_possible_transitions=True)
        self.m.fit(s, [list(y[i]) for i in ix]); return self

    def predict(self, P):
        s, ix = self._s(P); yp = self.m.predict(s); mg = self.m.predict_marginals(s)
        pred = np.empty(len(P), dtype=object); conf = np.zeros(len(P))
        for q, x in enumerate(ix):
            pred[x] = yp[q]; conf[x] = [mg[q][k][yp[q][k]] for k in range(len(x))]
        return pred, conf


class GB:
    def __init__(self, cols, compound=False):
        self.cols, self.compound = cols, compound
    TR = [t + '|' + r for t in TOD for r in ROOMS + ['unknown']]

    def _X(self, P):
        X = P[self.cols]
        if self.compound:
            tr = (P.tod + '|' + P.dom_room).to_numpy()
            X = pd.concat([X, neigh(P), pd.DataFrame({'tr=' + k: (tr == k).astype(float) for k in self.TR}, index=P.index)], axis=1)
        return X.to_numpy(float)

    def fit(self, P, y):
        self.m = HistGradientBoostingClassifier(random_state=0).fit(self._X(P), y); return self

    def predict(self, P):
        pr = self.m.predict_proba(self._X(P)); return self.m.classes_[pr.argmax(1)], pr.max(1)


MODELS = {'CRF: sensors only': lambda: CRF(FS['S0']), 'CRF: + temporal (S1)': lambda: CRF(FS['S1']),
          'CRF: + room fractions (S2a)': lambda: CRF(FS['S2a']), 'CRF: + temporal + spatial (S1+S2)': lambda: CRF(FS['S2']),
          'MMCFE full (S1+S2+S3 compound)': lambda: CRF(FS['S2'], compound=True),
          'MMCFE full, no transitions': lambda: CRF(FS['S2'], compound=True, notr=True),
          'GB: sensors only': lambda: GB(FS['S0']), 'Gradient boosting (full TSS)': lambda: GB(FS['S2'], compound=True)}

days = np.sort(Wn.day.unique()); blocks = np.array_split(days, 5)
pred = {m: np.empty(len(Wn), dtype=object) for m in MODELS}; conf = {m: np.zeros(len(Wn)) for m in MODELS}
y = Wn.label.to_numpy(dtype=object); fold_of = np.zeros(len(Wn), int)
for k, b in enumerate(blocks):
    te = Wn.day.isin(b).to_numpy(); tr = ~te; fold_of[te] = k
    pp = Prep().fit(Wn[tr]); Ptr = pp.transform(Wn[tr]).reset_index(drop=True); Pte = pp.transform(Wn[te]).reset_index(drop=True)
    for m, mk in MODELS.items():
        t1 = time.time(); mod = mk().fit(Ptr, y[tr]); p, c = mod.predict(Pte); pred[m][te] = p; conf[m][te] = c
        print(f'fold {k + 1}/5 {m:40s} acc={np.mean(p == y[te]):.3f} {time.time() - t1:.0f}s', flush=True)


def conf_by_day(p):
    gi, gu = pd.factorize(Wn.day); C = np.zeros((len(gu), len(ACTS), len(ACTS)))
    np.add.at(C, (gi, [A2I[v] for v in y], [A2I[v] for v in p]), 1); return C


N = {}
def put(k, v, f='{:.3f}'):
    N[k] = f.format(v) if not isinstance(v, str) else v
SH = {'CRF: sensors only': 's0', 'CRF: + temporal (S1)': 's1', 'CRF: + room fractions (S2a)': 's2a', 'CRF: + temporal + spatial (S1+S2)': 's2',
      'MMCFE full (S1+S2+S3 compound)': 'full', 'MMCFE full, no transitions': 'notr', 'GB: sensors only': 'gbs0', 'Gradient boosting (full TSS)': 'gb'}
Cs = {}; blk = fold_of[Wn.groupby('day').head(1).index]
summary = {}
for m in MODELS:
    C = conf_by_day(pred[m]); Cs[m] = C; a, f1, pc = acc_f1(C.sum(0)); bs = bootstrap(C, 1000, 0, hier=blk)
    lo, hi = ci(bs[:, 1]); k = SH[m]
    put(f'ar-{k}-f1', f1); put(f'ar-{k}-f1lo', lo); put(f'ar-{k}-f1hi', hi); put(f'ar-{k}-acc', a)
    e = ece(conf[m], pred[m] == y, 15)[0]; put(f'ar-{k}-ece', e)
    for i, act in enumerate(ACTS):
        put(f'ar-{k}-pc-{act}', pc[i])
    summary[m] = dict(f1=f1, lo=lo, hi=hi, acc=a, ece=e, pc=dict(zip(ACTS, pc)))
    print(f'{m:40s} acc={a:.3f} macroF1={f1:.3f} ({lo:.3f}-{hi:.3f}) ECE={e:.3f}', flush=True)
for a_, b_ in [('s1', 's0'), ('s2', 's1'), ('s2a', 's1'), ('s2', 's2a'), ('full', 's2'), ('full', 'notr'), ('full', 's0'), ('gb', 'full'), ('gb', 'gbs0')]:
    ma = [m for m in SH if SH[m] == a_][0]; mb = [m for m in SH if SH[m] == b_][0]
    d, (lo, hi) = paired_delta(Cs[ma], Cs[mb], 1000, 0, hier=blk)
    put(f'ar-d-{a_}-{b_}', d, '{:+.3f}'); put(f'ar-d-{a_}-{b_}-lo', lo, '{:+.3f}'); put(f'ar-d-{a_}-{b_}-hi', hi, '{:+.3f}')
put('ar-days', Wn.day.nunique(), '{:.0f}'); put('ar-daysall', ndays, '{:.0f}'); put('ar-windows', f'{len(Wn):,}')
put('ar-nclass', len(ACTS), '{:.0f}'); put('ar-other', 100 * (y == 'Other').mean(), '{:.1f}'); put('ar-mixed', 100 * (Wn.purity < 1).mean(), '{:.1f}')
put('ar-nsens', len(SENSOR_FEATS), '{:.0f}'); put('ar-folded', ', '.join(r.replace('_', ' ') for r in rare) if rare else 'none')
for a in ACTS:
    put(f'ar-n-{a}', int((y == a).sum()), '{:,}')
json.dump(N, open(f'{OUT}/aruba_numbers.json', 'w'), indent=0)
pickle.dump(dict(summary=summary, acts=ACTS, pred=pred, conf=conf, y=y, day=Wn.day.to_numpy(), fold=fold_of, rare=rare), open(f'{OUT}/aruba.pkl', 'wb'))
print('DONE', f'{time.time() - t0:.0f}s')
