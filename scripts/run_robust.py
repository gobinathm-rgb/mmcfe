"""Robustness to missing / faulty sensors (R1.4, R2.7, R3): modality-dropout training, partial outages, systematic errors."""
import sys, pickle, time; import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
from mmcfe.simulator import simulate, ADJACENT
from mmcfe.features import make_windows, SENSOR_FEATS, WEAR_FEATS, ROOM_FEATS
from mmcfe.pipeline import FoldPreprocessor, feature_sets, CRFModel, SKModel

data, meta, res, cfg = simulate()
data = data.sort_values(['resident_id', 'day', 'minute']).reset_index(drop=True)
WEAR_RAW = ['heart_rate', 'accel_x', 'accel_y', 'accel_z', 'step_count']


def corrupt(kind):
    d = data.copy(); rng = np.random.default_rng(123)
    nd = len(d) // 1440
    if kind == 'clean':
        pass
    elif kind == 'no_wearable':
        d[WEAR_RAW] = np.nan
    elif kind == 'no_location':
        d['room_observed'] = np.nan
    elif kind.startswith('loc_outage_'):          # contiguous outages covering ~p of each day
        p = float(kind.split('_')[-1]); m = np.zeros(len(d), bool)
        for k in range(nd):
            need = int(p * 1440); got = 0; day = np.zeros(1440, bool)
            while got < need:
                L = int(rng.integers(30, 180)); s = int(rng.integers(0, 1440 - L)); day[s:s + L] = True; got = day.sum()
            m[k * 1440:(k + 1) * 1440] = day
        d.loc[m, 'room_observed'] = np.nan
    elif kind == 'sys_living_as_bedroom':       # mis-placed sensor: half of Living Room minutes reported as Bedroom
        m = (d.room_true == 'Living Room') & (rng.random(len(d)) < 0.5)
        d.loc[m, 'room_observed'] = 'Bedroom'
    elif kind == 'sys_kitchen_dead':            # dead PIR battery in the kitchen
        d.loc[d.room_observed == 'Kitchen', 'room_observed'] = np.nan
    elif kind == 'sys_bathroom_blind':          # blind spot: bathroom presence attributed to the adjacent bedroom
        d.loc[d.room_observed == 'Bathroom', 'room_observed'] = 'Bedroom'
    return make_windows(d).sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)


KINDS = ['clean', 'no_wearable', 'no_location', 'loc_outage_0.1', 'loc_outage_0.25', 'loc_outage_0.5',
         'sys_living_as_bedroom', 'sys_kitchen_dead', 'sys_bathroom_blind']
WT = {k: corrupt(k) for k in KINDS}
W = WT['clean']
y = W.label.to_numpy(dtype=object)
SF_DO = SENSOR_FEATS + ['loc_frac']            # availability flag for the localisation modality


def augment(Wtr, rng):
    """Training-time modality dropout: a copy of every training day with modalities removed at random."""
    A = Wtr.copy(); A['day'] = A['day'] + 1000
    days = A[['resident_id', 'day']].drop_duplicates().to_numpy()
    for r, dday in days:
        m = (A.resident_id == r) & (A.day == dday)
        u = rng.random()
        if u < 0.25:                                    # whole-day location loss
            A.loc[m, ROOM_FEATS] = 0.0; A.loc[m, 'loc_frac'] = 0.0; A.loc[m, 'dom_room'] = 'unknown'
        elif u < 0.5:                                   # partial location outage (contiguous windows)
            idx = np.where(m)[0]; L = int(rng.integers(12, 48)); s = int(rng.integers(0, 96 - L)); sel = A.index[idx[s:s + L]]
            A.loc[sel, ROOM_FEATS] = 0.0; A.loc[sel, 'loc_frac'] = 0.0; A.loc[sel, 'dom_room'] = 'unknown'
        if rng.random() < 0.3:                          # whole-day wearable loss
            A.loc[m, [c for c in WEAR_FEATS if c != 'wear_frac']] = np.nan; A.loc[m, 'wear_frac'] = 0.0
    return A


res_pred = {}
for k_, r in enumerate(sorted(W.resident_id.unique())):
    t0 = time.time()
    tr = (W.resident_id != r).to_numpy(); te = (W.resident_id == r).to_numpy()
    rng = np.random.default_rng([7, k_])
    Wtr = W[tr].reset_index(drop=True)
    Waug = pd.concat([Wtr, augment(Wtr, rng)], ignore_index=True).sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
    prep = FoldPreprocessor().fit(Wtr); prep_do = FoldPreprocessor(SF_DO).fit(Wtr)
    fs, fsd = feature_sets(), feature_sets(SF_DO)
    Ptr, Pdo = prep.transform(Wtr).reset_index(drop=True), prep_do.transform(Waug).reset_index(drop=True)
    models = {
        'MMCFE': (CRFModel(fs['S2'], compound=True).fit(Ptr, Wtr.label.values), prep),
        'MMCFE + modality dropout': (CRFModel(fsd['S2'], compound=True).fit(Pdo, Waug.label.values), prep_do),
        'Gradient boosting': (SKModel('gb', fs['S2']).fit(Ptr, Wtr.label.values), prep),
        'Gradient boosting + modality dropout': (SKModel('gb', fsd['S2']).fit(Pdo, Waug.label.values), prep_do),
    }
    for kind in KINDS:
        Wte = WT[kind][WT[kind].resident_id == r].reset_index(drop=True)
        for name, (mdl, pp) in models.items():
            pred, conf, pr = mdl.predict(pp.transform(Wte).reset_index(drop=True))
            res_pred.setdefault((kind, name), []).append(pd.DataFrame({'resident_id': r, 'day': Wte.day, 'w': Wte.w, 'pred': pred, 'conf': conf,
                                                                       'loc_frac': Wte.loc_frac, 'label': Wte.label}))
    print(r, f'{time.time() - t0:.0f}s', flush=True)
res_pred = {k: pd.concat(v, ignore_index=True) for k, v in res_pred.items()}
pickle.dump(dict(pred=res_pred, kinds=KINDS, loc_true_match={k: float((WT[k].loc_frac).mean()) for k in KINDS}), open('results/robust.pkl', 'wb'))
print('DONE')
