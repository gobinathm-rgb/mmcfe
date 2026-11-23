"""Leave-one-resident-out and held-out-days protocols."""
import time
import numpy as np
import pandas as pd
from .features import SENSOR_FEATS, WEAR_FEATS, ROOM_FEATS, PRIOR_FEATS
from .pipeline import (FoldPreprocessor, feature_sets, CRFModel, SKModel, MajorityModel, RoomHourModel, HMMModel)
from .simulator import ACTIVITIES


def model_zoo(fs):
    """name -> factory(). Baselines use the Stage-2 feature set (sensor+temporal+location+prior)."""
    return {
        'CRF: sensors only': lambda: CRFModel(fs['S0']),
        'CRF: + temporal (S1)': lambda: CRFModel(fs['S1']),
        'CRF: + temporal + room fractions (S1+S2a)': lambda: CRFModel(fs['S2a']),
        'CRF: + temporal + prior only (S1+S2b)': lambda: CRFModel(fs['S2b']),
        'CRF: + temporal + spatial (S1+S2)': lambda: CRFModel(fs['S2']),
        'MMCFE full (S1+S2+S3 compound)': lambda: CRFModel(fs['S2'], compound=True),
        'MMCFE full, no transitions': lambda: CRFModel(fs['S2'], compound=True, no_transitions=True),
        'MMCFE causal (online)': lambda: CRFModel(fs['S2'], compound=True, causal=True),
        'Majority class': lambda: MajorityModel(),
        'Room × hour lookup': lambda: RoomHourModel(),
        'HMM (supervised, Gaussian)': lambda: HMMModel(fs['S2']),
        'Random Forest': lambda: SKModel('rf', fs['S2']),
        'Gradient boosting': lambda: SKModel('gb', fs['S2']),
        'MLP with ±2-window context': lambda: SKModel('mlp', fs['S2']),
        'GB: sensors only': lambda: SKModel('gb', fs['S0']),
        'GB: + temporal (S1)': lambda: SKModel('gb', fs['S1']),
        'GB: full TSS (S1+S2+S3 compound)': lambda: SKModel('gb', fs['S2'], compound=True),
    }


def mask_modality(P, which, prep):
    """Simulate total failure of a modality at test time (training unchanged)."""
    P = P.copy()
    if which == 'wearable':
        for c in WEAR_FEATS:
            if c == 'wear_frac':
                P['z_' + c] = (0.0 - prep.mu[c]) / prep.sd[c]; P['m_' + c] = 0.0
            else:
                P['z_' + c] = (prep.median[c] - prep.mu[c]) / prep.sd[c]; P['m_' + c] = 1.0
    elif which == 'location':
        for c in ROOM_FEATS:
            P[c] = 0.0
        for i, a in enumerate(ACTIVITIES):
            P['prior_' + a] = prep.marg[i]
        P['dom_room'] = 'unknown'
    return P


def run_protocol(W, models, protocol='LORO', mask=None, online_models=(), verbose=True, sensor_feats=SENSOR_FEATS):
    """Returns dict model -> DataFrame(pred, conf, prob_*) aligned with W, plus timing."""
    W = W.sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
    y = W['label'].to_numpy(dtype=object)
    if protocol == 'LORO':
        folds = [(W.resident_id != r, W.resident_id == r) for r in sorted(W.resident_id.unique())]
    else:
        folds = [(W.day <= 19, W.day >= 20)]
    out = {m: pd.DataFrame({'pred': np.empty(len(W), dtype=object), 'conf': np.nan}, index=W.index) for m in models}
    probs = {m: np.zeros((len(W), 9)) for m in models}
    timing = {m: {'t_fit': [], 't_inf_per_day': []} for m in models}
    for k, (tr, te) in enumerate(folds):
        tr, te = tr.to_numpy(), te.to_numpy()
        prep = FoldPreprocessor(sensor_feats).fit(W[tr])
        Ptr, Pte = prep.transform(W[tr]).reset_index(drop=True), prep.transform(W[te]).reset_index(drop=True)
        if mask:
            Pte = mask_modality(Pte, mask, prep)
        fs = feature_sets(sensor_feats)
        zoo = model_zoo(fs)
        for m in models:
            t0 = time.perf_counter()
            mdl = (models[m] if callable(models[m]) else zoo[m])()
            mdl.fit(Ptr, y[tr])
            if isinstance(mdl, CRFModel):
                pred, conf, pr = mdl.predict(Pte, online=(m in online_models or mdl.causal))
                timing[m]['t_inf_per_day'].append(mdl.t_inf_per_seq)
            else:
                pred, conf, pr = mdl.predict(Pte)
            timing[m]['t_fit'].append(getattr(mdl, 't_fit', np.nan))
            idx = np.where(te)[0]
            out[m].loc[idx, 'pred'] = pred; out[m].loc[idx, 'conf'] = conf; probs[m][idx] = pr
            if verbose:
                acc = (pred == y[te]).mean()
                print(f'  fold {k + 1}/{len(folds)} {m:<45} acc={acc:.3f} fit={time.perf_counter() - t0:.1f}s', flush=True)
    for m in models:
        for i, a in enumerate(ACTIVITIES):
            out[m]['p_' + a] = probs[m][:, i]
    return W, out, timing
