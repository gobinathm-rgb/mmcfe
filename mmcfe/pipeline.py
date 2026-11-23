"""Fold-wise preprocessing, TSS feature construction, CRF and baseline models."""
import time
import numpy as np
import pandas as pd
import sklearn_crfsuite
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from .simulator import ACTIVITIES, ROOMS
from .features import (SENSOR_FEATS, WEAR_FEATS, ROOM_FEATS, TEMPORAL_FEATS, PRIOR_FEATS, TOD, NEIGH_VARS,
                       leakage_audit)

A2I = {a: i for i, a in enumerate(ACTIVITIES)}
COUNT_COLS = [f'n|{r}|{a}' for r in ROOMS for a in ACTIVITIES]


# ----------------------------------------------------------------------------- preprocessing
class FoldPreprocessor:
    """Everything fitted here uses training windows only (median, mean/SD, room-activity matrix)."""

    def __init__(self, sensor_feats=SENSOR_FEATS):
        self.sf = list(sensor_feats)

    def fit(self, tr):
        X = tr[self.sf]
        self.median = X.median()
        Xi = X.fillna(self.median)
        self.mu, self.sd = Xi.mean(), Xi.std().replace(0, 1.0)
        n = tr[COUNT_COLS].sum().to_numpy().reshape(len(ROOMS), len(ACTIVITIES))
        self.M = (n + 1) / (n + 1).sum(1, keepdims=True)                    # Eq. (4), add-one smoothing
        self.marg = (n.sum(0) + 1) / (n.sum() + len(ACTIVITIES))           # fallback when no room is observed
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        X = df[self.sf]
        miss = X.isna()
        Z = (X.fillna(self.median) - self.mu) / self.sd
        for c in self.sf:
            out['z_' + c] = Z[c]
            out['m_' + c] = miss[c].astype(float)
        for c in TEMPORAL_FEATS:
            out[c] = df[c]
        f = df[ROOM_FEATS].to_numpy()
        for c in ROOM_FEATS:
            out[c] = df[c]
        num = f @ self.M
        tot = num.sum(1, keepdims=True)
        prior = np.where(tot > 0, num / np.where(tot > 0, tot, 1), self.marg)   # Eq. (5)
        for i, a in enumerate(ACTIVITIES):
            out['prior_' + a] = prior[:, i]
        out['dom_room'] = df['dom_room'].values
        out['tod'] = df['tod'].values
        out['resident_id'] = df['resident_id'].values
        out['day'] = df['day'].values
        out['w'] = df['w'].values
        return out


def feature_sets(sf=SENSOR_FEATS):
    s0 = ['z_' + c for c in sf] + ['m_' + c for c in sf]
    s1 = s0 + TEMPORAL_FEATS
    s1_room = s1 + ROOM_FEATS                  # Stage 2a: observed-room fractions only
    s1_prior = s1 + PRIOR_FEATS                # Stage 2b: learned prior only
    s2 = s1 + ROOM_FEATS + PRIOR_FEATS
    return dict(S0=s0, S1=s1, S2a=s1_room, S2b=s1_prior, S2=s2)


def add_neighbour_diffs(P, causal=False):
    """Previous/next-window differences within each resident-day (Stage-3 compound features)."""
    cols = {}
    g = P.groupby(['resident_id', 'day'], sort=False)
    for v in NEIGH_VARS:
        src = 'z_' + v if 'z_' + v in P else v
        x = P[src]
        cols['dprev_' + v] = (x - g[src].shift(1)).fillna(0.0)
        if not causal:
            cols['dnext_' + v] = (g[src].shift(-1) - x).fillna(0.0)
    return pd.DataFrame(cols, index=P.index)


# ----------------------------------------------------------------------------- CRF
def _seq_dicts(P, cols, compound=False, causal=False):
    X = P[cols].to_numpy(dtype=float)
    extra = add_neighbour_diffs(P, causal) if compound else None
    E = extra.to_numpy(dtype=float) if compound else None
    ecols = list(extra.columns) if compound else []
    trs = (P['tod'] + '|' + P['dom_room']).to_numpy() if compound else None
    feats = []
    for i in range(len(P)):
        d = {'bias': 1.0}
        d.update(zip(cols, X[i]))
        if compound:
            d.update(zip(ecols, E[i]))
            d['todroom=' + trs[i]] = 1.0
        feats.append(d)
    # split into resident-day sequences (P is ordered by resident, day, w)
    keys = (P['resident_id'].astype(str) + '#' + P['day'].astype(str)).to_numpy()
    seqs, idx, start = [], [], 0
    for i in range(1, len(P) + 1):
        if i == len(P) or keys[i] != keys[start]:
            seqs.append(feats[start:i]); idx.append(np.arange(start, i)); start = i
    return seqs, idx


def make_crf(max_iter=200):
    return sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=max_iter,
                                all_possible_transitions=True)


class CRFModel:
    def __init__(self, cols, compound=False, causal=False, no_transitions=False):
        self.cols, self.compound, self.causal, self.notr = cols, compound, causal, no_transitions

    def _seqs(self, P):
        seqs, idx = _seq_dicts(P, self.cols, self.compound, self.causal)
        if self.notr:   # every window its own length-1 sequence: no transition features can be learned
            flat = [s[i:i + 1] for s in seqs for i in range(len(s))]
            fidx = [np.array([j]) for ix in idx for j in ix]
            return flat, fidx
        return seqs, idx

    def fit(self, P, y):
        leakage_audit(self.cols)
        seqs, idx = self._seqs(P)
        ys = [list(y[ix]) for ix in idx]
        self.crf = make_crf()
        t = time.perf_counter(); self.crf.fit(seqs, ys); self.t_fit = time.perf_counter() - t
        return self

    def predict(self, P, online=False):
        seqs, idx = self._seqs(P)
        pred = np.empty(len(P), dtype=object); conf = np.zeros(len(P)); prob = np.zeros((len(P), len(ACTIVITIES)))
        t = time.perf_counter()
        if not online:
            yp = self.crf.predict(seqs)
            mg = self.crf.predict_marginals(seqs)
            for s, ix in enumerate(idx):
                pred[ix] = yp[s]
                for k, j in enumerate(ix):
                    prob[j] = [mg[s][k].get(a, 0.0) for a in ACTIVITIES]
                    conf[j] = mg[s][k].get(yp[s][k], 0.0)
        else:   # causal: label window t using only windows 1..t (forward-filtered marginal at t)
            tagger = self.crf.tagger_
            for s, ix in enumerate(idx):
                for k, j in enumerate(ix):
                    pre = seqs[s][:k + 1]
                    tagger.set([{kk: float(vv) for kk, vv in d.items()} for d in pre])
                    p = np.array([tagger.marginal(a, k) for a in ACTIVITIES])
                    prob[j] = p; pred[j] = ACTIVITIES[int(p.argmax())]; conf[j] = p.max()
        self.t_inf_per_seq = (time.perf_counter() - t) / max(1, len(seqs))
        return pred, conf, prob


# ----------------------------------------------------------------------------- baselines
def _ctx(Xdf, P, k=2):
    """Concatenate features of windows t-k..t+k within each resident-day (zero padded)."""
    X = Xdf.to_numpy(dtype=float)
    g = P.groupby(['resident_id', 'day'], sort=False).ngroup().to_numpy()
    blocks = [X]
    for o in [o for o in range(-k, k + 1) if o != 0]:
        sh = np.roll(X, -o, axis=0)
        gs = np.roll(g, -o)
        valid = (gs == g)
        if o > 0:
            valid[-o:] = False
        else:
            valid[:-o] = False
        sh[~valid] = 0.0
        blocks.append(sh)
    return np.hstack(blocks)


TODROOM = [t + '|' + r for t in TOD for r in ROOMS + ['unknown']]


def compound_matrix(P, causal=False):
    D = add_neighbour_diffs(P, causal)
    tr = (P['tod'] + '|' + P['dom_room']).to_numpy()
    OH = pd.DataFrame({'todroom=' + k: (tr == k).astype(float) for k in TODROOM}, index=P.index)
    return pd.concat([D, OH], axis=1)


class SKModel:
    def __init__(self, name, cols, seed=0, compound=False):
        self.name, self.cols, self.seed, self.compound = name, cols, seed, compound

    def _X(self, P):
        X = P[self.cols]
        if self.compound:
            X = pd.concat([X, compound_matrix(P)], axis=1)
        return _ctx(X, P) if self.name == 'mlp' else X.to_numpy(dtype=float)

    def fit(self, P, y):
        leakage_audit(self.cols)
        if self.name == 'gb':
            self.m = HistGradientBoostingClassifier(random_state=self.seed)
        elif self.name == 'rf':
            self.m = RandomForestClassifier(n_estimators=300, random_state=self.seed, n_jobs=1)
        elif self.name == 'mlp':
            self.m = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, random_state=self.seed, max_iter=300)
        t = time.perf_counter(); self.m.fit(self._X(P), y); self.t_fit = time.perf_counter() - t
        return self

    def predict(self, P):
        X = self._X(P)
        t = time.perf_counter()
        pr = self.m.predict_proba(X)
        self.t_inf_total = time.perf_counter() - t
        prob = np.zeros((len(P), len(ACTIVITIES)))
        for j, c in enumerate(self.m.classes_):
            prob[:, A2I[c]] = pr[:, j]
        pred = np.array(ACTIVITIES, dtype=object)[prob.argmax(1)]
        return pred, prob.max(1), prob


class MajorityModel:
    def fit(self, P, y):
        self.c = pd.Series(y).mode()[0]; self.t_fit = 0; return self

    def predict(self, P):
        n = len(P); prob = np.zeros((n, 9)); prob[:, A2I[self.c]] = 1
        return np.array([self.c] * n, dtype=object), np.ones(n), prob


class RoomHourModel:
    def fit(self, P, y):
        d = pd.DataFrame({'r': P['dom_room'].values, 'h': (P['w'].values * 15) // 60, 'y': y})
        self.rh = d.groupby(['r', 'h']).y.agg(lambda s: s.mode()[0]).to_dict()
        self.h = d.groupby('h').y.agg(lambda s: s.mode()[0]).to_dict()
        self.c = d.y.mode()[0]; self.t_fit = 0
        return self

    def predict(self, P):
        r, h = P['dom_room'].values, (P['w'].values * 15) // 60
        pred = np.array([self.rh.get((a, b), self.h.get(b, self.c)) for a, b in zip(r, h)], dtype=object)
        prob = np.zeros((len(P), 9)); prob[np.arange(len(P)), [A2I[p] for p in pred]] = 1
        return pred, np.ones(len(P)), prob


class HMMModel:
    """Supervised HMM: Gaussian naive-Bayes emissions, counted transitions (add-one), Viterbi per day."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, P, y):
        self.nb = GaussianNB().fit(P[self.cols].to_numpy(dtype=float), y)
        yi = np.array([A2I[v] for v in y])
        key = (P['resident_id'].astype(str) + '#' + P['day'].astype(str)).to_numpy()
        T = np.ones((9, 9)); pi = np.ones(9)
        for i in range(len(yi)):
            if i == 0 or key[i] != key[i - 1]:
                pi[yi[i]] += 1
            else:
                T[yi[i - 1], yi[i]] += 1
        self.logT = np.log(T / T.sum(1, keepdims=True)); self.logpi = np.log(pi / pi.sum())
        cls = [A2I[c] for c in self.nb.classes_]
        self.cls = cls; self.t_fit = 0
        return self

    def predict(self, P):
        lp = self.nb.predict_log_proba(P[self.cols].to_numpy(dtype=float)) - np.log(self.nb.class_prior_)
        E = np.full((len(P), 9), -1e9); E[:, self.cls] = lp
        key = (P['resident_id'].astype(str) + '#' + P['day'].astype(str)).to_numpy()
        pred = np.empty(len(P), dtype=object)
        starts = np.r_[0, np.where(key[1:] != key[:-1])[0] + 1, len(P)]
        for s0, s1 in zip(starts[:-1], starts[1:]):
            e = E[s0:s1]; n = len(e)
            dp = np.zeros((n, 9)); bp = np.zeros((n, 9), int)
            dp[0] = self.logpi + e[0]
            for t in range(1, n):
                sc = dp[t - 1][:, None] + self.logT
                bp[t] = sc.argmax(0); dp[t] = sc.max(0) + e[t]
            path = np.zeros(n, int); path[-1] = dp[-1].argmax()
            for t in range(n - 1, 0, -1):
                path[t - 1] = bp[t, path[t]]
            pred[s0:s1] = np.array(ACTIVITIES, dtype=object)[path]
        prob = np.zeros((len(P), 9)); prob[np.arange(len(P)), [A2I[p] for p in pred]] = 1
        return pred, np.ones(len(P)), prob
