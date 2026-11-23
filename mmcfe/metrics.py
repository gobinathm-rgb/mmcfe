"""Metrics and cluster / hierarchical bootstrap."""
import numpy as np
import pandas as pd
from .simulator import ACTIVITIES

A2I = {a: i for i, a in enumerate(ACTIVITIES)}


def confusions(y, p, groups):
    """Per-group 9x9 confusion matrices. groups: array of group ids (e.g. resident-day)."""
    gi, gu = pd.factorize(groups)
    yi = np.array([A2I[v] for v in y]); pi = np.array([A2I[v] for v in p])
    C = np.zeros((len(gu), 9, 9))
    np.add.at(C, (gi, yi, pi), 1)
    return C, gu


def acc_f1(C):
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    rs, cs = C.sum(-1), C.sum(-2)
    with np.errstate(all='ignore'):
        f1 = np.where(rs + cs > 0, 2 * tp / (rs + cs), 0.0)
    present = rs > 0
    macro = (f1 * present).sum(-1) / present.sum(-1)
    acc = tp.sum(-1) / C.sum((-2, -1))
    return acc, macro, f1


def point(y, p):
    C, _ = confusions(y, p, np.zeros(len(y)))
    a, m, f = acc_f1(C.sum(0))
    return float(a), float(m), f


def bootstrap(Cg, B=1000, seed=0, hier=None):
    """Bootstrap over groups (rows of Cg). If hier is given (resident id per group), resample residents
    first and then days within each sampled resident (two-stage hierarchical bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(Cg)
    out = np.zeros((B, 2))
    if hier is None:
        for b in range(B):
            s = rng.integers(0, n, n)
            a, m, _ = acc_f1(Cg[s].sum(0)); out[b] = a, m
    else:
        hier = np.asarray(hier); res = np.unique(hier)
        members = {r: np.where(hier == r)[0] for r in res}
        for b in range(B):
            rs = rng.choice(res, len(res))
            s = np.concatenate([rng.choice(members[r], len(members[r])) for r in rs])
            a, m, _ = acc_f1(Cg[s].sum(0)); out[b] = a, m
    return out


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def paired_delta(CgA, CgB, B=1000, seed=0, hier=None):
    rng = np.random.default_rng(seed); n = len(CgA); d = np.zeros(B)
    if hier is not None:
        hier = np.asarray(hier); res = np.unique(hier); members = {r: np.where(hier == r)[0] for r in res}
    for b in range(B):
        if hier is None:
            s = rng.integers(0, n, n)
        else:
            rs = rng.choice(res, len(res))
            s = np.concatenate([rng.choice(members[r], len(members[r])) for r in rs])
        d[b] = acc_f1(CgA[s].sum(0))[1] - acc_f1(CgB[s].sum(0))[1]
    full = acc_f1(CgA.sum(0))[1] - acc_f1(CgB.sum(0))[1]
    return float(full), ci(d)


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; n = len(conf)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
            rows.append((lo, hi, int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
    return e, rows


def brier(prob, y):
    Y = np.zeros_like(prob); Y[np.arange(len(y)), [A2I[v] for v in y]] = 1
    return float(((prob - Y) ** 2).sum(1).mean())
