"""Neural sequence baselines (R1.5): BiLSTM, BiLSTM-CRF, TCN, Transformer on the Stage-2 feature set, LORO."""
import sys, pickle, time; import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd, torch, torch.nn as nn
from mmcfe.simulator import ACTIVITIES
from mmcfe.pipeline import FoldPreprocessor, feature_sets
torch.set_num_threads(1)
W = pd.read_pickle('data/windows_default.pkl').sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
A2I = {a: i for i, a in enumerate(ACTIVITIES)}
RES = sorted(W.resident_id.unique()); COLS = feature_sets()['S2']; K = 9


class CRFLayer(nn.Module):
    def __init__(self, k):
        super().__init__(); self.T = nn.Parameter(torch.zeros(k, k)); self.s = nn.Parameter(torch.zeros(k)); self.e = nn.Parameter(torch.zeros(k))

    def nll(self, em, y):
        B, L, k = em.shape
        a = self.s + em[:, 0]
        for t in range(1, L):
            a = torch.logsumexp(a[:, :, None] + self.T[None] + em[:, t, None, :], 1)
        logZ = torch.logsumexp(a + self.e, 1)
        sc = self.s[y[:, 0]] + em[:, 0].gather(1, y[:, :1]).squeeze(1)
        for t in range(1, L):
            sc = sc + self.T[y[:, t - 1], y[:, t]] + em[:, t].gather(1, y[:, t:t + 1]).squeeze(1)
        sc = sc + self.e[y[:, -1]]
        return (logZ - sc).mean()

    def marginals(self, em):
        B, L, k = em.shape
        al = [self.s + em[:, 0]]
        for t in range(1, L):
            al.append(torch.logsumexp(al[-1][:, :, None] + self.T[None] + em[:, t, None, :], 1))
        be = [self.e.expand(B, k)]
        for t in range(L - 2, -1, -1):
            be.insert(0, torch.logsumexp(self.T[None] + (em[:, t + 1] + be[0])[:, None, :], 2))
        lm = torch.stack(al, 1) + torch.stack(be, 1)
        return torch.softmax(lm, -1)

    def viterbi(self, em):
        B, L, k = em.shape; d = self.s + em[:, 0]; bp = []
        for t in range(1, L):
            sc = d[:, :, None] + self.T[None]; bp.append(sc.argmax(1)); d = sc.max(1).values + em[:, t]
        d = d + self.e; path = [d.argmax(1)]
        for b in reversed(bp):
            path.insert(0, b.gather(1, path[0][:, None]).squeeze(1))
        return torch.stack(path, 1)


class Net(nn.Module):
    def __init__(self, kind, d, h=64):
        super().__init__(); self.kind = kind
        if kind in ('bilstm', 'bilstm_crf'):
            self.enc = nn.LSTM(d, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2); out = 2 * h
        elif kind == 'tcn':
            self.inp = nn.Conv1d(d, h, 1)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(h, h, 3, padding=dl, dilation=dl), nn.ReLU(), nn.Dropout(0.2),
                                                       nn.Conv1d(h, h, 3, padding=dl, dilation=dl), nn.ReLU()) for dl in [1, 2, 4, 8]]); out = h
        elif kind == 'transformer':
            self.inp = nn.Linear(d, h); self.pos = nn.Parameter(torch.zeros(1, 96, h))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 128, 0.1, batch_first=True), 2); out = h
        self.head = nn.Linear(out, K)
        self.crf = CRFLayer(K) if kind == 'bilstm_crf' else None

    def forward(self, x):
        if self.kind in ('bilstm', 'bilstm_crf'):
            z = self.enc(x)[0]
        elif self.kind == 'tcn':
            z = self.inp(x.transpose(1, 2))
            for b in self.blocks:
                z = z + b(z)
            z = z.transpose(1, 2)
        else:
            z = self.enc(self.inp(x) + self.pos)
        return self.head(z)


def to_seq(P, y):
    X = P[COLS].to_numpy(np.float32).reshape(-1, 96, len(COLS))
    Y = np.array([A2I[v] for v in y]).reshape(-1, 96)
    return torch.tensor(X), torch.tensor(Y)


def train(kind, Xtr, Ytr, Xva, Yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(kind, Xtr.shape[-1]); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    best, bstate, bad = 1e9, None, 0
    for ep in range(80):
        net.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 16):
            b = perm[i:i + 16]; em = net(Xtr[b])
            loss = net.crf.nll(torch.log_softmax(em, -1), Ytr[b]) if net.crf is not None else nn.functional.cross_entropy(em.reshape(-1, K), Ytr[b].reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5); opt.step()
        net.eval()
        with torch.no_grad():
            em = net(Xva)
            vl = (net.crf.nll(torch.log_softmax(em, -1), Yva) if net.crf is not None else nn.functional.cross_entropy(em.reshape(-1, K), Yva.reshape(-1))).item()
        if vl < best - 1e-4:
            best, bad = vl, 0; bstate = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(bstate); net.eval(); return net, ep + 1


KINDS = sys.argv[1:] or ['bilstm', 'bilstm_crf', 'tcn', 'transformer']
res = {k: {'pred': np.empty(len(W), dtype=object), 'conf': np.zeros(len(W)), 't_fit': [], 't_inf': [], 'epochs': [], 'params': 0} for k in KINDS}
for fi, r in enumerate(RES):
    tr = (W.resident_id != r).to_numpy(); te = ~tr
    Wtr = W[tr].reset_index(drop=True)
    va_res = RES[(fi + 1) % len(RES)]                    # one training resident held out for early stopping only
    pp = FoldPreprocessor().fit(Wtr[Wtr.resident_id != va_res])
    Ptr = pp.transform(Wtr).reset_index(drop=True); Pte = pp.transform(W[te]).reset_index(drop=True)
    vm = (Ptr.resident_id == va_res).to_numpy()
    Xa, Ya = to_seq(Ptr[~vm], Wtr.label.values[~vm]); Xv, Yv = to_seq(Ptr[vm], Wtr.label.values[vm]); Xt, _ = to_seq(Pte, W.label.values[te])
    for k in KINDS:
        t0 = time.perf_counter(); net, ep = train(k, Xa, Ya, Xv, Yv, seed=fi); tf = time.perf_counter() - t0
        with torch.no_grad():
            t0 = time.perf_counter(); em = net(Xt)
            if net.crf is not None:
                path = net.crf.viterbi(torch.log_softmax(em, -1)); P = net.crf.marginals(torch.log_softmax(em, -1))
            else:
                P = torch.softmax(em, -1); path = P.argmax(-1)
            ti = (time.perf_counter() - t0) / len(Xt)
        pred = np.array(ACTIVITIES, dtype=object)[path.reshape(-1).numpy()]
        conf = P.reshape(-1, K).gather(1, path.reshape(-1, 1)).squeeze(1).numpy()
        idx = np.where(te)[0]; res[k]['pred'][idx] = pred; res[k]['conf'][idx] = conf
        res[k]['t_fit'].append(tf); res[k]['t_inf'].append(ti); res[k]['epochs'].append(ep)
        res[k]['params'] = sum(p.numel() for p in net.parameters())
        print(r, k, f'acc={(pred == W.label.values[te]).mean():.3f} ep={ep} fit={tf:.0f}s', flush=True)
pickle.dump(res, open(f'results/dl_{"_".join(KINDS)}.pkl', 'wb'))
print('DONE')
