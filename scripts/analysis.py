"""Compile all tables, figures and the manuscript number macros (paper/numbers.tex) from saved results."""
import sys, os, pickle, json; import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from mmcfe.simulator import ACTIVITIES, ROOMS
from mmcfe.metrics import confusions, acc_f1, bootstrap, ci, paired_delta, ece, brier, A2I

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'TeX Gyre Termes', 'DejaVu Serif'], 'font.size': 8,
                     'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'savefig.dpi': 300,
                     'axes.edgecolor': '#555555', 'xtick.color': '#444444', 'ytick.color': '#444444', 'axes.labelcolor': '#222222'})
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GRAY = '#8a8a85'; INK = '#222222'; GRID = '#e6e6e3'
ACOL = dict(zip(['Rest', 'TV', 'Cooking', 'Eating', 'Prayer', 'Outdoor', 'Exercise', 'Hygiene'], CAT)); ACOL['Sleep'] = '#3b3b39'
FIG, TAB = 'figures', 'results/tables'
NUM = {}


def put(k, v, fmt='{:.3f}'):
    NUM[k] = fmt.format(v) if not isinstance(v, str) else v


def hier_groups(W):
    g = (W.resident_id + '#' + W.day.astype(str)).to_numpy()
    return g


def summarise(W, pred, B=1000):
    g = hier_groups(W)
    C, gu = confusions(W.label.to_numpy(), np.asarray(pred), g)
    res = np.array([k.split('#')[0] for k in gu])
    a, f, pc = acc_f1(C.sum(0))
    bh = bootstrap(C, B, hier=res); bd = bootstrap(C, B)
    per = {r: float(acc_f1(C[res == r].sum(0))[1]) for r in sorted(set(res))}
    pera = {r: float(acc_f1(C[res == r].sum(0))[0]) for r in sorted(set(res))}
    return dict(acc=float(a), f1=float(f), pc=pc, acc_h=ci(bh[:, 0]), f1_h=ci(bh[:, 1]), acc_d=ci(bd[:, 0]), f1_d=ci(bd[:, 1]),
                per=per, per_acc=pera, C=C, res=res)


# ------------------------------------------------------------------ main LORO / DAYS
W, OUT, TIM = pickle.load(open('results/oof_LORO.pkl', 'rb'))
y = W.label.to_numpy()
S = {m: summarise(W, OUT[m].pred) for m in OUT}
pickle.dump(S, open('results/summary_LORO.pkl', 'wb'))
ORDER_MAIN = ['Majority class', 'Room × hour lookup', 'HMM (supervised, Gaussian)', 'MLP with ±2-window context', 'MMCFE full (S1+S2+S3 compound)', 'Random Forest', 'Gradient boosting']
SHORT = {'CRF: sensors only': 's0', 'CRF: + temporal (S1)': 's1', 'CRF: + temporal + room fractions (S1+S2a)': 's2a',
         'CRF: + temporal + prior only (S1+S2b)': 's2b', 'CRF: + temporal + spatial (S1+S2)': 's2', 'MMCFE full (S1+S2+S3 compound)': 'full',
         'MMCFE full, no transitions': 'notr', 'MMCFE causal (online)': 'causal', 'Majority class': 'maj', 'Room × hour lookup': 'rh',
         'HMM (supervised, Gaussian)': 'hmm', 'Random Forest': 'rf', 'Gradient boosting': 'gb', 'MLP with ±2-window context': 'mlp',
         'GB: sensors only': 'gbs0', 'GB: + temporal (S1)': 'gbs1', 'GB: full TSS (S1+S2+S3 compound)': 'gbfull'}
for m, s in S.items():
    k = SHORT[m]
    put(f'{k}-acc', s['acc']); put(f'{k}-f1', s['f1'])
    put(f'{k}-f1lo', s['f1_h'][0]); put(f'{k}-f1hi', s['f1_h'][1]); put(f'{k}-acclo', s['acc_h'][0]); put(f'{k}-acchi', s['acc_h'][1])
    put(f'{k}-f1dlo', s['f1_d'][0]); put(f'{k}-f1dhi', s['f1_d'][1])
    v = np.array(list(s['per'].values())); put(f'{k}-resmean', v.mean()); put(f'{k}-ressd', v.std(ddof=1)); put(f'{k}-resmin', v.min()); put(f'{k}-resmax', v.max())
    put(f'{k}-accpct', 100 * s['acc'], '{:.1f}')
    for i, a in enumerate(ACTIVITIES):
        put(f'{k}-pc-{a}', s['pc'][i])
rows = []
for m in S:
    s = S[m]
    rows.append(dict(model=m, acc=s['acc'], acc_lo=s['acc_h'][0], acc_hi=s['acc_h'][1], f1=s['f1'], f1_lo=s['f1_h'][0], f1_hi=s['f1_h'][1],
                     f1_daylo=s['f1_d'][0], f1_dayhi=s['f1_d'][1], res_mean=np.mean(list(s['per'].values())), res_sd=np.std(list(s['per'].values()), ddof=1),
                     **{a: s['pc'][i] for i, a in enumerate(ACTIVITIES)}))
pd.DataFrame(rows).round(4).to_csv(f'{TAB}/results_LORO.csv', index=False)
pd.DataFrame({SHORT[m]: S[m]['per'] for m in S}).round(4).to_csv(f'{TAB}/per_resident_f1_LORO.csv')
pd.DataFrame({SHORT[m]: S[m]['per_acc'] for m in S}).round(4).to_csv(f'{TAB}/per_resident_acc_LORO.csv')
for r, v in S['MMCFE full (S1+S2+S3 compound)']['per'].items():
    put(f'full-res-{r}', v); put(f'gb-res-{r}', S['Gradient boosting']['per'][r]); put(f's0-res-{r}', S['CRF: sensors only']['per'][r])
    put(f'rf-res-{r}', S['Random Forest']['per'][r]); put(f'mlp-res-{r}', S['MLP with ±2-window context']['per'][r])
    put(f'full-resacc-{r}', S['MMCFE full (S1+S2+S3 compound)']['per_acc'][r])
gbwins = sum(S['Gradient boosting']['per'][r] > S['MMCFE full (S1+S2+S3 compound)']['per'][r] for r in S['Gradient boosting']['per'])
put('gb-wins', str(int(gbwins)))
worst = min(S['MMCFE full (S1+S2+S3 compound)']['per'], key=S['MMCFE full (S1+S2+S3 compound)']['per'].get); put('full-worst', worst)

# paired differences (hierarchical bootstrap)
def pdel(a, b):
    return paired_delta(S[a]['C'], S[b]['C'], 1000, 0, hier=S[a]['res'])
PAIRS = [('MMCFE full (S1+S2+S3 compound)', 'CRF: sensors only'), ('CRF: + temporal (S1)', 'CRF: sensors only'),
         ('CRF: + temporal + spatial (S1+S2)', 'CRF: + temporal (S1)'), ('CRF: + temporal + room fractions (S1+S2a)', 'CRF: + temporal (S1)'),
         ('CRF: + temporal + prior only (S1+S2b)', 'CRF: + temporal (S1)'), ('CRF: + temporal + spatial (S1+S2)', 'CRF: + temporal + room fractions (S1+S2a)'),
         ('MMCFE full (S1+S2+S3 compound)', 'CRF: + temporal + spatial (S1+S2)'), ('MMCFE full (S1+S2+S3 compound)', 'MMCFE full, no transitions'),
         ('MMCFE causal (online)', 'MMCFE full (S1+S2+S3 compound)'), ('MMCFE full (S1+S2+S3 compound)', 'Majority class'),
         ('MMCFE full (S1+S2+S3 compound)', 'Room × hour lookup'), ('MMCFE full (S1+S2+S3 compound)', 'HMM (supervised, Gaussian)'),
         ('MMCFE full (S1+S2+S3 compound)', 'MLP with ±2-window context'), ('MMCFE full (S1+S2+S3 compound)', 'Random Forest'),
         ('MMCFE full (S1+S2+S3 compound)', 'Gradient boosting'), ('GB: + temporal (S1)', 'GB: sensors only'),
         ('Gradient boosting', 'GB: + temporal (S1)'), ('GB: full TSS (S1+S2+S3 compound)', 'Gradient boosting'), ('Gradient boosting', 'GB: sensors only')]
prow = []
for a, b in PAIRS:
    d, (lo, hi) = pdel(a, b)
    prow.append(dict(a=a, b=b, delta=d, lo=lo, hi=hi, sig=(lo > 0) or (hi < 0)))
    k = f'd-{SHORT[a]}-{SHORT[b]}'
    put(k, d, '{:+.3f}'); put(k + '-lo', lo, '{:+.3f}'); put(k + '-hi', hi, '{:+.3f}')
pd.DataFrame(prow).round(4).to_csv(f'{TAB}/paired_LORO.csv', index=False)

# held-out days
if os.path.exists('results/oof_DAYS.pkl'):
    WD, OD, TD = pickle.load(open('results/oof_DAYS.pkl', 'rb'))
    te = (WD.day >= 20).to_numpy()
    for m in OD:
        s = summarise(WD[te].reset_index(drop=True), OD[m].pred.to_numpy()[te])
        k = SHORT[m]; put(f'{k}-dacc', s['acc']); put(f'{k}-df1', s['f1']); put(f'{k}-df1lo', s['f1_h'][0]); put(f'{k}-df1hi', s['f1_h'][1])

# timing (CRF inference per day from the main run)
for m in TIM:
    k = SHORT[m]
    put(f'{k}-tfit', np.nanmean(TIM[m]['t_fit']), '{:.1f}')
    if TIM[m]['t_inf_per_day']:
        put(f'{k}-tinf', 1000 * np.mean(TIM[m]['t_inf_per_day']), '{:.1f}')

# ------------------------------------------------------------------ confidence, calibration, boundaries
F = OUT['MMCFE full (S1+S2+S3 compound)']
conf = F.conf.to_numpy(float); corr = (F.pred.to_numpy() == y)
put('conf-correct', conf[corr].mean()); put('conf-wrong', conf[~corr].mean())
hi = conf >= 0.9; put('cov90', 100 * hi.mean(), '{:.1f}'); put('acc90', 100 * corr[hi].mean(), '{:.1f}'); put('accbelow90', 100 * corr[~hi].mean(), '{:.1f}')
single = (W.purity == 1).to_numpy()
put('acc-single', 100 * corr[single].mean(), '{:.1f}'); put('acc-mixed', 100 * corr[~single].mean(), '{:.1f}')
put('share-mixed', 100 * (~single).mean(), '{:.1f}'); put('err-mixed', 100 * (~single & ~corr).sum() / (~corr).sum(), '{:.1f}')
calib = []
rng = np.random.default_rng(0); g = hier_groups(W); res_of = W.resident_id.to_numpy()
for m in ['MMCFE full (S1+S2+S3 compound)', 'Gradient boosting', 'Random Forest', 'MLP with ±2-window context', 'MMCFE causal (online)']:
    P = OUT[m][['p_' + a for a in ACTIVITIES]].to_numpy(float); c = OUT[m].conf.to_numpy(float); ok = OUT[m].pred.to_numpy() == y
    e, _ = ece(c, ok, 15); b = brier(P, y)
    # hierarchical bootstrap of ECE / Brier
    R = sorted(set(res_of)); idx_r = {r: np.where(res_of == r)[0] for r in R}
    days_r = {r: [np.where((res_of == r) & (W.day.to_numpy() == d))[0] for d in range(30)] for r in R}
    eb, bb = [], []
    for _ in range(300):
        rs = rng.choice(R, len(R)); ii = np.concatenate([np.concatenate([days_r[r][j] for j in rng.integers(0, 30, 30)]) for r in rs])
        eb.append(ece(c[ii], ok[ii], 15)[0]); Yb = y[ii]; bb.append(brier(P[ii], Yb))
    calib.append(dict(model=m, ece=e, ece_lo=ci(eb)[0], ece_hi=ci(eb)[1], brier=b, brier_lo=ci(bb)[0], brier_hi=ci(bb)[1]))
    k = SHORT[m]; put(f'{k}-ece', e); put(f'{k}-ecelo', ci(eb)[0]); put(f'{k}-ecehi', ci(eb)[1]); put(f'{k}-brier', b); put(f'{k}-brierlo', ci(bb)[0]); put(f'{k}-brierhi', ci(bb)[1])
pd.DataFrame(calib).round(4).to_csv(f'{TAB}/calibration_LORO.csv', index=False)

# Figure: confidence panel (reliability with CIs, coverage curve, boundary windows)
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.3))
edges = np.linspace(0, 1, 11); mids, accs, los, his, ns = [], [], [], [], []
for lo_, hi_ in zip(edges[:-1], edges[1:]):
    msk = (conf > lo_) & (conf <= hi_)
    if msk.sum() < 20:
        continue
    bs = []
    for _ in range(300):
        rs = rng.choice(sorted(set(res_of)), 8); ii = np.concatenate([np.where(msk & (res_of == r))[0] for r in rs])
        bs.append(corr[ii].mean() if len(ii) else np.nan)
    mids.append(conf[msk].mean()); accs.append(corr[msk].mean()); los.append(np.nanpercentile(bs, 2.5)); his.append(np.nanpercentile(bs, 97.5)); ns.append(msk.sum())
ax[0].plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls='--')
ax[0].errorbar(mids, accs, yerr=[np.array(accs) - los, np.array(his) - accs], fmt='o', ms=4, color=CAT[0], ecolor=CAT[0], elinewidth=1, capsize=2)
for x_, y_, n_ in zip(mids, accs, ns):
    ax[0].annotate(f'{n_}', (x_, y_), textcoords='offset points', xytext=(0, -11), ha='center', fontsize=5.5, color='#555555')
ax[0].set_xlabel('Mean CRF confidence in bin'); ax[0].set_ylabel('Accuracy in bin'); ax[0].set_xlim(0.2, 1.02); ax[0].set_ylim(0.2, 1.02)
ax[0].set_title('(a) Reliability (95% resident-level CI)', loc='left')
taus = np.linspace(0.3, 0.99, 70); cov = [(conf >= t).mean() for t in taus]; acc_t = [corr[conf >= t].mean() for t in taus]
ax[1].plot(taus, acc_t, color=CAT[0], lw=2, label='Accuracy of retained windows'); ax[1].plot(taus, cov, color=CAT[1], lw=2, label='Share retained')
ax[1].set_xlabel(r'Confidence threshold $\tau$'); ax[1].set_ylim(0.5, 1.01); ax[1].legend(frameon=False, loc='lower left')
ax[1].set_title('(b) Accuracy vs. coverage', loc='left')
vals = [corr[single].mean(), corr[~single].mean()]
bars = ax[2].bar(['Single-activity', 'Mixed (boundary)'], vals, color=[CAT[0], CAT[1]], width=0.55, edgecolor='white', linewidth=2)
for bb_, v, sh, er in zip(bars, vals, [single.mean(), (~single).mean()], [(single & ~corr).sum() / (~corr).sum(), (~single & ~corr).sum() / (~corr).sum()]):
    ax[2].text(bb_.get_x() + bb_.get_width() / 2, v + 0.01, f'{100 * v:.1f}%\n{100 * sh:.1f}% of windows\n{100 * er:.1f}% of errors', ha='center', va='bottom', fontsize=6, color=INK)
ax[2].set_ylim(0, 1.25); ax[2].set_ylabel('Accuracy'); ax[2].set_title('(c) Boundary windows', loc='left')
for a_ in ax:
    a_.grid(axis='y', color=GRID, lw=0.5); a_.set_axisbelow(True)
plt.tight_layout(); plt.savefig(f'{FIG}/fig_confidence_boundaries.png', bbox_inches='tight'); plt.close()

# confusion matrix (full model)
C = S['MMCFE full (S1+S2+S3 compound)']['C'].sum(0); Cn = C / C.sum(1, keepdims=True)
pd.DataFrame(Cn, index=ACTIVITIES, columns=ACTIVITIES).round(3).to_csv(f'{TAB}/confusion_full_LORO.csv')
put('cm-ex-out', 100 * Cn[A2I['Exercise'], A2I['Outdoor']], '{:.0f}'); put('cm-pr-tv', 100 * Cn[A2I['Prayer'], A2I['TV']], '{:.0f}')
put('cm-ea-tv', 100 * Cn[A2I['Eating'], A2I['TV']], '{:.0f}'); put('cm-ea-rest', 100 * Cn[A2I['Eating'], A2I['Rest']], '{:.0f}')
fig, ax = plt.subplots(figsize=(3.5, 3.1))
im = ax.imshow(Cn, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(9)); ax.set_xticklabels(ACTIVITIES, rotation=45, ha='right'); ax.set_yticks(range(9)); ax.set_yticklabels(ACTIVITIES)
for i in range(9):
    for j in range(9):
        if Cn[i, j] >= 0.01:
            ax.text(j, i, f'{Cn[i, j]:.2f}', ha='center', va='center', fontsize=5.5, color='white' if Cn[i, j] > 0.55 else INK)
ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.spines[:].set_visible(False)
plt.colorbar(im, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig(f'{FIG}/fig_confusion_LORO.png', bbox_inches='tight'); plt.close()

# ablation figure
AB = ['CRF: sensors only', 'CRF: + temporal (S1)', 'CRF: + temporal + room fractions (S1+S2a)', 'CRF: + temporal + prior only (S1+S2b)',
      'CRF: + temporal + spatial (S1+S2)', 'MMCFE full (S1+S2+S3 compound)', 'MMCFE full, no transitions', 'MMCFE causal (online)']
LAB = ['Sensors only', '+ S1 temporal', '+ S1 + room\nfractions (S2a)', '+ S1 + prior\nonly (S2b)', '+ S1 + S2\n(fractions + prior)', 'Full MMCFE\n(+ S3)', 'Full, no\ntransitions', 'Full, causal\n(online)']
fig, ax = plt.subplots(figsize=(7.16, 2.3))
xs = np.arange(len(AB)); vals = [S[m]['f1'] for m in AB]
cols = [CAT[0]] * 6 + [GRAY, GRAY]
ax.bar(xs, vals, color=cols, width=0.6, edgecolor='white', linewidth=2)
ax.errorbar(xs, vals, yerr=[[v - S[m]['f1_h'][0] for v, m in zip(vals, AB)], [S[m]['f1_h'][1] - v for v, m in zip(vals, AB)]], fmt='none', ecolor=INK, elinewidth=0.8, capsize=2)
for x_, v, m in zip(xs, vals, AB):
    ax.text(x_ + 0.33, v - 0.012, f'{v:.3f}', ha='left', va='center', fontsize=6.5, color=INK)
ax.axhline(S['Gradient boosting']['f1'], color=CAT[1], lw=1.2, ls='--'); ax.text(-0.45, S['Gradient boosting']['f1'] + 0.004, f"Gradient boosting ({S['Gradient boosting']['f1']:.3f})", color=INK, fontsize=6.5, ha='left', va='bottom')
ax.set_xticks(xs); ax.set_xticklabels(LAB); ax.set_ylim(0.7, 0.95); ax.set_xlim(-0.6, len(AB) - 0.3); ax.set_ylabel('Macro-F1 (LORO)')
ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(f'{FIG}/fig_ablation_LORO.png', bbox_inches='tight'); plt.close()

# per-resident figure
PM = [('MMCFE full (S1+S2+S3 compound)', 'MMCFE (full)', CAT[0], 'o'), ('Gradient boosting', 'Gradient boosting', CAT[1], 's'),
      ('Random Forest', 'Random Forest', CAT[2], '^'), ('MLP with ±2-window context', 'MLP (±2 windows)', CAT[6], 'D'), ('CRF: sensors only', 'CRF, sensors only', GRAY, 'v')]
R = sorted(S['Gradient boosting']['per'])
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for i, (m, lab, c, mk) in enumerate(PM):
    ax.plot(np.arange(len(R)) + (i - 2) * 0.09, [S[m]['per'][r] for r in R], mk, color=c, ms=4.5, label=lab, markeredgecolor='white', markeredgewidth=0.6)
ax.set_xticks(range(len(R))); ax.set_xticklabels(R); ax.set_ylabel('Macro-F1 on held-out resident'); ax.set_xlabel('Test resident (LORO fold)')
ax.legend(frameon=False, ncol=2, loc='lower left', fontsize=6); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_ylim(0.6, 1.0)
plt.tight_layout(); plt.savefig(f'{FIG}/fig_per_resident.png', bbox_inches='tight'); plt.close()

# timeline (R02 day 3)
sel = ((W.resident_id == 'R02') & (W.day == 3)).to_numpy()
fig, ax = plt.subplots(figsize=(7.16, 1.25))
for row, lab in [(1, W.label.to_numpy()[sel]), (0, F.pred.to_numpy()[sel])]:
    for t, a in enumerate(lab):
        ax.barh(row, 1, left=t, color=ACOL[a], height=0.8, edgecolor='none')
ax.set_yticks([0, 1]); ax.set_yticklabels(['Predicted', 'True']); ax.set_xticks(range(0, 97, 8)); ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)])
ax.spines[['left', 'bottom']].set_visible(False); ax.tick_params(length=0)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ACOL[a]) for a in ACTIVITIES], labels=ACTIVITIES, ncol=9, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=6, handlelength=1)
plt.savefig(f'{FIG}/fig_timeline_heldout.png', bbox_inches='tight'); plt.close()
put('tl-acc', 100 * corr[sel].mean(), '{:.1f}')

# ------------------------------------------------------------------ optional sections
def opt(path):
    return os.path.exists(path)

if opt('results/oof_MASK.pkl'):
    WM, RM = pickle.load(open('results/oof_MASK.pkl', 'rb'))
    for mask, out in RM.items():
        for m, o in out.items():
            put(f'mask-{mask}-{SHORT[m]}', summarise(WM, o.pred, B=200)['f1'])

if opt('results/robust.pkl'):
    RB = pickle.load(open('results/robust.pkl', 'rb')); rows = []
    Wc = W[['resident_id', 'day', 'w']]
    for (kind, name), df in RB['pred'].items():
        df = df.sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
        s = summarise(df.rename(columns={}), df.pred, B=300)
        rows.append(dict(kind=kind, model=name, f1=s['f1'], lo=s['f1_h'][0], hi=s['f1_h'][1], acc=s['acc'], conf=df.conf.mean()))
        k = {'MMCFE': 'mm', 'MMCFE + modality dropout': 'mmdo', 'Gradient boosting': 'gb', 'Gradient boosting + modality dropout': 'gbdo'}[name]
        put(f'rb-{kind}-{k}', s['f1']); put(f'rb-{kind}-{k}-lo', s['f1_h'][0]); put(f'rb-{kind}-{k}-hi', s['f1_h'][1])
    RBT = pd.DataFrame(rows); RBT.round(4).to_csv(f'{TAB}/robustness_LORO.csv', index=False)
    for nm, k in [('MMCFE', 'mm'), ('MMCFE + modality dropout', 'mmdo'), ('Gradient boosting', 'gb'), ('Gradient boosting + modality dropout', 'gbdo')]:
        f_ = lambda kd: RBT[(RBT.kind == kd) & (RBT.model == nm)].f1.iloc[0]
        put(f'd-rbloss-{k}', f_('clean') - f_('loc_outage_0.5')); put(f'd-rbnoloc-{k}', f_('clean') - f_('no_location'))
    clean = RB['pred'][('clean', 'MMCFE')].sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
    for kind in ['loc_outage_0.25', 'no_location', 'sys_living_as_bedroom', 'sys_kitchen_dead', 'sys_bathroom_blind']:
        for name, k in [('MMCFE', 'mm'), ('MMCFE + modality dropout', 'mmdo')]:
            df = RB['pred'][(kind, name)].sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
            cl = RB['pred'][('clean', name)].sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
            put(f'rbconf-{kind}-{k}', df.conf.mean()); put(f'rbconf-clean-{k}', cl.conf.mean())
            # day-level failure flag: mean daily confidence of the corrupted day vs the same clean day
            dc = df.groupby(['resident_id', 'day']).conf.mean().to_numpy(); dk = cl.groupby(['resident_id', 'day']).conf.mean().to_numpy()
            auc = roc_auc_score(np.r_[np.ones(len(dc)), np.zeros(len(dk))], -np.r_[dc, dk]); put(f'rbauc-{kind}-{k}', auc, '{:.2f}')
    # figure: macro-F1 vs location outage
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    xs = [0, 0.1, 0.25, 0.5, 1.0]; kinds = ['clean', 'loc_outage_0.1', 'loc_outage_0.25', 'loc_outage_0.5', 'no_location']
    for (name, c, mk, ls) in [('MMCFE', CAT[0], 'o', '-'), ('MMCFE + modality dropout', CAT[0], 'o', '--'), ('Gradient boosting', CAT[1], 's', '-'), ('Gradient boosting + modality dropout', CAT[1], 's', '--')]:
        v = [RBT[(RBT.kind == k) & (RBT.model == name)].f1.iloc[0] for k in kinds]
        ax.plot(xs, v, ls, marker=mk, color=c, lw=2, ms=4.5, label=name, markeredgecolor='white', markeredgewidth=0.6)
    ax.set_xlabel('Share of each test day without localisation'); ax.set_ylabel('Macro-F1 (LORO)'); ax.set_xticks(xs); ax.set_xticklabels(['0', '10%', '25%', '50%', '100%'])
    ax.legend(frameon=False, fontsize=6, loc='lower left'); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(f'{FIG}/fig_robustness.png', bbox_inches='tight'); plt.close()

if opt('results/inner.pkl'):
    IN = pickle.load(open('results/inner.pkl', 'rb'))
    # hybrid & permutation
    hy = pd.concat(IN['hybrid']).sort_values('idx'); pr = np.empty(len(W), dtype=object); pr[hy.idx] = hy.pred.to_numpy()
    sh = summarise(W, pr); put('hyb-f1', sh['f1']); put('hyb-f1lo', sh['f1_h'][0]); put('hyb-f1hi', sh['f1_h'][1]); put('hyb-acc', sh['acc'])
    S['hybrid'] = sh
    d, (lo, hi_) = paired_delta(sh['C'], S['Gradient boosting']['C'], 1000, 0, hier=sh['res']); put('d-hyb-gb', d, '{:+.3f}'); put('d-hyb-gb-lo', lo, '{:+.3f}'); put('d-hyb-gb-hi', hi_, '{:+.3f}')
    d, (lo, hi_) = paired_delta(sh['C'], S['MMCFE full (S1+S2+S3 compound)']['C'], 1000, 0, hier=sh['res']); put('d-hyb-full', d, '{:+.3f}'); put('d-hyb-full-lo', lo, '{:+.3f}'); put('d-hyb-full-hi', hi_, '{:+.3f}')
    pm = pd.concat(IN['perm']).sort_values('idx'); pp_ = np.empty(len(W), dtype=object); pp_[pm.idx] = pm.pred.to_numpy()
    sp = summarise(W, pp_, B=200); put('perm-f1', sp['f1']); v = np.array(list(sp['per'].values())); put('perm-min', v.min()); put('perm-max', v.max())
    pd.DataFrame({'real': S['MMCFE full (S1+S2+S3 compound)']['per'], 'permuted': sp['per']}).round(4).to_csv(f'{TAB}/permutation_all_folds.csv')
    # pre-specified threshold rule chosen on inner (training-resident) predictions only
    grid = np.round(np.arange(0.50, 0.995, 0.01), 2); TARGET = 0.98; rows = []
    for r in sorted(W.resident_id.unique()):
        cs = np.concatenate([IN['inner_conf'][k] for k in IN['inner_conf'] if k[0] == r]); ok_ = np.concatenate([IN['inner_correct'][k] for k in IN['inner_correct'] if k[0] == r])
        tau = next((t for t in grid if ok_[cs >= t].mean() >= TARGET), grid[-1])
        m_ = (W.resident_id == r).to_numpy(); keep = conf[m_] >= tau
        rows.append(dict(resident=r, tau=tau, coverage=keep.mean(), acc=corr[m_][keep].mean(), acc_rest=corr[m_][~keep].mean()))
    TH = pd.DataFrame(rows); TH.round(4).to_csv(f'{TAB}/threshold_innercv.csv', index=False)
    taus_r = dict(zip(TH.resident, TH.tau)); keep_all = np.array([conf[i] >= taus_r[r] for i, r in enumerate(res_of)])
    put('tau-min', TH.tau.min(), '{:.2f}'); put('tau-max', TH.tau.max(), '{:.2f}'); put('tau-median', TH.tau.median(), '{:.2f}')
    put('th-cov', 100 * keep_all.mean(), '{:.1f}'); put('th-acc', 100 * corr[keep_all].mean(), '{:.1f}'); put('th-accrest', 100 * corr[~keep_all].mean(), '{:.1f}')
    put('th-covmin', 100 * TH.coverage.min(), '{:.1f}'); put('th-covmax', 100 * TH.coverage.max(), '{:.1f}'); put('th-accmin', 100 * TH.acc.min(), '{:.1f}'); put('th-accmax', 100 * TH.acc.max(), '{:.1f}')
    pa = []
    for a in ACTIVITIES:
        msk = y == a
        pa.append(dict(activity=a, support=int(msk.sum()), coverage=keep_all[msk].mean(), acc_retained=corr[msk & keep_all].mean() if (msk & keep_all).any() else np.nan,
                       recall_all=corr[msk].mean()))
        put(f'th-cov-{a}', 100 * keep_all[msk].mean(), '{:.1f}'); put(f'th-acc-{a}', 100 * corr[msk & keep_all].mean(), '{:.1f}')
    pd.DataFrame(pa).round(4).to_csv(f'{TAB}/threshold_per_activity.csv', index=False)

if opt('results/anomaly.pkl'):
    AN = pickle.load(open('results/anomaly.pkl', 'rb'))
    WA, meta = AN['W'], AN['meta']
    IND = {'missed_meals': (lambda lab: np.isin(lab, ['Eating']), 44, 86, -1), 'prolonged_inactivity': (lambda lab: np.isin(lab, ['Rest', 'Sleep']), 36, 72, +1),
           'night_wandering': (lambda lab: ~np.isin(lab, ['Sleep', 'Hygiene']), 0, 16, +1), 'skipped_morning_routine': (lambda lab: np.isin(lab, ['Hygiene', 'Prayer']), 16, 40, -1)}

    def day_indicators(lab, conf_=None, tau=None):
        """minutes of each indicator per resident-day; with gating, use confident windows only and rescale by coverage"""
        L = lab.reshape(-1, 96); out = {}; cov = {}
        Cf = conf_.reshape(-1, 96) if conf_ is not None else None
        for k, (fn, a, b, _) in IND.items():
            hit = fn(L[:, a:b]).astype(float)
            if Cf is None:
                out[k] = 15 * hit.sum(1); cov[k] = np.ones(len(L))
            else:
                ok_ = Cf[:, a:b] >= tau; n = ok_.sum(1); cov[k] = n / (b - a)
                out[k] = np.where(n > 0, 15 * (hit * ok_).sum(1) / np.maximum(n, 1) * (b - a), np.nan)
        return pd.DataFrame(out), pd.DataFrame(cov)

    keys = WA[['resident_id', 'day']].drop_duplicates().reset_index(drop=True)
    an = keys.merge(meta[['resident_id', 'day', 'anomaly', 'day_type']], on=['resident_id', 'day'])
    lab_true = WA.label.to_numpy()

    def detect(ind, cov=None, zthr=3.0, mincov=0.5):
        alerts = np.zeros(len(keys), bool); deferred = np.zeros(len(keys), bool); which = np.array([''] * len(keys), dtype=object)
        for r in keys.resident_id.unique():
            base = (keys.resident_id == r) & (keys.day < 30); test = (keys.resident_id == r) & (keys.day >= 30)
            for k, (_, a, b, sgn) in IND.items():
                v = ind[k].to_numpy(); med = np.nanmedian(v[base]); mad = 1.4826 * np.nanmedian(np.abs(v[base] - med)); sc = max(mad, 15.0)
                z = sgn * (v - med) / sc; fire = test.to_numpy() & (z > zthr)
                if cov is not None:
                    low = cov[k].to_numpy() < mincov
                    deferred |= fire & low; fire = fire & ~low
                which[fire & ~alerts] = k; alerts |= fire
        deferred &= ~alerts
        return alerts, deferred, which

    rows = []
    T = (keys.day >= 30).to_numpy(); isan = (an.anomaly != '').to_numpy()
    variants = {'Ground-truth labels (oracle)': day_indicators(lab_true)}
    for mname in AN['preds']:
        pdf = AN['preds'][mname].sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
        lab = pdf.pred.to_numpy(); cf = pdf.conf.to_numpy(float)
        variants[f'{mname}, ungated'] = day_indicators(lab)
        variants[f'{mname}, gated (τ=0.90)'] = day_indicators(lab, cf, 0.90)
    for vname, (ind, cov) in variants.items():
        al, de, wh = detect(ind, None if 'gated' not in vname else cov)
        n_norm = (T & ~isan).sum(); n_an = (T & isan).sum()
        fa = (al & T & ~isan).sum(); tp = (al & T & isan).sum(); dfr = (de & T).sum()
        typed = {t: ((al & T & (an.anomaly == t).to_numpy()).sum(), (T & (an.anomaly == t).to_numpy()).sum()) for t in IND}
        rows.append(dict(variant=vname, false_alerts=int(fa), normal_days=int(n_norm), fa_per100=100 * fa / n_norm, detected=int(tp), anomalous_days=int(n_an),
                         recall=tp / n_an, deferred=int(dfr), precision=tp / max(1, tp + fa), **{f'rec_{t}': v[0] / max(1, v[1]) for t, v in typed.items()}))
    ANT = pd.DataFrame(rows); ANT.round(4).to_csv(f'{TAB}/anomaly_detection.csv', index=False)
    for i, rw in ANT.iterrows():
        k = ['oracle', 'mm-ug', 'mm-g', 'gb-ug', 'gb-g'][i] if len(ANT) == 5 else f'v{i}'
        put(f'an-{k}-fa', rw.false_alerts, '{:.0f}'); put(f'an-{k}-fa100', rw.fa_per100, '{:.1f}'); put(f'an-{k}-rec', 100 * rw.recall, '{:.1f}')
        put(f'an-{k}-det', rw.detected, '{:.0f}'); put(f'an-{k}-def', rw.deferred, '{:.0f}'); put(f'an-{k}-prec', 100 * rw.precision, '{:.1f}')
        for t in IND:
            put(f'an-{k}-rec-{t}', 100 * rw[f'rec_{t}'], '{:.0f}')
    put('an-normal', int(ANT.normal_days.iloc[0]), '{:.0f}'); put('an-anom', int(ANT.anomalous_days.iloc[0]), '{:.0f}')

dl_files = [f for f in os.listdir('results') if f.startswith('dl_') and f.endswith('.pkl')]
DLN = {'bilstm': 'BiLSTM', 'bilstm_crf': 'BiLSTM-CRF', 'tcn': 'TCN', 'transformer': 'Transformer'}
for f in dl_files:
    D = pickle.load(open('results/' + f, 'rb'))
    for k, v in D.items():
        s = summarise(W, v['pred']); S[DLN[k]] = s
        put(f'{k}-f1', s['f1']); put(f'{k}-acc', s['acc']); put(f'{k}-f1lo', s['f1_h'][0]); put(f'{k}-f1hi', s['f1_h'][1])
        put(f'{k}-tfit', np.mean(v['t_fit']), '{:.1f}'); put(f'{k}-tinf', 1000 * np.mean(v['t_inf']), '{:.1f}'); put(f'{k}-params', v['params'], '{:,}')
        vv = np.array(list(s['per'].values())); put(f'{k}-ressd', vv.std(ddof=1))
        d, (lo, hi_) = paired_delta(S['MMCFE full (S1+S2+S3 compound)']['C'], s['C'], 1000, 0, hier=s['res'])
        put(f'd-full-{k}', d, '{:+.3f}'); put(f'd-full-{k}-lo', lo, '{:+.3f}'); put(f'd-full-{k}-hi', hi_, '{:+.3f}')
        c_ = np.asarray(v['conf'], float); ok_ = np.asarray(v['pred']) == y; put(f'{k}-ece', ece(c_, ok_, 15)[0])
        for i, a in enumerate(ACTIVITIES):
            put(f'{k}-pc-{a}', s['pc'][i])

if opt('results/cost.pkl'):
    CO = pickle.load(open('results/cost.pkl', 'rb'))
    for k, (m_, s_) in CO['cost'].items():
        nm, met = k.split('|'); kk = {'MMCFE': 'mm', 'Gradient boosting': 'gb'}[nm]
        put(f'cost-{kk}-{met}', m_, '{:,.0f}' if met in ('nonzero_weights', 'total_weights', 'tree_nodes') else '{:.1f}')
    for nm, (rho, ov) in CO['stability'].items():
        kk = {'MMCFE': 'mm', 'Gradient boosting': 'gb'}[nm]; put(f'stab-{kk}-rho', rho, '{:.2f}'); put(f'stab-{kk}-top10', 100 * ov, '{:.0f}')
    for k, (m_, s_) in CO['faith'].items():
        nm, kk = k.split('|'); put(f'faith-{ {"MMCFE": "mm", "Gradient boosting": "gb"}[nm]}-{kk}', m_, '{:.3f}')
    json.dump({'top_crf': CO['top_crf'], 'top_gb': CO['top_gb']}, open(f'{TAB}/top_features.json', 'w'), indent=1)

sens_files = sorted(f for f in os.listdir('results/sens') if f.endswith('.pkl'))
if sens_files:
    rows = []
    for f in sens_files:
        D = pickle.load(open('results/sens/' + f, 'rb')); sm = D['summary']; name = f[:-4]
        row = dict(config=name, n_family=D['info']['n_family'], mixed=D['info']['mixed'])
        for m, v in sm.items():
            row[SHORT[m]] = v['f1']; row[SHORT[m] + '_lo'] = v['f1_ci'][0]; row[SHORT[m] + '_hi'] = v['f1_ci'][1]
            row[SHORT[m] + '_ressd'] = np.std(list(v['per_res'].values()), ddof=1)
        rows.append(row)
    base = S
    rows.append(dict(config='seed42', n_family=3, mixed=float((W.purity < 1).mean()), **{SHORT[m]: S[m]['f1'] for m in ['CRF: sensors only', 'CRF: + temporal (S1)', 'CRF: + temporal + spatial (S1+S2)', 'MMCFE full (S1+S2+S3 compound)', 'Gradient boosting']},
                     **{SHORT[m] + '_lo': S[m]['f1_h'][0] for m in ['CRF: sensors only', 'CRF: + temporal (S1)', 'CRF: + temporal + spatial (S1+S2)', 'MMCFE full (S1+S2+S3 compound)', 'Gradient boosting']},
                     **{SHORT[m] + '_hi': S[m]['f1_h'][1] for m in ['CRF: sensors only', 'CRF: + temporal (S1)', 'CRF: + temporal + spatial (S1+S2)', 'MMCFE full (S1+S2+S3 compound)', 'Gradient boosting']},
                     **{SHORT[m] + '_ressd': np.std(list(S[m]['per'].values()), ddof=1) for m in ['CRF: sensors only', 'CRF: + temporal (S1)', 'CRF: + temporal + spatial (S1+S2)', 'MMCFE full (S1+S2+S3 compound)', 'Gradient boosting']}))
    ST = pd.DataFrame(rows); ST.round(4).to_csv(f'{TAB}/sensitivity.csv', index=False)
    for _, r in ST.iterrows():
        c = r.config.replace('.', 'p').replace('_', '')
        for k in ['s0', 's1', 's2', 'full', 'gb']:
            if k in r and pd.notna(r[k]):
                put(f'sens-{c}-{k}', r[k])
        if pd.notna(r.get('s1', np.nan)):
            put(f'sens-{c}-dS1', r['s1'] - r['s0'], '{:+.3f}')
        if pd.notna(r.get('s2', np.nan)) and pd.notna(r.get('s1', np.nan)):
            put(f'sens-{c}-dS2', r['s2'] - r['s1'], '{:+.3f}')
        put(f'sens-{c}-dS3', r['full'] - r['s2'], '{:+.3f}'); put(f'sens-{c}-gap', r['gb'] - r['full'], '{:+.3f}')
    pops = ST[ST.config.str.match(r'^seed\d+$')]
    for k in ['s0', 's1', 's2', 'full', 'gb']:
        put(f'pop-{k}-mean', pops[k].mean()); put(f'pop-{k}-sd', pops[k].std(ddof=1)); put(f'pop-{k}-min', pops[k].min()); put(f'pop-{k}-max', pops[k].max())
    put('pop-n', str(len(pops)))
    gap = pops.gb - pops.full; put('pop-gap-mean', gap.mean(), '{:+.3f}'); put('pop-gap-min', gap.min(), '{:+.3f}'); put('pop-gap-max', gap.max(), '{:+.3f}')
    allc = ST.dropna(subset=['s1']) if 's1' in ST else ST
    put('sens-stage-order-holds', str(int(((allc.s2 - allc.s1) > (allc.s1 - allc.s0)).sum())) + '/' + str(len(allc)))
    put('sens-s3pos', str(int((ST.full - ST.s2 > 0).sum())) + '/' + str(len(ST)))
    put('sens-gbwins', str(int((ST.gb > ST.full).sum())) + '/' + str(len(ST)))
    gp = ST.gb - ST.full; put('sens-gap-min', gp.min(), '{:+.3f}'); put('sens-gap-max', gp.max(), '{:+.3f}')
    put('sens-full-min', ST.full.min()); put('sens-full-max', ST.full.max()); put('sens-n', str(len(ST)))
    put('sens-ncfg', str(len(ST) - 1))
    # figure: noise curves
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw=dict(width_ratios=[1.1, 1.4]))
    for k, lab, c in [('s0', 'CRF, sensors only', GRAY), ('s1', '+ S1', CAT[2]), ('s2', '+ S1 + S2', CAT[6]), ('full', 'Full MMCFE', CAT[0]), ('gb', 'Gradient boosting', CAT[1])]:
        xs_, mu, lo_, hi_ = [], [], [], []
        for nz in [0.5, 1.0, 1.5, 2.0]:
            sub = ST[ST.config.isin([f'noise{nz}_seed42', f'noise{nz}_seed1'])] if nz != 1.0 else ST[ST.config.isin(['seed42', 'seed1'])]
            if len(sub) and k in sub:
                xs_.append(nz); mu.append(sub[k].mean()); lo_.append(sub[k].min()); hi_.append(sub[k].max())
        ax[0].plot(xs_, mu, '-o', color=c, lw=2, ms=4, label=lab, markeredgecolor='white', markeredgewidth=0.6)
        ax[0].fill_between(xs_, lo_, hi_, color=c, alpha=0.12, lw=0)
    ax[0].set_xlabel('Simulator noise level (default 1.0)'); ax[0].set_ylabel('Macro-F1 (LORO)'); ax[0].legend(frameon=False, fontsize=6); ax[0].set_xticks([0.5, 1.0, 1.5, 2.0])
    ax[0].set_title('(a) Noise (mean and range over 2 seeds)', loc='left'); ax[0].grid(axis='y', color=GRID, lw=0.5); ax[0].set_axisbelow(True)
    CF = [('seed42', 'Seed 42\n(main)'), ('seed1', 'Seed 1'), ('seed2', 'Seed 2'), ('seed3', 'Seed 3'), ('seed4', 'Seed 4'), ('seed5', 'Seed 5'), ('family_none', 'All live\nalone'),
          ('family_all', 'All with\nfamily'), ('locerr10', 'Loc. error\n10%'), ('locerr20', 'Loc. error\n20%'), ('actfreq0.6', 'Activity\nfreq. ×0.6'), ('actfreq1.4', 'Activity\nfreq. ×1.4'), ('residents16', '16\nresidents')]
    CF = [c for c in CF if c[0] in set(ST.config)]
    xs = np.arange(len(CF))
    for j, (k, lab, c, mk) in enumerate([('s0', 'CRF, sensors only', GRAY, 'v'), ('full', 'Full MMCFE', CAT[0], 'o'), ('gb', 'Gradient boosting', CAT[1], 's')]):
        v = [ST[ST.config == cc].iloc[0][k] for cc, _ in CF]
        lo_ = [ST[ST.config == cc].iloc[0].get(k + '_lo', np.nan) for cc, _ in CF]; hi_ = [ST[ST.config == cc].iloc[0].get(k + '_hi', np.nan) for cc, _ in CF]
        ax[1].errorbar(xs + (j - 1) * 0.18, v, yerr=[np.array(v) - lo_, np.array(hi_) - v], fmt=mk, color=c, ms=4, elinewidth=0.8, capsize=0, label=lab, markeredgecolor='white', markeredgewidth=0.5)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([l for _, l in CF], fontsize=5.5); ax[1].set_ylabel('Macro-F1 (LORO, 95% CI)'); ax[1].legend(frameon=False, fontsize=6, ncol=3, loc='lower left')
    ax[1].set_title('(b) Independent populations and configurations', loc='left'); ax[1].grid(axis='y', color=GRID, lw=0.5); ax[1].set_axisbelow(True)
    plt.tight_layout(); plt.savefig(f'{FIG}/fig_sensitivity.png', bbox_inches='tight'); plt.close()

if opt('results/realism.json'):
    for k, v in json.load(open('results/realism.json')).items():
        put(k, v)

# ------------------------------------------------------------------ write macros
if 'd-full-gb' in NUM:
    NUM['gb-gap'] = '{:.3f}'.format(-float(NUM['d-full-gb'])); NUM['gb-gap-lo'] = '{:.3f}'.format(-float(NUM['d-full-gb-hi'])); NUM['gb-gap-hi'] = '{:.3f}'.format(-float(NUM['d-full-gb-lo']))
if 'pop-gap-mean' in NUM:
    NUM['pop-gap-abs'] = '{:.3f}'.format(float(NUM['pop-gap-mean']))
if opt('results/aruba_numbers.json'):
    NUM.update(json.load(open('results/aruba_numbers.json')))
if opt('results/har70_numbers.json'):
    NUM.update(json.load(open('results/har70_numbers.json')))
if opt('results/interp_numbers.json'):
    NUM.update({k: v for k, v in json.load(open('results/interp_numbers.json')).items() if isinstance(v, str)})
with open('results/numbers.tex', 'w') as fh:
    fh.write('% AUTO-GENERATED by scripts/analysis.py -- do not edit\n\\makeatletter\n')
    for k, v in sorted(NUM.items()):
        fh.write('\\@namedef{r@%s}{%s}\n' % (k, v.replace('%', '\\%')))
    fh.write('\\makeatother\n\\newcommand{\\R}[1]{\\ifcsname r@#1\\endcsname\\csname r@#1\\endcsname\\else\\textbf{??#1}\\fi}\n')
json.dump(NUM, open('results/numbers.json', 'w'), indent=0, ensure_ascii=False)
print(len(NUM), 'numbers written')
