"""Figure: what the CRF exposes to a reviewer, caregiver or engineer (R1.2, R3.4).
(a) learned room-activity matrix M, (b) learned transition weights, (c) the evidence behind one held-out prediction."""
import os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mmcfe.simulator import ACTIVITIES, ROOMS
from mmcfe.pipeline import FoldPreprocessor, feature_sets, CRFModel, _seq_dicts

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'], 'font.size': 7, 'axes.titlesize': 7.5,
                     'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.spines.top': False, 'axes.spines.right': False})
INK, GRAY = '#222222', '#8a8a85'; POS, NEG = '#2a78d6', '#e34948'
W = pd.read_pickle('data/windows_default.pkl').sort_values(['resident_id', 'day', 'w']).reset_index(drop=True)
HELD = 'R02'
tr, te = W[W.resident_id != HELD].reset_index(drop=True), W[W.resident_id == HELD].reset_index(drop=True)
pp = FoldPreprocessor().fit(tr); Ptr, Pte = pp.transform(tr).reset_index(drop=True), pp.transform(te).reset_index(drop=True)
cols = feature_sets()['S2']
m = CRFModel(cols, compound=True).fit(Ptr, tr.label.values)
pred, conf, prob = m.predict(Pte)

# choose a correct, confident Eating window at lunch time
cand = np.where((pred == 'Eating') & (te.label.values == 'Eating') & (te.w.between(50, 56)) & (conf > 0.8) & (conf < 0.97))[0]
j = int(cand[len(cand) // 2]); day = te.day[j]; wi = te.w[j]
seqs, idx = _seq_dicts(Pte, cols, True, False)
s = [k for k, ix in enumerate(idx) if j in ix][0]; feats = seqs[s][list(idx[s]).index(j)]
SF = m.crf.state_features_
def score(lab):
    return {a: v * SF.get((a, lab), 0.0) for a, v in feats.items() if SF.get((a, lab), 0.0) != 0 and v != 0}
ce, alt = score('Eating'), None
second = sorted(ACTIVITIES, key=lambda a: -prob[j][ACTIVITIES.index(a)])[1]
ca = score(second)
diff = {a: ce.get(a, 0) - ca.get(a, 0) for a in set(ce) | set(ca)}
NAME = {'z_hr_mean': 'Heart rate (mean)', 'z_amag_sd': 'Wrist movement (SD)', 'z_az_mean': 'Wrist orientation', 'z_steps_sum': 'Steps',
        'z_motion_rate': 'PIR motion rate', 'z_plug_k': 'Kitchen plug', 'z_plug_tv': 'TV plug', 'z_light_mean': 'Light level', 'z_light_max': 'Light level (max)', 'z_amag_mean': 'Wrist movement (mean)',
        'z_temp_diff': 'Room–home temp. diff.', 'z_room_temp': 'Room temperature', 'z_hum_mean': 'Humidity', 'cos_t': 'Time of day (cos)',
        'sin_t': 'Time of day (sin)', 'bias': 'Bias'}
def nice(a):
    if a.startswith('todroom='): t, r = a[8:].split('|'); return f'{t} × {r}'
    if a.startswith('loc_'): return f'Observed room: {a[4:]}'
    if a.startswith('prior_'): return f'Room prior for {a[6:]}'
    if a.startswith('tod_'): return f'Period: {a[4:]}'
    if a.startswith('dprev_') or a.startswith('dnext_'): return ('Δ prev. ' if a.startswith('dprev') else 'Δ next ') + NAME.get('z_' + a[6:], a[6:].replace('loc_', 'room: '))
    if a.startswith('m_'): return 'Missing: ' + NAME.get('z_' + a[2:], a[2:])
    return NAME.get(a, a.replace('z_', ''))
top = sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:9][::-1]

fig = plt.figure(figsize=(7.16, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.0], wspace=1.05)
# (a) room-activity matrix
ax = fig.add_subplot(gs[0]); M = pp.M
im = ax.imshow(M, cmap='Blues', vmin=0, vmax=M.max(), aspect='auto')
ax.set_yticks(range(len(ROOMS))); ax.set_yticklabels(ROOMS); ax.set_xticks(range(9)); ax.set_xticklabels(ACTIVITIES, rotation=60, ha='right')
for r in range(M.shape[0]):
    for a in range(M.shape[1]):
        if M[r, a] >= 0.10:
            ax.text(a, r, f'{M[r, a]:.2f}'.lstrip('0'), ha='center', va='center', fontsize=5, color='white' if M[r, a] > 0.5 * M.max() else INK)
ax.set_title('(a) Room–activity matrix $\\mathbf{M}$', loc='left')
# (b) transition weights
ax = fig.add_subplot(gs[1]); T = np.zeros((9, 9))
for (a, b), w in m.crf.transition_features_.items():
    T[ACTIVITIES.index(a), ACTIVITIES.index(b)] = w
lim = np.abs(T).max(); im = ax.imshow(T, cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
ax.set_xticks(range(9)); ax.set_xticklabels(ACTIVITIES, rotation=60, ha='right'); ax.set_yticks(range(9)); ax.set_yticklabels(ACTIVITIES)
ax.set_xlabel('to activity', labelpad=1); ax.set_ylabel('from activity', labelpad=1)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=5.5); cb.set_label('weight', fontsize=5.5, labelpad=1)
ax.set_title('(b) Learned transition weights', loc='left')
# (c) evidence for one prediction
ax = fig.add_subplot(gs[2])
v = [d for _, d in top]; ax.barh(range(len(top)), v, color=[POS if x > 0 else NEG for x in v], height=0.62)
ax.set_yticks(range(len(top))); ax.set_yticklabels([nice(a) for a, _ in top]); ax.axvline(0, color=INK, lw=0.6)
ax.set_xlabel(f'contribution: Eating minus {second}', labelpad=1)
hh, mm = divmod(int(wi) * 15, 60)
ax.set_title(f'(c) Why "Eating" at {hh:02d}:{mm:02d}, not "{second}"\n     (R02, day {day}; confidence {conf[j]:.2f})', loc='left', pad=8)
ax.grid(axis='x', color='#e6e6e3', lw=0.5); ax.set_axisbelow(True)
plt.savefig('figures/fig_interpretability.png', dpi=300, bbox_inches='tight'); plt.close()
json.dump({'interp-time': f'{hh:02d}:{mm:02d}', 'interp-day': str(int(day)), 'interp-second': second, 'interp-conf': f'{conf[j]:.2f}',
           'interp-top': [nice(a) for a, _ in top[::-1][:3]]}, open('results/interp_numbers.json', 'w'))
print('saved', hh, mm, day, second, conf[j], [nice(a) for a, _ in top[::-1]])
