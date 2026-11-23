"""Fig. 1 (architecture) and Fig. 2 (simulated routines)."""
import sys; import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mmcfe.simulator import simulate, ACTIVITIES
exec(open('scripts/analysis.py').read().split('FIG, TAB')[0].split('from mmcfe.metrics')[0])
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'], 'font.size': 8})
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
ACOL = dict(zip(['Rest', 'TV', 'Cooking', 'Eating', 'Prayer', 'Outdoor', 'Exercise', 'Hygiene'], CAT)); ACOL['Sleep'] = '#3b3b39'
data, meta, res, cfg = simulate(N_RESIDENTS=5, N_DAYS=7)
fig, axes = plt.subplots(2, 1, figsize=(7.16, 2.9), sharex=True)
for ax, rid in zip(axes, ['R01', 'R05']):
    for di in range(7):
        x = data[(data.resident_id == rid) & (data.day == di)].activity.values
        ch = np.r_[0, np.where(x[1:] != x[:-1])[0] + 1, 1440]
        for s0, s1 in zip(ch[:-1], ch[1:]):
            ax.barh(di, s1 - s0, left=s0, color=ACOL[x[s0]], height=0.8, edgecolor='none')
        dt = meta[(meta.resident_id == rid) & (meta.day == di)].day_type.iloc[0]
        ax.text(1446, di, dt, va='center', fontsize=6, color='#444444')
    r = [q for q in res if q['resident_id'] == rid][0]
    ax.set_ylabel(f"{rid} (age {r['age']})\nday", fontsize=7); ax.invert_yaxis(); ax.set_yticks(range(7))
    ax.spines[['top', 'right']].set_visible(False)
axes[1].set_xticks(range(0, 1441, 120)); axes[1].set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)]); axes[1].set_xlim(0, 1440)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ACOL[a]) for a in ACTIVITIES], labels=ACTIVITIES, ncol=9, loc='lower center', frameon=False, fontsize=6.5, handlelength=1)
plt.tight_layout(rect=[0, 0.07, 0.97, 1]); plt.savefig('figures/fig_routines.png', dpi=300, bbox_inches='tight'); plt.close()

# architecture
fig, ax = plt.subplots(figsize=(7.16, 3.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis('off')
INK = '#222222'


def box(x, y, w, h, text, fc, ec='#555555', fs=5.9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, weight='bold' if bold else 'normal', linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#555555', lw=0.8, mutation_scale=7))


mods = ['Wearable\n(HR, accel., steps)', 'Ambient\n(PIR, door)', 'Appliance\n(kitchen & TV plugs)', 'Environment\n(light, temp., humidity)', 'Localisation\n(observed room, noisy)']
for i, t in enumerate(mods):
    box(1, 34 - i * 7.2, 15, 5.6, t, '#eef3fb')
box(21, 6, 14, 33.6, 'Windowing\n15-min windows\n\n21 sensor features\n+ missingness\nindicators\n\nfold-wise median\nimputation and\nstandardisation\n(training folds only)', '#f5f5f2')
for i in range(5):
    arrow(16.3, 36.8 - i * 7.2, 20.7, 36.8 - i * 7.2)
box(40, 29, 17, 9.6, 'Stage 1 – Temporal\nsin/cos of clock time\ntime-of-day period', '#e8f6f0', bold=False)
box(40, 17.2, 17, 9.6, 'Stage 2 – Spatial\nroom fractions +\nprior p(activity | room)\nfitted on training residents', '#fdf0e6')
box(40, 5.4, 17, 9.6, 'Stage 3 – Semantic\nlinear-chain CRF over 96\nwindows; time×room and\nneighbour differences', '#efedfb')
for yy in [33.8, 22, 10.2]:
    arrow(35.3, yy, 39.7, yy)
arrow(48.5, 28.7, 48.5, 27.1); arrow(48.5, 16.9, 48.5, 15.3)
box(62, 17.2, 16, 21.4, 'Outputs per window\n\npredicted activity\nCRF marginal\nconfidence\n\nretrospective (full-day)\nor causal (online)\ndecoding', '#f5f5f2')
arrow(57.3, 10.2, 61.7, 22)
box(62, 5.4, 16, 9.6, 'Confidence gate\nτ chosen on training\nresidents only', '#fdf7e3')
arrow(70, 16.9, 70, 15.3)
box(82, 17.2, 17, 21.4, 'Daily Digital Twin\n(structured profile)\n\nminutes per activity\nand period, room\noccupancy, mean HR,\nwearable coverage,\nconfidence', '#eef3fb')
box(82, 5.4, 17, 9.6, 'Downstream personalised\nanomaly detection\n(alerts / deferral\nfor review)', '#fdeceb')
arrow(78.3, 27.9, 81.7, 27.9); arrow(78.3, 10.2, 81.7, 10.2); arrow(90.5, 16.9, 90.5, 15.3)
ax.text(1, 0.8, 'Fitted on training data only: imputation, scaling, room–activity matrix, CRF weights, confidence threshold. Ground-truth room and label purity never enter the features (automated audit).',
        fontsize=5.8, color='#444444')
plt.savefig('figures/fig_architecture.png', dpi=300, bbox_inches='tight'); plt.close()
print('ok')
