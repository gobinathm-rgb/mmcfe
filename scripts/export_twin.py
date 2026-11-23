"""Export the Digital Twin (Section III-F) from out-of-fold LORO predictions of the full MMCFE model."""
import os, sys, json, pickle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
from mmcfe.simulator import ACTIVITIES

W, out, _ = pickle.load(open('results/oof_LORO.pkl', 'rb'))
o = out['MMCFE full (S1+S2+S3 compound)']
os.makedirs('twin', exist_ok=True)
win = pd.DataFrame({'resident_id': W.resident_id, 'day': W.day, 'window': W.w,
                    'start': [f'{(15 * w) // 60:02d}:{(15 * w) % 60:02d}' for w in W.w], 'time_of_day': W.tod,
                    'predicted_activity': o.pred.values, 'confidence': o.conf.round(4).values,
                    'dominant_observed_room': W.dom_room.values, 'hr_mean': W.hr_mean.round(1).values, 'wearable_coverage': W.wear_frac.round(3).values})
win.to_csv('twin/digital_twin_windows.csv.gz', index=False)
days = []
for (r, d), g in win.groupby(['resident_id', 'day']):
    days.append(dict(resident_id=r, day=int(d), mean_confidence=round(float(g.confidence.mean()), 4),
                     minutes_per_activity={a: int(15 * (g.predicted_activity == a).sum()) for a in ACTIVITIES},
                     minutes_per_activity_by_period={t: {a: int(15 * ((g.time_of_day == t) & (g.predicted_activity == a)).sum()) for a in ACTIVITIES}
                                                     for t in ['Night', 'Morning', 'Afternoon', 'Evening']},
                     room_minutes_by_period={t: {k: int(15 * v) for k, v in g[g.time_of_day == t].dominant_observed_room.value_counts().items()}
                                             for t in ['Night', 'Morning', 'Afternoon', 'Evening']}))
json.dump(days, open('twin/digital_twin_days.json', 'w'), indent=1)
print(len(win), 'windows,', len(days), 'resident-days')
