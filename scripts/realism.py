"""Summary statistics of the main simulated dataset for comparison with published reference values."""
import sys, json; import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
from mmcfe.simulator import simulate
d, meta, res, cfg = simulate()
g = d.groupby(['resident_id', 'day'])
worn = g.heart_rate.apply(lambda s: s.notna().mean() > 0.8)
steps = g.step_count.sum()[worn]
night = d[(d.activity == 'Sleep')].groupby(['resident_id', 'day']).size() / 60
nap = d[(d.activity == 'Rest') & (d.room_true == 'Bedroom') & d.minute.between(720, 1080)].groupby(['resident_id', 'day']).size().reindex(night.index, fill_value=0) / 60
prayer = d[d.activity == 'Prayer'].groupby(['resident_id', 'day']).size().reindex(night.index, fill_value=0)
out = {'sim-steps-med': f'{steps.median():,.0f}', 'sim-steps-p10': f'{steps.quantile(.1):,.0f}', 'sim-steps-p90': f'{steps.quantile(.9):,.0f}',
       'sim-sleep-h': f'{night.mean():.1f}', 'sim-nap-h': f'{nap.mean():.1f}', 'sim-prayer': f'{prayer.mean():.0f}',
       'sim-cadence': f"{d[(d.activity == 'Outdoor') & (d.step_count > 0)].step_count.median():.0f}",
       'sim-hr-sleep': f"{d[d.activity == 'Sleep'].heart_rate.mean():.0f}", 'sim-hr-sed': f"{d[d.activity.isin(['Rest', 'TV', 'Eating', 'Prayer'])].heart_rate.mean():.0f}"}
json.dump(out, open('results/realism.json', 'w'), indent=1); print(out)
