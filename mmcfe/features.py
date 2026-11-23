"""Windowing (15-min, non-overlapping) and the 21 sensor features of Section III-B."""
import numpy as np
import pandas as pd
from .simulator import ACTIVITIES, ROOMS

W = 15
NW = 96
SENSOR_FEATS = ['hr_mean', 'hr_sd', 'hr_min', 'hr_max', 'amag_mean', 'amag_sd', 'az_mean',
                'steps_sum', 'steps_max', 'wear_frac',
                'motion_rate', 'door_events', 'plug_k', 'plug_tv', 'light_mean', 'light_max',
                'hum_mean', 'hum_change', 'room_temp', 'home_temp', 'temp_diff']
WEAR_FEATS = SENSOR_FEATS[:10]
ROOM_FEATS = ['loc_' + r for r in ROOMS]
TOD = ['Night', 'Morning', 'Afternoon', 'Evening']


def _nan_stat(x, fn):
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return fn(x, axis=-1)


def _first_last_diff(x):
    """Last valid minus first valid value within each window (NaN if <2 valid)."""
    valid = ~np.isnan(x)
    n = valid.sum(-1)
    idx = np.arange(x.shape[-1])
    first = np.where(valid, idx, 10 ** 6).min(-1)
    last = np.where(valid, idx, -1).max(-1)
    fv = np.take_along_axis(x, np.clip(first, 0, x.shape[-1] - 1)[..., None], -1)[..., 0]
    lv = np.take_along_axis(x, np.clip(last, 0, x.shape[-1] - 1)[..., None], -1)[..., 0]
    out = lv - fv
    out[n < 2] = np.nan
    return out


def make_windows(data):
    """Minute data (resident-days of 1440 rows, in order) -> window table (one row per 15-min window)."""
    data = data.sort_values(['resident_id', 'day', 'minute'], kind='stable')
    keys = data[['resident_id', 'day']].drop_duplicates().reset_index(drop=True)
    nd = len(keys)
    shp = (nd, NW, W)

    def col(c):
        return data[c].to_numpy(dtype=float).reshape(shp).copy()

    hr = col('heart_rate')
    ax, ay, az = col('accel_x'), col('accel_y'), col('accel_z')
    amag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    st = col('step_count')
    worn = ~np.isnan(hr)
    out = {}
    out['hr_mean'] = _nan_stat(hr, np.nanmean)
    out['hr_sd'] = _nan_stat(hr, np.nanstd)
    out['hr_min'] = _nan_stat(hr, np.nanmin)
    out['hr_max'] = _nan_stat(hr, np.nanmax)
    out['amag_mean'] = _nan_stat(amag, np.nanmean)
    out['amag_sd'] = _nan_stat(amag, np.nanstd)
    out['az_mean'] = _nan_stat(az, np.nanmean)
    ss = np.nansum(st, -1); ss[worn.sum(-1) == 0] = np.nan
    out['steps_sum'] = ss
    out['steps_max'] = _nan_stat(st, np.nanmax)
    out['wear_frac'] = worn.mean(-1)
    out['motion_rate'] = _nan_stat(col('motion_detected'), np.nanmean)
    dr = col('door_open'); ds = np.nansum(dr, -1); ds[np.isnan(dr).all(-1)] = np.nan
    out['door_events'] = ds
    # kitchen / TV plugs: the per-plug channels inherit the missingness of the combined plug channel
    pmiss = np.isnan(col('smart_plug_on'))
    pk = col('plug_kitchen'); pk[pmiss] = np.nan
    pt = col('plug_tv'); pt[pmiss] = np.nan
    out['plug_k'] = _nan_stat(pk, np.nanmean)
    out['plug_tv'] = _nan_stat(pt, np.nanmean)
    li = col('ambient_light')
    out['light_mean'] = _nan_stat(li, np.nanmean)
    out['light_max'] = _nan_stat(li, np.nanmax)
    hu = col('humidity')
    out['hum_mean'] = _nan_stat(hu, np.nanmean)
    out['hum_change'] = _first_last_diff(hu)
    out['room_temp'] = _nan_stat(col('temperature_sensor'), np.nanmean)
    out['home_temp'] = _nan_stat(col('env_temperature'), np.nanmean)
    out['temp_diff'] = out['room_temp'] - out['home_temp']
    loc = data['room_observed'].to_numpy(dtype=object).reshape(shp)
    for r in ROOMS:
        out['loc_' + r] = (loc == r).mean(-1)
    loc_obs = np.array([[pd.notna(v) for v in row] for row in loc.reshape(-1, W)]).reshape(shp)
    out['loc_frac'] = loc_obs.mean(-1)
    fr = np.stack([out['loc_' + r] for r in ROOMS], -1)
    dom = np.array(ROOMS, dtype=object)[fr.argmax(-1)]
    dom[fr.sum(-1) == 0] = 'unknown'
    # labels
    act = data['activity'].to_numpy(dtype=object).reshape(shp)
    alpha = sorted(ACTIVITIES)                                      # pandas mode() breaks ties alphabetically
    codes = np.stack([(act == a).sum(-1) for a in alpha], -1)
    lab = np.array(alpha, dtype=object)[codes.argmax(-1)]
    purity = codes.max(-1) / W
    # minute counts of (observed room, activity) per window -> used to fit the Stage-2 matrix on training minutes only
    for r in ROOMS:
        for a in ACTIVITIES:
            out[f'n|{r}|{a}'] = ((loc == r) & (act == a)).sum(-1).astype(float)
    rows = {k: v.reshape(-1) for k, v in out.items()}
    tab = pd.DataFrame(rows)
    tab.insert(0, 'w', np.tile(np.arange(NW), nd))
    tab.insert(0, 'day', np.repeat(keys.day.to_numpy(), NW))
    tab.insert(0, 'resident_id', np.repeat(keys.resident_id.to_numpy(), NW))
    tab['dom_room'] = dom.reshape(-1)
    tab['label'] = lab.reshape(-1)
    tab['purity'] = purity.reshape(-1)
    s = tab.w * W
    tab['sin_t'] = np.sin(2 * np.pi * s / 1440)
    tab['cos_t'] = np.cos(2 * np.pi * s / 1440)
    tab['tod'] = np.array(TOD)[(s // 360).to_numpy()]
    for t in TOD:
        tab['tod_' + t] = (tab.tod == t).astype(float)
    tab['hour'] = s // 60
    return tab


TEMPORAL_FEATS = ['sin_t', 'cos_t'] + ['tod_' + t for t in TOD]
PRIOR_FEATS = ['prior_' + a for a in ACTIVITIES]
NEIGH_VARS = ['hr_mean', 'light_mean', 'motion_rate', 'steps_sum', 'plug_k', 'plug_tv'] + ROOM_FEATS
FORBIDDEN = {'label', 'purity', 'room_true', 'activity'}


def leakage_audit(cols):
    bad = [c for c in cols if c in FORBIDDEN or 'true' in c or 'purity' in c or c == 'label']
    if bad:
        raise AssertionError(f'Leakage audit failed: {bad}')
