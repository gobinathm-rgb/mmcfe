"""Statistical simulator of Indian senior routines (v2).

Faithful port of Phase0_v2_Data_Engineering.ipynb. With default settings the
random-number stream, and therefore every generated value, is identical to the
dataset used in the manuscript. Extra keyword arguments (all defaulting to the
original behaviour) support the sensitivity analyses requested in review.
"""
import numpy as np
import pandas as pd

ACTIVITIES = ['Sleep', 'Prayer', 'Hygiene', 'Outdoor', 'Exercise', 'Cooking', 'Eating', 'Rest', 'TV']
ROOMS = ['Bedroom', 'Living Room', 'Kitchen', 'Bathroom', 'Balcony']
ADJACENT = {'Bedroom': ['Living Room', 'Bathroom'], 'Living Room': ['Bedroom', 'Kitchen', 'Balcony'],
            'Kitchen': ['Living Room'], 'Bathroom': ['Bedroom'], 'Balcony': ['Living Room']}

PARAMS = {
    'Sleep':    dict(I=0.0, steps=0.0,  acc=0.02, pm=0.06, up=False),
    'Rest':     dict(I=0.2, steps=0.3,  acc=0.04, pm=0.12, up=False),
    'TV':       dict(I=0.3, steps=0.6,  acc=0.06, pm=0.25, up=True),
    'Prayer':   dict(I=0.6, steps=1.5,  acc=0.10, pm=0.35, up=True),
    'Eating':   dict(I=0.7, steps=1.5,  acc=0.12, pm=0.45, up=True),
    'Hygiene':  dict(I=1.3, steps=6.0,  acc=0.20, pm=0.70, up=True),
    'Cooking':  dict(I=1.5, steps=10.0, acc=0.22, pm=0.75, up=True),
    'Outdoor':  dict(I=2.3, steps=40.0, acc=0.35, pm=0.50, up=True),
    'Exercise': dict(I=2.6, steps=25.0, acc=0.40, pm=0.80, up=True),
}
DAY_TYPE_P = dict(normal=0.70, fasting=0.07, festival=0.05, visitor=0.13, unwell=0.05)

DEFAULT_CONFIG = dict(
    SEED=42, N_RESIDENTS=8, N_DAYS=30, START_DATE='2026-04-05',
    P_FAMILY=0.6,       # share of residents living with family
    NOISE=1.0,          # behavioural variability / class overlap
    LOC_ERR=0.03,       # share of minutes the localisation sensor reports an adjacent room
    LOC_MISS=0.01,      # share of minutes with no localisation reading
    ACT_FREQ=1.0,       # multiplier on the inclusion probability of optional activity blocks
    FORCE_FAMILY=None,  # None: sample; True/False: force every resident's household type
    ANOMALIES=None,     # {(resident_index, day): type} behavioural anomalies injected for downstream validation
)

ANOMALY_TYPES = ['missed_meals', 'prolonged_inactivity', 'night_wandering', 'skipped_morning_routine']


def inject_anomaly(A, R, kind, rng):
    """Overwrite the sampled schedule with a clinically motivated deviation (before signals are generated)."""
    A, R = A.copy(), R.copy(); m = np.arange(1440)
    if kind == 'missed_meals':                       # no lunch/tea/dinner eating or cooking
        sel = np.isin(A, ['Eating', 'Cooking']) & (m >= 660) & (m < 1290)
        A[sel] = 'Rest'; R[sel] = 'Living Room'
    elif kind == 'prolonged_inactivity':             # stays in bed through the day
        sel = (m >= 540) & (m < 1080) & ~np.isin(A, ['Hygiene'])
        A[sel] = 'Rest'; R[sel] = 'Bedroom'
    elif kind == 'night_wandering':                  # awake and moving between rooms at night
        s0 = int(rng.integers(60, 180)); L = int(rng.integers(60, 120))
        for t in range(s0, s0 + L, 15):
            A[t:t + 15] = 'TV' if rng.random() < 0.5 else 'Rest'
            R[t:t + 15] = 'Living Room' if rng.random() < 0.7 else 'Kitchen'
    elif kind == 'skipped_morning_routine':          # no morning bath or prayer
        sel = np.isin(A, ['Hygiene', 'Prayer']) & (m >= 240) & (m < 600)
        A[sel] = 'Rest'; R[sel] = 'Bedroom'
    return A, R


def make_resident(rid, rng, cfg):
    age = int(rng.integers(60, 86))
    res = dict(
        resident_id=rid, age=age,
        hr_rest=float(rng.normal(68, 6)),
        hr_gain=float(rng.uniform(7, 13)),
        mobility=float(np.clip(rng.normal(1.0 - (age - 60) * 0.012, 0.12), 0.45, 1.2)),
        wake=float(rng.normal(345, 25)), bed=float(rng.normal(1320, 30)),
        p_prayer_m=float(rng.uniform(0.6, 0.98)), p_prayer_e=float(rng.uniform(0.4, 0.9)),
        prayer_room=str(rng.choice(['Bedroom', 'Living Room'])),
        bath_before_prayer=bool(rng.random() < 0.6),
        p_cooks=float(rng.uniform(0.25, 0.95)),
        p_induction=float(rng.uniform(0.3, 0.9)),
        eats_in_living=float(rng.uniform(0.1, 0.6)),
        tv_in_bedroom=float(rng.uniform(0.0, 0.4)),
        p_walk=float(rng.uniform(0.3, 0.85)), p_exercise=float(rng.uniform(0.2, 0.8)),
        exercise_room=str(rng.choice(['Living Room', 'Balcony'], p=[0.7, 0.3])),
        yoga_style=bool(rng.random() < 0.5),
        p_nap=float(rng.uniform(0.5, 0.95)), tv_lover=float(rng.uniform(0.2, 0.8)),
        night_voids=float(0.3 + (age - 60) * 0.05),
        wear_compliance=float(rng.uniform(0.85, 0.99)),
        lives_with_family=bool(rng.random() < cfg['P_FAMILY']),
    )
    if cfg.get('FORCE_FAMILY') is not None:
        res['lives_with_family'] = bool(cfg['FORCE_FAMILY'])
    return res


def _fill(A, R, t0, t1, res, rng):
    t = t0
    while t < t1:
        d = int(rng.integers(20, 60))
        if rng.random() < res['tv_lover'] * 0.6:
            a, r = 'TV', 'Living Room'
        else:
            a, r = 'Rest', ('Living Room' if rng.random() < 0.75 else 'Bedroom')
        A[t:min(t + d, t1)] = a
        R[t:min(t + d, t1)] = r
        t += d


def sample_day(res, day_type, rng, cfg):
    A = np.array(['Sleep'] * 1440, dtype=object)
    R = np.array(['Bedroom'] * 1440, dtype=object)
    wake = int(np.clip(rng.normal(res['wake'], 20), 270, 450))
    bed = int(np.clip(rng.normal(res['bed'], 25), 1230, 1410))
    blocks = []
    f = cfg['ACT_FREQ']

    def add(p, *b, optional=True):
        p = min(1.0, p * f) if optional else p
        if rng.random() < p:
            blocks.append(b)

    unwell, fasting, festival = day_type == 'unwell', day_type == 'fasting', day_type == 'festival'
    pr_mult = 2.0 if festival else (1.4 if fasting else 1.0)
    bath = (wake + 5, 5, 22, 7, 'Hygiene', 'Bathroom')
    pray = (wake + 5, 5, 25 * pr_mult, 8, 'Prayer', res['prayer_room'])
    order = [bath, pray] if res['bath_before_prayer'] else [pray, bath]
    for blk in order:
        if blk[4] == 'Prayer':
            add(res['p_prayer_m'], *blk)
        else:
            add(0.95, *blk, optional=False)
    add(0 if unwell else res['p_walk'], wake + 70, 20, 30, 10, 'Outdoor', 'Balcony')
    add(0 if unwell else res['p_exercise'], wake + 100, 25, 35, 12, 'Exercise', res['exercise_room'])
    cook_p = min(1, res['p_cooks'] * (1.3 if festival else 1.0))
    add(cook_p, 500, 20, 30 * (1.5 if festival else 1), 10, 'Cooking', 'Kitchen')
    add(0.95, 545, 20, 20, 6, 'Eating', 'Living Room' if rng.random() < res['eats_in_living'] else 'Kitchen', optional=False)
    add(res['tv_lover'], 640, 30, 50, 20, 'TV', 'Living Room')
    if not fasting:
        add(cook_p, 715, 20, 45 * (1.5 if festival else 1), 12, 'Cooking', 'Kitchen')
        add(0.97, 790, 20, 25, 7, 'Eating', 'Living Room' if rng.random() < res['eats_in_living'] else 'Kitchen', optional=False)
    add(min(1, res['p_nap'] + (0.2 if unwell else 0)), 830, 20, 100 if not unwell else 150, 30, 'Rest', 'Bedroom')
    add(0.8, 990, 15, 15, 5, 'Eating', 'Kitchen' if rng.random() < 0.5 else 'Living Room', optional=False)
    add(0 if unwell else res['p_walk'], 1035, 20, 30, 10, 'Outdoor', 'Balcony')
    add(res['p_prayer_e'] * (1.2 if festival else 1), 1115, 15, 25 * pr_mult, 8, 'Prayer', res['prayer_room'])
    add(res['tv_lover'], 1150, 20, 60, 20, 'TV', 'Bedroom' if rng.random() < res['tv_in_bedroom'] else 'Living Room')
    add(cook_p, 1180, 20, 45 * (1.4 if festival else 1), 12, 'Cooking', 'Kitchen')
    add(0.97, 1245, 20, 25, 7, 'Eating', 'Living Room' if rng.random() < res['eats_in_living'] else 'Kitchen', optional=False)
    s = []
    for sm, ss, dm, ds, a, r in blocks:
        st = int(rng.normal(sm, ss)); du = int(max(5, rng.normal(dm, ds)))
        s.append((st, du, a, r))
    s.sort(key=lambda x: x[0])
    t = wake
    for st, du, a, r in s:
        st = max(st, t)
        if st >= bed:
            break
        if st > t:
            _fill(A, R, t, st, res, rng)
        en = min(st + du, bed); A[st:en] = a; R[st:en] = r; t = en
    if t < bed:
        _fill(A, R, t, bed, res, rng)
    for _ in range(rng.poisson(res['night_voids'])):
        m = int(rng.choice(np.r_[0:wake - 15, bed + 15:1435]))
        d = int(rng.integers(4, 12)); A[m:m + d] = 'Hygiene'; R[m:m + d] = 'Bathroom'
    return A, R


def _bouts(A, R):
    change = np.r_[True, (A[1:] != A[:-1]) | (R[1:] != R[:-1])]
    return np.cumsum(change) - 1


def generate_signals(res, day_idx, day_type, A, R, rng, cfg):
    T = 1440; minute = np.arange(T); hour = minute / 60
    b = _bouts(A, R); nb = b.max() + 1
    NZ = cfg['NOISE']
    I = np.array([PARAMS[a]['I'] for a in A])
    if res['yoga_style']:
        I = np.where(A == 'Exercise', 1.6, I)
    I = np.clip(I + rng.normal(0, 0.35 * NZ, nb)[b], 0, None)
    unwell = day_type == 'unwell'
    target = res['hr_rest'] + res['hr_gain'] * I * res['mobility'] ** 0.5 + (5 if unwell else 0) \
        - 4 * (A == 'Sleep') + rng.normal(0, 2.5, nb)[b]
    hr = np.empty(T); hr[0] = target[0]; e = 0.0
    for t in range(1, T):
        e = 0.8 * e + rng.normal(0, 1.2)
        hr[t] = hr[t - 1] + 0.2 * (target[t] - hr[t - 1]) + e * 0.5
    hr = np.clip(hr, 42, 160).round(1)
    acc_sd = np.array([PARAMS[a]['acc'] for a in A]) * res['mobility'] * rng.lognormal(0, 0.4 * NZ, nb)[b]
    if res['yoga_style']:
        acc_sd = np.where(A == 'Exercise', 0.18 * res['mobility'], acc_sd)
    up = np.array([PARAMS[a]['up'] for a in A])
    tilt = np.where(up[np.unique(b, return_index=True)[1]], rng.uniform(0, 0.5, nb), rng.uniform(0.6, 1.5, nb))
    az_ = rng.uniform(0, 2 * np.pi, nb)
    g = np.c_[np.sin(tilt) * np.cos(az_), np.sin(tilt) * np.sin(az_), np.cos(tilt)][b]
    acc = (g + rng.normal(0, 1, (T, 3)) * acc_sd[:, None]).round(3)
    rate = np.array([PARAMS[a]['steps'] for a in A]) * res['mobility']
    if res['yoga_style']:
        rate = np.where(A == 'Exercise', 3.0, rate)
    steps = rng.poisson(rate * rng.lognormal(0, 0.5 * NZ, nb)[b])
    worn = np.ones(T, bool)
    if rng.random() > res['wear_compliance']:
        worn[:] = False
    if rng.random() < 0.7:
        c0 = int(rng.integers(600, 1200)); worn[c0:c0 + int(rng.integers(40, 90))] = False
    bath = (A == 'Hygiene') & (minute > 300) & (minute < 1380)
    if rng.random() < 0.8:
        worn[bath] = False
    pm = np.clip(np.array([PARAMS[a]['pm'] for a in A]) + rng.normal(0, 0.12 * NZ, nb)[b], 0.02, 0.95)
    vis = day_type in ('visitor', 'festival')
    if vis:
        pm = np.where((R == 'Living Room') & (minute > 600) & (minute < 1260), np.maximum(pm, 0.55), pm)
    fam = res['lives_with_family']
    if fam:
        shared = np.isin(R, ['Living Room', 'Kitchen']) & (minute > 360) & (minute < 1320)
        pm = np.where(shared, np.maximum(pm, rng.uniform(0.25, 0.6)), pm)
    motion = (rng.random(T) < pm).astype(float)
    trans = np.r_[False, R[1:] != R[:-1]]
    door = ((trans & (rng.random(T) < 0.7)) | (rng.random(T) < 0.02)).astype(float)
    plug_k = np.zeros(T); plug_tv = np.zeros(T)
    for k in range(nb):
        idx = np.where(b == k)[0]; a, r = A[idx[0]], R[idx[0]]
        if a == 'Cooking' and rng.random() < res['p_induction']:
            on = idx[int(len(idx) * rng.uniform(0, 0.3)): int(len(idx) * rng.uniform(0.6, 1.0))]
            plug_k[on] = 1
        if a == 'TV' and rng.random() < 0.97:
            plug_tv[idx] = 1
        if a in ('Eating', 'Rest') and r == 'Living Room' and rng.random() < (0.6 if fam else 0.3):
            plug_tv[idx] = 1
    if vis or fam:
        ev0 = int(rng.integers(1080, 1170)); plug_tv[ev0:ev0 + int(rng.integers(30, 120))] = 1
    if fam:
        for c0 in (int(rng.normal(500, 20)), int(rng.normal(715, 20)), int(rng.normal(1180, 20))):
            if rng.random() < 0.5 and not (A[c0:c0 + 30] == 'Cooking').any():
                plug_k[c0:c0 + int(rng.integers(15, 45))] = 1
    plug = np.maximum(plug_k, plug_tv)
    doy = day_idx
    cloud = rng.uniform(0.4, 1.0)
    sunrise, sunset = 6.0 + rng.normal(0, 0.1), 18.4 + rng.normal(0, 0.1)
    day = np.clip(np.sin(np.pi * (hour - sunrise) / (sunset - sunrise)), 0, None)
    room_f = np.array([{'Balcony': 2.2, 'Living Room': 1.0, 'Kitchen': 0.8, 'Bedroom': 0.55, 'Bathroom': 0.4}[r] for r in R])
    curtains = np.where((A == 'Rest') & (R == 'Bedroom'), 0.35, 1.0)
    lamp = np.where(np.isin(A, ['Sleep']), rng.uniform(2, 15, T),
                    np.where(A == 'Rest', rng.uniform(10, 120, T), rng.uniform(90, 250, T)))
    light = np.maximum(500 * cloud * day * room_f * curtains, lamp * (day < 0.3)) + rng.normal(0, 15, T)
    light = np.clip(light, 0, None).round(1)
    t_base = 29 + 0.04 * doy + rng.normal(0, 1.0) + 3 * np.sin(np.pi * (hour - 9) / 12)
    env_temp = (t_base + rng.normal(0, 0.3, T)).round(2)
    room_temp = env_temp + np.where((A == 'Cooking') & (R == 'Kitchen'), 1.5, 0) \
        + np.where(R == 'Bedroom', -0.8, 0) + rng.normal(0, 0.4, T)
    hum = 60 + 0.1 * doy + rng.normal(0, 3) - 6 * np.sin(np.pi * (hour - 9) / 12) + rng.normal(0, 1.2, T)
    spike = np.zeros(T)
    for t in np.where(bath)[0]:
        spike[t:t + 60] += 0.4 * np.exp(-np.arange(min(60, T - t)) / 20)
    hum = np.clip(hum + np.minimum(spike, 12) + 3 * ((A == 'Cooking')), 30, 95).round(1)
    loc = R.copy()
    for t in np.where(trans)[0]:
        lag = int(rng.integers(0, 3)); loc[t:t + lag] = R[t - 1]
    err = rng.random(T) < cfg['LOC_ERR']
    loc[err] = [rng.choice(ADJACENT[r]) for r in R[err]]
    loc = loc.astype(object); loc[rng.random(T) < cfg['LOC_MISS']] = np.nan
    out = dict(heart_rate=hr, accel_x=acc[:, 0], accel_y=acc[:, 1], accel_z=acc[:, 2], step_count=steps.astype(float),
               motion_detected=motion, door_open=door, temperature_sensor=room_temp.round(2),
               smart_plug_on=plug, plug_kitchen=plug_k, plug_tv=plug_tv,
               ambient_light=light, env_temperature=env_temp, humidity=hum, room_observed=loc)
    for c in ['heart_rate', 'accel_x', 'accel_y', 'accel_z', 'step_count']:
        out[c] = np.where(worn, out[c], np.nan)
    for c in ['motion_detected', 'door_open', 'temperature_sensor', 'smart_plug_on', 'ambient_light', 'env_temperature', 'humidity']:
        x = out[c].astype(float); x[rng.random(T) < 0.01] = np.nan
        if rng.random() < 0.05:
            s0 = int(rng.integers(0, 1300)); x[s0:s0 + int(rng.integers(30, 120))] = np.nan
        out[c] = x
    return out


def simulate(**overrides):
    """Generate the full minute-level dataset. Returns (data, day_meta, residents)."""
    cfg = dict(DEFAULT_CONFIG, **overrides)
    residents = [make_resident(f'R{i + 1:02d}', np.random.default_rng([cfg['SEED'], i]), cfg)
                 for i in range(cfg['N_RESIDENTS'])]
    dates = pd.date_range(cfg['START_DATE'], periods=cfg['N_DAYS'], freq='D')
    rows, meta = [], []
    for ri, res in enumerate(residents):
        for di, date in enumerate(dates):
            rng = np.random.default_rng([cfg['SEED'], ri, di, 7]); an = None
            dtype = 'visitor' if (date.dayofweek == 6 and rng.random() < 0.5) else \
                str(rng.choice(list(DAY_TYPE_P), p=list(DAY_TYPE_P.values())))
            A, R = sample_day(res, dtype, rng, cfg)
            an = (cfg['ANOMALIES'] or {}).get((ri, di))
            if an:
                A, R = inject_anomaly(A, R, an, np.random.default_rng([cfg['SEED'], ri, di, 99]))
            sig = generate_signals(res, di, dtype, A, R, rng, cfg)
            df = pd.DataFrame(sig)
            df.insert(0, 'minute', np.arange(1440))
            df.insert(0, 'day', di); df.insert(0, 'resident_id', res['resident_id'])
            df['activity'] = A; df['room_true'] = R
            rows.append(df)
            meta.append(dict(resident_id=res['resident_id'], day=di, date=str(date.date()), day_type=dtype, anomaly=an or ''))
    data = pd.concat(rows, ignore_index=True)
    return data, pd.DataFrame(meta), residents, cfg


FILES = {
    'sensor_data.csv': ['motion_detected', 'door_open', 'temperature_sensor', 'smart_plug_on', 'plug_kitchen', 'plug_tv'],
    'wearable_data.csv': ['heart_rate', 'accel_x', 'accel_y', 'accel_z', 'step_count'],
    'environment_data.csv': ['ambient_light', 'env_temperature', 'humidity'],
    'location_map.csv': ['room_observed'],
    'activity_labels.csv': ['activity', 'room_true'],
}


def save_csv(data, meta, residents, cfg, out_dir):
    """Write the five modality files exactly as in the original notebook."""
    import json
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = data.copy()
    start = pd.Timestamp(cfg['START_DATE'])
    d.insert(2, 'timestamp', (start + pd.to_timedelta(d.day, 'D') + pd.to_timedelta(d.minute, 'min')).dt.strftime('%Y-%m-%d %H:%M'))
    key = ['resident_id', 'day', 'timestamp']
    for f, cols in FILES.items():
        d[key + cols].to_csv(out / f, index=False)
    meta.to_csv(out / 'day_metadata.csv', index=False)
    json.dump(dict(config=cfg, params=PARAMS, residents=residents), open(out / 'residents.json', 'w'), indent=2)
