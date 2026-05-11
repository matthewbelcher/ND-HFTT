import glob
import os
import pandas as pd
import numpy as np

# Fields
ETF_TICKER  = 'VOO'
DATE_STR    = '20240717'
DATA_DIR    = f'data_v2/{DATE_STR}'
OUT_DIR     = f'data_v2_rescaled/{DATE_STR}'

files = sorted(glob.glob(f'{DATA_DIR}/{ETF_TICKER}_{DATE_STR}_*.parquet'))
if not files:
    raise FileNotFoundError(f'No parquet files found in {DATA_DIR}')

COLS       = ['time_m', 'etf_bid', 'etf_ask', 'etf_mid', 'synth_bid', 'synth_ask', 'synth_mid']
SYNTH_COLS = ['synth_bid', 'synth_ask', 'synth_mid']

PLOT_AFTER_STR  = '093500'
PLOT_AFTER_TIME = f'{PLOT_AFTER_STR[:2]}:{PLOT_AFTER_STR[2:4]}:{PLOT_AFTER_STR[4:6]}'
PLOT_UNTIL_STR  = '160000'
PLOT_UNTIL_TIME = f'{PLOT_UNTIL_STR[:2]}:{PLOT_UNTIL_STR[2:4]}:{PLOT_UNTIL_STR[4:6]}'

DATE       = f'{DATE_STR[:4]}-{DATE_STR[4:6]}-{DATE_STR[6:]}'
PLOT_AFTER = pd.Timestamp(f'{DATE} {PLOT_AFTER_TIME}')
PLOT_UNTIL = pd.Timestamp(f'{DATE} {PLOT_UNTIL_TIME}')

# Rescale after spread persists past threshold for this amount of time 
RESCALE_THRESHOLD_S = 30.0
# Spread band
RESCALE_THRESHOLD_SPREAD = 0.05

os.makedirs(OUT_DIR, exist_ok=True)

# Pass 1: accumulate mid prices
print('Pass 1')
chunks = []
for f in files:
    df = pd.read_parquet(f, columns=['time_m', 'etf_mid', 'synth_mid']).sort_values('time_m').reset_index(drop=True)
    df['time_m'] = pd.to_datetime(df['time_m'])
    if PLOT_AFTER is not None:
        df = df[df['time_m'] >= PLOT_AFTER]
    if PLOT_UNTIL is not None:
        df = df[df['time_m'] <= PLOT_UNTIL]
    if df.empty:
        continue
    chunks.append(df)

mids = pd.concat(chunks, ignore_index=True).sort_values('time_m').reset_index(drop=True)
print(f'  {len(mids):,} ticks')

# Pass 2: compute scale factors at points breaching threshold
print('Pass 2')
first_valid = mids.dropna(subset=['etf_mid', 'synth_mid']).iloc[0]
factor        = first_valid['etf_mid'] / first_valid['synth_mid'] if first_valid['synth_mid'] != 0 else 1.0
last_zero_t   = mids['time_m'].iloc[0]
factors       = np.ones(len(mids))
rescale_log   = [{'time': first_valid['time_m'], 'factor': factor}]
print(f'  Initial rescale at {first_valid["time_m"]}: factor={factor:.6f}')

for i, row in mids.iterrows():
    if pd.isna(row['etf_mid']) or pd.isna(row['synth_mid']):
        factors[i] = factor
        continue

    spread_corrected = row['etf_mid'] - row['synth_mid'] * factor

    if abs(spread_corrected) <= RESCALE_THRESHOLD_SPREAD:
        last_zero_t = row['time_m']

    time_since_zero = (row['time_m'] - last_zero_t).total_seconds()
    if time_since_zero > RESCALE_THRESHOLD_S and row['synth_mid'] != 0:
        factor = row['etf_mid'] / row['synth_mid']
        last_zero_t = row['time_m']
        rescale_log.append({'time': row['time_m'], 'factor': factor})
        print(f'  Rescaled at {row["time_m"]}: factor={factor:.6f}')

    factors[i] = factor

mids['factor'] = factors
print(f'  {len(rescale_log)} rescale events')

# Pass 3: apply scale factor and save
print('Pass 3')
for f in files:
    df = pd.read_parquet(f, columns=COLS).sort_values('time_m').reset_index(drop=True)
    df['time_m'] = pd.to_datetime(df['time_m'])
    if PLOT_AFTER is not None:
        df = df[df['time_m'] >= PLOT_AFTER]
    if PLOT_UNTIL is not None:
        df = df[df['time_m'] <= PLOT_UNTIL]
    if df.empty:
        continue

    df = pd.merge_asof(
        df.sort_values('time_m'),
        mids[['time_m', 'factor']].sort_values('time_m'),
        on='time_m',
        direction='backward'
    )
    df['factor'] = df['factor'].fillna(1.0)
    df[SYNTH_COLS] = df[SYNTH_COLS].mul(df['factor'], axis=0)

    out = df.drop(columns='factor')
    fname = os.path.basename(f)
    out.to_parquet(f'{OUT_DIR}/{fname}')

print(f'Saved {OUT_DIR}/')
print(f'Rescale events:')
for e in rescale_log:
    print(f'{e["time"]} | factor={e["factor"]:.6f}')
