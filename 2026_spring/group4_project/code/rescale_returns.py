import glob
import os
import pandas as pd

# Fields
ETF_TICKER  = 'SPXL'
DATE_STR    = '20240717'
DATA_DIR    = f'data_v2/{DATE_STR}'
OUT_DIR     = f'data_v2_rescaled/{DATE_STR}'

files = sorted(glob.glob(f'{DATA_DIR}/{ETF_TICKER}_{DATE_STR}_*.parquet'))
if not files:
    raise FileNotFoundError(f'No parquet files found in {DATA_DIR}')

COLS       = ['time_m', 'etf_bid', 'etf_ask', 'etf_mid', 'synth_bid', 'synth_ask', 'synth_mid']
SYNTH_COLS = ['synth_bid', 'synth_ask', 'synth_mid']

PLOT_AFTER_STR  = '094500'
PLOT_AFTER_TIME = f'{PLOT_AFTER_STR[:2]}:{PLOT_AFTER_STR[2:4]}:{PLOT_AFTER_STR[4:6]}'
PLOT_UNTIL_STR  = '160000'
PLOT_UNTIL_TIME = f'{PLOT_UNTIL_STR[:2]}:{PLOT_UNTIL_STR[2:4]}:{PLOT_UNTIL_STR[4:6]}'

DATE       = f'{DATE_STR[:4]}-{DATE_STR[4:6]}-{DATE_STR[6:]}'
PLOT_AFTER = pd.Timestamp(f'{DATE} {PLOT_AFTER_TIME}')
PLOT_UNTIL = pd.Timestamp(f'{DATE} {PLOT_UNTIL_TIME}')

LEVERAGE = 3.0

os.makedirs(OUT_DIR, exist_ok=True)

# Pass 1: find the initial anchor prices (first valid row after market open)
print('Pass 1: finding anchor prices')
first_valid = None
for f in files:
    chunk = pd.read_parquet(f, columns=['time_m', 'etf_mid', 'synth_mid'])
    chunk['time_m'] = pd.to_datetime(chunk['time_m'])
    chunk = chunk[chunk['time_m'] >= PLOT_AFTER].sort_values('time_m').reset_index(drop=True)
    candidates = chunk.dropna(subset=['etf_mid', 'synth_mid'])
    if not candidates.empty:
        first_valid = candidates.iloc[0]
        break
if first_valid is None:
    raise ValueError(f'No valid row with etf_mid and synth_mid found at or after {PLOT_AFTER_TIME}')

etf_mid_0   = first_valid['etf_mid']
synth_mid_0 = first_valid['synth_mid']
anchor_time = first_valid['time_m']

print(f'  Anchor at {anchor_time}')
print(f'  etf_mid_0={etf_mid_0:.4f}  synth_mid_0={synth_mid_0:.4f}')

# Pass 2: apply 3x return rescaling and save
# rescaled = etf_mid_0 * (1 + LEVERAGE * (synth_col / synth_mid_0 - 1))
#          = etf_mid_0 * (LEVERAGE * synth_col / synth_mid_0 - (LEVERAGE - 1))
print('Pass 2: rescaling and saving')
for f in files:
    df = pd.read_parquet(f, columns=COLS).sort_values('time_m').reset_index(drop=True)
    df['time_m'] = pd.to_datetime(df['time_m'])
    df = df[(df['time_m'] >= PLOT_AFTER) & (df['time_m'] <= PLOT_UNTIL)]
    if df.empty:
        continue

    df[SYNTH_COLS] = etf_mid_0 * (LEVERAGE * df[SYNTH_COLS] / synth_mid_0 - (LEVERAGE - 1))

    fname = os.path.basename(f)
    df.to_parquet(f'{OUT_DIR}/{fname}')
    print(f'  Saved {fname}')

print(f'Done. Output: {OUT_DIR}/')
