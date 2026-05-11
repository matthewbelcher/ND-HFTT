import glob
import json
import os
import pandas as pd
import numpy as np

DATE_STR  = '20240708'
DATA_DIR  = f'data_etf_pairs/{DATE_STR}'
TICKERS   = ['SPY', 'VOO', 'IVV']

RESAMPLE_MS = 1

files = sorted(glob.glob(f'{DATA_DIR}/etf_pairs_{DATE_STR}_*.parquet'))
if not files:
    raise FileNotFoundError(f'No parquet files found in {DATA_DIR}')

PLOT_AFTER_STR  = '093500'
PLOT_AFTER_TIME = f'{PLOT_AFTER_STR[:2]}:{PLOT_AFTER_STR[2:4]}:{PLOT_AFTER_STR[4:6]}'
PLOT_UNTIL_STR  = '160000'
PLOT_UNTIL_TIME = f'{PLOT_UNTIL_STR[:2]}:{PLOT_UNTIL_STR[2:4]}:{PLOT_UNTIL_STR[4:6]}'

DATE       = f'{DATE_STR[:4]}-{DATE_STR[4:6]}-{DATE_STR[6:]}'
PLOT_AFTER = pd.Timestamp(f'{DATE} {PLOT_AFTER_TIME}')
PLOT_UNTIL = pd.Timestamp(f'{DATE} {PLOT_UNTIL_TIME}')

OUT_PATH = f'results/etf_pairs_{DATE_STR}_{PLOT_AFTER_STR}_{PLOT_UNTIL_STR}_stats.json'
os.makedirs('results', exist_ok=True)

MID_COLS = [f'{t}_mid' for t in TICKERS]
BID_COLS = [f'{t}_bid' for t in TICKERS]
ASK_COLS = [f'{t}_ask' for t in TICKERS]

chunks = []
for f in files:
    df = pd.read_parquet(f).sort_values('time_m').reset_index(drop=True)
    df['time_m'] = pd.to_datetime(df['time_m'])
    if PLOT_AFTER is not None:
        df = df[df['time_m'] >= PLOT_AFTER]
    if PLOT_UNTIL is not None:
        df = df[df['time_m'] <= PLOT_UNTIL]
    if df.empty:
        continue
    chunks.append(df)

raw = pd.concat(chunks, ignore_index=True).sort_values('time_m').reset_index(drop=True)
print(f'  {len(raw):,} raw ticks')

mids = (
    raw.set_index('time_m')[MID_COLS]
    .resample(f'{RESAMPLE_MS}ms')
    .last()
    .ffill()
    .dropna()
)
print(f'  {len(mids):,} ticks after {RESAMPLE_MS}ms resample')

ba_stats = {}
for t in TICKERS:
    ba = (raw[f'{t}_ask'] - raw[f'{t}_bid']).dropna()
    ba_stats[t] = {
        'mean':   float(ba.mean()),
        'median': float(ba.median()),
        'std':    float(ba.std()),
    }

# Return correlation 
rets = mids.pct_change().dropna()
corr = rets.corr()
corr.columns = TICKERS
corr.index   = TICKERS
print(corr.to_string())

# Pairwise bps spread stats
print('Pairwise Mid Spread (bps)')
SPREAD_PAIRS = [(a, b) for i, a in enumerate(TICKERS) for b in TICKERS[i+1:]]

anchor = {t: mids[f'{t}_mid'].iloc[0] for t in TICKERS}
norm_mids = {t: (mids[f'{t}_mid'] / anchor[t] - 1) * 10_000 for t in TICKERS}

spread_stats = {}
for a, b in SPREAD_PAIRS:
    spread_bps = norm_mids[a] - norm_mids[b]
    key = f'{a}_vs_{b}'
    spread_stats[key] = {
        'mean_bps':   float(spread_bps.mean()),
        'std_bps':    float(spread_bps.std()),
        'min_bps':    float(spread_bps.min()),
        'max_bps':    float(spread_bps.max()),
        'pct_pos':    float((spread_bps > 0).mean() * 100),
    }
    print(f'  {a}−{b}: mean={spread_bps.mean():.3f}bps  std={spread_bps.std():.3f}bps'
          f'  min={spread_bps.min():.3f}  max={spread_bps.max():.3f}'
          f'  % {a} rich={((spread_bps > 0).mean()*100):.1f}%')

# Save 
results = {
    'date':           DATE,
    'window_start':   PLOT_AFTER.isoformat(),
    'window_end':     PLOT_UNTIL.isoformat(),
    'resample_ms':    RESAMPLE_MS,
    'tickers':        TICKERS,
    'bid_ask_spread': ba_stats,
    'return_corr':    {f'{a}_vs_{b}': float(corr.loc[a, b])
                       for a, b in SPREAD_PAIRS},
    'spread_bps':     spread_stats,
}

with open(OUT_PATH, 'w') as fh:
    json.dump(results, fh, indent=2)
print(f'Saved to {OUT_PATH}')
