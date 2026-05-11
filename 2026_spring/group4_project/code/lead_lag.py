import glob
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ETF_TICKER  = 'SPY'
DATE_STR    = '20240715'
DATA_DIR    = f'data_v2/{DATE_STR}'

files = sorted(glob.glob(f'{DATA_DIR}/{ETF_TICKER}_{DATE_STR}_*.parquet'))
if not files:
    raise FileNotFoundError(f'No parquet files found in {DATA_DIR}')

COLS = ['time_m', 'etf_mid', 'synth_mid']

PLOT_AFTER_STR  = '093500'
PLOT_AFTER_TIME = f'{PLOT_AFTER_STR[:2]}:{PLOT_AFTER_STR[2:4]}:{PLOT_AFTER_STR[4:6]}'
PLOT_UNTIL_STR  = '160000'
PLOT_UNTIL_TIME = f'{PLOT_UNTIL_STR[:2]}:{PLOT_UNTIL_STR[2:4]}:{PLOT_UNTIL_STR[4:6]}'

MAX_LAG     = 100
RESAMPLE_MS = 1

DATE       = f'{DATE_STR[:4]}-{DATE_STR[4:6]}-{DATE_STR[6:]}'
PLOT_AFTER = pd.Timestamp(f'{DATE} {PLOT_AFTER_TIME}')
PLOT_UNTIL = pd.Timestamp(f'{DATE} {PLOT_UNTIL_TIME}')
OUT_PATH   = f'results/{ETF_TICKER}_{DATE_STR}_{PLOT_AFTER_STR}_{PLOT_UNTIL_STR}_lead_lag_{RESAMPLE_MS}ms.json'
PLOT_PATH  = f'plots/{ETF_TICKER}_{DATE_STR}_{PLOT_AFTER_STR}_{PLOT_UNTIL_STR}_lead_lag_{RESAMPLE_MS}ms.png'
os.makedirs('results', exist_ok=True)
os.makedirs('plots',   exist_ok=True)

# Load dataset
chunks = []
for f in files:
    df = pd.read_parquet(f, columns=COLS).sort_values('time_m').reset_index(drop=True)
    df['time_m'] = pd.to_datetime(df['time_m'])
    if PLOT_AFTER is not None:
        df = df[df['time_m'] >= PLOT_AFTER]
    if PLOT_UNTIL is not None:
        df = df[df['time_m'] <= PLOT_UNTIL]
    if df.empty:
        continue
    chunks.append(df)

mids = pd.concat(chunks, ignore_index=True).sort_values('time_m').reset_index(drop=True)
print(f'Total ticks: {len(mids):,}')

# Resample
mids = (
    mids.set_index('time_m')
    .resample(f'{RESAMPLE_MS}ms')
    .last()
    .ffill()
    .dropna()
    .reset_index()
)
print(f'After {RESAMPLE_MS}ms resample: {len(mids):,} ticks')

# Compute returns
mids['etf_ret']   = mids['etf_mid'].pct_change()
mids['synth_ret'] = mids['synth_mid'].pct_change()
mids = mids.dropna()

# Cross-correlation
# Positive lag: ETF leads basket
# Negative lag: basket leads ETF
lags  = range(-MAX_LAG, MAX_LAG + 1)
corrs = []
for lag in lags:
    corr = mids['etf_ret'].corr(mids['synth_ret'].shift(lag))
    corrs.append(corr)

corr_series = pd.Series(corrs, index=list(lags))
peak_lag    = int(corr_series.idxmax())
peak_corr   = float(corr_series.max())
lag_unit    = f'{RESAMPLE_MS}ms'
peak_ms     = peak_lag * RESAMPLE_MS

print(f'Peak correlation: {peak_corr:.4f} at lag {peak_lag} ({lag_unit})')

# Plot 
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(list(lags), corrs,
       color=['tomato' if l > 0 else 'steelblue' if l < 0 else 'grey' for l in lags])
ax.axvline(0,        color='black', linewidth=0.8, linestyle='--')
ax.axvline(peak_lag, color='green', linewidth=1.2, linestyle=':',
           label=f'Peak lag correlation')
ax.set_xlabel(f'Lag ({lag_unit})')
ax.set_ylabel('Cross-correlation')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f'Plot saved to {PLOT_PATH}')
plt.show()

# Save results
results = {
    'etf_ticker':  ETF_TICKER,
    'date':        DATE,
    'window_start': PLOT_AFTER.isoformat(),
    'window_end':   PLOT_UNTIL.isoformat(),
    'data_dir':    DATA_DIR,
    'resample_ms': RESAMPLE_MS,
    'max_lag':     MAX_LAG,
    'total_ticks': int(len(mids)),
    'peak_lag':    peak_lag,
    'peak_lag_ms': peak_ms,
    'peak_corr':   peak_corr,
    'cross_corr':  {str(lag): float(c) for lag, c in zip(lags, corrs)},
}

with open(OUT_PATH, 'w') as fh:
    json.dump(results, fh, indent=2)
print(f'Results saved to {OUT_PATH}')
