import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATE_STR  = '20240708'
DATA_DIR  = f'data_etf_pairs/{DATE_STR}'
TICKERS   = ['SPY', 'VOO', 'IVV']
REF       = 'SPY'    # spread denominator

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
OUT_PATH_PRICE  = f'plots/etf_pairs_{DATE_STR}_{PLOT_AFTER_STR}_{PLOT_UNTIL_STR}_price.png'
OUT_PATH_SPREAD = f'plots/etf_pairs_{DATE_STR}_{PLOT_AFTER_STR}_{PLOT_UNTIL_STR}_spread.png'

MID_COLS    = [f'{t}_mid' for t in TICKERS]
SPREAD_PAIRS = [(a, b) for i, a in enumerate(TICKERS) for b in TICKERS[i+1:]]

COLORS = {'SPY': 'steelblue', 'VOO': 'tomato', 'IVV': 'seagreen'}
PAIR_COLORS = ['purple', 'darkorange', 'teal']

fig_price,  ax_price  = plt.subplots(figsize=(18, 5))
fig_spread, ax_spread = plt.subplots(figsize=(18, 5))

anchor = {}
first  = True

for f in files:
    df = pd.read_parquet(f).sort_values('time_m').reset_index(drop=True)
    df['time_m'] = pd.to_datetime(df['time_m'])
    if PLOT_AFTER is not None:
        df = df[df['time_m'] >= PLOT_AFTER]
    if PLOT_UNTIL is not None:
        df = df[df['time_m'] <= PLOT_UNTIL]
    if df.empty:
        continue

    for t in TICKERS:
        col = f'{t}_mid'
        if t not in anchor:
            first_valid = df[col].dropna()
            if len(first_valid):
                anchor[t] = first_valid.iloc[0]

    if len(anchor) < len(TICKERS):
        continue

    norm = {}
    for t in TICKERS:
        col = f'{t}_mid'
        norm[t] = (df[col] / anchor[t] - 1) * 10_000
        ax_price.plot(df['time_m'], norm[t],
                      color=COLORS[t], linewidth=0.5,
                      label=t if first else None)

    for (a, b), color in zip(SPREAD_PAIRS, PAIR_COLORS):
        spread_bps = norm[a] - norm[b]
        ax_spread.plot(df['time_m'], spread_bps,
                       color=color, linewidth=0.5,
                       label=f'{a}−{b} (bps)' if first else None)

    first = False

ax_price.axhline(0, color='black', linewidth=0.6, linestyle='--')
ax_price.set_ylabel('Return from open (bps)')
ax_price.legend(fontsize=9)
ax_price.grid(True, alpha=0.3)
ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax_price.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax_price.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
fig_price.autofmt_xdate()
fig_price.tight_layout()
fig_price.savefig(OUT_PATH_PRICE, dpi=150)
print(f'Saved price to {OUT_PATH_PRICE}')

ax_spread.axhline(0, color='black', linewidth=0.6, linestyle='--')
ax_spread.set_ylabel('Mid-to-mid spread (bps)')
ax_spread.legend(fontsize=9)
ax_spread.grid(True, alpha=0.3)
ax_spread.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax_spread.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax_spread.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
fig_spread.autofmt_xdate()
fig_spread.tight_layout()
fig_spread.savefig(OUT_PATH_SPREAD, dpi=150)
print(f'Saved spread to {OUT_PATH_SPREAD}')

plt.show()
