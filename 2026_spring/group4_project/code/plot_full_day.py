import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ETF_TICKER  = 'SPXL'
DATE_STR    = '20240715'
DATA_DIR    = f'data_v2_rescaled/{DATE_STR}'

files = sorted(glob.glob(f'{DATA_DIR}/{ETF_TICKER}_{DATE_STR}_*.parquet'))
if not files:
    raise FileNotFoundError(f'No parquet files found in {DATA_DIR}')

COLS = ['time_m', 'etf_bid', 'etf_ask', 'etf_mid', 'synth_bid', 'synth_ask', 'synth_mid']

fig, ax = plt.subplots(figsize=(18, 7))

SERIES = [
    ('etf_bid',   'steelblue',      0.6, 'solid',  'ETF Bid'),
    ('etf_ask',   'cornflowerblue', 0.6, 'dashed', 'ETF Ask'),
    ('etf_mid',   'blue',           0.8, 'solid',  'ETF Mid'),
    ('synth_bid', 'tomato',         0.6, 'solid',  'Basket Bid'),
    ('synth_ask', 'salmon',         0.6, 'dashed', 'Basket Ask'),
    ('synth_mid', 'firebrick',      0.8, 'solid',  'Basket Mid'),
]

SCALE_AFTER_STR = '000000'
SCALE_AFTER_TIME = f'{SCALE_AFTER_STR[:2]}:{SCALE_AFTER_STR[2:4]}:{SCALE_AFTER_STR[4:6]}'
PLOT_AFTER_STR = '095000'
PLOT_AFTER_TIME = f'{PLOT_AFTER_STR[:2]}:{PLOT_AFTER_STR[2:4]}:{PLOT_AFTER_STR[4:6]}'
PLOT_UNTIL_STR  = '160000'
PLOT_UNTIL_TIME = f'{PLOT_UNTIL_STR[:2]}:{PLOT_UNTIL_STR[2:4]}:{PLOT_UNTIL_STR[4:6]}'

DATE        = f'{DATE_STR[:4]}-{DATE_STR[4:6]}-{DATE_STR[6:]}'
SCALE_AT    = None  # set to e.g. pd.Timestamp(f'{DATE} {SCALE_AFTER_TIME}') to enable
PLOT_AFTER  = pd.Timestamp(f'{DATE} {PLOT_AFTER_TIME}')
PLOT_UNTIL  = pd.Timestamp(f'{DATE} {PLOT_UNTIL_TIME}')
SYNTH_COLS  = ['synth_bid', 'synth_ask', 'synth_mid']
OUT_PATH    = f'plots/{ETF_TICKER}_{DATE_STR}_{PLOT_AFTER_STR}_{PLOT_UNTIL_STR}_rescaled.png'


scale = None
if SCALE_AT is None:
    print('  Scale disabled — plotting raw prices')
else:
    for f in files:
        df = pd.read_parquet(f, columns=COLS).sort_values('time_m')
        df['time_m'] = pd.to_datetime(df['time_m'])
        df = df[df['time_m'] >= SCALE_AT]
        idx = df[['etf_mid', 'synth_mid']].dropna().index
        if len(idx):
            row = df.loc[idx[0]]
            scale = row['etf_mid'] / row['synth_mid']
            print(f'  Scale = {scale:.6f} (set at {row["time_m"]})')
            break

first = True
for i, f in enumerate(files):
    df = pd.read_parquet(f, columns=COLS).sort_values('time_m')
    df['time_m'] = pd.to_datetime(df['time_m'])
    if PLOT_AFTER is not None:
        df = df[df['time_m'] >= PLOT_AFTER]
    if PLOT_UNTIL is not None:
        df = df[df['time_m'] <= PLOT_UNTIL]
    if df.empty:
        continue

    if scale is not None:
        df[SYNTH_COLS] = df[SYNTH_COLS] * scale

    for col, color, lw, ls, label in SERIES:
        ax.plot(df['time_m'], df[col],
                color=color, linewidth=lw, linestyle=ls,
                label=label if first else None)
    first = False
    if (i + 1) % 100 == 0:
        print(f'  {i + 1}/{len(files)} files plotted')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
fig.autofmt_xdate()

ax.set_ylabel('Price ($)')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f'Saved to {OUT_PATH}')
plt.show()
