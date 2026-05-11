import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ETF_TICKER  = 'SPY'
DATE_STR    = '20240715'
DATA_DIR    = f'data_v2_rescaled/{DATE_STR}'

files = sorted(glob.glob(f'{DATA_DIR}/{ETF_TICKER}_{DATE_STR}_*.parquet'))
if not files:
    raise FileNotFoundError(f'No parquet files found in {DATA_DIR}')

COLS = ['time_m', 'etf_mid', 'synth_mid']

PLOT_AFTER_STR = '094500'
PLOT_AFTER_TIME = f'{PLOT_AFTER_STR[:2]}:{PLOT_AFTER_STR[2:4]}:{PLOT_AFTER_STR[4:6]}'
PLOT_UNTIL_STR = '160000'
PLOT_UNTIL_TIME = f'{PLOT_UNTIL_STR[:2]}:{PLOT_UNTIL_STR[2:4]}:{PLOT_UNTIL_STR[4:6]}'

DATE       = f'{DATE_STR[:4]}-{DATE_STR[4:6]}-{DATE_STR[6:]}'
PLOT_AFTER = pd.Timestamp(f'{DATE} {PLOT_AFTER_TIME}')
PLOT_UNTIL = pd.Timestamp(f'{DATE} {PLOT_UNTIL_TIME}')
OUT_PATH   = f'plots/{ETF_TICKER}_{DATE_STR}_{PLOT_AFTER_STR}_{PLOT_UNTIL_STR}_spread_rescaled.png'

fig, ax = plt.subplots(figsize=(18, 5))

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

    df['spread'] = df['etf_mid'] - df['synth_mid']
    ax.plot(df['time_m'], df['spread'],
            color='purple', linewidth=0.6,
            label='ETF Weighted Mid-Price − Underlying Basket Weighted Mid-Price' if first else None)
    first = False
    if (i + 1) % 100 == 0:
        print(f'  {i + 1}/{len(files)} files plotted')

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
fig.autofmt_xdate()

ax.set_ylabel('Spread ($)')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f'Saved to {OUT_PATH}')
