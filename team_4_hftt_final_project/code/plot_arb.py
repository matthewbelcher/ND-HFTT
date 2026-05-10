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

COLS = ['time_m', 'etf_bid', 'etf_ask', 'etf_mid', 'synth_bid', 'synth_ask', 'synth_mid']

PLOT_AFTER_STR = '094500'
PLOT_AFTER_TIME = f'{PLOT_AFTER_STR[:2]}:{PLOT_AFTER_STR[2:4]}:{PLOT_AFTER_STR[4:6]}'
PLOT_UNTIL_STR = '160000'
PLOT_UNTIL_TIME = f'{PLOT_UNTIL_STR[:2]}:{PLOT_UNTIL_STR[2:4]}:{PLOT_UNTIL_STR[4:6]}'

DATE       = f'{DATE_STR[:4]}-{DATE_STR[4:6]}-{DATE_STR[6:]}'
PLOT_AFTER = pd.Timestamp(f'{DATE} {PLOT_AFTER_TIME}')
PLOT_UNTIL = pd.Timestamp(f'{DATE} {PLOT_UNTIL_TIME}')
OUT_PATH   = f'plots/{ETF_TICKER}_{DATE_STR}_{PLOT_AFTER_STR}_{PLOT_UNTIL_STR}_arb_rescaled.png'

SYNTH_COLS = ['synth_bid', 'synth_ask', 'synth_mid']

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

    df['buy_spy_sell_basket'] = df['synth_bid'] - df['etf_ask']   # buy SPY ask, sell basket bid
    df['buy_basket_sell_spy'] = df['etf_bid']   - df['synth_ask'] # buy basket ask, sell SPY bid

    ax.plot(df['time_m'], df['buy_spy_sell_basket'],
            color='steelblue', linewidth=0.6,
            label='Buy SPY ask / Sell basket bid' if first else None)
    ax.plot(df['time_m'], df['buy_basket_sell_spy'],
            color='tomato', linewidth=0.6,
            label='Buy basket ask / Sell SPY bid' if first else None)
    first = False
    if (i + 1) % 100 == 0:
        print(f'  {i + 1}/{len(files)} files plotted')

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
fig.autofmt_xdate()

ax.set_ylabel('Profit per share ($)')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f'Saved to {OUT_PATH}')
