import glob
import json
import os
import pandas as pd
import numpy as np

ETF_TICKER  = 'SPY'
DATE_STR    = '20240715'
DATA_DIR    = f'data_v2_rescaled/{DATE_STR}'

# Minimum spread
MIN_SPREAD  = 0.02   # dollars per share

files = sorted(glob.glob(f'{DATA_DIR}/{ETF_TICKER}_{DATE_STR}_*.parquet'))
if not files:
    raise FileNotFoundError(f'No parquet files found in {DATA_DIR}')

COLS = ['time_m', 'etf_bid', 'etf_ask', 'etf_mid', 'synth_bid', 'synth_ask', 'synth_mid']

PLOT_AFTER_STR  = '093500'
PLOT_AFTER_TIME = f'{PLOT_AFTER_STR[:2]}:{PLOT_AFTER_STR[2:4]}:{PLOT_AFTER_STR[4:6]}'
PLOT_UNTIL_STR  = '160000'
PLOT_UNTIL_TIME = f'{PLOT_UNTIL_STR[:2]}:{PLOT_UNTIL_STR[2:4]}:{PLOT_UNTIL_STR[4:6]}'

DATE       = f'{DATE_STR[:4]}-{DATE_STR[4:6]}-{DATE_STR[6:]}'
PLOT_AFTER = pd.Timestamp(f'{DATE} {PLOT_AFTER_TIME}')
PLOT_UNTIL = pd.Timestamp(f'{DATE} {PLOT_UNTIL_TIME}')

OUT_PATH = f'results/{ETF_TICKER}_{DATE_STR}_{PLOT_AFTER_STR}_{PLOT_UNTIL_STR}_arb_stats_rescaled.json'
os.makedirs('results', exist_ok=True)

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
    mid = pd.DataFrame({
        'time_m':    df['time_m'].values,
        'disloc':    (df['etf_mid'] - df['synth_mid']).values,   # Positive indicates ETF rich, negative indicates ETF cheap
    })
    chunks.append(mid)

mids = pd.concat(chunks, ignore_index=True).sort_values('time_m').reset_index(drop=True)
print(f'Total ticks: {len(mids):,}')
print(f'Min dislocation threshold: ${MIN_SPREAD:.4f}')

def detect_episodes(series, times, min_spread=0.0):
    active = series > min_spread
    group  = (active != active.shift()).cumsum()
    episodes = []
    for _, grp in series.groupby(group):
        if (grp <= min_spread).all():
            continue
        t        = times.loc[grp.index]
        duration = (t.iloc[-1] - t.iloc[0]).total_seconds()
        episodes.append({
            'start':           t.iloc[0],
            'end':             t.iloc[-1],
            'duration_s':      duration,
            'entry_disloc':    grp.iloc[0],
            'peak_disloc':     grp.max(),
            'mean_disloc':     grp.mean(),
        })
    return pd.DataFrame(episodes)


def print_stats(label, episodes):
    if episodes.empty:
        print(f'No opportunities found')
        return
    total_s = episodes['duration_s'].sum()
    print(f'  Opportunities:              {len(episodes):>8,}')
    print(f'  Total time dislocated:      {total_s:.6f}s  ({total_s/60:.4f} min)')
    print(f'  Mean duration:              {episodes["duration_s"].mean()*1000:.6f}ms')
    print(f'  Median duration:            {episodes["duration_s"].median()*1000:.6f}ms')
    print(f'  Max duration:               {episodes["duration_s"].max()*1000:.6f}ms')
    print(f'  Mean entry disloc:         ${episodes["entry_disloc"].mean():>8.4f}')
    print(f'  Mean peak disloc:          ${episodes["peak_disloc"].mean():>8.4f}')
    print(f'  Mean disloc (while active):${episodes["mean_disloc"].mean():>8.4f}')
    print(f'  Total gross profit (1sh, entry): ${episodes["entry_disloc"].sum():>8.4f}')


ep_rich  = detect_episodes( mids['disloc'], mids['time_m'], MIN_SPREAD)
ep_cheap = detect_episodes(-mids['disloc'], mids['time_m'], MIN_SPREAD)

print_stats('ETF rich  (sell ETF, buy basket)', ep_rich)
print_stats('ETF cheap (buy ETF, sell basket)', ep_cheap)

all_ep      = pd.concat([ep_rich, ep_cheap], ignore_index=True)
window_s    = (PLOT_UNTIL - PLOT_AFTER).total_seconds()
total_arb_s  = all_ep['duration_s'].sum() if not all_ep.empty else 0.0
total_profit = all_ep['entry_disloc'].sum() if not all_ep.empty else 0.0

print(f'  Total opportunities:            {len(all_ep):>8,}')
print(f'  Total time dislocated:          {total_arb_s:>8.1f}s  ({total_arb_s/60:.1f} min)')
print(f'  % of window dislocated:         {100*total_arb_s/window_s:>8.2f}%')
print(f'  Total gross profit (1sh, entry): ${total_profit:>8.4f}')

def leg_dict(ep):
    if ep.empty:
        return {}
    return {
        'n_opportunities':        int(len(ep)),
        'total_time_s':           float(ep['duration_s'].sum()),
        'mean_duration_s':        float(ep['duration_s'].mean()),
        'median_duration_s':      float(ep['duration_s'].median()),
        'max_duration_s':         float(ep['duration_s'].max()),
        'mean_entry_disloc':      float(ep['entry_disloc'].mean()),
        'mean_peak_disloc':       float(ep['peak_disloc'].mean()),
        'mean_disloc_in_arb':     float(ep['mean_disloc'].mean()),
        'total_gross_profit_1sh': float(ep['entry_disloc'].sum()),
    }

results = {
    'etf_ticker':    ETF_TICKER,
    'date':          DATE,
    'window_start':  PLOT_AFTER.isoformat(),
    'window_end':    PLOT_UNTIL.isoformat(),
    'min_spread':    MIN_SPREAD,
    'total_ticks':   int(len(mids)),
    'etf_rich':      leg_dict(ep_rich),
    'etf_cheap':     leg_dict(ep_cheap),
    'combined': {
        'n_opportunities':         int(len(all_ep)),
        'total_time_s':            float(total_arb_s),
        'pct_window_dislocated':   float(100 * total_arb_s / window_s),
        'total_gross_profit_1sh':  float(total_profit),
    },
}

with open(OUT_PATH, 'w') as fh:
    json.dump(results, fh, indent=2)
print(f'Saved to {OUT_PATH}')
