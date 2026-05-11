#!/usr/bin/env python3
"""
BTC Cross-Momentum Signal Analysis — Full Dataset
==================================================
Parses all 199 Kalshi+BTC session pairs from bigdata, resamples to 1s,
computes BTC price momentum, and measures whether it predicts Kalshi
mid-price direction.

Signals tested (all backward-looking, no lookahead):
  btc_mom_5s   - BTC return over last 5s
  btc_mom_10s  - BTC return over last 10s
  btc_mom_30s  - BTC return over last 30s
  btc_mom_60s  - BTC return over last 60s
  btc_accel    - btc_mom_5s - btc_mom_10s (is momentum building?)
  btc_x_obi   - btc_mom_10s * obi1 (BTC momentum × order book alignment)

Uses merge_plot.py from Kalshi-Orderbook-Predictor to reconstruct the book.
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent   # Kalshi-Orderbook-Predictor/
DATA_DIRS = [
    ROOT.parent / 'bigdata',  # HFT/bigdata/ — main data store
    ROOT,                      # CSVs loose in project root (fallback)
]
OUTPUT = ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)
CACHE  = OUTPUT / 'btc_mom_cache'
CACHE.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))  # analysis/ — for merge_plot
import merge_plot

HORIZONS    = [5, 10, 30]
BTC_WINDOWS = [5, 10, 30, 60]
RESAMPLE    = '1s'

# ── Colors ─────────────────────────────────────────────────────────────────────
BG, PANEL, TEXT = '#0f1117', '#151821', '#c8d0e0'
BLUE, GREEN, RED, ORANGE = '#00d4ff', '#00e676', '#ff1744', '#ff6b35'


# ── 1. Collect all session pairs ───────────────────────────────────────────────

def find_pairs():
    pairs = []
    seen  = set()
    for d in DATA_DIRS:
        for kp in sorted(d.glob('KXBTC15M-*.csv')):
            bp = kp.parent / f'BTC-{kp.name}'
            if bp.exists() and kp.stem not in seen:
                pairs.append((kp, bp))
                seen.add(kp.stem)
    return pairs


# ── 2. Parse one session → 1s resampled DataFrame ─────────────────────────────

def load_session(kpath: Path, bpath: Path) -> pd.DataFrame | None:
    cache_path = CACHE / f'{kpath.stem}.parquet'
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        kdf, _ = merge_plot.parse_kalshi(str(kpath))
        bdf    = merge_plot.parse_btc(str(bpath),
                                      kdf.index[0]  - merge_plot.BTC_BUFFER,
                                      kdf.index[-1] + merge_plot.BTC_BUFFER)
        tick   = merge_plot.asof_join(kdf, bdf)
    except Exception as e:
        print(f'  [skip] {kpath.stem}: {e}')
        return None

    if len(tick) < 200 or 'btc_mid' not in tick.columns:
        return None

    # Resample to 1s: last known value in each second
    tick_clean = tick[['mid', 'obi1', 'obi3', 'obi5', 'btc_mid']].copy()
    tick_clean = tick_clean[tick_clean['mid'].notna()]
    s = tick_clean.resample(RESAMPLE).last().ffill()
    if len(s) < 60:
        return None

    # BTC momentum (rolling return over N seconds)
    for w in BTC_WINDOWS:
        s[f'btc_mom_{w}s'] = s['btc_mid'].pct_change(w)

    s['btc_accel'] = s['btc_mom_5s'] - s['btc_mom_10s']
    s['btc_x_obi'] = s['btc_mom_10s'] * s['obi1']

    # Forward Kalshi returns (what we're trying to predict)
    for h in HORIZONS:
        s[f'future_mid_{h}s']  = s['mid'].shift(-h)
        s[f'future_dir_{h}s']  = np.sign(s[f'future_mid_{h}s'] - s['mid'])

    s['contract'] = kpath.stem
    s.to_parquet(cache_path)
    return s


# ── 3. Load all sessions ───────────────────────────────────────────────────────

pairs = find_pairs()
print(f'Found {len(pairs)} session pairs. Loading…')

frames = []
for i, (kp, bp) in enumerate(pairs):
    df = load_session(kp, bp)
    if df is not None:
        frames.append(df)
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{len(pairs)} loaded ({len(frames)} ok)')

if not frames:
    print('ERROR: No sessions loaded.')
    sys.exit(1)

data = pd.concat(frames)
print(f'\nTotal 1s rows: {len(data):,}  across {data["contract"].nunique()} sessions')

MOM_COLS = [f'btc_mom_{w}s' for w in BTC_WINDOWS] + ['btc_accel', 'btc_x_obi']


# ── 4. IC Analysis ─────────────────────────────────────────────────────────────

print('\n── IC Analysis (Spearman corr: BTC signal → future Kalshi direction) ──')
print(f'{"Signal":<18}', end='')
for h in HORIZONS:
    print(f'  IC@{h}s   p-val ', end='')
print()
print('─' * (18 + 15 * len(HORIZONS)))

ic_table = {}
for sig in MOM_COLS:
    ic_table[sig] = {}
    row = f'{sig:<18}'
    for h in HORIZONS:
        tgt   = f'future_dir_{h}s'
        valid = data[[sig, tgt]].dropna()
        if len(valid) < 100:
            row += f'  {"n/a":>6}   {"":>6} '
            continue
        ic, p = scipy_stats.spearmanr(valid[sig], valid[tgt])
        ic_table[sig][h] = (ic, p)
        flag = '**' if p < 0.01 else ('*' if p < 0.05 else '  ')
        row += f'  {ic:+.4f}{flag} {p:.4f} '
    print(row)

# Baseline: OBI alone
print(f'\n── Baseline: OBI1 alone ──')
for h in HORIZONS:
    tgt   = f'future_dir_{h}s'
    valid = data[['obi1', tgt]].dropna()
    ic, p = scipy_stats.spearmanr(valid['obi1'], valid[tgt])
    print(f'  IC@{h}s = {ic:+.4f}  p={p:.4f}')


# ── 5. Lead-lag sweep ──────────────────────────────────────────────────────────

print('\n── Lead-Lag: how many seconds does Kalshi lag BTC? ──')
LAGS = list(range(0, 31))
xcorr_sum = np.zeros(len(LAGS))
n_sess    = 0

for contract, grp in data.groupby('contract'):
    grp = grp.dropna(subset=['btc_mom_10s', 'mid'])
    if len(grp) < 60:
        continue
    btc_s = grp['btc_mom_10s'].values
    kal_s = grp['mid'].pct_change().values
    for li, lag in enumerate(LAGS):
        a, b = (btc_s[:-lag], kal_s[lag:]) if lag > 0 else (btc_s, kal_s)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 30:
            continue
        c, _ = scipy_stats.spearmanr(a[mask], b[mask])
        xcorr_sum[li] += 0 if np.isnan(c) else c
    n_sess += 1

if n_sess > 0:
    xcorr = xcorr_sum / n_sess
    best  = int(np.argmax(np.abs(xcorr)))
    print(f'  Sessions averaged: {n_sess}')
    for lag in [0, 1, 2, 3, 5, 10, 20]:
        print(f'  Lag={lag:2d}s  corr={xcorr[lag]:+.4f}')
    print(f'  → Best lag: {best}s  (corr={xcorr[best]:+.4f})')


# ── 6. Conditional: OBI+BTC aligned vs opposed ────────────────────────────────

print('\n── OBI hit rate: BTC momentum ALIGNED vs OPPOSED ──')
for h in HORIZONS:
    tgt   = f'future_dir_{h}s'
    valid = data[['obi1', 'btc_mom_10s', tgt]].dropna()
    obi_d = np.sign(valid['obi1'])
    btc_d = np.sign(valid['btc_mom_10s'])
    def hr(mask):
        sub = valid[mask]
        return (np.sign(sub['obi1']) == sub[tgt]).mean(), len(sub)
    hr_al, n_al = hr(obi_d == btc_d)
    hr_op, n_op = hr(obi_d != btc_d)
    print(f'  @{h}s  aligned={hr_al:.3f} (n={n_al:,})  opposed={hr_op:.3f} (n={n_op:,})'
          f'  Δ={hr_al-hr_op:+.3f}')


# ── 7. Large BTC move analysis ─────────────────────────────────────────────────

print('\n── Large BTC moves → Kalshi hit rate ──')
print(f'{"Threshold":>12}  {"N":>7}  {"Hit@5s":>8}  {"Hit@10s":>8}  {"Hit@30s":>8}')
thresholds = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
for thr in thresholds:
    fired = data[data['btc_mom_10s'].abs() >= thr]
    if len(fired) < 20:
        continue
    pred = np.sign(fired['btc_mom_10s'])
    cols = {h: (pred == fired[f'future_dir_{h}s']).mean() for h in HORIZONS}
    n    = (~fired['btc_mom_10s'].isna()).sum()
    print(f'  {thr:>10.4f}  {n:>7,}  '
          + '  '.join(f'{cols[h]:>8.3f}' for h in HORIZONS))


# ── 8. Combined signal: btc_mom + OBI score ───────────────────────────────────

print('\n── Combined score = btc_mom_10s × obi1 (same-direction amplification) ──')
data['combined'] = data['btc_mom_10s'] * data['obi1']
for h in HORIZONS:
    tgt   = f'future_dir_{h}s'
    valid = data[['combined', tgt]].dropna()
    ic, p = scipy_stats.spearmanr(valid['combined'], valid[tgt])
    print(f'  IC@{h}s = {ic:+.4f}  p={p:.4f}')


# ── 9. IC by time-into-contract (does momentum matter more at certain times?) ──

print('\n── IC by minute-into-contract (btc_mom_10s → future_dir_10s) ──')
data['elapsed_min'] = data.groupby('contract').cumcount() // 60
tgt = 'future_dir_10s'
for minute in range(0, 15):
    bucket = data[data['elapsed_min'] == minute][['btc_mom_10s', tgt]].dropna()
    if len(bucket) < 50:
        continue
    ic, p = scipy_stats.spearmanr(bucket['btc_mom_10s'], bucket[tgt])
    bar = '█' * int(abs(ic) * 100)
    print(f'  Min {minute:2d}  IC={ic:+.4f}  n={len(bucket):>6,}  {bar}')


# ── 10. Plot ───────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12), facecolor=BG)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

def ax_style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=9, pad=6)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#2a2f45')
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color='#1e2130', linewidth=0.5)

# Panel A: IC heatmap
ax = fig.add_subplot(gs[0, 0])
sig_labels = MOM_COLS + ['obi1_baseline']
ic_matrix  = []
for s in MOM_COLS:
    ic_matrix.append([ic_table.get(s, {}).get(h, (np.nan,))[0] for h in HORIZONS])
# add OBI baseline row
for h in HORIZONS:
    pass  # already printed above; approximate here
obi_ics = []
for h in HORIZONS:
    tgt   = f'future_dir_{h}s'
    valid = data[['obi1', tgt]].dropna()
    ic, _ = scipy_stats.spearmanr(valid['obi1'], valid[tgt])
    obi_ics.append(ic)
ic_matrix.append(obi_ics)

mat = np.array(ic_matrix)
im  = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=-0.20, vmax=0.20)
ax.set_xticks(range(len(HORIZONS))); ax.set_xticklabels([f'{h}s' for h in HORIZONS], color=TEXT, fontsize=8)
ax.set_yticks(range(len(sig_labels))); ax.set_yticklabels(sig_labels, color=TEXT, fontsize=7)
for i in range(len(sig_labels)):
    for j in range(len(HORIZONS)):
        v = mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:+.3f}', ha='center', va='center', fontsize=7,
                    color='black' if abs(v) > 0.10 else TEXT)
fig.colorbar(im, ax=ax, fraction=0.04)
ax_style(ax, 'IC Heatmap — BTC Momentum vs Kalshi Direction')

# Panel B: Lag sweep
ax = fig.add_subplot(gs[0, 1])
if n_sess > 0:
    colors = [BLUE if v >= 0 else RED for v in xcorr]
    ax.bar(LAGS, xcorr, color=colors)
    ax.axvline(best, color=ORANGE, linestyle='--', linewidth=1.2, label=f'best lag={best}s')
    ax.axhline(0, color='#2a2f45', linewidth=0.8)
    ax.set_xlabel('BTC leads Kalshi by N seconds')
    ax.set_ylabel('Avg Spearman IC')
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL)
ax_style(ax, 'Lead-Lag: btc_mom_10s → Kalshi 1s return')

# Panel C: BTC momentum distribution by Kalshi outcome
ax = fig.add_subplot(gs[1, 0])
valid = data[['btc_mom_10s', 'future_dir_10s']].dropna()
up   = valid[valid['future_dir_10s'] ==  1]['btc_mom_10s']
down = valid[valid['future_dir_10s'] == -1]['btc_mom_10s']
bins = np.linspace(-0.004, 0.004, 50)
ax.hist(up,   bins=bins, alpha=0.6, color=GREEN, label=f'Kalshi UP (n={len(up):,})',   density=True)
ax.hist(down, bins=bins, alpha=0.6, color=RED,   label=f'Kalshi DOWN (n={len(down):,})', density=True)
ax.set_xlabel('btc_mom_10s')
ax.set_ylabel('Density')
ax.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL)
ax_style(ax, 'BTC Momentum Dist. by Kalshi @10s Outcome')

# Panel D: OBI hit rate aligned vs opposed
ax = fig.add_subplot(gs[1, 1])
hr_al_list, hr_op_list = [], []
for h in HORIZONS:
    tgt   = f'future_dir_{h}s'
    valid = data[['obi1', 'btc_mom_10s', tgt]].dropna()
    obi_d = np.sign(valid['obi1']); btc_d = np.sign(valid['btc_mom_10s'])
    hr_al_list.append((np.sign(valid.loc[obi_d == btc_d, 'obi1']) == valid.loc[obi_d == btc_d, tgt]).mean())
    hr_op_list.append((np.sign(valid.loc[obi_d != btc_d, 'obi1']) == valid.loc[obi_d != btc_d, tgt]).mean())
x = np.arange(len(HORIZONS)); w = 0.3
ax.bar(x - w/2, hr_al_list, width=w, color=GREEN, label='OBI aligned w/ BTC', alpha=0.85)
ax.bar(x + w/2, hr_op_list, width=w, color=RED,   label='OBI opposed to BTC', alpha=0.85)
ax.axhline(1/3, color='white', linestyle='--', linewidth=0.8, alpha=0.5, label='random (33%)')
ax.set_xticks(x); ax.set_xticklabels([f'{h}s' for h in HORIZONS], color=TEXT)
ax.set_ylabel('OBI Hit Rate'); ax.set_ylim(0.20, 0.45)
ax.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL)
ax_style(ax, 'OBI Hit Rate: Aligned vs Opposed to BTC Momentum')

# Panel E: IC by minute-into-contract
ax = fig.add_subplot(gs[2, 0])
minute_ics = []
for minute in range(15):
    bucket = data[data['elapsed_min'] == minute][['btc_mom_10s', 'future_dir_10s']].dropna()
    if len(bucket) < 50:
        minute_ics.append(np.nan)
    else:
        ic, _ = scipy_stats.spearmanr(bucket['btc_mom_10s'], bucket['future_dir_10s'])
        minute_ics.append(ic)
ax.bar(range(15), minute_ics, color=[BLUE if v and v >= 0 else RED for v in minute_ics])
ax.axhline(0, color='#2a2f45', linewidth=0.8)
ax.set_xlabel('Minute into 15-min contract')
ax.set_ylabel('IC (btc_mom_10s → future_dir_10s)')
ax_style(ax, 'BTC Momentum IC by Time in Contract')

# Panel F: large move hit rates
ax = fig.add_subplot(gs[2, 1])
thr_vals, hr5_vals, hr10_vals, hr30_vals, n_vals = [], [], [], [], []
for thr in thresholds:
    fired = data[data['btc_mom_10s'].abs() >= thr].dropna(subset=[f'future_dir_{h}s' for h in HORIZONS])
    if len(fired) < 20: continue
    pred = np.sign(fired['btc_mom_10s'])
    thr_vals.append(thr); n_vals.append(len(fired))
    hr5_vals.append((pred == fired['future_dir_5s']).mean())
    hr10_vals.append((pred == fired['future_dir_10s']).mean())
    hr30_vals.append((pred == fired['future_dir_30s']).mean())
x = np.arange(len(thr_vals))
ax.plot(x, hr5_vals,  'o-', color=BLUE,   label='@5s',  linewidth=1.5)
ax.plot(x, hr10_vals, 's-', color=GREEN,  label='@10s', linewidth=1.5)
ax.plot(x, hr30_vals, '^-', color=ORANGE, label='@30s', linewidth=1.5)
ax.axhline(1/3, color='white', linestyle='--', linewidth=0.8, alpha=0.5, label='random')
ax.set_xticks(x); ax.set_xticklabels([f'{t:.4f}\n(n={n:,})' for t, n in zip(thr_vals, n_vals)],
                                      color=TEXT, fontsize=6)
ax.set_ylabel('Hit rate'); ax.set_xlabel('|btc_mom_10s| threshold')
ax.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL)
ax_style(ax, 'Hit Rate by BTC Move Size')

plt.suptitle(f'BTC Cross-Momentum — {data["contract"].nunique()} sessions, {len(data):,} 1s bars',
             color=TEXT, fontsize=12, y=0.99)
out = OUTPUT / 'btc_cross_momentum.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'\nFigure → {out}')
