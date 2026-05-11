"""
obi_signal_test.py  —  OBI delta → price-move predictive test
==============================================================

Tests whether a sudden spike in OBI (order book imbalance) predicts a
Kalshi mid-price move in the same direction within a short forward window.

Methodology
-----------
1. Compute d_obi = rolling change in OBI over a backward window (SIGNAL_WINDOW).
2. A signal fires when |d_obi| >= SIGNAL_THRESHOLD.
   - Positive d_obi → predict YES price rises  (buy signal)
   - Negative d_obi → predict YES price falls  (sell signal)
3. Look forward up to HORIZON seconds; record the max favourable and max
   adverse move in the Kalshi mid.
4. A signal "hits" if the mid moves >= MIN_TICK in the predicted direction
   within the horizon, WITHOUT first moving >= MIN_TICK adversely.

Input
-----
A merged CSV produced by merge_plot.py / the notebook, which must contain
columns: ts (index), mid, obi.

Usage
-----
    python obi_signal_test.py <merged_csv> [options]

    python obi_signal_test.py completed-data/merged_KXBTC15M-26MAR162145-45.csv
    python obi_signal_test.py merged.csv --horizon 2.0 --threshold 0.15 --depth 3

Options
-------
    --horizon     Forward look window in seconds          (default: 1.0)
    --threshold   Minimum |d_obi| to fire a signal        (default: 0.10)
    --signal-win  Backward window for delta calc (secs)   (default: 0.5)
    --min-tick    Minimum mid move to count as a hit ($)  (default: 0.01)
    --depth       Override OBI column suffix, e.g. 3 for obi3 or 'obi' for
                  the parametric column                   (default: obi)
    --cooldown    Seconds to suppress re-firing after signal (default: 1.0)
    --out-dir     Output directory for plot + CSV          (default: same as input)
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from pathlib import Path
from datetime import timezone


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_test(
    df: pd.DataFrame,
    obi_col:     str   = 'obi',
    horizon:     float = 1.0,
    signal_win:  float = 0.5,
    threshold:   float = 0.10,
    min_tick:    float = 0.01,
    cooldown:    float = 1.0,
) -> pd.DataFrame:
    """
    Returns a DataFrame of signal events with outcome labels.

    Columns in result:
        ts          : signal timestamp
        obi_before  : OBI value at start of signal window
        obi_at      : OBI value at signal time
        d_obi       : obi_at - obi_before
        direction   : +1 (predict rise) or -1 (predict fall)
        mid_at      : Kalshi mid at signal time
        mid_fwd     : best mid reached in the predicted direction within horizon
        fwd_move    : mid_fwd - mid_at  (signed, positive = correct)
        hit         : True if fwd_move >= min_tick without first breaching adversely
        adverse     : True if adverse move >= min_tick fired first (clean reversal)
    """
    if obi_col not in df.columns:
        raise ValueError(
            f"Column '{obi_col}' not found. Available: {list(df.columns)}")
    if 'mid' not in df.columns:
        raise ValueError("Column 'mid' not found in merged dataframe.")

    ser   = df[obi_col].dropna()
    mid_s = df['mid'].ffill()   # forward-fill so lookup is always defined

    horizon_td   = pd.Timedelta(seconds=horizon)
    signal_win_td = pd.Timedelta(seconds=signal_win)

    events = []
    # Use first timestamp minus a large offset rather than Timestamp.min,
    # which overflows pandas Timedelta arithmetic.
    last_signal_ts = ser.index[0] - pd.Timedelta(days=1)

    ts_arr  = ser.index.to_numpy()
    obi_arr = ser.to_numpy()

    for i, (ts, obi_now) in enumerate(zip(ts_arr, obi_arr)):
        # Cooldown guard
        if ts - last_signal_ts < pd.Timedelta(seconds=cooldown):
            continue

        # Find OBI value signal_win seconds ago
        lo = ts - signal_win_td
        past = ser.loc[lo:ts]
        if len(past) < 2:
            continue
        obi_before = past.iloc[0]
        d_obi = obi_now - obi_before

        if abs(d_obi) < threshold:
            continue

        direction = 1 if d_obi > 0 else -1
        mid_now   = mid_s.asof(ts)
        if pd.isna(mid_now):
            continue

        # Scan forward
        fwd_slice = mid_s.loc[ts : ts + horizon_td]
        if len(fwd_slice) < 2:
            continue

        # Walk through each forward tick checking for hit vs adverse
        hit       = False
        adverse   = False
        best_fwd  = mid_now

        for fwd_ts, fwd_mid in fwd_slice.iloc[1:].items():
            move = (fwd_mid - mid_now) * direction   # positive = correct
            if move >= min_tick:
                hit = True
                best_fwd = fwd_mid
                break
            if move <= -min_tick:
                adverse = True
                best_fwd = fwd_mid
                break

        events.append({
            'ts'         : ts,
            'obi_before' : obi_before,
            'obi_at'     : obi_now,
            'd_obi'      : d_obi,
            'direction'  : direction,
            'mid_at'     : mid_now,
            'mid_fwd'    : best_fwd,
            'fwd_move'   : (best_fwd - mid_now) * direction,
            'hit'        : hit,
            'adverse'    : adverse,
        })

        last_signal_ts = ts

    return pd.DataFrame(events).set_index('ts') if events else pd.DataFrame()


def summarise(results: pd.DataFrame, params: dict) -> str:
    if results.empty:
        return "No signals fired. Try lowering --threshold or --signal-win.\n"

    n        = len(results)
    n_hit    = results['hit'].sum()
    n_adv    = results['adverse'].sum()
    n_none   = n - n_hit - n_adv
    hit_rate = n_hit / n if n else 0
    avg_fwd  = results['fwd_move'].mean()
    avg_hit  = results.loc[results['hit'],  'fwd_move'].mean() if n_hit else float('nan')
    avg_adv  = results.loc[results['adverse'], 'fwd_move'].mean() if n_adv else float('nan')

    # By direction
    buy  = results[results['direction'] ==  1]
    sell = results[results['direction'] == -1]

    lines = [
        "═" * 58,
        "  OBI Delta Signal Test",
        "─" * 58,
        f"  OBI column   : {params['obi_col']}",
        f"  Signal window: {params['signal_win']}s   Horizon: {params['horizon']}s",
        f"  Threshold     : |d_obi| >= {params['threshold']}",
        f"  Min tick move : {params['min_tick']}",
        f"  Cooldown      : {params['cooldown']}s",
        "─" * 58,
        f"  Total signals : {n}",
        f"    BUY  (d_obi > 0) : {len(buy):>4}",
        f"    SELL (d_obi < 0) : {len(sell):>4}",
        "─" * 58,
        f"  Hit rate      : {hit_rate:.1%}  ({n_hit}/{n})",
        f"  Adverse rate  : {n_adv/n:.1%}  ({n_adv}/{n})",
        f"  No-move rate  : {n_none/n:.1%}  ({n_none}/{n})",
        "─" * 58,
        f"  Avg fwd move  : {avg_fwd:+.5f}  (all signals)",
        f"  Avg hit move  : {avg_hit:+.5f}  (hits only)",
        f"  Avg adv move  : {avg_adv:+.5f}  (adverse only)",
    ]

    # Buy / sell breakdown
    for label, sub in [("BUY", buy), ("SELL", sell)]:
        if len(sub) == 0:
            continue
        hr = sub['hit'].sum() / len(sub)
        lines.append(f"  {label} hit rate   : {hr:.1%}  (n={len(sub)})")

    lines.append("═" * 58)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_signals(df: pd.DataFrame, results: pd.DataFrame,
                 obi_col: str, ticker: str, out_path: str,
                 horizon: float):
    BG     = '#0f1117'
    AX_BG  = '#151821'
    GRID   = '#1e2130'
    TEXT   = '#c8d0e0'
    BLUE   = '#00d4ff'
    GREEN  = '#00e676'
    RED    = '#ff1744'
    YELLOW = '#ffd600'
    ORANGE = "#f3a311e6"
    PURPLE = "#d400ff"
    SPINE  = '#2a2f45'

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs  = gridspec.GridSpec(2, 1, hspace=0.08, top=0.93, bottom=0.08,
                            left=0.07, right=0.97)
    ax_mid = fig.add_subplot(gs[0])
    ax_obi = fig.add_subplot(gs[1], sharex=ax_mid)

    for ax in (ax_mid, ax_obi):
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.5, linestyle='--', alpha=0.7)
        for sp in ax.spines.values():
            sp.set_color(SPINE); sp.set_linewidth(0.5)

    # ── Panel 1: mid price + signal markers ─────────────────────────────────
    mid = df['mid'].dropna()
    ax_mid.plot(mid.index, mid.values, color=BLUE, lw=1.0, alpha=0.9,
                label='Kalshi mid')

    if not results.empty:
        hits    = results[results['hit']]
        adverse = results[results['adverse']]
        misses  = results[~results['hit'] & ~results['adverse']]

        def _scatter(sub, color, marker, label, zorder=5):
            if len(sub):
                ax_mid.scatter(sub.index, sub['mid_at'],
                               c=color, marker=marker, s=50, zorder=zorder,
                               label=label, linewidths=0)

        _scatter(hits,   GREEN,  '^', f'Hit (n={len(hits)})')
        _scatter(adverse, RED,    'v', f'Adverse (n={len(adverse)})')
        _scatter(misses,  YELLOW, 'o', f'No-move (n={len(misses)})', zorder=4)

        # Horizon bars for hits
        for ts, row in hits.iterrows():
            ax_mid.annotate('', xy=(ts + pd.Timedelta(seconds=horizon), row['mid_fwd']),
                            xytext=(ts, row['mid_at']),
                            arrowprops=dict(arrowstyle='->', color=GREEN,
                                            lw=0.8, alpha=0.5))

    ax_mid.set_ylabel('YES price ($)', color=TEXT, fontsize=9)
    ax_mid.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.2f}'))
    ax_mid.legend(loc='upper left', fontsize=7.5, framealpha=0.3,
                  labelcolor=TEXT, facecolor=AX_BG, edgecolor=SPINE)
    ax_mid.set_title(f'OBI Signal Test  —  {ticker}', color=TEXT,
                     fontsize=12, fontweight='bold', pad=10)
    ax_mid.tick_params(labelbottom=False)

    # ── Panel 2: OBI + d_obi signal markers ─────────────────────────────────
    obi = df[obi_col].dropna()
    colors_bar = np.where(obi.values >= 0, GREEN, RED)
    ax_obi.bar(obi.index, obi.values,
               width=pd.Timedelta(seconds=0.8),
               color=colors_bar, alpha=0.60, linewidth=0)
    ax_obi.axhline(0, color='#ffffff', lw=0.5, alpha=0.3)

    if not results.empty:
        buy_sigs  = results[results['direction'] ==  1]
        sell_sigs = results[results['direction'] == -1]
        if len(buy_sigs):
            ax_obi.scatter(buy_sigs.index, buy_sigs['obi_at'],
                           c= PURPLE , marker='^', s=55, zorder=5,
                           label='Buy signal', linewidths=0)
        if len(sell_sigs):
            ax_obi.scatter(sell_sigs.index, sell_sigs['obi_at'],
                           c= ORANGE,   marker='v', s=55, zorder=5,
                           label='Sell signal', linewidths=0)
        ax_obi.legend(loc='upper left', fontsize=7.5, framealpha=0.3,
                      labelcolor=TEXT, facecolor=AX_BG, edgecolor=SPINE)

    ax_obi.set_ylabel(f'{obi_col}', color=TEXT, fontsize=9)
    ax_obi.set_ylim(-1.05, 1.05)
    ax_obi.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+.1f}'))

    ax_obi.xaxis.set_major_formatter(
        mdates.DateFormatter('%H:%M', tz=timezone.utc))
    ax_obi.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    plt.setp(ax_obi.xaxis.get_majorticklabels(),
             rotation=0, ha='center', color=TEXT, fontsize=8)
    ax_obi.set_xlabel('Time (UTC)', color=TEXT, fontsize=9)

    plt.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Chart]   -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Micro-price signal test
# ─────────────────────────────────────────────────────────────────────────────

def run_microprice_test(
    df: pd.DataFrame,
    horizon:   float = 1.0,
    threshold: float = 0.003,
    min_tick:  float = 0.01,
    cooldown:  float = 1.0,
) -> pd.DataFrame:
    """
    Signal fires when abs(microprice - mid) >= threshold.

    Direction:
        microprice > mid  →  book leans YES  →  BUY  (+1)
        microprice < mid  →  book leans NO   →  SELL (-1)

    Requires a 'microprice' column in df, produced by merge_plot.py when
    a calibrated g_star.json is present.

    Columns in result:
        ts           signal timestamp
        mid_at       Kalshi mid at signal time
        mp_at        micro-price at signal time
        deviation    microprice - mid  (signed)
        direction    +1 or -1
        mid_fwd      best mid reached in predicted direction within horizon
        fwd_move     (mid_fwd - mid_at) * direction  (positive = correct)
        hit          True if fwd_move >= min_tick without adverse first
        adverse      True if adverse move >= min_tick fired first
    """
    required = {'mid', 'microprice'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Columns {missing} not found. "
            f"Run merge_plot.py with a g_star.json present, or add the "
            f"microprice column via calibrate_microprice.add_microprice().")

    mid_s = df['mid'].ffill()
    mp_s  = df['microprice'].ffill()

    valid = mid_s.notna() & mp_s.notna()
    mid_s = mid_s[valid]
    mp_s  = mp_s[valid]

    if mid_s.empty:
        return pd.DataFrame()

    horizon_td     = pd.Timedelta(seconds=horizon)
    ts_arr         = mid_s.index.to_numpy()
    last_signal_ts = mid_s.index[0] - pd.Timedelta(days=1)
    events         = []

    for ts in ts_arr:
        if ts - last_signal_ts < pd.Timedelta(seconds=cooldown):
            continue

        mid_now = mid_s.asof(ts)
        mp_now  = mp_s.asof(ts)
        if pd.isna(mid_now) or pd.isna(mp_now):
            continue

        deviation = mp_now - mid_now
        if abs(deviation) < threshold:
            continue

        direction = 1 if deviation > 0 else -1

        fwd_slice = mid_s.loc[ts : ts + horizon_td]
        if len(fwd_slice) < 2:
            continue

        hit      = False
        adverse  = False
        best_fwd = mid_now

        for _, fwd_mid in fwd_slice.iloc[1:].items():
            move = (fwd_mid - mid_now) * direction
            if move >= min_tick:
                hit      = True
                best_fwd = fwd_mid
                break
            if move <= -min_tick:
                adverse  = True
                best_fwd = fwd_mid
                break

        events.append({
            'ts'       : ts,
            'mid_at'   : mid_now,
            'mp_at'    : mp_now,
            'deviation': deviation,
            'direction': direction,
            'mid_fwd'  : best_fwd,
            'fwd_move' : (best_fwd - mid_now) * direction,
            'hit'      : hit,
            'adverse'  : adverse,
        })

        last_signal_ts = ts

    return pd.DataFrame(events).set_index('ts') if events else pd.DataFrame()


def summarise_microprice(results: pd.DataFrame, params: dict) -> str:
    """Print summary statistics for a microprice signal test run."""
    if results.empty:
        return "No microprice signals fired. Try lowering --threshold.\n"

    n        = len(results)
    n_hit    = results['hit'].sum()
    n_adv    = results['adverse'].sum()
    n_none   = n - n_hit - n_adv
    hit_rate = n_hit / n

    buy  = results[results['direction'] ==  1]
    sell = results[results['direction'] == -1]

    lines = [
        "═" * 58,
        "  Micro-Price Deviation Signal Test",
        "─" * 58,
        f"  Threshold    : |mp - mid| >= {params['threshold']}",
        f"  Horizon      : {params['horizon']}s",
        f"  Min tick     : {params['min_tick']}",
        f"  Cooldown     : {params['cooldown']}s",
        "─" * 58,
        f"  Total signals: {n}",
        f"    BUY  (mp > mid): {len(buy):>4}",
        f"    SELL (mp < mid): {len(sell):>4}",
        "─" * 58,
        f"  Hit rate     : {hit_rate:.1%}  ({n_hit}/{n})",
        f"  Adverse rate : {n_adv/n:.1%}  ({n_adv}/{n})",
        f"  No-move rate : {n_none/n:.1%}  ({n_none}/{n})",
        "─" * 58,
        f"  Avg deviation: {results['deviation'].abs().mean():+.6f}",
        f"  Avg fwd move : {results['fwd_move'].mean():+.5f}",
    ]
    for label, sub in [("BUY", buy), ("SELL", sell)]:
        if len(sub):
            hr = sub['hit'].sum() / len(sub)
            lines.append(f"  {label} hit rate  : {hr:.1%}  (n={len(sub)})")
    lines.append("═" * 58)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Test whether OBI delta predicts Kalshi mid-price moves.')
    ap.add_argument('merged_csv',
                    help='Merged CSV produced by merge_plot.py')
    ap.add_argument('--signal',     type=str,   default='obi',
                    choices=['obi', 'microprice'],
                    help="Signal type: 'obi' (default) or 'microprice'")
    ap.add_argument('--horizon',    type=float, default=1.0,
                    help='Forward look window in seconds (default 1.0)')
    ap.add_argument('--threshold',  type=float, default=0.10,
                    help='Min |d_obi| to fire (obi mode) or min |mp-mid| in $ (microprice mode)')
    ap.add_argument('--signal-win', type=float, default=0.5,
                    help='Backward window for OBI delta in seconds (default 0.5, obi mode only)')
    ap.add_argument('--min-tick',   type=float, default=0.01,
                    help='Min mid move ($) to count as a hit (default 0.01)')
    ap.add_argument('--cooldown',   type=float, default=1.0,
                    help='Seconds to suppress re-firing (default 1.0)')
    ap.add_argument('--depth',      type=str,   default='obi',
                    help="OBI column: 'obi' (parametric), '1','3','5','10' for obi1 etc.")
    ap.add_argument('--out-dir',
                    help='Output directory (default: same dir as merged_csv)')
    args = ap.parse_args()

    path = Path(args.merged_csv)
    if not path.exists():
        print(f"[Error] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Infer ticker from filename — must happen before anything else
    stem   = path.stem
    ticker = stem[len('merged_'):] if stem.startswith('merged_') else stem

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(path, index_col='ts', parse_dates=['ts'])
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.sort_index()

    print(f"[Loaded]  {len(df):,} rows  ticker={ticker}")

    # ── Run test ─────────────────────────────────────────────────────────────
    if args.signal == 'microprice':
        results = run_microprice_test(
            df,
            horizon   = args.horizon,
            threshold = args.threshold,
            min_tick  = args.min_tick,
            cooldown  = args.cooldown,
        )
        mp_params = dict(threshold=args.threshold, horizon=args.horizon,
                         min_tick=args.min_tick, cooldown=args.cooldown)
        print(summarise_microprice(results, mp_params))
    else:
        obi_col = args.depth if args.depth.startswith('obi') else f'obi{args.depth}'
        params  = dict(obi_col=obi_col, horizon=args.horizon,
                       signal_win=args.signal_win, threshold=args.threshold,
                       min_tick=args.min_tick, cooldown=args.cooldown)
        results = run_test(df, **params)
        print(summarise(results, params))

    if results.empty:
        sys.exit(0)

    # ── Save results CSV ─────────────────────────────────────────────────────
    csv_out = out_dir / f'signals_{ticker}.csv'
    results.to_csv(csv_out)
    print(f"[Output]  -> {csv_out}")

    # ── Plot (obi mode only) ──────────────────────────────────────────────────
    if args.signal == 'obi':
        obi_col   = args.depth if args.depth.startswith('obi') else f'obi{args.depth}'
        chart_out = out_dir / f'signals_{ticker}.png'
        plot_signals(df, results, obi_col, ticker, str(chart_out), args.horizon)


if __name__ == '__main__':
    main()