"""
merge_and_plot.py  —  Kalshi order book replay + BTC as-of join + visualization
================================================================================

Merges an arbitrary Kalshi 15M market window with its paired BTC feed.
BTC file is assumed to be named  BTC-<kalshi_ticker>.csv  in the same directory
unless --btc is passed explicitly.

Usage:
    python merge_and_plot.py <kalshi_csv> [--btc <btc_csv>] [--out-dir <dir>]

Examples:
    python merge_and_plot.py KXBTC15M-26MAR161930-30.csv
    python merge_and_plot.py rawdata/KXBTC15M-26MAR162000-00.csv --out-dir results/

Outputs (written to --out-dir, default = same dir as kalshi_csv):
    merged_<ticker>.csv    tick-level DataFrame (one row per Kalshi delta)
    chart_<ticker>.png     3-panel visualization

─────────────────────────────────────────────────────────────────────────────
Design notes on BTC price source
─────────────────────────────────────────────────────────────────────────────
We record two Coinbase channels: `ticker` and `market_trades`.

`ticker`        fires on every best-bid/ask change (every ~200ms).  Each message
                carries the current best_bid, best_ask, and last trade price.
                This is the right source for mid-quote: it reflects the live
                order book state, not just completed prints.

`market_trades` fires only when a trade executes.  It gives you the exact
                matched price and size, which is useful for volume analysis
                but noisy as a mid-price reference (trades print at bid or ask,
                so alternating buy/sell prints give a sawtooth that overstates
                realized volatility).

Decision: use `ticker` channel exclusively, with btc_mid = (best_bid + best_ask) / 2.
This matches what you would use in a live signal: the current mid-quote, not
the last print. Filter out the initial snapshot event (type == "snapshot") —
it carries stale trades history, not a live book update.
"""

import sys
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import timezone

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_LEVELS  = 100    # price levels $0.01 … $0.99
TICK      = 0.01   # cents

# ── OBI parameters ────────────────────────────────────────────────────────────
# OBI_DEPTH : how many price levels (per side) to include in the imbalance calc.
#             1 = top-of-book only; 10 = deep book.  Try 3–5 for signal work. Max 99 total levesl 
OBI_DEPTH = 5

# OBI_DECAY : exponential decay applied across levels (weight ∝ exp(-decay * rank)).
#             0.0 = uniform (original behaviour).
#             0.5 = mild down-weighting of deeper levels.
#             1.0 = aggressive down-weighting (top level ~2.7× weight of level 2).
OBI_DECAY = 0.5

# How much BTC data on either side of the Kalshi window to keep.
# Buffer applied around the Kalshi window when clipping BTC data.
# 5s is enough — the collector pre-connects ~60s early to catch the opening
# snapshot, but we only need a tiny buffer to guarantee a BTC tick exists
# before the first Kalshi delta at market open.
BTC_BUFFER = pd.Timedelta(seconds=5)

# Path to a calibrated g_star.json produced by calibrate_microprice.py.
# Set to None (or leave the file absent) to skip microprice computation.
# When running from analysis/, this resolves to analysis/g_star.json.
GSTAR_PATH = 'g_star.json'

# ─────────────────────────────────────────────────────────────────────────────
# Timestamp parser  (handles both 6-digit and 9-digit sub-seconds) 
# ─────────────────────────────────────────────────────────────────────────────

_TS_RE = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z')

def parse_ts(s: str) -> pd.Timestamp:
    m = _TS_RE.match(s)
    if not m:
        raise ValueError(f"Unparseable timestamp: {s!r}")
    base = m.group(1)
    frac = m.group(2) or '0'   # no sub-seconds -> treat as 0
    us = int(frac[:6].ljust(6, '0'))
    return pd.Timestamp(base, tz='UTC') + pd.Timedelta(microseconds=us)

# ─────────────────────────────────────────────────────────────────────────────
# Array-based Kalshi order book
# ─────────────────────────────────────────────────────────────────────────────

def _price_to_idx(p: str) -> int:
    """'0.17' → 16  (0-based; index i ↔ price (i+1)*$0.01)"""
    return round(float(p) * 100) - 1


class OrderBook:
    """
    Two 100-slot arrays, one per side.

    yes_qty[i]  dollar-qty resting on YES side at price (i+1)*$0.01
    no_qty[i]   dollar-qty resting on NO  side at price (i+1)*$0.01

    Mirror convention:
        best_yes_bid  = highest i with yes_qty[i] > 0  → price in [0.01, 0.99]
        best_no_bid   = highest i with no_qty[i]  > 0  → price in [0.01, 0.99]
        best_yes_ask  = 1 - best_no_bid   (NO buyer at q ↔ YES seller at 1-q)
        spread        = best_yes_ask - best_yes_bid
        mirror check  = best_yes_bid + best_no_bid  ≈  0.99  (1-tick spread)
    """

    def __init__(self):
        self.yes_qty   = np.zeros(N_LEVELS, dtype=np.float64)
        self.no_qty    = np.zeros(N_LEVELS, dtype=np.float64)
        self.ready     = False

    def apply_snapshot(self, msg: dict):
        self.yes_qty[:] = 0
        self.no_qty[:]  = 0
        for p, q in msg.get('yes_dollars_fp', []):
            idx = _price_to_idx(p)
            if 0 <= idx < N_LEVELS:
                self.yes_qty[idx] = float(q)
        for p, q in msg.get('no_dollars_fp', []):
            idx = _price_to_idx(p)
            if 0 <= idx < N_LEVELS:
                self.no_qty[idx] = float(q)
        self.ready = True

    def apply_delta(self, msg: dict):
        idx   = _price_to_idx(msg['price_dollars'])
        delta = float(msg['delta_fp'])
        if not (0 <= idx < N_LEVELS):
            return
        if msg['side'] == 'yes':
            self.yes_qty[idx] = max(0.0, self.yes_qty[idx] + delta)
        else:
            self.no_qty[idx]  = max(0.0, self.no_qty[idx]  + delta)

    # ── Derived quantities ────────────────────────────────────────────

    def best_yes_bid(self) -> float | None:
        idxs = np.flatnonzero(self.yes_qty)
        if idxs.size == 0:
            return None
        return round((idxs[-1] + 1) * TICK, 2)

    def best_yes_ask(self) -> float | None:
        """Best YES ask = 1 - best NO bid (highest NO price with depth)."""
        idxs = np.flatnonzero(self.no_qty)
        if idxs.size == 0:
            return None
        return round(1.0 - (idxs[-1] + 1) * TICK, 2)

    def mid(self) -> float | None:
        b, a = self.best_yes_bid(), self.best_yes_ask()
        return round((b + a) / 2, 4) if b is not None and a is not None else None

    def spread(self) -> float | None:
        b, a = self.best_yes_bid(), self.best_yes_ask()
        return round(a - b, 2) if b is not None and a is not None else None

    def obi(self, depth: int = OBI_DEPTH, decay: float = OBI_DECAY) -> float | None:
        """
        OBI = (yes_weighted - no_weighted) / (yes_weighted + no_weighted)

        Compares the `depth` most aggressive YES bids against the `depth` most
        aggressive NO bids.  Positive => market leans YES; Negative => market leans NO.

        decay > 0 applies exponential down-weighting by rank so that level k
        (0-indexed from top) gets weight exp(-decay * k).  decay=0 is uniform
        (original behaviour).
        """
        yes_idx = sorted([i for i in range(N_LEVELS) if self.yes_qty[i] > 0],
                         reverse=True)[:depth]
        no_idx  = sorted([i for i in range(N_LEVELS) if self.no_qty[i]  > 0],
                         reverse=True)[:depth]

        if decay == 0.0:
            y = sum(self.yes_qty[i] for i in yes_idx)
            n = sum(self.no_qty[i]  for i in no_idx)
        else:
            y = sum(self.yes_qty[i] * np.exp(-decay * k)
                    for k, i in enumerate(yes_idx))
            n = sum(self.no_qty[i]  * np.exp(-decay * k)
                    for k, i in enumerate(no_idx))

        return (y - n) / (y + n) if (y + n) > 0 else None

    def total_depth(self) -> tuple[float, float]:
        return float(self.yes_qty.sum()), float(self.no_qty.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Micro-price helpers (requires a calibrated g_star.json from calibrate_microprice.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_gstar(path: str | None) -> dict | None:
    """Load a calibrated G* lookup table. Returns None if path is absent/None."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[G*]      {path} not found — microprice column will be skipped")
        return None
    cal = json.loads(p.read_text())
    print(f"[G*]      loaded {path}  "
          f"(n_imb={cal['n_imb']}, max_spread={cal['max_spread']})")
    return cal


def lookup_microprice(mid: float,
                      obi1: float,
                      spread: float,
                      cal: dict | None) -> float | None:
    """Return microprice = mid + G*[s_bkt][i_bkt], or None if cal is absent."""
    if cal is None or mid is None:
        return None
    n    = cal['n_imb']
    ms   = cal['max_spread']
    tick = cal.get('tick', 0.01)

    i_bkt = int((obi1 + 1.0) / 2.0 * n)
    i_bkt = max(0, min(n - 1, i_bkt))

    s_bkt = max(1, round(spread / tick)) - 1
    s_bkt = min(s_bkt, ms - 1)

    adj = cal['g_star'][s_bkt][i_bkt]
    return round(mid + adj, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Parse Kalshi CSV → tick-level DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def parse_kalshi(path: str,
                 cal: dict | None = None) -> tuple[pd.DataFrame, str]:
    book   = OrderBook()
    rows   = []
    ticker = None

    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw) #Reads into the json
            except json.JSONDecodeError:
                continue

            t = msg.get('type')
            if t == 'orderbook_snapshot':
                inner  = msg['msg']
                ticker = inner.get('market_ticker')
                book.apply_snapshot(inner)

            elif t == 'orderbook_delta':
                if not book.ready:
                    continue
                inner = msg['msg']
                book.apply_delta(inner)
                y_depth, n_depth = book.total_depth()
                rows.append({
                    'ts'       : parse_ts(inner['ts']),
                    'seq'      : msg.get('seq'),
                    'side'     : inner['side'],
                    'price'    : float(inner['price_dollars']),
                    'delta'    : float(inner['delta_fp']),
                    'mid'      : book.mid(),
                    'bid'      : book.best_yes_bid(),
                    'ask'      : book.best_yes_ask(),
                    'spread'   : book.spread(),
                    # Parametric OBI -- controlled by OBI_DEPTH / OBI_DECAY at module top
                    'obi'      : book.obi(OBI_DEPTH, OBI_DECAY),
                    # Legacy fixed-depth columns (uniform weighting) kept for backward compat
                    'obi1'     : book.obi(1,  0.0),
                    'obi3'     : book.obi(3,  0.0),
                    'obi5'     : book.obi(5,  0.0),
                    'obi10'    : book.obi(10, 0.0),
                    'yes_depth': y_depth,
                    'no_depth' : n_depth,
                    'microprice': lookup_microprice(
                                      book.mid(), book.obi(1, 0.0),
                                      book.spread() or 0.01, cal),
                })

    if not rows:
        raise RuntimeError(f"No orderbook_delta rows in {path}")

    df = (pd.DataFrame(rows)
            .set_index('ts')
            .sort_index())

    print(f"[Kalshi]  {len(df):,} deltas  "
          f"{df.index[0].strftime('%H:%M:%S')} → {df.index[-1].strftime('%H:%M:%S')} UTC  "
          f"ticker={ticker}")
    return df, ticker


# ─────────────────────────────────────────────────────────────────────────────
# Parse BTC CSV → ticker-channel DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def parse_btc(path: str,
              clip_start: pd.Timestamp,
              clip_end:   pd.Timestamp) -> pd.DataFrame:
    """
    Extract only `ticker` channel update events from the Coinbase feed.

    Why ticker-only, why update-only:
      - `market_trades` gives last-print prices which alternate between bid
        and ask on each trade, producing artificial micro-noise. Mid-quote
        from the order book (best_bid + best_ask)/2 is a cleaner reference.
      - The initial `snapshot` event in the ticker channel is a copy of the
        last known state before subscription; its timestamp is the WebSocket
        connect time, not a real market event. We skip it to avoid a stale
        price being as-of joined into early Kalshi rows.
    """
    rows = []
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get('channel') != 'ticker':
                continue

            ts_raw = msg.get('timestamp', '')
            if not ts_raw:
                continue
            ts = parse_ts(ts_raw)

            # Clip to Kalshi window (exact, no buffer — anything outside is noise)
            if ts < clip_start or ts > clip_end:
                continue

            for ev in msg.get('events', []):
                # Skip the stale snapshot delivered at subscription time
                if ev.get('type') != 'update':
                    continue
                for tk in ev.get('tickers', []):
                    bid_s = tk.get('best_bid')
                    ask_s = tk.get('best_ask')
                    if bid_s is None or ask_s is None:
                        continue
                    bid, ask = float(bid_s), float(ask_s)
                    rows.append({
                        'ts'     : ts,
                        'btc_bid': bid,
                        'btc_ask': ask,
                        'btc_mid': (bid + ask) / 2,
                    })

    if not rows:
        raise RuntimeError(f"No BTC ticker-update rows in [{clip_start}, {clip_end}] in {path}")

    df = (pd.DataFrame(rows)
            .set_index('ts')
            .sort_index())
    df = df[~df.index.duplicated(keep='last')]   # keep last if same microsecond

    print(f"[BTC]     {len(df):,} ticker updates  "
          f"{df.index[0].strftime('%H:%M:%S')} → {df.index[-1].strftime('%H:%M:%S')} UTC  "
          f"clipped to [{clip_start.strftime('%H:%M:%S')}, {clip_end.strftime('%H:%M:%S')}]")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# As-of join
# ─────────────────────────────────────────────────────────────────────────────

def asof_join(kalshi_df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the most recent preceding BTC mid-quote to each Kalshi delta.
    Strictly backward — no lookahead, no interpolation.
    """
    merged = pd.merge_asof(
        kalshi_df.reset_index().sort_values('ts'),
        btc_df.reset_index().sort_values('ts'),
        on='ts',
        direction='backward',
    ).set_index('ts')

    n_nan = merged['btc_mid'].isna().sum()
    if n_nan:
        print(f"[Join]    {n_nan} rows without a preceding BTC tick (first few messages)")
    print(f"[Join]    {len(merged):,} merged rows")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame, ticker: str,
         market_open: pd.Timestamp, market_close: pd.Timestamp,
         out_path: str):

    BG      = '#0f1117'
    AX_BG   = '#151821'
    GRID    = '#1e2130'
    TEXT    = '#c8d0e0'
    BLUE    = '#00d4ff'
    ORANGE  = '#ff6b35'
    GREEN   = '#00e676'
    RED     = '#ff1744'
    PURPLE  = '#d400ff'
    SPINE   = '#2a2f45'

    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    gs  = gridspec.GridSpec(3, 1, hspace=0.06,
                            top=0.93, bottom=0.07, left=0.07, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    for ax in axes:
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.5, linestyle='--', alpha=0.7)
        for sp in ax.spines.values():
            sp.set_color(SPINE); sp.set_linewidth(0.5)
        ax.axvspan(market_open, market_close, color='#ffffff', alpha=0.025)
        ax.axvline(market_open,  color='#ffffff', alpha=0.18, lw=0.8, ls='--')
        ax.axvline(market_close, color='#ffffff', alpha=0.18, lw=0.8, ls='--')

    ts = df.index

    # ── Panel 1: Kalshi mid + bid/ask band ───────────────────────────────────
    ax = axes[0]
    ax.fill_between(ts, df['bid'], df['ask'], color=BLUE, alpha=0.10)
    ax.plot(ts, df['mid'], color=BLUE,  lw=1.3, label='Kalshi mid (YES)', zorder=3)
    if 'microprice' in df.columns:
        mp = df['microprice'].dropna()
        ax.plot(mp.index, mp.values, color=PURPLE, lw=0.8, alpha=0.75,
                label='Micro-price', linestyle='--', zorder=4)
    ax.plot(ts, df['bid'], color=GREEN, lw=0.6, alpha=0.65, label='Best bid')
    ax.plot(ts, df['ask'], color=RED,   lw=0.6, alpha=0.65, label='Best ask')
    ax.set_ylabel('YES price ($)', color=TEXT, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.2f}'))
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.25,
              labelcolor=TEXT, facecolor=AX_BG, edgecolor=SPINE)
    ax.set_title(f'Market Replay  —  {ticker}', color=TEXT,
                 fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(labelbottom=False)
    for label, xpos in [('open', market_open), ('resolution', market_close)]:
        ax.annotate(label, xy=(xpos, ax.get_ylim()[0]),
                    xytext=(3, 4), textcoords='offset points',
                    color='#ffffff', alpha=0.35, fontsize=6.5)

    # ── Panel 2: BTC mid-price ───────────────────────────────────────────────
    ax = axes[1]
    btc = df['btc_mid'].dropna()
    ax.plot(btc.index, btc.values, color=ORANGE, lw=1.0, alpha=0.9,
            label='BTC-USD mid')
    ax.set_ylabel('BTC-USD ($)', color=TEXT, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.25,
              labelcolor=TEXT, facecolor=AX_BG, edgecolor=SPINE)
    ax.tick_params(labelbottom=False)

    # ── Panel 3: OBI (3-level) bar chart ────────────────────────────────────
    ax = axes[2]
    obi = df['obi'].dropna()  # parametric OBI (depth=OBI_DEPTH, decay=OBI_DECAY)

    # Downsample purely for rendering speed — data is not modified
    if len(obi) > 3000:
        step = len(obi) // 3000
        obi  = obi.iloc[::step]

    colors = np.where(obi.values >= 0, GREEN, RED)
    ax.bar(obi.index, obi.values,
           width=pd.Timedelta(seconds=1.2),
           color=colors, alpha=0.75, linewidth=0)
    ax.axhline(0, color='#ffffff', lw=0.5, alpha=0.35)
    ax.set_ylabel(f'OBI (depth={OBI_DEPTH}, decay={OBI_DECAY})', color=TEXT, fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+.1f}'))

    # ── Shared x-axis ────────────────────────────────────────────────────────
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter('%H:%M', tz=timezone.utc))
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(),
             rotation=0, ha='center', color=TEXT, fontsize=8)
    axes[-1].set_xlabel('Time (UTC)', color=TEXT, fontsize=9)

    plt.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Chart]   → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Merge a Kalshi 15M CSV with its paired BTC feed and plot.')
    ap.add_argument('kalshi_csv',
                    help='Path to the Kalshi CSV, e.g. KXBTC15M-26MAR161930-30.csv')
    ap.add_argument('--btc',
                    help='Path to the BTC CSV (default: BTC-<kalshi_csv> in same dir)')
    ap.add_argument('--out-dir',
                    help='Output directory (default: same dir as kalshi_csv)')
    args = ap.parse_args()

    kalshi_path = Path(args.kalshi_csv)

    # Derive BTC path from naming convention if not given
    btc_path = Path(args.btc) if args.btc else \
               kalshi_path.parent / f'BTC-{kalshi_path.name}'
    if not btc_path.exists():
        print(f"[Error] BTC file not found: {btc_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else kalshi_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse Kalshi ─────────────────────────────────────────────────────────
    cal = load_gstar(GSTAR_PATH)
    kalshi_df, ticker = parse_kalshi(str(kalshi_path), cal=cal)

    # Window = exact range of Kalshi deltas (the recorder already handles buffers)
    market_open  = kalshi_df.index[0]
    market_close = kalshi_df.index[-1]

    # ── Parse BTC (clipped to Kalshi window) ────────────────────────────────
    btc_df = parse_btc(str(btc_path),
                       market_open  - BTC_BUFFER,
                       market_close + BTC_BUFFER)

    # ── Merge ────────────────────────────────────────────────────────────────
    merged = asof_join(kalshi_df, btc_df)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_out = out_dir / f'merged_{ticker}.csv'
    merged.to_csv(csv_out)
    print(f"[Output]  → {csv_out}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    chart_out = out_dir / f'chart_{ticker}.png'
    plot(merged, ticker, market_open, market_close, str(chart_out))

    # ── Sanity stats ─────────────────────────────────────────────────────────
    print('\n── Sanity ────────────────────────────────────────────────────')
    print(merged[['mid', 'spread', 'obi', 'btc_mid']].describe().round(4))
    mirror = (merged['bid'] + merged['no_depth'].apply(lambda _: 0)  # just for shape
              + 0).copy()   # placeholder — real check below
    mirror = (merged['bid'] + (1.0 - merged['ask'])).dropna()
    print(f'\nMirror check  best_yes_bid + best_no_bid ≈ 0.99')
    print(f'  mean={mirror.mean():.4f}  std={mirror.std():.5f}  '
          f'min={mirror.min():.4f}  max={mirror.max():.4f}')


if __name__ == '__main__':
    main()