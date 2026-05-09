"""
signal_runner.py
================
Loads all sessions from one or more data directories, runs every registered
signal, and returns a unified results DataFrame.

Usage
-----
    python signal_runner.py --data-dir completed-data/ --out results.parquet

    # Multiple data directories
    python signal_runner.py --data-dir data1/ data2/ --out results.parquet

    # Force recalibrate G* before running
    python signal_runner.py --data-dir completed-data/ --recalibrate

In Python (e.g. from dashboard.py):
    from signal_runner import run_all
    results_df, session_stats = run_all(data_dirs=['completed-data/'])
"""

import argparse
import glob
import sys
import warnings
from pathlib import Path

import pandas as pd

import merge_plot
from signals import (
    OBIDeltaSignal,
    MicropriceSignal,
    SpreadFilteredOBI,
    TimeWindowOBI,
)
import calibrate_microprice

warnings.filterwarnings('ignore', category=FutureWarning)


# ── Signal registry ───────────────────────────────────────────────────────────
# Add new signals here — the runner and dashboard pick them up automatically.

def build_signals(gstar_path: str = 'g_star.json') -> list:
    """Instantiate all signals with their default parameters."""
    return [
        OBIDeltaSignal(
            obi_col        = 'obi3',
            horizon        = 1.0,
            signal_win     = 0.25,
            threshold      = 0.40,
            min_tick       = 0.01,
            cooldown       = 0.5,
        ),
        MicropriceSignal(
            gstar_path = gstar_path,
            threshold  = 0.001,
            horizon    = 1.0,
            min_tick   = 0.01,
            cooldown   = 0.5,
        ),
        SpreadFilteredOBI(
            max_spread_ticks = 1,
            obi_col          = 'obi3',
            horizon          = 1.0,
            signal_win       = 0.25,
            threshold        = 0.40,
            min_tick         = 0.01,
            cooldown         = 0.5,
        ),
        TimeWindowOBI(
            start_sec  = 60,
            end_sec    = 840,
            obi_col    = 'obi3',
            horizon    = 1.0,
            signal_win = 0.25,
            threshold  = 0.40,
            min_tick   = 0.01,
            cooldown   = 0.5,
        ),
    ]


# ── Session loading ───────────────────────────────────────────────────────────

def find_session_pairs(data_dirs: list[str]) -> list[tuple[Path, Path]]:
    """
    Find all (kalshi_csv, btc_csv) pairs across data_dirs.
    Expects BTC files named BTC-<kalshi_stem>.csv in the same directory.
    """
    pairs = []
    for d in data_dirs:
        for kpath in sorted(Path(d).glob('KXBTC15M-*.csv')):
            bpath = kpath.parent / f'BTC-{kpath.name}'
            if bpath.exists():
                pairs.append((kpath, bpath))
            else:
                print(f'  [skip] no BTC file for {kpath.name}', file=sys.stderr)
    return pairs


def load_session(kpath: Path, bpath: Path,
                 cache_dir: Path | None = None) -> pd.DataFrame | None:
    """
    Parse one Kalshi+BTC pair into a merged DataFrame.

    If cache_dir is set, looks for a pre-cached parquet at
    cache_dir/<ticker>.parquet. If found, loads it directly (fast).
    If not found, parses the raw CSVs and saves the result to cache.
    This means each session is only replayed once no matter how many
    times you run the dashboard, calibration, or signal runner.
    """
    ticker_name = kpath.stem

    # ── Try cache first ───────────────────────────────────────────────────────
    REQUIRED_COLS = {'mid', 'obi1', 'spread', 'ticker'}
    if cache_dir is not None:
        cache_path = cache_dir / f'{ticker_name}.parquet'
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                # Invalidate cache if columns added since it was written
                if REQUIRED_COLS.issubset(df.columns):
                    return df
                else:
                    missing = REQUIRED_COLS - set(df.columns)
                    print(f'  [recache] {ticker_name}: missing {missing}',
                          file=sys.stderr)
                    cache_path.unlink()   # delete stale cache
            except Exception as e:
                print(f'  [cache miss] {ticker_name}: {e}', file=sys.stderr)

    # ── Parse from raw CSVs ───────────────────────────────────────────────────
    try:
        kdf, _ = merge_plot.parse_kalshi(str(kpath))
        bdf    = merge_plot.parse_btc(
            str(bpath),
            kdf.index[0]  - merge_plot.BTC_BUFFER,
            kdf.index[-1] + merge_plot.BTC_BUFFER,
        )
        mdf = merge_plot.asof_join(kdf, bdf)
        mdf['ticker']       = ticker_name
        mdf['market_open']  = kdf.index[0]
        mdf['market_close'] = kdf.index[-1]

        # ── Save to cache ─────────────────────────────────────────────────────
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f'{ticker_name}.parquet'
            try:
                mdf.to_parquet(cache_path)
            except Exception as e:
                print(f'  [cache write failed] {ticker_name}: {e}',
                      file=sys.stderr)

        return mdf
    except Exception as e:
        print(f'  [error] {ticker_name}: {e}', file=sys.stderr)
        return None


# ── Calibration ───────────────────────────────────────────────────────────────

def recalibrate(
    sessions: list[pd.DataFrame],
    gstar_path: str   = 'g_star.json',
    n_imb:      int   = 10,
    max_spread: int   = 3,
    min_count:  int   = 30,
    train_frac: float = 0.8,
) -> None:
    """Re-estimate G* using the first train_frac of sessions."""
    tickers_sorted = sorted({df['ticker'].iloc[0] for df in sessions})
    split          = max(1, int(len(tickers_sorted) * train_frac))
    train_set      = set(tickers_sorted[:split])

    all_records = []
    for df in sessions:
        if df['ticker'].iloc[0] not in train_set:
            continue
        sub  = df[['mid', 'obi1', 'spread']].dropna()
        recs = calibrate_microprice.extract_transitions(sub, n_imb, max_spread)
        all_records.extend(recs)

    sym            = calibrate_microprice.symmetrize(all_records, n_imb)
    Q, T, R, K, c = calibrate_microprice.build_matrices(sym, n_imb, max_spread, min_count)
    G_star         = calibrate_microprice.solve_g_star(Q, T, R, K, c, min_count)
    calibrate_microprice.save_g_star(G_star, n_imb, max_spread, c, Path(gstar_path))
    print(f'[calibrate] G* saved to {gstar_path}  '
          f'range=[{G_star.min():+.6f}, {G_star.max():+.6f}]')


# ── Main runner ───────────────────────────────────────────────────────────────

def run_all(
    data_dirs:   list[str],
    gstar_path:  str  = 'g_star.json',
    recalibrate_gstar: bool = False,
    train_frac:  float = 0.8,
    cache_dir:   str | None = '../results/session_cache',
    verbose:     bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all sessions, run all signals, return:
        results_df    — one row per signal event, all signals combined
        session_stats — per-signal per-session summary statistics

    Sessions are cached to cache_dir as parquet after first parse.
    On subsequent runs each session loads in ~0.1s instead of ~10s.
    Set cache_dir=None to disable caching.
    """
    cache = Path(cache_dir) if cache_dir else None

    # ── Load sessions ─────────────────────────────────────────────────────────
    pairs = find_session_pairs(data_dirs)
    if not pairs:
        raise RuntimeError(f"No session pairs found in {data_dirs}")

    cached_count = 0
    if cache:
        cached_count = sum(1 for kp, _ in pairs
                           if (cache / f'{kp.stem}.parquet').exists())

    print(f'[runner] Found {len(pairs)} session pairs  '
          f'({cached_count} already cached)')

    sessions = []
    for kpath, bpath in pairs:
        df = load_session(kpath, bpath, cache_dir=cache)
        if df is not None:
            sessions.append(df)
    print(f'[runner] Loaded {len(sessions)} sessions')

    # ── (Re-)calibrate G* if requested ───────────────────────────────────────
    if recalibrate_gstar:
        print('[runner] Recalibrating G*...')
        recalibrate(sessions, gstar_path=gstar_path, train_frac=train_frac)

    # ── Build signals ─────────────────────────────────────────────────────────
    signal_list = build_signals(gstar_path=gstar_path)

    # Fit signals that need calibration (MicropriceSignal auto-loads from disk)
    for sig in signal_list:
        if hasattr(sig, 'load'):
            sig.load()

    # ── Run all signals on all sessions ───────────────────────────────────────
    all_results = []

    for df in sessions:
        ticker = df['ticker'].iloc[0]
        for sig in signal_list:
            try:
                res = sig.evaluate(df)
                if not res.empty:
                    res = res.copy()
                    res['ticker'] = ticker
                    all_results.append(res)
            except Exception as e:
                if verbose:
                    print(f'  [warn] {sig.name} on {ticker}: {e}', file=sys.stderr)

    if not all_results:
        return pd.DataFrame(), pd.DataFrame()

    results_df = pd.concat(all_results).sort_index()

    # ── Compute per-signal per-session statistics ─────────────────────────────
    def _agg(g):
        n   = len(g)
        hr  = g['hit'].mean()
        ar  = g['adverse'].mean()
        fwd = g['fwd_move'].mean()
        return pd.Series({
            'n_signals'   : n,
            'hit_rate'    : round(hr, 4),
            'adverse_rate': round(ar, 4),
            'avg_fwd_move': round(fwd, 5),
            'no_move_rate': round(1 - hr - ar, 4),
        })

    session_stats = (
        results_df
        .groupby(['signal_name', 'ticker'])
        .apply(_agg, include_groups=False)
        .reset_index()
    )

    return results_df, session_stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Run all signals across all sessions.')
    ap.add_argument('--data-dir',     nargs='+', default=['../data'],
                    help='One or more directories containing session CSV pairs '
                         '(default: ../data)')
    ap.add_argument('--gstar',        default='g_star.json',
                    help='Path to g_star.json (default: g_star.json)')
    ap.add_argument('--out',          default='../results/results.parquet',
                    help='Output parquet path (default: ../results/results.parquet)')
    ap.add_argument('--recalibrate',  action='store_true',
                    help='Recalibrate G* from data before running signals')
    ap.add_argument('--cache-dir',    default='../results/session_cache',
                    help='Directory to cache merged session parquets '
                         '(default: ../results/session_cache). '
                         'Pass --cache-dir "" to disable.')
    args = ap.parse_args()

    results_df, session_stats = run_all(
        data_dirs          = args.data_dir,
        gstar_path         = args.gstar,
        recalibrate_gstar  = args.recalibrate,
        train_frac         = args.train_frac,
        cache_dir          = args.cache_dir or None,
    )

    if results_df.empty:
        print('[runner] No results — check your data directory and signal config.')
        sys.exit(1)

    # Save
    out = Path(args.out)
    results_df.to_parquet(out)
    stats_out = out.with_name(out.stem + '_stats.parquet')
    session_stats.to_parquet(stats_out)

    # Print summary table
    summary = (
        results_df
        .groupby('signal_name')
        .apply(lambda g: pd.Series({
            'n_signals'   : len(g),
            'n_sessions'  : g['ticker'].nunique(),
            'hit_rate'    : f"{g['hit'].mean():.1%}",
            'adverse_rate': f"{g['adverse'].mean():.1%}",
            'avg_fwd_move': f"{g['fwd_move'].mean():+.5f}",
        }), include_groups=False)
    )
    print('\n── Signal Summary ────────────────────────────────────────────')
    print(summary.to_string())
    print(f'\n[runner] Results → {out}')
    print(f'[runner] Stats   → {stats_out}')


if __name__ == '__main__':
    main()