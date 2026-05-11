"""
Part 3 -- iNAV Calculator (Andrew)
Derives the True Intraday Net Asset Value for SPY, IVV, VOO, SPXL, and SPXS
from the per-ETF parquet files produced by build_dataset.py.

Data layout assumed (data_v2_rescaled/):
    <YYYYMMDD>/<ETF>_<YYYYMMDD>_<HHMMSS>_<HHMMSS>.parquet
Each file has generic columns:
    time_m, etf_bid, etf_ask, etf_mid, synth_bid, synth_ask, synth_mid
where synth_* is the S&P 500 basket NAV already rescaled to that ETF's open.

For unleveraged trackers (SPY/IVV/VOO, leverage = 1) the per-ETF rescaled
synth IS the iNAV (a 1x scale factor preserves the arbitrage relationship),
so the calculator just renames synth_* -> {etf}_inav_*.

For leveraged ETFs (SPXL = +3, SPXS = -3) the linear rescaling produces the
WRONG iNAV intraday; instead we anchor to the open and propagate the basket's
percent return at the leverage factor:
    inav_X = etf_mid_open * (1 + L * (synth_X - synth_X_open) / synth_X_open)
For inverse ETFs the bid/ask sides of synth flip when applied to the ETF
because the ETF's lower side corresponds to the basket's upper side.

Output : part3_output_rescaled/inav_<YYYYMMDD>_<etf>.parquet  (one per date+ETF)
         Columns: time_m, {etf}_bid, {etf}_ask, {etf}_mid,
                  {etf}_inav_bid, {etf}_inav_ask, {etf}_inav_mid,
                  inav_compute_ns
"""

import glob
import os
import re
import time

import numpy as np
import pandas as pd

# -- Configuration -------------------------------------------------------------

DATA_DIR   = 'data_v2_rescaled'
OUTPUT_DIR = 'part3_output_rescaled'

# Daily leverage factor relative to the S&P 500 basket return.
LEVERAGE = {'spy': 1.0, 'ivv': 1.0, 'voo': 1.0, 'spxl': 3.0, 'spxs': -3.0}
ETFS     = list(LEVERAGE.keys())
N_STOCKS = 500
N_TRIALS = 100_000

# Filename pattern: TICKER_YYYYMMDD_*.parquet (case-insensitive ticker).
_FNAME_RE = re.compile(r'^(?P<etf>[A-Za-z]+)_(?P<date>\d{8})_', re.IGNORECASE)

# -- Discovery -----------------------------------------------------------------

def find_data_dirs(root: str) -> list[str]:
    """Return subdirectories of `root` that contain parquet files (one per
    trading day), or [root] itself if it directly holds parquets."""
    subdirs = sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and glob.glob(os.path.join(root, d, '*.parquet'))
    )
    if subdirs:
        return subdirs
    if glob.glob(os.path.join(root, '*.parquet')):
        return [root]
    raise FileNotFoundError(f'No parquet files found in {root!r} or its subdirectories')


def group_by_etf(data_dir: str) -> dict[str, list[str]]:
    """Group parquet files in `data_dir` by ETF prefix in the filename."""
    by_etf: dict[str, list[str]] = {}
    for f in sorted(glob.glob(os.path.join(data_dir, '*.parquet'))):
        m = _FNAME_RE.match(os.path.basename(f))
        if not m:
            continue
        by_etf.setdefault(m.group('etf').lower(), []).append(f)
    return by_etf

# -- iNAV Computation ----------------------------------------------------------

def compute_inav(df: pd.DataFrame, etf: str, leverage: float) -> pd.DataFrame:
    """
    Build a DataFrame with time_m, the ETF quotes, and the iNAV columns.

    Input df has generic columns (etf_bid/ask/mid, synth_bid/ask/mid) where
    synth_* is already pre-rescaled to this ETF's opening price.
    """
    out = pd.DataFrame({
        'time_m':       df['time_m'],
        f'{etf}_bid':   df['etf_bid'],
        f'{etf}_ask':   df['etf_ask'],
        f'{etf}_mid':   df['etf_mid'],
    })

    if leverage == 1.0:
        out[f'{etf}_inav_bid'] = df['synth_bid']
        out[f'{etf}_inav_ask'] = df['synth_ask']
        out[f'{etf}_inav_mid'] = df['synth_mid']
        return out

    idx = df['etf_mid'].first_valid_index()
    if idx is None:
        raise ValueError(f'No valid etf_mid data to anchor leveraged iNAV for {etf.upper()}')
    etf_open = float(df.loc[idx, 'etf_mid'])
    s_mid_o  = float(df.loc[idx, 'synth_mid'])
    s_bid_o  = float(df.loc[idx, 'synth_bid'])
    s_ask_o  = float(df.loc[idx, 'synth_ask'])

    pct_mid = (df['synth_mid'] - s_mid_o) / s_mid_o
    if leverage > 0:
        pct_bid = (df['synth_bid'] - s_bid_o) / s_bid_o
        pct_ask = (df['synth_ask'] - s_ask_o) / s_ask_o
    else:
        pct_bid = (df['synth_ask'] - s_ask_o) / s_ask_o
        pct_ask = (df['synth_bid'] - s_bid_o) / s_bid_o

    out[f'{etf}_inav_bid'] = etf_open * (1 + leverage * pct_bid)
    out[f'{etf}_inav_ask'] = etf_open * (1 + leverage * pct_ask)
    out[f'{etf}_inav_mid'] = etf_open * (1 + leverage * pct_mid)
    return out

# -- Compute Latency Benchmark -------------------------------------------------

def benchmark_inav_kernel(n_stocks: int = N_STOCKS, n_trials: int = N_TRIALS) -> float:
    """
    Mean nanoseconds for the core iNAV arithmetic:
        iNAV = dot(prices, shares) / shares_outstanding
    Lower-bound compute latency; production systems add unmarshalling and
    scheduling overhead.
    """
    rng    = np.random.default_rng(42)
    prices = rng.uniform(50, 500, n_stocks)
    shares = rng.integers(1_000_000, 50_000_000, n_stocks).astype(np.float64)
    shrout = 9_500_000_000.0

    for _ in range(1_000):           # warm-up cache
        _ = np.dot(prices, shares) / shrout

    t0 = time.perf_counter_ns()
    for _ in range(n_trials):
        _ = np.dot(prices, shares) / shrout
    return (time.perf_counter_ns() - t0) / n_trials

# -- Main ----------------------------------------------------------------------

def process_dir(data_dir: str, output_dir: str, compute_ns: float) -> list[str]:
    by_etf = group_by_etf(data_dir)
    if not by_etf:
        print(f'  No matching parquet files in {data_dir!r}')
        return []

    print(f'  Found ETFs: {sorted(by_etf.keys())}')
    written = []
    for etf in sorted(by_etf):
        files = by_etf[etf]
        if etf not in LEVERAGE:
            print(f'  {etf.upper()}: not in LEVERAGE config -- skipping ({len(files)} files)')
            continue
        leverage = LEVERAGE[etf]

        df = (pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
                .sort_values('time_m')
                .reset_index(drop=True))

        try:
            result = compute_inav(df, etf, leverage)
        except ValueError as e:
            print(f'  {etf.upper()}: {e} -- skipping')
            continue

        result['inav_compute_ns'] = compute_ns

        date_tag = result['time_m'].dt.date.iloc[0].strftime('%Y%m%d')
        out_path = os.path.join(output_dir, f'inav_{date_tag}_{etf}.parquet')
        result.to_parquet(out_path, index=False)

        spread = result[f'{etf}_inav_ask'] - result[f'{etf}_inav_bid']
        print(f'  {etf.upper()}  L={leverage:+.1f}  files={len(files)}  rows={len(result):,}  '
              f'iNAV mid [{result[f"{etf}_inav_mid"].min():.3f}, '
              f'{result[f"{etf}_inav_mid"].max():.3f}]  '
              f'mean_spread={spread.mean():.4f}  -> {out_path}')
        written.append(out_path)
    return written


def run(data_dir: str = DATA_DIR, output_dir: str = OUTPUT_DIR) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    dirs = find_data_dirs(data_dir)

    compute_ns = benchmark_inav_kernel()
    print(f'iNAV compute benchmark  ({N_STOCKS} stocks, {N_TRIALS:,} trials): '
          f'{compute_ns:.1f} ns  ({compute_ns / 1_000:.3f} us) per tick')

    written = []
    for d in dirs:
        print(f'\n=== {d} ===')
        written.extend(process_dir(d, output_dir, compute_ns))

    print(f'\nWrote {len(written)} per-(date, ETF) parquet(s) to {output_dir!r}')
    return written


if __name__ == '__main__':
    run()
