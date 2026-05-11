"""
Part 3 -- Misvaluation Analysis (Andrew)
Quantifies the magnitude, frequency, and duration of ETF price-to-iNAV gaps
for SPY, IVV, and VOO.

Gap definitions (per ETF):
    basis         = ETF_mid  - iNAV_mid   (signed, $)
    basis_bps     = basis / iNAV_mid * 10_000
    arb_sell_etf  = ETF_bid  - iNAV_ask   (profit/share if ETF is overpriced)
    arb_buy_etf   = iNAV_bid - ETF_ask    (profit/share if ETF is underpriced)

Input  : part3_output_rescaled/inav_<YYYYMMDD>_<etf>.parquet  (per date+ETF, from inav_calculator.py)
         Fallback: data_v2_rescaled/<YYYYMMDD>/<ETF>_*.parquet  (raw per-(date, ETF) files)
Output : part3_output_rescaled/misvaluation_summary.csv  (rows for every date+ETF)
         part3_output_rescaled/plots/basis_bps_distribution.png      (per-ETF panels, all dates)
         part3_output_rescaled/plots/gap_duration_distribution.png   (per-ETF x 2 legs)
         part3_output_rescaled/plots/basis_timeseries_by_date.png    (date x ETF grid)
"""

import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- Configuration -------------------------------------------------------------

INAV_DIR     = 'part3_output_rescaled'
FALLBACK_DIR = 'data_v2_rescaled'
OUTPUT_DIR   = 'part3_output_rescaled'
LEVERAGE     = {'spy': 1.0, 'ivv': 1.0, 'voo': 1.0, 'spxl': 3.0, 'spxs': -3.0}
ETFS         = list(LEVERAGE.keys())
BPS_THRESH   = [1, 2, 5, 10]

# -- Data Loading --------------------------------------------------------------

_INAV_FNAME_RE = re.compile(r'^inav_(?P<date>\d{8})_(?P<etf>[A-Za-z]+)\.parquet$')
_RAW_FNAME_RE  = re.compile(r'^(?P<etf>[A-Za-z]+)_(?P<date>\d{8})_', re.IGNORECASE)


def find_date_etf_inputs(inav_dir: str,
                         fallback_dir: str) -> list[tuple[str, str, list[str]]]:
    """Return [(date_tag, etf_lower, [parquet_paths]), ...].

    Prefers inav_<date>_<etf>.parquet outputs; falls back to raw
    <ETF>_<date>_*.parquet files inside date-named subdirectories.
    """
    out: list[tuple[str, str, list[str]]] = []
    for f in sorted(glob.glob(os.path.join(inav_dir, 'inav_*.parquet'))):
        m = _INAV_FNAME_RE.match(os.path.basename(f))
        if m:
            out.append((m.group('date'), m.group('etf').lower(), [f]))
    if out:
        return out

    if os.path.isdir(fallback_dir):
        for d in sorted(os.listdir(fallback_dir)):
            full = os.path.join(fallback_dir, d)
            if not os.path.isdir(full):
                continue
            by_etf: dict[str, list[str]] = {}
            for f in sorted(glob.glob(os.path.join(full, '*.parquet'))):
                m = _RAW_FNAME_RE.match(os.path.basename(f))
                if m:
                    by_etf.setdefault(m.group('etf').lower(), []).append(f)
            for etf, files in sorted(by_etf.items()):
                out.append((d, etf, files))
        if out:
            return out

    raise FileNotFoundError(
        'No parquet files found. Run inav_calculator.py first.')


def load_files_for_etf(files: list[str], etf: str) -> pd.DataFrame:
    """Load and concat parquets, renaming generic raw-data columns
    (etf_bid/ask/mid) to ETF-prefixed columns expected by the analysis."""
    df = (pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            .sort_values('time_m')
            .reset_index(drop=True))
    rename = {f'etf_{c}': f'{etf}_{c}'
              for c in ('bid', 'ask', 'mid')
              if f'etf_{c}' in df.columns and f'{etf}_{c}' not in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df

# -- iNAV Column Injection (fallback when calculator has not yet been run) -----

def _ensure_inav_cols(df: pd.DataFrame, etf: str) -> pd.DataFrame:
    """Return df (possibly a copy) with {etf}_inav_bid/ask/mid guaranteed."""
    if f'{etf}_inav_mid' in df.columns:
        return df
    leverage = LEVERAGE.get(etf, 1.0)
    if leverage != 1.0:
        raise KeyError(
            f'No iNAV columns for leveraged ETF {etf.upper()} (L={leverage:+.1f}). '
            'Leveraged iNAV must be computed by inav_calculator.py first -- '
            'the on-the-fly fallback only handles unleveraged trackers.')
    if 'synth_mid' not in df.columns:
        raise KeyError(
            f'No iNAV columns for {etf.upper()} and no synth_* fallback. '
            'Run part3_inav_calculator.py first.')
    # synth_* in the rescaled raw data is already aligned to this ETF's open,
    # so for L=1 it IS the iNAV (no further scaling needed).
    out = df.copy()
    out[f'{etf}_inav_bid'] = df['synth_bid']
    out[f'{etf}_inav_ask'] = df['synth_ask']
    out[f'{etf}_inav_mid'] = df['synth_mid']
    return out

# -- Gap Computation -----------------------------------------------------------

def compute_gaps(df: pd.DataFrame, etf: str) -> pd.DataFrame:
    df   = _ensure_inav_cols(df, etf)
    bid  = f'{etf}_bid';   ask  = f'{etf}_ask';   mid  = f'{etf}_mid'
    ibid = f'{etf}_inav_bid'; iask = f'{etf}_inav_ask'; imid = f'{etf}_inav_mid'

    mask = df[mid].notna() & df[imid].notna()
    d    = df[mask].reset_index(drop=True)

    out = pd.DataFrame({
        'time_m':       d['time_m'],
        'etf_mid':      d[mid],
        'etf_bid':      d[bid],
        'etf_ask':      d[ask],
        'inav_mid':     d[imid],
        'inav_bid':     d[ibid],
        'inav_ask':     d[iask],
        'basis':        d[mid]  - d[imid],
        'basis_bps':   (d[mid]  - d[imid]) / d[imid] * 10_000,
        'arb_sell_etf': d[bid]  - d[iask],   # profit if ETF overpriced
        'arb_buy_etf':  d[ibid] - d[ask],    # profit if ETF underpriced
    })
    out['arb_spread'] = np.maximum(
        np.maximum(out['arb_sell_etf'], out['arb_buy_etf']), 0)
    return out.reset_index(drop=True)

# -- Episode Detection ---------------------------------------------------------

def detect_episodes(times: pd.Series, signal: pd.Series,
                    threshold: float = 0.0) -> pd.DataFrame:
    """
    Find contiguous runs where signal > threshold.
    Returns columns: start, end, duration_ms, peak, mean.
    """
    above = (signal > threshold).values
    if not above.any():
        return pd.DataFrame(columns=['start', 'end', 'duration_ms', 'peak', 'mean'])

    episodes, in_ep = [], False
    for i, v in enumerate(above):
        if v and not in_ep:
            s, in_ep = i, True
        elif not v and in_ep:
            e = i - 1
            episodes.append(_ep_record(times, signal, s, e))
            in_ep = False
    if in_ep:
        episodes.append(_ep_record(times, signal, s, len(above) - 1))
    return pd.DataFrame(episodes)


def _ep_record(times, signal, s, e) -> dict:
    dur = (times.iloc[e] - times.iloc[s]).total_seconds() * 1_000
    return {
        'start':       times.iloc[s],
        'end':         times.iloc[e],
        'duration_ms': dur,
        'peak':        signal.iloc[s:e + 1].max(),
        'mean':        signal.iloc[s:e + 1].mean(),
    }

# -- Summary Statistics --------------------------------------------------------

def compute_summary(gaps: pd.DataFrame, etf: str) -> dict:
    n = len(gaps)
    s = {'etf': etf.upper(), 'n_rows': n}

    for col, lbl in [('basis',     'basis_$'),
                     ('basis_bps', 'basis_bps'),
                     ('arb_spread','arb_spread_$')]:
        v = gaps[col]
        s.update({
            f'{lbl}_mean': v.mean(),
            f'{lbl}_std':  v.std(),
            f'{lbl}_p50':  v.median(),
            f'{lbl}_p95':  v.quantile(0.95),
            f'{lbl}_p99':  v.quantile(0.99),
            f'{lbl}_max':  v.abs().max(),
        })

    for thr in BPS_THRESH:
        s[f'pct_above_{thr}bps'] = 100 * (gaps['basis_bps'].abs() > thr).mean()

    for leg, col in [('sell_etf', 'arb_sell_etf'), ('buy_etf', 'arb_buy_etf')]:
        eps = detect_episodes(gaps['time_m'], gaps[col])
        s[f'{leg}_n_episodes'] = len(eps)
        if not eps.empty:
            s[f'{leg}_dur_median_ms'] = eps['duration_ms'].median()
            s[f'{leg}_dur_p95_ms']    = eps['duration_ms'].quantile(0.95)
            s[f'{leg}_dur_max_ms']    = eps['duration_ms'].max()
            s[f'{leg}_peak_mean_$']   = eps['peak'].mean()
    return s

# -- Plots ---------------------------------------------------------------------

def _save(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path}')


def plot_basis_distribution_all(per_etf_gaps: dict, plot_dir: str) -> None:
    """One figure, 3 panels (one per ETF), histogram of basis_bps pooled across all dates."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, etf in zip(axes, ETFS):
        days = per_etf_gaps.get(etf, {})
        if not days:
            ax.set_visible(False)
            continue
        all_basis = pd.concat([g['basis_bps'] for g in days.values()])
        ax.hist(all_basis, bins=80, color='steelblue',
                alpha=0.8, edgecolor='white', lw=0.3)
        ax.axvline(0, color='k', lw=0.9)
        for thr, color in zip(BPS_THRESH, ['gold', 'orange', 'tomato', 'darkred']):
            ax.axvline( thr, color=color, lw=0.9, ls='--', label=f'+/-{thr} bps')
            ax.axvline(-thr, color=color, lw=0.9, ls='--')
        ax.set_xlabel('Basis (bps)')
        ax.set_title(f'{etf.upper()}  ({len(days)} days, n={len(all_basis):,})')
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
    fig.suptitle('Basis Distribution -- All Trading Days (ETF_mid - iNAV_mid)')
    plt.tight_layout()
    _save(fig, os.path.join(plot_dir, 'basis_bps_distribution.png'))


def plot_gap_durations_all(per_etf_gaps: dict, plot_dir: str) -> None:
    """One figure, 3x2 panels (rows=ETFs, cols=arb leg), pooled across dates."""
    fig, axes = plt.subplots(len(ETFS), 2, figsize=(12, 4 * len(ETFS)), sharex=True)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    for row, etf in enumerate(ETFS):
        days = per_etf_gaps.get(etf, {})
        for col, (signal_col, label, color) in enumerate([
            ('arb_sell_etf', 'Sell ETF / Buy Basket', 'tomato'),
            ('arb_buy_etf',  'Buy ETF / Sell Basket', 'steelblue'),
        ]):
            ax = axes[row, col]
            durs = []
            for g in days.values():
                eps = detect_episodes(g['time_m'], g[signal_col])
                if not eps.empty:
                    durs.extend(eps['duration_ms'].tolist())
            if durs:
                ax.hist(durs, bins=40, color=color,
                        alpha=0.8, edgecolor='white', lw=0.3)
                med = float(np.median(durs))
                p95 = float(np.percentile(durs, 95))
                ax.axvline(med, color='k', lw=1.0, ls='--',
                           label=f'med={med:.1f} ms  p95={p95:.1f} ms  n={len(durs)}')
                ax.legend(fontsize=8)
            ax.set_title(f'{etf.upper()} -- {label}')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            if row == len(ETFS) - 1:
                ax.set_xlabel('Episode Duration (ms)')
    fig.suptitle('Arbitrage Gap Episode Durations -- All Trading Days')
    plt.tight_layout()
    _save(fig, os.path.join(plot_dir, 'gap_duration_distribution.png'))


def plot_basis_timeseries_grid(per_etf_gaps: dict, plot_dir: str) -> None:
    """Small multiples: rows=dates, cols=ETFs. Lets you spot intraday patterns
    across days without producing one file per (date, etf)."""
    dates = sorted({d for days in per_etf_gaps.values() for d in days})
    if not dates:
        return
    fig, axes = plt.subplots(len(dates), len(ETFS),
                             figsize=(5 * len(ETFS), 2.4 * len(dates)),
                             sharex='col')
    axes = np.atleast_2d(axes)
    if axes.shape != (len(dates), len(ETFS)):
        axes = axes.reshape(len(dates), len(ETFS))
    for r, date_tag in enumerate(dates):
        for c, etf in enumerate(ETFS):
            ax = axes[r, c]
            g = per_etf_gaps.get(etf, {}).get(date_tag)
            if g is None or g.empty:
                ax.set_visible(False)
                continue
            ax.plot(g['time_m'], g['basis_bps'], lw=0.5, color='steelblue')
            ax.axhline(0, color='k', lw=0.7)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            if r == 0:
                ax.set_title(etf.upper())
            if c == 0:
                ax.set_ylabel(f'{date_tag}\nbasis (bps)', fontsize=9)
    fig.suptitle('Basis (bps) Over Time -- Rows=Date, Columns=ETF')
    fig.autofmt_xdate()
    plt.tight_layout()
    _save(fig, os.path.join(plot_dir, 'basis_timeseries_by_date.png'))

# -- Main ----------------------------------------------------------------------

def run(inav_dir: str = INAV_DIR, fallback_dir: str = FALLBACK_DIR,
        output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    inputs = find_date_etf_inputs(inav_dir, fallback_dir)
    by_date: dict[str, list[str]] = {}
    for d, e, _ in inputs:
        by_date.setdefault(d, []).append(e)
    print(f'Found {len(inputs)} (date, ETF) inputs across {len(by_date)} day(s):')
    for d, es in by_date.items():
        print(f'  {d}: {es}')

    summaries    = []
    per_etf_gaps = {etf: {} for etf in ETFS}

    for date_tag, etf, files in inputs:
        if etf not in LEVERAGE:
            print(f'\n[{date_tag} {etf.upper()}] not in LEVERAGE config -- skipping')
            continue
        print(f'\n[{date_tag} {etf.upper()}]')
        df = load_files_for_etf(files, etf)
        print(f'  Loaded {len(files)} file(s) -> {len(df):,} rows '
              f'[{df["time_m"].min()} - {df["time_m"].max()}]')

        if f'{etf}_mid' not in df.columns:
            print(f'  market data columns absent -- skipping')
            continue
        gaps = compute_gaps(df, etf)
        per_etf_gaps[etf][date_tag] = gaps
        s = compute_summary(gaps, etf)
        s['date'] = date_tag
        summaries.append(s)
        print(f'  basis_bps mean={s["basis_bps_mean"]:.3f}  '
              f'std={s["basis_bps_std"]:.3f}  '
              f'|max|={s["basis_bps_max"]:.3f}  '
              f'>1bps={s["pct_above_1bps"]:.1f}%')

    print('\n-- Combined plots ---------------------------------------------')
    plot_basis_distribution_all(per_etf_gaps, plot_dir)
    plot_gap_durations_all(per_etf_gaps, plot_dir)
    plot_basis_timeseries_grid(per_etf_gaps, plot_dir)

    out  = pd.DataFrame(summaries)
    if not out.empty:
        cols = ['date', 'etf'] + [c for c in out.columns if c not in ('date', 'etf')]
        out  = out[cols]
    path = os.path.join(output_dir, 'misvaluation_summary.csv')
    out.to_csv(path, index=False)
    print(f'\nSaved -> {path}  ({len(out)} rows)')
    return out


if __name__ == '__main__':
    run()
