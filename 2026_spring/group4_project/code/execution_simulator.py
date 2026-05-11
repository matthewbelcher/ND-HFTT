"""
Part 3 -- Execution Latency & Arbitrage Viability Simulator (Andrew)
Benchmarks Time-to-Compute against Duration of Misvaluation and simulates
a trading engine to determine the actual viability of an HFT arbitrage strategy.

Latency pipeline (co-located server, one round trip):
    [feed receipt] -> [iNAV compute] -> [decision] -> [order routing]
    -> [exchange matching] -> [fill confirmation]

Three scenarios are modeled:
    Co-located Optimistic  -- best-case co-lo (~16 uss total)
    Co-located Realistic   -- typical co-lo  (~100 uss total)
    Remote (1 ms DC)       -- off-site server (~1.6 ms total)

NOTE: Dataset resolution is ~25 ms.  Gap durations measured here are
      lower-bound estimates -- nanosecond TAQ data will reveal sub-ms structure.

Input  : part3_output_rescaled/inav_<YYYYMMDD>_<etf>.parquet  (per date+ETF, from inav_calculator.py)
         Fallback: data_v2_rescaled/<YYYYMMDD>/<ETF>_*.parquet  (raw per-(date, ETF) files)
Output : part3_output_rescaled/execution_viability.csv  (rows for every date+ETF+scenario)
         part3_output_rescaled/plots/latency_vs_gaps.png             (per-ETF panels, all dates)
         part3_output_rescaled/plots/spread_magnitude_distribution.png (per-ETF panels, all dates)
         part3_output_rescaled/plots/pnl_by_date.png                 (per-date bars by ETF)
"""

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

FEE_PER_SHARE  = 0.003   # $/share per leg (typical exchange fee)
SLIPPAGE_BPS   = 0.5    # assumed execution slippage per leg (bps of notional)
MIN_ARB_BPS    = 1.0    # minimum basis (bps) required to attempt a trade

# -- Latency Model -------------------------------------------------------------

@dataclass
class LatencyModel:
    name:        str
    feed_us:     float   # market data feed delivery
    compute_us:  float   # iNAV arithmetic kernel
    decision_us: float   # signal check + order construction
    routing_us:  float   # order routing to exchange
    matching_us: float   # exchange order matching engine
    confirm_us:  float   # fill confirmation receipt

    @property
    def total_us(self) -> float:
        return (self.feed_us + self.compute_us + self.decision_us
                + self.routing_us + self.matching_us + self.confirm_us)

    @property
    def total_ms(self) -> float:
        return self.total_us / 1_000

    def print_breakdown(self) -> None:
        steps = [
            ('Feed receipt',      self.feed_us),
            ('iNAV computation',  self.compute_us),
            ('Decision logic',    self.decision_us),
            ('Order routing',     self.routing_us),
            ('Exchange matching', self.matching_us),
            ('Fill confirmation', self.confirm_us),
        ]
        print(f'\n  -- {self.name} --')
        for label, us in steps:
            print(f'    {label:<22}  {us:8.1f} us')
        print(f'    {"TOTAL":<22}  {self.total_us:8.1f} us  ({self.total_ms:.3f} ms)')


_BASE_SCENARIOS: List[LatencyModel] = [
    LatencyModel('Co-located (Optimistic)',
                 feed_us=1,   compute_us=1,  decision_us=0.5,
                 routing_us=5,   matching_us=10,  confirm_us=5),
    LatencyModel('Co-located (Realistic)',
                 feed_us=5,   compute_us=5,  decision_us=2,
                 routing_us=20,  matching_us=50,  confirm_us=20),
    LatencyModel('Remote (1 ms DC proximity)',
                 feed_us=500, compute_us=10, decision_us=5,
                 routing_us=500, matching_us=100, confirm_us=500),
]

# -- Data Loading --------------------------------------------------------------

_INAV_FNAME_RE = re.compile(r'^inav_(?P<date>\d{8})_(?P<etf>[A-Za-z]+)\.parquet$')
_RAW_FNAME_RE  = re.compile(r'^(?P<etf>[A-Za-z]+)_(?P<date>\d{8})_', re.IGNORECASE)


def find_date_etf_inputs(inav_dir: str,
                         fallback_dir: str) -> List[Tuple[str, str, List[str]]]:
    """Return [(date_tag, etf_lower, [parquet_paths]), ...]."""
    out: List[Tuple[str, str, List[str]]] = []
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
            by_etf: Dict[str, List[str]] = {}
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


def load_files_for_etf(files: List[str], etf: str) -> Tuple[pd.DataFrame, Optional[float]]:
    """Returns (df, benchmarked_compute_ns | None). Renames generic etf_*
    columns to {etf}_* if the raw fallback layout was used."""
    df = (pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            .sort_values('time_m')
            .reset_index(drop=True))
    rename = {f'etf_{c}': f'{etf}_{c}'
              for c in ('bid', 'ask', 'mid')
              if f'etf_{c}' in df.columns and f'{etf}_{c}' not in df.columns}
    if rename:
        df = df.rename(columns=rename)
    compute_ns = (float(df['inav_compute_ns'].iloc[0])
                  if 'inav_compute_ns' in df.columns else None)
    return df, compute_ns


def build_scenarios(benchmarked_ns: Optional[float]) -> List[LatencyModel]:
    """Override the placeholder compute_us with the benchmarked value if available."""
    if benchmarked_ns is None:
        return _BASE_SCENARIOS
    us = benchmarked_ns / 1_000
    return [
        LatencyModel(l.name, l.feed_us, us, l.decision_us,
                     l.routing_us, l.matching_us, l.confirm_us)
        for l in _BASE_SCENARIOS
    ]

# -- iNAV Column Injection (fallback when calculator has not been run) ---------

def _ensure_inav_cols(df: pd.DataFrame, etf: str) -> pd.DataFrame:
    if f'{etf}_inav_mid' in df.columns:
        return df
    leverage = LEVERAGE.get(etf, 1.0)
    if leverage != 1.0:
        raise KeyError(
            f'No iNAV columns for leveraged ETF {etf.upper()} (L={leverage:+.1f}). '
            'Leveraged iNAV must be computed by inav_calculator.py first -- '
            'the on-the-fly fallback only handles unleveraged trackers.')
    if 'synth_mid' not in df.columns:
        raise KeyError(f'No iNAV data for {etf.upper()}. '
                       'Run part3_inav_calculator.py first.')
    # synth_* in the rescaled raw data is already aligned to this ETF's open,
    # so for L=1 it IS the iNAV (no further scaling needed).
    out = df.copy()
    out[f'{etf}_inav_bid'] = df['synth_bid']
    out[f'{etf}_inav_ask'] = df['synth_ask']
    out[f'{etf}_inav_mid'] = df['synth_mid']
    return out

# -- Arbitrage Spread Extraction -----------------------------------------------

def get_arb_spreads(df: pd.DataFrame, etf: str) -> pd.DataFrame:
    df   = _ensure_inav_cols(df, etf)
    bid  = f'{etf}_bid';    ask  = f'{etf}_ask';    mid  = f'{etf}_mid'
    ibid = f'{etf}_inav_bid'; iask = f'{etf}_inav_ask'; imid = f'{etf}_inav_mid'

    mask = df[mid].notna() & df[imid].notna()
    d    = df[mask].reset_index(drop=True)

    out = pd.DataFrame({
        'time_m':       d['time_m'].values,
        'etf_bid':      d[bid].values,
        'etf_ask':      d[ask].values,
        'etf_mid':      d[mid].values,
        'inav_bid':     d[ibid].values,
        'inav_ask':     d[iask].values,
        'inav_mid':     d[imid].values,
        'arb_sell_etf': d[bid].values  - d[iask].values,
        'arb_buy_etf':  d[ibid].values - d[ask].values,
    })
    out['arb_spread'] = np.maximum(
        np.maximum(out['arb_sell_etf'], out['arb_buy_etf']), 0)
    out['basis_bps']  = (out['etf_mid'] - out['inav_mid']) / out['inav_mid'] * 10_000
    return out

# -- Viability Assessment ------------------------------------------------------

def assess_viability(gaps: pd.DataFrame, lat: LatencyModel, etf: str,
                      slippage_bps: float = SLIPPAGE_BPS) -> dict:
    """
    For each positive arb_spread row, estimate whether the opportunity
    would survive long enough to be executed at the given latency.

    Because the dataset has ~25 ms resolution, we cannot directly observe
    whether a gap lasts longer than our sub-ms latency.  We therefore:
      (a) report how many ticks show a positive spread (lower bound on
          opportunity count -- shorter gaps would not appear at all), and
      (b) flag ticks where the spread exceeds MIN_ARB_BPS as 'viable'.
    With nanosecond TAQ data, gap duration can be measured precisely.
    """
    pos  = gaps[gaps['arb_spread'] > 0].copy()
    n    = len(gaps)
    fee      = FEE_PER_SHARE * 2                            # both legs
    slippage = slippage_bps / 10_000 * pos['inav_mid'] * 2  # both legs

    pos['gross']    = pos['arb_spread']
    pos['slippage'] = slippage
    pos['net']      = pos['gross'] - fee - slippage
    min_spread      = MIN_ARB_BPS / 10_000 * pos['inav_mid']
    pos['viable']   = pos['gross'] > min_spread

    data_interval_ms = (gaps['time_m'].diff()
                                      .dt.total_seconds()
                                      .median() * 1_000)

    return {
        'etf':                    etf.upper(),
        'latency_scenario':       lat.name,
        'total_latency_us':       lat.total_us,
        'total_latency_ms':       lat.total_ms,
        'data_resolution_ms':     round(data_interval_ms, 3),
        'n_rows':                 n,
        'n_positive_spread':      len(pos),
        'pct_positive_spread':    round(100 * len(pos) / n, 2),
        'n_viable_opportunities': int(pos['viable'].sum()),
        'pct_viable_of_positive': round(100 * pos['viable'].mean(), 2) if len(pos) else 0,
        'mean_gross_$':           round(pos['gross'].mean(), 4) if len(pos) else 0,
        'mean_net_$':             round(pos['net'].mean(),   4) if len(pos) else 0,
        'max_gross_$':            round(pos['gross'].max(),  4) if len(pos) else 0,
        'total_gross_$':          round(pos['gross'].sum(),  4),
        'total_net_$':            round(pos['net'].sum(),    4),
        'slippage_bps':           slippage_bps,
        'mean_slippage_$':       round(pos['slippage'].mean(), 4) if len(pos) else 0,
        'n_net_positive':         int((pos['net'] > 0).sum()),
    }

# -- Gap Duration Utility ------------------------------------------------------

def _gap_episode_durations(gaps: pd.DataFrame) -> List[float]:
    """Collect duration (ms) of each contiguous run of positive arb_spread."""
    above = (gaps['arb_spread'] > 0).values
    durs, in_ep = [], False
    for i, v in enumerate(above):
        if v and not in_ep:
            s, in_ep = i, True
        elif not v and in_ep:
            dur = (gaps.loc[i - 1, 'time_m'] - gaps.loc[s, 'time_m']
                   ).total_seconds() * 1_000
            durs.append(dur)
            in_ep = False
    if in_ep:
        dur = (gaps.iloc[-1]['time_m'] - gaps.loc[s, 'time_m']
               ).total_seconds() * 1_000
        durs.append(dur)
    return durs

# -- Plots ---------------------------------------------------------------------

def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path}')


def plot_latency_vs_gaps_all(per_etf_gaps: dict, scenarios: List[LatencyModel],
                              plot_dir: str) -> None:
    """One figure, 3 panels (one per ETF). Histogram of gap episode durations
    pooled across all dates with vertical lines at each latency scenario."""
    fig, axes = plt.subplots(1, len(ETFS), figsize=(5 * len(ETFS), 5), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    colors = ['tomato', 'darkorange', 'purple']
    for ax, etf in zip(axes, ETFS):
        days = per_etf_gaps.get(etf, {})
        all_durs = []
        for g in days.values():
            all_durs.extend(_gap_episode_durations(g))
        if not all_durs:
            ax.set_visible(False)
            continue
        ax.hist(all_durs, bins=40, color='steelblue',
                alpha=0.8, edgecolor='white', lw=0.3,
                label=f'Gap durations (n={len(all_durs)})')
        for lat, color in zip(scenarios, colors):
            ax.axvline(lat.total_ms, color=color, lw=1.2, ls='--',
                       label=f'{lat.name}: {lat.total_ms:.2f} ms')
        ax.set_xlabel('Episode Duration (ms)')
        ax.set_title(f'{etf.upper()}  ({len(days)} days)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel('Count')
    fig.suptitle('Arbitrage Gap Duration vs Execution Latency '
                 '(left of a latency line = too short to execute)')
    plt.tight_layout()
    _save(fig, os.path.join(plot_dir, 'latency_vs_gaps.png'))


def plot_spread_distribution_all(per_etf_gaps: dict, plot_dir: str,
                                 slippage_bps: float = SLIPPAGE_BPS) -> None:
    """One figure, 3 panels. Histogram of positive arb_spread (cents/share)
    pooled across dates, with break-even lines for fees + slippage."""
    fig, axes = plt.subplots(1, len(ETFS), figsize=(5 * len(ETFS), 5), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, etf in zip(axes, ETFS):
        days = per_etf_gaps.get(etf, {})
        if not days:
            ax.set_visible(False)
            continue
        pos = pd.concat([g[g['arb_spread'] > 0] for g in days.values()])
        if pos.empty:
            ax.set_visible(False)
            continue
        spread_cents       = pos['arb_spread'] * 100
        slippage_per_share = slippage_bps / 10_000 * pos['inav_mid'].mean() * 2
        total_cost_cents   = (FEE_PER_SHARE * 2 + slippage_per_share) * 100
        ax.hist(spread_cents, bins=60, color='steelblue',
                alpha=0.8, edgecolor='white', lw=0.3)
        ax.axvline(FEE_PER_SHARE * 2 * 100, color='orange', lw=1.0, ls='--',
                   label=f'Fee only: {FEE_PER_SHARE*2*100:.1f}c')
        ax.axvline(total_cost_cents, color='tomato', lw=1.0, ls='--',
                   label=f'Fee+slip: {total_cost_cents:.2f}c')
        ax.set_xlabel('Arb Spread (cents/share)')
        ax.set_title(f'{etf.upper()}  (positive spread ticks: {len(pos):,})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel('Count')
    fig.suptitle('Positive Arbitrage Spread Distribution -- All Trading Days')
    plt.tight_layout()
    _save(fig, os.path.join(plot_dir, 'spread_magnitude_distribution.png'))


def plot_pnl_by_date(per_etf_gaps: dict, plot_dir: str,
                     slippage_bps: float = SLIPPAGE_BPS) -> None:
    """Bar chart of total post-cost P&L per (date, ETF) -- compact summary
    that replaces N per-day cumulative timelines."""
    rows = []
    for etf, days in per_etf_gaps.items():
        for date_tag, g in days.items():
            pos = g[g['arb_spread'] > 0]
            if pos.empty:
                rows.append({'date': date_tag, 'etf': etf.upper(), 'net_$': 0.0})
                continue
            slippage = slippage_bps / 10_000 * pos['inav_mid'] * 2
            net = (pos['arb_spread'] - 2 * FEE_PER_SHARE - slippage).sum()
            rows.append({'date': date_tag, 'etf': etf.upper(), 'net_$': float(net)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Reorder ETF columns to match the canonical ETFS order, dropping any absent.
    canonical = [e.upper() for e in ETFS]
    cols      = [c for c in canonical if c in df['etf'].unique()]
    pivot     = (df.pivot(index='date', columns='etf', values='net_$')
                   .reindex(columns=cols).sort_index())
    palette   = ['steelblue', 'tomato', 'mediumseagreen', 'purple', 'darkorange']
    fig, ax   = plt.subplots(figsize=(max(8, 1.2 * len(pivot)), 5))
    pivot.plot(kind='bar', ax=ax, width=0.8, color=palette[:len(cols)])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('Total post-cost P&L ($/share)')
    ax.set_xlabel('Date')
    ax.set_title(f'Simulated Arbitrage P&L by Date and ETF\n'
                 f'(fee={FEE_PER_SHARE*2:.3f}$/share both legs; '
                 f'slippage={slippage_bps} bps/leg)')
    ax.legend(title='ETF', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, os.path.join(plot_dir, 'pnl_by_date.png'))

# -- Main ----------------------------------------------------------------------

def run(inav_dir: str = INAV_DIR, fallback_dir: str = FALLBACK_DIR,
        output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    inputs = find_date_etf_inputs(inav_dir, fallback_dir)
    by_date: Dict[str, List[str]] = {}
    for d, e, _ in inputs:
        by_date.setdefault(d, []).append(e)
    print(f'Found {len(inputs)} (date, ETF) inputs across {len(by_date)} day(s):')
    for d, es in by_date.items():
        print(f'  {d}: {es}')

    all_results: List[dict] = []
    per_etf_gaps            = {etf: {} for etf in ETFS}
    last_compute_ns         = None

    for date_tag, etf, files in inputs:
        if etf not in LEVERAGE:
            print(f'\n[{date_tag} {etf.upper()}] not in LEVERAGE config -- skipping')
            continue
        print(f'\n[{date_tag} {etf.upper()}]')
        df, compute_ns = load_files_for_etf(files, etf)
        if compute_ns is not None:
            last_compute_ns = compute_ns
        scenarios = build_scenarios(compute_ns)
        bench = (f'  compute_ns={compute_ns:.1f}'
                 if compute_ns is not None else '  (no benchmark in data)')
        print(f'  Loaded {len(files)} file(s) -> {len(df):,} rows '
              f'[{df["time_m"].min()} - {df["time_m"].max()}]{bench}')

        if f'{etf}_mid' not in df.columns:
            print(f'  market data columns absent -- skipping')
            continue

        gaps  = get_arb_spreads(df, etf)
        per_etf_gaps[etf][date_tag] = gaps
        n_pos = (gaps['arb_spread'] > 0).sum()
        durs  = _gap_episode_durations(gaps)
        print(f'  pos_ticks={n_pos:,}/{len(gaps):,} '
              f'({100 * n_pos / max(len(gaps),1):.1f}%)  '
              f'episodes={len(durs)}'
              + (f'  med={np.median(durs):.1f}ms p95={np.percentile(durs,95):.1f}ms'
                 if durs else ''))

        for lat in scenarios:
            r = assess_viability(gaps, lat, etf, SLIPPAGE_BPS)
            r['date'] = date_tag
            all_results.append(r)

    final_scenarios = build_scenarios(last_compute_ns)
    print('\nLatency Scenarios (using benchmarked compute from latest data):')
    for lat in final_scenarios:
        lat.print_breakdown()

    print('\n-- Combined plots ---------------------------------------------')
    plot_latency_vs_gaps_all(per_etf_gaps, final_scenarios, plot_dir)
    plot_spread_distribution_all(per_etf_gaps, plot_dir, SLIPPAGE_BPS)
    plot_pnl_by_date(per_etf_gaps, plot_dir, SLIPPAGE_BPS)

    results = pd.DataFrame(all_results)
    if not results.empty:
        cols = ['date', 'etf', 'latency_scenario'] + \
               [c for c in results.columns
                if c not in ('date', 'etf', 'latency_scenario')]
        results = results[cols]
    path = os.path.join(output_dir, 'execution_viability.csv')
    results.to_csv(path, index=False)
    print(f'\nSaved -> {path}  ({len(results)} rows)')

    if not results.empty:
        print('\n-- Viability Summary -------------------------------------------')
        summary_cols = ['date', 'etf', 'latency_scenario', 'total_latency_ms',
                        'n_viable_opportunities', 'mean_net_$', 'total_net_$']
        print(results[summary_cols].to_string(index=False))
    return results


if __name__ == '__main__':
    run()
