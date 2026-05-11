"""
calibrate_microprice.py  —  Stoikov micro-price calibration
============================================================

Pools merged_*.csv files, estimates Markov transition matrices Q, T, R
from (imbalance_bucket, spread_ticks) states, solves for G*, and saves
the resulting lookup table to g_star.json.

The output g_star.json can be loaded at inference time (Python or C++) to
compute:   microprice = mid + G_star[imbalance_bucket, spread_bucket]

Usage
-----
    # Calibrate on all merged CSVs in a directory
    python calibrate_microprice.py --data-dir completed-data/

    # Calibrate on specific files
    python calibrate_microprice.py --files merged_A.csv merged_B.csv ...

    # Tune discretization
    python calibrate_microprice.py --data-dir data/ --n-imb 10 --max-spread 3

Options
-------
    --data-dir      Directory containing merged_*.csv files
    --files         Explicit list of merged CSV paths (overrides --data-dir)
    --n-imb         Number of imbalance buckets          (default: 10)
    --max-spread    Max spread in ticks to model; wider → last bucket (default: 3)
    --out           Output path for g_star.json          (default: g_star.json)
    --min-count     Min transitions per state to trust; below → G*=0 (default: 30)
    --plot          Save a heatmap of G* to g_star_heatmap.png

Notes on discretization
-----------------------
  imbalance I = obi1 ∈ [-1, 1].  We map this to buckets 0..n-1.
    bucket = clip(floor((I + 1) / 2 * n_imb), 0, n_imb - 1)
    bucket 0 = full NO lean, bucket n-1 = full YES lean.

  spread in ticks: Kalshi tick = $0.01.
    spread_tick = round(spread / 0.01).  Clipped to [1, max_spread].
    bucket 0 = 1-tick spread, bucket max_spread-1 = max_spread-tick spread.

Symmetrization (critical for convergence — see Stoikov Theorem 3.1)
--------------------------------------------------------------------
For every observed transition (I, S, I', S', dM) we also add its mirror
(-I, S, -I', S', -dM).  This enforces B*G1 = 0 so the micro-price sum
converges.
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Discretization helpers
# ─────────────────────────────────────────────────────────────────────────────

def imb_to_bucket(I: float, n: int) -> int:
    """Map imbalance in [-1,1] to bucket index in [0, n-1]."""
    b = int((I + 1.0) / 2.0 * n)
    return max(0, min(n - 1, b))


def spread_to_bucket(spread: float, max_spread: int) -> int:
    """Map spread ($) to tick bucket in [0, max_spread-1]. Clips at max_spread."""
    ticks = max(1, round(spread / 0.01))
    return min(ticks, max_spread) - 1   # 0-indexed


def state_idx(i_bkt: int, s_bkt: int, n_imb: int) -> int:
    return i_bkt * 1 + s_bkt * n_imb   # column-major: spread varies slowly


# ─────────────────────────────────────────────────────────────────────────────
# Load + validate a single merged CSV
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED = {'mid', 'obi1', 'spread'}

def load_merged(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, index_col='ts', parse_dates=['ts'])
    except Exception as e:
        print(f"  [skip] {path.name}: read error — {e}")
        return None

    missing = REQUIRED - set(df.columns)
    if missing:
        print(f"  [skip] {path.name}: missing columns {missing}")
        return None

    df = df[list(REQUIRED)].dropna()
    if len(df) < 200:
        print(f"  [skip] {path.name}: only {len(df)} clean rows")
        return None

    print(f"  [ok]   {path.name}: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Build transition observations from one session
# ─────────────────────────────────────────────────────────────────────────────

def extract_transitions(df: pd.DataFrame,
                        n_imb: int,
                        max_spread: int) -> list[dict]:
    """
    Returns a list of transition records:
        i_bkt, s_bkt, i_bkt_next, s_bkt_next, dm
    where dm = mid_next - mid  (0 if mid unchanged, ±0.005 for half-tick moves).

    Only mid-price *change* events count as 'absorbing'; all other events are
    transient (dm == 0, new state = next imbalance/spread).
    """
    mid    = df['mid'].values
    obi1   = df['obi1'].values
    spread = df['spread'].values
    n      = len(df)

    records = []
    for t in range(n - 1):
        i_bkt = imb_to_bucket(obi1[t],   n_imb)
        s_bkt = spread_to_bucket(spread[t], max_spread)
        dm    = mid[t + 1] - mid[t]

        i_bkt_next = imb_to_bucket(obi1[t + 1],   n_imb)
        s_bkt_next = spread_to_bucket(spread[t + 1], max_spread)

        records.append({
            'i': i_bkt, 's': s_bkt,
            'i_next': i_bkt_next, 's_next': s_bkt_next,
            'dm': round(dm, 6),
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Symmetrize
# ─────────────────────────────────────────────────────────────────────────────

def symmetrize(records: list[dict], n_imb: int) -> list[dict]:
    """
    For each record add its mirror: flip imbalance buckets and negate dm.
    i_mirror = (n_imb - 1) - i
    This enforces B*G1 = 0 (Stoikov Theorem 3.1) guaranteeing convergence.
    """
    mirrored = []
    for r in records:
        mirrored.append(r)
        mirrored.append({
            'i':      (n_imb - 1) - r['i'],
            's':       r['s'],
            'i_next': (n_imb - 1) - r['i_next'],
            's_next':  r['s_next'],
            'dm':     -r['dm'],
        })
    return mirrored


# ─────────────────────────────────────────────────────────────────────────────
# Build Q, T, R matrices
# ─────────────────────────────────────────────────────────────────────────────

def build_matrices(records: list[dict],
                   n_imb: int,
                   max_spread: int,
                   min_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """
    Returns (Q, T, R, K, counts) where:
        Q   [nm x nm]  transient → transient  (dm == 0)
        T   [nm x nm]  transient → absorb-and-return  (dm != 0, gives new state)
        R   [nm x 4]   transient → absorbing  (dm != 0, gives dm value)
        K   [4]        the 4 absorbing state dm values: -0.01, -0.005, +0.005, +0.01
        counts [nm]    total transitions from each state (for min_count filtering)
    """
    nm = n_imb * max_spread
    K_vals = np.array([-0.01, -0.005, 0.005, 0.01])

    # Raw count matrices
    Q_raw = np.zeros((nm, nm), dtype=np.float64)
    T_raw = np.zeros((nm, nm), dtype=np.float64)
    R_raw = np.zeros((nm, 4),  dtype=np.float64)

    for r in records:
        from_idx = r['i'] + r['s'] * n_imb
        to_idx   = r['i_next'] + r['s_next'] * n_imb
        dm       = r['dm']

        if abs(dm) < 1e-7:
            # Transient: mid didn't move
            Q_raw[from_idx, to_idx] += 1
        else:
            # Absorbing: mid moved — goes into both T and R
            T_raw[from_idx, to_idx] += 1
            # Find closest K value
            k_idx = int(np.argmin(np.abs(K_vals - dm)))
            R_raw[from_idx, k_idx] += 1

    # Row-normalize to get probabilities
    counts = Q_raw.sum(axis=1) + T_raw.sum(axis=1) + R_raw.sum(axis=1)

    Q = np.zeros_like(Q_raw)
    T = np.zeros_like(T_raw)
    R = np.zeros_like(R_raw)

    for i in range(nm):
        total = counts[i]
        if total >= min_count:
            Q[i] = Q_raw[i] / total
            T[i] = T_raw[i] / total
            R[i] = R_raw[i] / total

    return Q, T, R, K_vals, counts


# ─────────────────────────────────────────────────────────────────────────────
# Solve for G* = micro-price adjustment vector
# ─────────────────────────────────────────────────────────────────────────────

def solve_g_star(Q: np.ndarray,
                 T: np.ndarray,
                 R: np.ndarray,
                 K: np.ndarray,
                 counts: np.ndarray,
                 min_count: int,
                 n_iter: int = 200) -> np.ndarray:
    """
    Stoikov iterative solution:
        G1  = (I - Q)^{-1} R K
        B   = (I - Q)^{-1} T
        G*  = sum_{k=0}^{inf} B^k G1  ≈  iterative sum until convergence

    States with fewer than min_count observations get G*=0 (trust floor).
    We use the power-series form B^k G1 which converges fast in practice
    (Stoikov notes: "converges very fast").
    """
    nm = Q.shape[0]
    I  = np.eye(nm)

    # (I - Q) might be singular for sparse states — add tiny regularization
    IQ = I - Q
    try:
        IQ_inv = np.linalg.inv(IQ)
    except np.linalg.LinAlgError:
        IQ_inv = np.linalg.pinv(IQ)

    # G1: first-order micro-price adjustment
    G1 = IQ_inv @ (R @ K)          # shape [nm]

    # B matrix
    B  = IQ_inv @ T                 # shape [nm x nm]

    # Power series: G* = G1 + B*G1 + B^2*G1 + ...
    G_star = G1.copy()
    Bk_G1  = G1.copy()
    for _ in range(n_iter):
        Bk_G1  = B @ Bk_G1
        delta  = np.abs(Bk_G1).max()
        G_star += Bk_G1
        if delta < 1e-10:
            break

    # Zero out states with insufficient data
    G_star[counts < min_count] = 0.0

    return G_star


# ─────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_g_star(G_star: np.ndarray,
                n_imb: int,
                max_spread: int,
                counts: np.ndarray,
                out_path: Path):
    """
    Save G* as a JSON with metadata so it's self-describing.

    Layout: g_star[s_bucket][i_bucket]  (spread outer, imbalance inner)
    so you can do:  adj = g_star[s_bkt][i_bkt]
    """
    nm = n_imb * max_spread
    g_matrix = []
    for s in range(max_spread):
        row = []
        for i in range(n_imb):
            idx = i + s * n_imb
            row.append(float(G_star[idx]))
        g_matrix.append(row)

    count_matrix = []
    for s in range(max_spread):
        row = []
        for i in range(n_imb):
            idx = i + s * n_imb
            row.append(int(counts[idx]))
        count_matrix.append(row)

    payload = {
        'n_imb':      n_imb,
        'max_spread': max_spread,
        'tick':       0.01,
        # g_star[s_bucket][i_bucket] → microprice adjustment in $
        'g_star':     g_matrix,
        # counts[s_bucket][i_bucket] → number of transitions observed (diagnostic)
        'counts':     count_matrix,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[Saved]   {out_path}  (n_imb={n_imb}, max_spread={max_spread})")


# ─────────────────────────────────────────────────────────────────────────────
# Optional diagnostic heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(G_star: np.ndarray,
                 counts: np.ndarray,
                 n_imb: int,
                 max_spread: int,
                 out_path: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("[Plot]    matplotlib not available, skipping heatmap.")
        return

    # Reshape to (max_spread, n_imb)
    grid = np.zeros((max_spread, n_imb))
    cnt  = np.zeros((max_spread, n_imb))
    for s in range(max_spread):
        for i in range(n_imb):
            idx      = i + s * n_imb
            grid[s, i] = G_star[idx]
            cnt[s, i]  = counts[idx]

    BG   = '#0f1117'
    TEXT = '#c8d0e0'

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor=BG)

    # G* heatmap
    ax = axes[0]
    ax.set_facecolor(BG)
    vmax = np.abs(grid).max() or 0.005
    im = ax.imshow(grid, aspect='auto', origin='lower',
                   cmap='RdYlGn', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label='G* ($)')
    ax.set_xlabel('Imbalance bucket (0=full NO, n-1=full YES)', color=TEXT)
    ax.set_ylabel('Spread bucket (0=1-tick, ...)', color=TEXT)
    ax.set_title('Micro-price adjustment G*', color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)

    # Count heatmap (log scale for readability)
    ax = axes[1]
    ax.set_facecolor(BG)
    cnt_log = np.log1p(cnt)
    im2 = ax.imshow(cnt_log, aspect='auto', origin='lower', cmap='Blues')
    fig.colorbar(im2, ax=ax, label='log(1 + count)')
    ax.set_xlabel('Imbalance bucket', color=TEXT)
    ax.set_ylabel('Spread bucket', color=TEXT)
    ax.set_title('Transition counts (log scale)', color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot]    -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Raw session loader (merge on the fly without writing CSVs)
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_sessions(raw_dirs: list[Path],
                      n_imb: int,
                      max_spread: int,
                      cache_dir: Path | None = None) -> tuple[list[dict], int]:
    """
    Find KXBTC15M-*.csv / BTC-KXBTC15M-*.csv pairs in raw_dirs, merge each
    pair (or load from cache), extract transitions, return (all_records, n_loaded).

    If cache_dir is set, merged DataFrames are cached as parquet after first
    parse. Subsequent runs skip the slow order-book replay entirely.
    """
    try:
        import merge_plot
    except ImportError:
        print("[Error] merge_plot.py not found. "
              "Run from analysis/ directory.", file=sys.stderr)
        sys.exit(1)

    pairs: list[tuple[Path, Path]] = []
    for d in raw_dirs:
        for kpath in sorted(d.glob('KXBTC15M-*.csv')):
            bpath = d / f'BTC-{kpath.name}'
            if bpath.exists():
                pairs.append((kpath, bpath))
            else:
                print(f"  [skip] no BTC file for {kpath.name}", file=sys.stderr)

    if not pairs:
        print(f"[Error] No KXBTC15M-*.csv / BTC-KXBTC15M-*.csv pairs found "
              f"in {[str(d) for d in raw_dirs]}", file=sys.stderr)
        sys.exit(1)

    cached_count = 0
    if cache_dir:
        cached_count = sum(1 for kp, _ in pairs
                           if (cache_dir / f'{kp.stem}.parquet').exists())

    print(f"[Raw]     Found {len(pairs)} session pairs  "
          f"({cached_count} already cached)")

    all_records: list[dict] = []
    loaded = 0

    for kpath, bpath in pairs:
        ticker_name = kpath.stem

        # ── Try cache first ───────────────────────────────────────────────────
        mdf = None
        if cache_dir is not None:
            cache_path = cache_dir / f'{ticker_name}.parquet'
            if cache_path.exists():
                try:
                    mdf = pd.read_parquet(cache_path)
                    if mdf.index.tz is None:
                        mdf.index = mdf.index.tz_localize('UTC')
                    # Invalidate if required columns missing
                    if not {'mid', 'obi1', 'spread'}.issubset(mdf.columns):
                        mdf = None
                        cache_path.unlink()
                except Exception:
                    mdf = None

        # ── Parse from raw if not cached ──────────────────────────────────────
        if mdf is None:
            try:
                kdf, _ = merge_plot.parse_kalshi(str(kpath))
                bdf    = merge_plot.parse_btc(
                    str(bpath),
                    kdf.index[0]  - merge_plot.BTC_BUFFER,
                    kdf.index[-1] + merge_plot.BTC_BUFFER,
                )
                mdf = merge_plot.asof_join(kdf, bdf)

                if cache_dir is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        mdf.to_parquet(cache_dir / f'{ticker_name}.parquet')
                    except Exception as e:
                        print(f"  [cache write failed] {ticker_name}: {e}",
                              file=sys.stderr)
            except Exception as e:
                print(f"  [error] {ticker_name}: {e}", file=sys.stderr)
                continue

        try:
            sub  = mdf[['mid', 'obi1', 'spread']].dropna()
            recs = extract_transitions(sub, n_imb, max_spread)
            all_records.extend(recs)
            loaded += 1
        except Exception as e:
            print(f"  [error extracting] {ticker_name}: {e}", file=sys.stderr)

    return all_records, loaded


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Calibrate Stoikov micro-price from Kalshi order book data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Raw data (KXBTC15M-*.csv + BTC-KXBTC15M-*.csv pairs) — most common case
  python calibrate_microprice.py --raw-dir ~/bigdata/ --plot

  # Multiple raw directories
  python calibrate_microprice.py --raw-dir ~/bigdata/ ../data/ --plot

  # Pre-merged CSVs (merged_*.csv from merge_plot.py)
  python calibrate_microprice.py --data-dir completed-data/ --plot

  # Explicit merged file list
  python calibrate_microprice.py --files merged_A.csv merged_B.csv --plot
        """)

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--raw-dir',  type=Path, nargs='+',
                     help='Directory/directories of raw KXBTC15M-*.csv + '
                          'BTC-KXBTC15M-*.csv pairs (merged in memory)')
    src.add_argument('--data-dir', type=Path,
                     help='Directory of pre-merged merged_*.csv files')
    src.add_argument('--files',    type=Path, nargs='+',
                     help='Explicit list of pre-merged CSV paths')

    ap.add_argument('--cache-dir',  type=Path,
                    default=Path('../results/session_cache'),
                    help='Session parquet cache dir (default: ../results/session_cache). '
                         'Shared with signal_runner and dashboard — build once, use everywhere.')
    ap.add_argument('--n-imb',      type=int,   default=10,
                    help='Number of imbalance buckets (default 10)')
    ap.add_argument('--max-spread', type=int,   default=3,
                    help='Max spread in ticks to model (default 3)')
    ap.add_argument('--out',        type=Path,  default=Path('g_star.json'),
                    help='Output JSON path (default g_star.json)')
    ap.add_argument('--min-count',  type=int,   default=30,
                    help='Min transitions per state to trust (default 30)')
    ap.add_argument('--plot',       action='store_true',
                    help='Save diagnostic heatmap alongside JSON')
    args = ap.parse_args()

    # ── Load sessions and extract transitions ────────────────────────────────
    all_records: list[dict] = []
    loaded = 0

    if args.raw_dir:
        # Raw pairs — merge in memory, no disk writes
        all_records, loaded = load_raw_sessions(
            args.raw_dir, args.n_imb, args.max_spread,
            cache_dir=args.cache_dir)

    elif args.data_dir:
        # Pre-merged CSVs
        paths = sorted(args.data_dir.glob('merged_*.csv'))
        if not paths:
            print(f"[Error] No merged_*.csv files found in {args.data_dir}\n"
                  f"        If your data is raw (unmerged), use --raw-dir instead.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[Files]   {len(paths)} merged CSV(s) found")
        for p in paths:
            df = load_merged(p)
            if df is None:
                continue
            all_records.extend(extract_transitions(df, args.n_imb, args.max_spread))
            loaded += 1

    else:
        # Explicit file list
        print(f"[Files]   {len(args.files)} file(s) specified")
        for p in args.files:
            df = load_merged(p)
            if df is None:
                continue
            all_records.extend(extract_transitions(df, args.n_imb, args.max_spread))
            loaded += 1

    if not all_records:
        print("[Error] No transitions extracted. Check your data directory.",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n[Data]    {loaded} sessions loaded, "
          f"{len(all_records):,} raw transitions")

    # ── Symmetrize ───────────────────────────────────────────────────────────
    sym_records = symmetrize(all_records, args.n_imb)
    print(f"[Sym]     {len(sym_records):,} transitions after symmetrization")

    # ── Build matrices ───────────────────────────────────────────────────────
    Q, T, R, K, counts = build_matrices(
        sym_records, args.n_imb, args.max_spread, args.min_count)

    nm = args.n_imb * args.max_spread
    covered = (counts >= args.min_count).sum()
    print(f"[Matrix]  {nm} states total, "
          f"{covered} with >= {args.min_count} transitions "
          f"({covered/nm:.0%} coverage)")

    # ── Solve G* ─────────────────────────────────────────────────────────────
    print("[Solve]   Running power-series iteration for G*...")
    G_star = solve_g_star(Q, T, R, K, counts, args.min_count)

    g_min  = G_star[counts >= args.min_count].min() if covered else 0
    g_max  = G_star[counts >= args.min_count].max() if covered else 0
    g_std  = G_star[counts >= args.min_count].std() if covered else 0
    print(f"[G*]      range=[{g_min:+.6f}, {g_max:+.6f}]  std={g_std:.6f}")
    print(f"          (should be ≲ half-spread ≈ ±0.005 for 1-tick markets)")

    # ── Save ─────────────────────────────────────────────────────────────────
    save_g_star(G_star, args.n_imb, args.max_spread, counts, args.out)

    # ── Optional plot ────────────────────────────────────────────────────────
    if args.plot:
        heatmap_path = args.out.with_name(args.out.stem + '_heatmap.png')
        plot_heatmap(G_star, counts, args.n_imb, args.max_spread, heatmap_path)

    print("\n[Done]")
    print(f"  Load in Python:  import json; cal = json.load(open('{args.out}'))")
    print( "  Lookup:          adj = cal['g_star'][s_bkt][i_bkt]")
    print( "  Microprice:      microprice = mid + adj")


if __name__ == '__main__':
    main()