"""
signals/microprice.py
=====================
Signal 2 — Stoikov Micro-Price Deviation

Fires when |microprice - mid| >= threshold.

Requires a calibrated g_star.json produced by calibrate_microprice.py.
Call fit() with your training sessions before evaluate().

NOTE: On Kalshi KXBTC15M markets, G* is empirically capped at ~±$0.0022
(sub-tick on the $0.01 grid). This means threshold must be <= 0.002 to
fire any signals at all. At that level, hit rate (~35%) is below the OBI
delta signal (~41%), so this signal is currently a useful benchmark rather
than a primary trading signal. See report Section 5 for full analysis.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseSignal

# Import calibrate_microprice helpers (assumed to be in the project root)
try:
    import calibrate_microprice as _cal_mod
    _HAS_CAL = True
except ImportError:
    _HAS_CAL = False


class MicropriceSignal(BaseSignal):

    name = 'microprice'

    def __init__(
        self,
        gstar_path: str   = 'g_star.json',
        n_imb:      int   = 10,
        max_spread: int   = 3,
        min_count:  int   = 30,
        threshold:  float = 0.001,   # |mp - mid| in $ to fire
        horizon:    float = 1.0,
        min_tick:   float = 0.01,
        cooldown:   float = 0.5,
    ):
        self.gstar_path = gstar_path
        self.n_imb      = n_imb
        self.max_spread = max_spread
        self.min_count  = min_count
        self.threshold  = threshold
        self.horizon    = horizon
        self.min_tick   = min_tick
        self.cooldown   = cooldown
        self._cal       = None   # populated by fit() or load()

    # ── calibration ──────────────────────────────────────────────────────────

    def fit(self, sessions: list[pd.DataFrame]) -> None:
        """Estimate G* from training sessions and save to gstar_path."""
        if not _HAS_CAL:
            raise ImportError("calibrate_microprice module not found.")

        all_records = []
        for df in sessions:
            sub = df[['mid', 'obi1', 'spread']].dropna()
            if len(sub) < 200:
                continue
            recs = _cal_mod.extract_transitions(sub, self.n_imb, self.max_spread)
            all_records.extend(recs)

        if not all_records:
            raise RuntimeError("No transitions extracted — check your session data.")

        sym = _cal_mod.symmetrize(all_records, self.n_imb)
        Q, T, R, K, counts = _cal_mod.build_matrices(
            sym, self.n_imb, self.max_spread, self.min_count)
        G_star = _cal_mod.solve_g_star(Q, T, R, K, counts, self.min_count)
        _cal_mod.save_g_star(G_star, self.n_imb, self.max_spread,
                             counts, Path(self.gstar_path))
        self._cal = json.loads(Path(self.gstar_path).read_text())
        print(f"[{self.name}] G* fitted and saved to {self.gstar_path}  "
              f"range=[{G_star.min():+.6f}, {G_star.max():+.6f}]")

    def load(self) -> bool:
        """Load a previously fitted G* from disk. Returns True if successful."""
        p = Path(self.gstar_path)
        if not p.exists():
            return False
        self._cal = json.loads(p.read_text())
        return True

    # ── microprice lookup ─────────────────────────────────────────────────────

    def _add_microprice(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised microprice column injection."""
        if self._cal is None:
            raise RuntimeError(
                f"[{self.name}] Not calibrated. Call fit() or load() first.")
        if _HAS_CAL:
            return _cal_mod.add_microprice(df, self._cal)
        # Fallback: manual lookup
        n    = self._cal['n_imb']
        ms   = self._cal['max_spread']
        tick = self._cal.get('tick', 0.01)
        g    = np.array(self._cal['g_star'])
        df   = df.copy()
        i_bkt = ((df['obi1'] + 1.0) / 2.0 * n).clip(0, n - 1).astype(int)
        spread = df['spread'] / tick
        valid  = np.isfinite(spread)
        s_bkt  = np.full(len(df), -1, dtype=int)
        s_bkt[valid] = (spread[valid].round().clip(1, ms).astype(int) - 1)
        adj  = np.full(len(df), np.nan)
        mask = s_bkt >= 0
        adj[mask] = g[s_bkt[mask], i_bkt.values[mask]]
        df['microprice'] = np.where(df['mid'].notna(), df['mid'] + adj, np.nan)
        return df

    # ── evaluate ─────────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._cal is None and not self.load():
            raise RuntimeError(
                f"[{self.name}] No calibration found at {self.gstar_path}. "
                f"Call fit() first or run calibrate_microprice.py.")

        if 'microprice' not in df.columns:
            df = self._add_microprice(df)

        mid_s = df['mid'].ffill()
        mp_s  = df['microprice'].ffill()

        valid = mid_s.notna() & mp_s.notna()
        mid_s = mid_s[valid]
        mp_s  = mp_s[valid]

        if mid_s.empty:
            return pd.DataFrame()

        horizon_td  = pd.Timedelta(seconds=self.horizon)
        cooldown_td = pd.Timedelta(seconds=self.cooldown)
        last_ts     = mid_s.index[0] - pd.Timedelta(days=1)
        events      = []

        for ts in mid_s.index:
            if ts - last_ts < cooldown_td:
                continue

            mid_now = mid_s.asof(ts)
            mp_now  = mp_s.asof(ts)
            if pd.isna(mid_now) or pd.isna(mp_now):
                continue

            deviation = mp_now - mid_now
            if abs(deviation) < self.threshold:
                continue

            direction = 1 if deviation > 0 else -1
            hit, adverse, best_fwd = self._scan_forward(
                mid_s, ts, direction, horizon_td, self.min_tick)

            events.append({
                'ts'         : ts,
                'direction'  : direction,
                'mid_at'     : mid_now,
                'mid_fwd'    : best_fwd,
                'fwd_move'   : (best_fwd - mid_now) * direction,
                'hit'        : hit,
                'adverse'    : adverse,
                'signal_name': self.name,
                # signal-specific metadata
                'mp_at'      : mp_now,
                'deviation'  : deviation,
            })
            last_ts = ts

        if not events:
            return pd.DataFrame()
        return pd.DataFrame(events).set_index('ts')