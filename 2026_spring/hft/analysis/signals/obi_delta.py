"""
signals/obi_delta.py
====================
Signal 1 — OBI Delta

Fires when OBI changes by more than `threshold` within a backward
window of `signal_win` seconds.

Positive delta → predict YES price rise  (BUY,  direction=+1)
Negative delta → predict YES price fall  (SELL, direction=-1)

This is the baseline signal established in the exploratory analysis.
Best depth found: obi3 (3-level, uniform weighting).
"""

import pandas as pd
import numpy as np
from .base import BaseSignal


class OBIDeltaSignal(BaseSignal):

    name = 'obi_delta'

    def __init__(
        self,
        obi_col:    str   = 'obi3',   # which OBI column to use
        horizon:    float = 1.0,       # forward look window (seconds)
        signal_win: float = 0.25,      # backward delta window (seconds)
        threshold:  float = 0.40,      # min |d_obi| to fire
        min_tick:   float = 0.01,      # min mid move to count as hit ($)
        cooldown:   float = 0.5,       # suppress re-firing (seconds)
    ):
        self.obi_col    = obi_col
        self.horizon    = horizon
        self.signal_win = signal_win
        self.threshold  = threshold
        self.min_tick   = min_tick
        self.cooldown   = cooldown

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.obi_col not in df.columns:
            raise ValueError(
                f"[{self.name}] Column '{self.obi_col}' not found. "
                f"Available: {list(df.columns)}")

        obi_s = df[self.obi_col].dropna()
        mid_s = df['mid'].ffill()

        horizon_td    = pd.Timedelta(seconds=self.horizon)
        signal_win_td = pd.Timedelta(seconds=self.signal_win)
        cooldown_td   = pd.Timedelta(seconds=self.cooldown)

        last_ts = obi_s.index[0] - pd.Timedelta(days=1)
        events  = []

        for ts, obi_now in zip(obi_s.index, obi_s.values):
            if ts - last_ts < cooldown_td:
                continue

            past = obi_s.loc[ts - signal_win_td : ts]
            if len(past) < 2:
                continue

            d_obi = obi_now - past.iloc[0]
            if abs(d_obi) < self.threshold:
                continue

            mid_now = mid_s.asof(ts)
            if pd.isna(mid_now):
                continue

            direction = 1 if d_obi > 0 else -1
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
                'obi_before' : past.iloc[0],
                'obi_at'     : obi_now,
                'd_obi'      : d_obi,
                'obi_col'    : self.obi_col,
            })
            last_ts = ts

        if not events:
            return pd.DataFrame()
        return pd.DataFrame(events).set_index('ts')