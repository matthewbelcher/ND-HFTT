"""
signals/time_window.py
======================
Signal 4 — Time-Windowed OBI Delta

Same as OBIDeltaSignal but only fires within a specific time band
measured in seconds elapsed since market open.

Hypothesis: OBI predictiveness varies across the 15-minute window.
  - First ~60s: initial book formation, wide spreads, price discovery.
    Market makers are placing initial quotes — OBI is noisy.
  - Middle 1–14 min: stable book, OBI should be most predictive.
  - Last ~60s: thin book, aggressive directional bets as resolution
    approaches. OBI may over-fire on one-sided pressure.

By default this signal targets the middle band (60s–840s elapsed),
excluding the first and last minutes. You can override `start_sec` and
`end_sec` to test any sub-window.

Usage:
    # Test only the last 2 minutes
    sig = TimeWindowOBI(start_sec=720, end_sec=900)

    # Test only the first 90 seconds
    sig = TimeWindowOBI(start_sec=0, end_sec=90, name='early_obi')
"""

import pandas as pd
from .obi_delta import OBIDeltaSignal


class TimeWindowOBI(OBIDeltaSignal):

    name = 'time_window_obi'

    def __init__(
        self,
        start_sec:  int   = 60,    # seconds after market open to start firing
        end_sec:    int   = 840,   # seconds after market open to stop firing (14 min)
        obi_col:    str   = 'obi3',
        horizon:    float = 1.0,
        signal_win: float = 0.25,
        threshold:  float = 0.40,
        min_tick:   float = 0.01,
        cooldown:   float = 0.5,
    ):
        super().__init__(
            obi_col=obi_col, horizon=horizon, signal_win=signal_win,
            threshold=threshold, min_tick=min_tick, cooldown=cooldown)
        self.start_sec = start_sec
        self.end_sec   = end_sec

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        # Compute elapsed seconds from market open (first timestamp in session)
        market_open = df.index[0]
        elapsed = (df.index - market_open).total_seconds()

        # Filter to the target time band
        mask    = (elapsed >= self.start_sec) & (elapsed <= self.end_sec)
        df_band = df[mask]

        if len(df_band) < 10:
            return pd.DataFrame()

        results = super().evaluate(df_band)

        if results.empty:
            return results

        results = results.copy()
        results['signal_name']  = self.name
        results['elapsed_start'] = self.start_sec
        results['elapsed_end']   = self.end_sec

        # Add elapsed time to each signal event for downstream analysis
        results['elapsed_sec'] = (
            results.index - market_open).total_seconds()

        return results