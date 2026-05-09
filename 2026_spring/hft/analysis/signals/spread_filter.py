"""
signals/spread_filter.py
========================
Signal 3 — Spread-Filtered OBI Delta

Identical to OBIDeltaSignal but only fires when the spread is at or
below `max_spread_ticks` ticks at the moment of signal.

Hypothesis: OBI is only informative when market makers are actively
posting tight quotes. Wide spread states (multi-tick repricing) add
noise that degrades hit rate. Filtering to 1-tick spread states should
increase precision at the cost of fewer signals.

From the exploratory data:
  spread mean  = $0.0153  (mean > 1 tick — wide spread states are common)
  spread std   = $0.0123
  spread min   = $0.00    (book momentarily empty on one side)
  spread max   = $0.98    (near-zero probability markets)

Expected effect: ~30-40% fewer signals, hit rate increase of ~3-7pp.
"""

import pandas as pd
from .obi_delta import OBIDeltaSignal


class SpreadFilteredOBI(OBIDeltaSignal):

    name = 'spread_filtered_obi'

    def __init__(
        self,
        max_spread_ticks: int   = 1,     # only fire when spread <= this many ticks
        obi_col:          str   = 'obi3',
        horizon:          float = 1.0,
        signal_win:       float = 0.25,
        threshold:        float = 0.40,
        min_tick:         float = 0.01,
        cooldown:         float = 0.5,
    ):
        super().__init__(
            obi_col=obi_col, horizon=horizon, signal_win=signal_win,
            threshold=threshold, min_tick=min_tick, cooldown=cooldown)
        self.max_spread_ticks = max_spread_ticks
        self.max_spread_dollars = max_spread_ticks * 0.01

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Run parent signal, then filter results to tight-spread moments only
        # We do this by filtering the input df first so the parent's scan
        # is over tight-spread rows only.
        if 'spread' not in df.columns:
            raise ValueError(f"[{self.name}] 'spread' column not found.")

        # Mask: keep rows where spread <= threshold OR spread is NaN
        # (NaN spread = book not yet initialised, the parent handles this)
        tight = df[df['spread'].isna() | (df['spread'] <= self.max_spread_dollars)]

        if len(tight) < 10:
            return pd.DataFrame()

        results = super().evaluate(tight)

        if results.empty:
            return results

        # Tag with this signal's name (parent tags with 'obi_delta')
        results = results.copy()
        results['signal_name'] = self.name
        results['spread_filter'] = self.max_spread_dollars

        return results