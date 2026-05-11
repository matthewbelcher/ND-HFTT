"""
signals/
========
A small library of order-book signals for Kalshi KXBTC15M markets.

Every signal is a subclass of BaseSignal with two methods:
    fit(sessions: list[pd.DataFrame])   — optional calibration step
    evaluate(df: pd.DataFrame) -> pd.DataFrame  — run on one merged session

The evaluate() result always has the same columns so the dashboard and
signal_runner can treat every signal identically:

    ts (index)   signal timestamp
    direction    +1 (predict rise) or -1 (predict fall)
    mid_at       Kalshi mid at signal time
    mid_fwd      best mid reached in predicted direction within horizon
    fwd_move     (mid_fwd - mid_at) * direction  (positive = correct)
    hit          True if fwd_move >= min_tick without adverse first
    adverse      True if adverse move >= min_tick fired first
    signal_name  name of the signal that fired
    [extra cols] signal-specific metadata (e.g. d_obi, deviation, spread_at)
"""

from .base import BaseSignal
from .obi_delta import OBIDeltaSignal
from .microprice import MicropriceSignal
from .spread_filter import SpreadFilteredOBI
from .time_window import TimeWindowOBI

__all__ = [
    'BaseSignal',
    'OBIDeltaSignal',
    'MicropriceSignal',
    'SpreadFilteredOBI',
    'TimeWindowOBI',
]