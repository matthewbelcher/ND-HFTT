"""
signals/base.py
===============
Abstract base class that every signal must implement.
"""

from abc import ABC, abstractmethod
import pandas as pd


# Standard columns every evaluate() result must contain
RESULT_COLS = [
    'direction', 'mid_at', 'mid_fwd', 'fwd_move', 'hit', 'adverse', 'signal_name'
]


class BaseSignal(ABC):
    """
    Subclass this to add a new signal.

    Minimal implementation:
        class MySignal(BaseSignal):
            name = 'my_signal'

            def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
                # ... compute events ...
                return events_df   # must contain RESULT_COLS

    Optional:
        def fit(self, sessions: list[pd.DataFrame]) -> None:
            # called once with all training sessions before evaluate()
    """

    # Subclasses set this — used as the display name everywhere
    name: str = 'unnamed_signal'

    # ── shared forward-scan helper ────────────────────────────────────────────

    @staticmethod
    def _scan_forward(
        mid_s: pd.Series,
        ts: pd.Timestamp,
        direction: int,
        horizon_td: pd.Timedelta,
        min_tick: float,
    ) -> tuple[bool, bool, float]:
        """
        Walk forward up to horizon_td seconds from ts.
        Returns (hit, adverse, best_fwd_mid).
        """
        fwd_slice = mid_s.loc[ts : ts + horizon_td]
        if len(fwd_slice) < 2:
            return False, False, mid_s.asof(ts)

        mid_now  = fwd_slice.iloc[0]
        best_fwd = mid_now
        hit = adverse = False

        for _, fwd_mid in fwd_slice.iloc[1:].items():
            move = (fwd_mid - mid_now) * direction
            if move >= min_tick:
                hit      = True
                best_fwd = fwd_mid
                break
            if move <= -min_tick:
                adverse  = True
                best_fwd = fwd_mid
                break

        return hit, adverse, best_fwd

    # ── interface ─────────────────────────────────────────────────────────────

    def fit(self, sessions: list[pd.DataFrame]) -> None:
        """
        Optional calibration step. Called once with all training sessions.
        Default: no-op (signals that need calibration override this).
        """
        pass

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the signal on one merged session DataFrame.

        Parameters
        ----------
        df : merged session DataFrame from merge_plot.asof_join()
             Must have at minimum: ts (index), mid, spread, obi, obi1, obi3,
             obi5, obi10, btc_mid.

        Returns
        -------
        pd.DataFrame with index=ts and at minimum RESULT_COLS present.
        Returns an empty DataFrame if no signals fired.
        """
        ...

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name!r})'