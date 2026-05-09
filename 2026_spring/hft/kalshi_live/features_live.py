"""Rolling order-book features with configurable sample rate (default 20 Hz).

Lookbacks are scaled by wall-clock so that at e.g. 20 Hz, ``mom_5s`` still uses
mid from ~5 seconds ago (matching 1 Hz training semantics for the GBM).

BTC momentum features (btc_mom_5s, btc_mom_10s, btc_accel) were added after
cross-momentum analysis showed IC of +0.14 at 5s horizon, improving GBM AUC
from 0.643 → 0.674 at 10s and from 0.652 → 0.704 at 5s (augmented model:
ensemble_gbm_10s_btcmom.joblib). These features require add_btc_price() to
be called on every Coinbase ticker or trade price update.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from kalshi_live.orderbook import compute_ob_from_book

# GBM feature order — must match ensemble_with_btc_mom.ALL_FEATS exactly.
# btc_mom_* appended after teammate's original 16 features.
FEATS: List[str] = [
    "obi_1",
    "obi_3",
    "obi_5",
    "obi_10",
    "spread",
    "mid_price",
    "depth_skew",
    "best_level_imbalance",
    "toxicity_10s",
    "depth_hhi",
    "mom_5s",
    "mom_10s",
    "mr_30s",
    "obi_vel_1s",
    "obi_vel_5s",
    "btc_vol_imbalance",
    # ── BTC cross-momentum features (augmented model) ──────────────────────
    "btc_mom_5s",   # BTC pct return over last 5s  (IC +0.14 @ 5s horizon)
    "btc_mom_10s",  # BTC pct return over last 10s (IC +0.13 @ 5s horizon)
    "btc_accel",    # btc_mom_5s − btc_mom_10s: is momentum accelerating?
]


class LiveFeatureTracker:
    """Samples the reconstructed book at ``sample_rate_hz``; builds GBM feature rows."""

    def __init__(
        self,
        sample_rate_hz: float = 20.0,
        btc_imbalance_window: float = 900.0,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.sample_rate_hz = float(sample_rate_hz)
        self.sample_interval = 1.0 / self.sample_rate_hz
        self.btc_imbalance_window = btc_imbalance_window
        self._last_sample_mono: float = 0.0
        # ≥90 s of history at the chosen rate (for mr_30s, moms, toxicity windows)
        _max_rows = int(max(120, round(self.sample_rate_hz * 90)))
        self._rows: Deque[Dict[str, float]] = deque(maxlen=_max_rows)
        self._btc_trades: Deque[Tuple[float, str, float]] = deque()
        # BTC price history for momentum: (wall_time, price). 90s keeps 60s lookback.
        self._btc_prices: Deque[Tuple[float, float]] = deque()
        self._prev_yes_depth: Optional[float] = None
        self._prev_no_depth: Optional[float] = None

    def _rows_for_seconds(self, seconds: float) -> int:
        """How many samples span ``seconds`` wall time at the current sample rate."""
        return max(1, int(round(seconds * self.sample_rate_hz)))

    def add_btc_trade(self, ts_wall: float, side: str, size: float) -> None:
        if size <= 0:
            return
        self._btc_trades.append((ts_wall, side.upper(), size))

    def add_btc_price(self, ts_wall: float, price: float) -> None:
        """Record a BTC price tick for momentum computation."""
        if price > 0:
            self._btc_prices.append((ts_wall, price))

    def _btc_price_n_ago(self, now_wall: float, seconds: float) -> Optional[float]:
        """Return the most recent BTC price that is at least ``seconds`` old."""
        cutoff = now_wall - seconds
        result: Optional[float] = None
        for ts, px in self._btc_prices:
            if ts <= cutoff:
                result = px
        return result

    def _prune_btc(self, now_wall: float) -> None:
        cutoff_imb = now_wall - self.btc_imbalance_window
        while self._btc_trades and self._btc_trades[0][0] < cutoff_imb:
            self._btc_trades.popleft()
        cutoff_mom = now_wall - 90.0  # 90s keeps enough history for 60s momentum
        while self._btc_prices and self._btc_prices[0][0] < cutoff_mom:
            self._btc_prices.popleft()

    def btc_vol_imbalance(self, now_wall: float) -> float:
        self._prune_btc(now_wall)
        buy = sell = 0.0
        for _, side, sz in self._btc_trades:
            if side == "BUY":
                buy += sz
            elif side == "SELL":
                sell += sz
        tot = buy + sell
        if tot <= 0:
            return 0.0
        return (buy - sell) / tot

    def maybe_sample(
        self,
        yes_book: Dict[float, float],
        no_book: Dict[float, float],
        now_mono: Optional[float] = None,
        now_wall: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        now_mono = now_mono if now_mono is not None else time.monotonic()
        now_wall = now_wall if now_wall is not None else time.time()
        if now_mono - self._last_sample_mono < self.sample_interval:
            return None

        if not yes_book or not no_book:
            return None

        base = compute_ob_from_book(yes_book, no_book)
        if base["best_yes_bid"] <= 0 and base["best_no_bid"] <= 0:
            return None

        self._last_sample_mono = now_mono

        yes_d = base["yes_depth_total"]
        no_d = base["no_depth_total"]
        ych = np.nan
        nch = np.nan
        if self._prev_yes_depth is not None and self._prev_no_depth is not None:
            ych = yes_d - self._prev_yes_depth
            nch = no_d - self._prev_no_depth
        self._prev_yes_depth = yes_d
        self._prev_no_depth = no_d

        abs_y = abs(ych) if ych == ych else 0.0
        abs_n = abs(nch) if nch == nch else 0.0
        tox = abs_y / (abs_y + abs_n) if (abs_y + abs_n) > 0 else np.nan

        row = dict(base)
        row["yes_depth_change"] = ych
        row["no_depth_change"] = nch
        row["toxicity"] = tox
        self._rows.append(row)

        return self._finalize_row(now_wall)

    def _finalize_row(self, now_wall: float) -> Dict[str, float]:
        r = self._rows[-1]
        mid = float(r["mid_price"])
        yes_depth_total = float(r["yes_depth_total"])
        no_depth_total = float(r["no_depth_total"])
        bysq = float(r["best_yes_bid_qty"])
        bnq = float(r["best_no_bid_qty"])

        depth_skew = (no_depth_total - yes_depth_total) / (
            yes_depth_total + no_depth_total
        ) if (yes_depth_total + no_depth_total) > 0 else 0.0
        best_level_imbalance = (bysq - bnq) / (bysq + bnq) if (bysq + bnq) > 0 else 0.0
        denom = yes_depth_total**2 + no_depth_total**2
        depth_hhi = (bysq**2 + bnq**2) / denom if denom > 0 else np.nan

        mids = [float(x["mid_price"]) for x in self._rows]
        obi1s = [float(x["obi_1"]) for x in self._rows]

        lag5 = self._rows_for_seconds(5.0)
        lag10 = self._rows_for_seconds(10.0)
        lag30 = self._rows_for_seconds(30.0)
        lag1 = self._rows_for_seconds(1.0)

        mom_5s = mid - mids[-(lag5 + 1)] if len(mids) > lag5 else np.nan
        mom_10s = mid - mids[-(lag10 + 1)] if len(mids) > lag10 else np.nan
        tail30 = mids[-lag30:] if len(mids) >= lag30 else mids
        mr_30s = float(np.mean(tail30)) - mid if tail30 else np.nan

        obi_vel_1s = (
            obi1s[-1] - obi1s[-(lag1 + 1)] if len(obi1s) > lag1 else np.nan
        )
        obi_vel_5s = (
            obi1s[-1] - obi1s[-(lag5 + 1)] if len(obi1s) > lag5 else np.nan
        )

        tox_series = [x["toxicity"] for x in self._rows if x.get("toxicity") == x.get("toxicity")]
        win_tox = self._rows_for_seconds(10.0)
        if len(tox_series) >= 3:
            chunk = tox_series[-win_tox:] if win_tox else tox_series
            toxicity_10s = float(np.mean(chunk))
        else:
            toxicity_10s = np.nan

        btc_imb = self.btc_vol_imbalance(now_wall)

        # ── BTC price momentum ────────────────────────────────────────────────
        # Matches training semantics: pct_change over N 1s bars resampled from
        # raw BTC ticks (btc_cross_momentum.py). Live uses wall-clock lookback.
        btc_mom_5s = btc_mom_10s = btc_accel = np.nan
        if self._btc_prices:
            now_px = self._btc_prices[-1][1]
            px_5s  = self._btc_price_n_ago(now_wall, 5.0)
            px_10s = self._btc_price_n_ago(now_wall, 10.0)
            if px_5s is not None and px_5s > 0:
                btc_mom_5s = (now_px - px_5s) / px_5s
            if px_10s is not None and px_10s > 0:
                btc_mom_10s = (now_px - px_10s) / px_10s
            if not (btc_mom_5s != btc_mom_5s) and not (btc_mom_10s != btc_mom_10s):
                btc_accel = btc_mom_5s - btc_mom_10s

        out = {
            **r,
            "depth_skew": depth_skew,
            "best_level_imbalance": best_level_imbalance,
            "depth_hhi": depth_hhi,
            "mom_5s": mom_5s,
            "mom_10s": mom_10s,
            "mr_30s": mr_30s,
            "obi_vel_1s": obi_vel_1s,
            "obi_vel_5s": obi_vel_5s,
            "toxicity_10s": toxicity_10s,
            "btc_vol_imbalance": btc_imb,
            "btc_mom_5s": btc_mom_5s,
            "btc_mom_10s": btc_mom_10s,
            "btc_accel": btc_accel,
        }
        return out

    def feature_vector(self, row: Dict[str, Any]) -> Optional[np.ndarray]:
        vec = []
        for name in FEATS:
            v = row.get(name)
            if v is None or (isinstance(v, float) and (v != v)):  # NaN
                return None
            vec.append(float(v))
        return np.array(vec, dtype=np.float64).reshape(1, -1)


def kalshi_fee(price: float, qty: int, fee_type: str = "maker") -> float:
    rate = 0.07 if fee_type == "taker" else 0.0175
    return float(np.ceil(rate * qty * price * (1 - price) * 100) / 100)
