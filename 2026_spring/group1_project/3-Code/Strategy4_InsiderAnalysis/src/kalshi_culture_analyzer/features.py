from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import Config
from .storage import Storage


def classify_aggressive_side(
    price: Optional[float],
    snapshot: Optional[dict],
    tolerance: float,
) -> Optional[str]:
    if price is None or snapshot is None:
        return None
    yes_bid = snapshot.get("yes_bid")
    yes_ask = snapshot.get("yes_ask")
    if yes_ask is not None and price >= (yes_ask - tolerance):
        return "yes"
    if yes_bid is not None and price <= (yes_bid + tolerance):
        return "no"
    return None


def compute_market_features(
    storage: Storage,
    market_ticker: str,
    ts_ms: int,
    config: Config,
) -> Dict[str, float]:
    snapshot_row = storage.get_snapshot_before(market_ticker, ts_ms)
    if not snapshot_row:
        return {}

    snapshot = dict(snapshot_row)

    trailing_window_ms = config.large_trade.trailing_volume_seconds() * 1000
    trailing_start = ts_ms - trailing_window_ms
    trades = storage.get_trades(market_ticker, since_ms=trailing_start)

    aggressive_yes_count = 0
    aggressive_yes_contracts = 0.0
    aggressive_burst_count = 0
    aggressive_burst_contracts = 0.0
    aggressive_step_contracts = 0.0
    max_aggressive_yes_trade = 0.0

    burst_window_ms = config.aggressive.burst_window_minutes * 60 * 1000
    burst_start = ts_ms - burst_window_ms
    step_window_ms = config.step_change.step_window_hours * 3600 * 1000
    step_start = ts_ms - step_window_ms

    trailing_volume = 0.0

    for trade in trades:
        count = trade["count"] or 0.0
        trailing_volume += count
        taker_side = trade["taker_side"]
        if not taker_side:
            snap = storage.get_snapshot_before(market_ticker, trade["ts"])
            taker_side = classify_aggressive_side(trade["price"], snap, config.aggressive.price_tolerance)

        if taker_side == "yes":
            aggressive_yes_count += 1
            aggressive_yes_contracts += count
            if count > max_aggressive_yes_trade:
                max_aggressive_yes_trade = count
            if trade["ts"] >= burst_start:
                aggressive_burst_count += 1
                aggressive_burst_contracts += count
            if trade["ts"] >= step_start:
                aggressive_step_contracts += count

    mid_change = None
    start_snapshot = storage.get_snapshot_before(market_ticker, step_start)
    if start_snapshot and snapshot.get("mid") is not None:
        start_mid = start_snapshot["mid"]
        if start_mid is not None:
            mid_change = snapshot["mid"] - start_mid

    return {
        "yes_bid": snapshot.get("yes_bid") or 0.0,
        "yes_ask": snapshot.get("yes_ask") or 0.0,
        "yes_bid_size": snapshot.get("yes_bid_size") or 0.0,
        "yes_ask_size": snapshot.get("yes_ask_size") or 0.0,
        "yes_mid": snapshot.get("mid") or 0.0,
        "spread": snapshot.get("spread") or 0.0,
        "volume": snapshot.get("volume") or 0.0,
        "open_interest": snapshot.get("open_interest") or 0.0,
        "trailing_volume": trailing_volume,
        "aggressive_yes_count": aggressive_yes_count,
        "aggressive_yes_contracts": aggressive_yes_contracts,
        "aggressive_burst_count": aggressive_burst_count,
        "aggressive_burst_contracts": aggressive_burst_contracts,
        "aggressive_step_contracts": aggressive_step_contracts,
        "max_aggressive_yes_trade": max_aggressive_yes_trade,
        "mid_change_window": mid_change if mid_change is not None else 0.0,
    }


def compute_event_favorites(
    storage: Storage,
    event_ticker: str,
    ts_ms: int,
) -> Dict[str, Dict[str, float]]:
    markets = storage.get_markets_for_event(event_ticker)
    mids: List[Tuple[str, float]] = []
    for market in markets:
        snap = storage.get_snapshot_before(market["market_ticker"], ts_ms)
        if not snap:
            continue
        mid = snap["mid"]
        if mid is None:
            continue
        mids.append((market["market_ticker"], float(mid)))

    if not mids:
        return {}

    mids.sort(key=lambda x: x[1], reverse=True)
    top_mid = mids[0][1]
    second_mid = mids[1][1] if len(mids) > 1 else None
    strength = top_mid - second_mid if second_mid is not None else top_mid

    features: Dict[str, Dict[str, float]] = {}
    for idx, (ticker, mid) in enumerate(mids, start=1):
        features[ticker] = {
            "favorite_rank": float(idx),
            "is_favorite": 1.0 if idx == 1 else 0.0,
            "favorite_strength": strength if idx == 1 else 0.0,
        }
    return features

