from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

from .config import Config
from .features import compute_event_favorites, compute_market_features
from .kalshi_client import KalshiClient, build_client_from_config
from .market_utils import extract_settlement_value, get_price_dollars, parse_iso_ts
from .rules import config_hash, evaluate_rules
from .storage import Storage
from .ticker_parser import TickerEntry, parse_ticker_file
from .utils import utc_now_ms


def expand_events(
    client: KalshiClient,
    storage: Storage,
    ticker_entries: List[TickerEntry],
    use_historical: bool = False,
) -> List[str]:
    market_tickers: List[str] = []
    for entry in ticker_entries:
        try:
            event_resp = client.get_event(entry.ticker, with_nested_markets=True)
        except Exception:
            continue
        event = event_resp.get("event", event_resp)
        storage.upsert_event(entry.ticker, event, tags=entry.tags)
        markets = event.get("markets") or []

        if not markets:
            markets = _fetch_all_markets(client, entry.ticker, use_historical=use_historical)

        for market in markets:
            market_ticker = market.get("ticker") or market.get("market_ticker")
            if not market_ticker:
                continue
            market_data = {
                "title": market.get("title"),
                "status": market.get("status"),
                "close_time": market.get("close_time"),
                "result": market.get("result"),
                "settlement_value": extract_settlement_value(market),
            }
            storage.upsert_market(market_ticker, entry.ticker, market_data)
            market_tickers.append(market_ticker)
    return market_tickers


def _fetch_all_markets(client: KalshiClient, event_ticker: str, use_historical: bool = False) -> List[Dict[str, Any]]:
    markets: List[Dict[str, Any]] = []
    cursor = None
    while True:
        resp = (
            client.get_historical_markets(event_ticker=event_ticker, cursor=cursor)
            if use_historical
            else client.get_markets(event_ticker=event_ticker, cursor=cursor)
        )
        batch = resp.get("markets") or []
        markets.extend(batch)
        cursor = resp.get("cursor")
        if not cursor:
            break
    return markets


def _snapshot_from_market(market: Dict[str, Any]) -> Dict[str, Any]:
    yes_bid = get_price_dollars(market, "yes_bid")
    yes_ask = get_price_dollars(market, "yes_ask")
    no_bid = get_price_dollars(market, "no_bid")
    no_ask = get_price_dollars(market, "no_ask")

    # Compute missing asks from opposite bids if available
    if yes_ask is None and no_bid is not None:
        yes_ask = 1.0 - no_bid
    if no_ask is None and yes_bid is not None:
        no_ask = 1.0 - yes_bid

    mid = None
    if yes_bid is not None and yes_ask is not None:
        mid = (yes_bid + yes_ask) / 2.0
    spread = None
    if yes_bid is not None and yes_ask is not None:
        spread = yes_ask - yes_bid

    return {
        "market_ticker": market.get("ticker") or market.get("market_ticker"),
        "ts": utc_now_ms(),
        "snapshot_type": "bbo",
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "mid": mid,
        "spread": spread,
        "volume": market.get("volume") or market.get("volume_24h"),
        "open_interest": market.get("open_interest"),
        "yes_bid_size": market.get("yes_bid_size") or market.get("yes_bid_size_fp"),
        "yes_ask_size": market.get("yes_ask_size") or market.get("yes_ask_size_fp"),
        "raw": market,
    }


def _fetch_trades(client: KalshiClient, market_ticker: str, since_ts: Optional[int]) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    cursor = None
    while True:
        resp = client.get_trades(market_ticker, min_ts=since_ts, cursor=cursor)
        batch = resp.get("trades") or []
        trades.extend(batch)
        cursor = resp.get("cursor")
        if not cursor:
            break
    return trades


def _trade_to_row(trade: Dict[str, Any]) -> Dict[str, Any]:
    ts = trade.get("ts")
    if ts is None:
        ts = parse_iso_ts(trade.get("created_time"))
    price = get_price_dollars(trade, "price")
    trade_id = trade.get("trade_id") or trade.get("id") or f"{trade.get('ticker')}:{ts}:{price}"
    return {
        "trade_id": str(trade_id),
        "market_ticker": trade.get("ticker") or trade.get("market_ticker"),
        "ts": ts or utc_now_ms(),
        "price": price,
        "count": trade.get("count"),
        "taker_side": trade.get("taker_side"),
        "is_inferred": False,
        "raw": trade,
    }


def _infer_trade_from_volume(prev_snapshot: Dict[str, Any], snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not prev_snapshot:
        return None
    prev_vol = prev_snapshot.get("volume")
    curr_vol = snapshot.get("volume")
    if prev_vol is None or curr_vol is None:
        return None
    delta = curr_vol - prev_vol
    if delta <= 0:
        return None
    prev_mid = prev_snapshot.get("mid")
    curr_mid = snapshot.get("mid")
    taker_side = "yes" if (curr_mid or 0) >= (prev_mid or 0) else "no"
    price = snapshot.get("yes_ask") if taker_side == "yes" else snapshot.get("yes_bid")
    trade_id = f"INF:{snapshot.get('market_ticker')}:{snapshot.get('ts')}:{taker_side}:{delta}"
    return {
        "trade_id": trade_id,
        "market_ticker": snapshot.get("market_ticker"),
        "ts": snapshot.get("ts"),
        "price": price,
        "count": delta,
        "taker_side": taker_side,
        "is_inferred": True,
        "raw": {"source": "inferred_volume_delta"},
    }


def collect(
    tickers_file: str,
    db_path: str,
    config: Config,
    poll_minutes: Optional[float] = None,
    use_historical: bool = False,
) -> None:
    storage = Storage(db_path)
    client = build_client_from_config(config)

    entries = parse_ticker_file(tickers_file)
    market_tickers = set(expand_events(client, storage, entries, use_historical=use_historical))

    poll_interval = poll_minutes or config.polling.poll_minutes

    while True:
        loop_start = time.time()
        # Refresh markets list per event in case of new markets
        market_tickers.update(expand_events(client, storage, entries, use_historical=use_historical))

        for market_ticker in sorted(market_tickers):
            try:
                market_resp = client.get_market(market_ticker)
            except Exception:
                continue
            market = market_resp.get("market", market_resp)
            snapshot = _snapshot_from_market(market)
            storage.insert_snapshot(snapshot)

            state = storage.get_collector_state(market_ticker)
            last_trade_ts = state["last_trade_ts"] if state else None
            trades_raw = []
            try:
                trades_raw = _fetch_trades(client, market_ticker, last_trade_ts)
            except Exception:
                trades_raw = []

            for trade in trades_raw:
                storage.insert_trade(_trade_to_row(trade))

            # Inference fallback if no trades returned
            if not trades_raw:
                prev_snapshot = storage.get_snapshot_before(market_ticker, snapshot["ts"] - 1)
                inferred = _infer_trade_from_volume(prev_snapshot, snapshot)
                if inferred:
                    storage.insert_trade(inferred)

            if trades_raw:
                newest_ts = max(_trade_to_row(t)["ts"] for t in trades_raw)
                storage.update_collector_state(market_ticker, last_trade_ts=newest_ts)

            storage.update_collector_state(market_ticker, last_snapshot_ts=snapshot["ts"])

            # Compute features + score
            features = compute_market_features(storage, market_ticker, snapshot["ts"], config)
            storage.insert_features(market_ticker, snapshot["ts"], "market", features)

            # Event-level favorites
            event_features_map = {}
            for entry in entries:
                event_features_map.update(compute_event_favorites(storage, entry.ticker, snapshot["ts"]))
            event_features = event_features_map.get(market_ticker, {})

            score, rules, explanation = evaluate_rules(market_ticker, features, event_features, config)
            storage.insert_score(market_ticker, snapshot["ts"], config_hash(config), score, rules, explanation)

        # Adaptive polling (simple): if any market had trades in burst window, reduce interval
        if config.polling.adaptive:
            trade_bursts = 0
            for market_ticker in sorted(market_tickers):
                features = compute_market_features(storage, market_ticker, utc_now_ms(), config)
                if features.get("aggressive_burst_count", 0) >= config.polling.adaptive_trade_burst_count:
                    trade_bursts += 1
            if trade_bursts > 0:
                poll_interval = config.polling.adaptive_min_minutes
            else:
                poll_interval = config.polling.adaptive_max_minutes

        elapsed = time.time() - loop_start
        sleep_for = max(0.0, poll_interval * 60 - elapsed)
        time.sleep(sleep_for)
