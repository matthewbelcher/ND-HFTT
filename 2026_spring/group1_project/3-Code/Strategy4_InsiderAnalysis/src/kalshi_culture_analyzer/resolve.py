from __future__ import annotations

from typing import Any, Dict, Optional

from .market_utils import extract_settlement_value
from .config import Config
from .kalshi_client import KalshiClient, build_client_from_config
from .storage import Storage
from .utils import utc_now_ms


def resolve_outcomes(db_path: str, config: Config) -> None:
    storage = Storage(db_path)
    client = build_client_from_config(config)

    for market in storage.list_markets():
        if market["result"]:
            continue
        try:
            resp = client.get_market(market["market_ticker"])
        except Exception:
            continue
        market_obj = resp.get("market", resp)
        result = market_obj.get("result")
        if result:
            settlement_value = extract_settlement_value(market_obj)
            storage.upsert_market(market["market_ticker"], market["event_ticker"], {
                "title": market_obj.get("title"),
                "status": market_obj.get("status"),
                "close_time": market_obj.get("close_time"),
                "result": result,
                "settlement_value": settlement_value,
            })
            storage.upsert_outcome(market["market_ticker"], result, settlement_value, utc_now_ms())
