"""Resolve the live BTC 15m Kalshi market ticker via the public REST API."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_SERIES = "KXBTC15M"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def fetch_market_by_ticker(
    ticker: str,
    base_url: str = DEFAULT_BASE,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """GET /markets?tickers=… — returns the market dict for ``ticker``."""
    params = {"tickers": ticker, "limit": "20"}
    url = f"{base_url.rstrip('/')}/markets?{urlencode(params)}"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for m in data.get("markets") or []:
        if m.get("ticker") == ticker:
            return m
    raise LookupError(f"No market returned for ticker {ticker!r}")


def fetch_open_markets_for_series(
    series_ticker: str = DEFAULT_SERIES,
    base_url: str = DEFAULT_BASE,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Paginate GET /markets?series_ticker=…&status=open."""
    markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        params: Dict[str, str] = {
            "series_ticker": series_ticker,
            "status": "open",
            "limit": "1000",
        }
        if cursor:
            params["cursor"] = cursor
        url = f"{base_url.rstrip('/')}/markets?{urlencode(params)}"
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        batch = data.get("markets") or []
        markets.extend(batch)
        cursor = data.get("cursor") or None
        if not cursor or not batch:
            break
    return markets


def pick_current_btc15m_ticker(
    series_ticker: str = DEFAULT_SERIES,
    base_url: str = DEFAULT_BASE,
    timeout: float = 30.0,
) -> str:
    """
    Choose the open BTC 15m market to stream.

    Kalshi normally has one ``open`` contract at a time. If several are returned,
    we prefer a market whose trading window contains "now" (``open_time`` <= now
    < ``close_time``); if there is still a tie, we take the one with the earliest
    ``close_time`` (the interval ending soonest).
    """
    markets = fetch_open_markets_for_series(series_ticker, base_url, timeout)
    if not markets:
        raise RuntimeError(
            f"No open markets for series {series_ticker!r}. "
            "Try again when a window is trading, or pass --market explicitly."
        )

    now = _utcnow()
    parsed: List[tuple[Dict[str, Any], datetime, datetime]] = []
    for m in markets:
        try:
            close_t = _parse_ts(m["close_time"])
            open_t = _parse_ts(m.get("open_time") or m["close_time"])
        except (KeyError, TypeError, ValueError):
            continue
        parsed.append((m, open_t, close_t))

    if not parsed:
        raise RuntimeError("Open markets list could not be parsed (missing times).")

    not_yet_closed = [(m, ot, ct) for m, ot, ct in parsed if ct > now]
    if not not_yet_closed:
        raise RuntimeError(
            f"Series {series_ticker!r}: all listed open markets already have close_time in the past."
        )

    in_window = [(m, ot, ct) for m, ot, ct in not_yet_closed if ot <= now < ct]
    pool = in_window if in_window else not_yet_closed

    def sort_key(item: tuple[Dict[str, Any], datetime, datetime]):
        m, ot, ct = item
        # Prefer the active window; then soonest close (current 15m bar).
        in_w = 0 if (ot <= now < ct) else 1
        return (in_w, ct)

    pool.sort(key=sort_key)
    best = pool[0][0]
    return str(best["ticker"])


def describe_discovery(ticker: str, series_ticker: str) -> str:
    return f"Auto-selected {series_ticker} market: {ticker}"


def market_close_datetime(market: Dict[str, Any]) -> datetime:
    """Parse ``close_time`` from a market object (UTC)."""
    return _parse_ts(market["close_time"])


def market_open_datetime(market: Dict[str, Any]) -> datetime:
    """Parse ``open_time`` from a market object (UTC); falls back to ``close_time`` if missing."""
    raw = market.get("open_time") or market["close_time"]
    return _parse_ts(raw)
