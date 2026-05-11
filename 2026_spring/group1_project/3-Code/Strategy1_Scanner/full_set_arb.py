#!/usr/bin/env python3
"""
Full-set arbitrage for Kalshi range events.

Given an event (event_ticker or event_url), computes risk-free arbitrage by
buying YES (or NO) across all ranges. Uses the full market order book (walks
every price level by size) and includes all fees: per-level taker/maker fees
and series fee multiplier where applicable. Profit is cost (premium + fees) vs
payout; budget sizing finds the number of sets that maximizes profit within cap.

Inputs: event_url or event_ticker, fee_mode (taker/maker), optional budget_dollars,
        contracts_per_set, orderbooks JSON, market metadata JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

# Series fee cache: avoid hitting API more than once per day per series
SERIES_FEE_CACHE_MAX_AGE_HOURS = 24

# For partial cover: only count as arb opportunity if no excluded outcome has implied prob >= this
MIN_PROB_TO_REQUIRE_TRADEABLE = 0.01
SERIES_FEE_CACHE_FILENAME = "series_fee_cache.json"

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# --- Fee and rounding ---

def ceil_to_cent(x: float) -> float:
    """Round up to the next $0.01."""
    return math.ceil(x * 100) / 100.0

def fee_per_trade(
    c: int,
    p: float,
    fee_mode: Literal["taker", "maker"],
    fee_multiplier: float = 1.0,
) -> float:
    """Fee for one trade: C contracts at price P (dollars). Uses Kalshi series fee_multiplier when set."""
    rate = (0.07 if fee_mode == "taker" else 0.0175) * fee_multiplier
    raw = rate * c * p * (1.0 - p)
    return ceil_to_cent(raw)

def fee_per_trade_best_case(
    total_premium: float,
    total_contracts: int,
    fee_mode: Literal["taker", "maker"],
    fee_multiplier: float = 1.0,
) -> float:
    """Best-case fee: single trade at VWAP (one ceil_to_cent). Uses Kalshi series fee_multiplier when set."""
    if total_contracts <= 0:
        return 0.0
    vwap = total_premium / total_contracts
    rate = (0.07 if fee_mode == "taker" else 0.0175) * fee_multiplier
    raw = rate * total_contracts * vwap * (1.0 - vwap)
    return ceil_to_cent(raw)

# --- Orderbook parsing ---

def _parse_price(p: Any) -> float:
    if isinstance(p, (int, float)):
        if p <= 1:
            return float(p)
        return p / 100.0  # cents to dollars
    s = str(p).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)

def _parse_size(sz: Any) -> int:
    if isinstance(sz, (int, float)):
        return int(sz)
    return int(Decimal(str(sz)))

def get_yes_ladder_taker(orderbook: dict, ticker: str) -> list[tuple[float, int, str]]:
    """
    Market-buy YES (taker): consume NO ladder, transform price.
    """
    ob = orderbook.get("orderbook", orderbook)
    no_lvls = ob.get("no_dollars") or ob.get("no") or []
    out: list[tuple[float, int, str]] = []
    for px, sz in no_lvls:
        no_price = _parse_price(px)       # dollars
        size = _parse_size(sz)
        yes_price = 1.0 - no_price
        if size > 0 and 0.0 <= yes_price <= 1.0:
            out.append((round(yes_price * 100) / 100.0, size, "1-no_level"))
    out.sort(key=lambda x: (x[0], -x[1]))
    return out


def get_no_ladder_taker(orderbook: dict, ticker: str) -> list[tuple[float, int, str]]:
    """
    Market-buy NO (taker): consume YES ladder, transform price.
    """
    ob = orderbook.get("orderbook", orderbook)
    yes_lvls = ob.get("yes_dollars") or ob.get("yes") or []
    out: list[tuple[float, int, str]] = []
    for px, sz in yes_lvls:
        yes_price = _parse_price(px)      # dollars
        size = _parse_size(sz)
        no_price = 1.0 - yes_price
        if size > 0 and 0.0 <= no_price <= 1.0:
            out.append((round(no_price * 100) / 100.0, size, "1-yes_level"))
    out.sort(key=lambda x: (x[0], -x[1]))
    return out


def fill_from_ladder_with_fees(
    ladder: list[tuple[float, int, str]],
    contracts: int,
    fee_mode: Literal["taker", "maker"],
    fee_multiplier: float = 1.0,
) -> tuple[list[tuple[float, int, str]], float, float, float, int, int]:
    """
    Returns (fills, premium, fee_worst, fee_best, filled, unfilled).
    fee_multiplier: from series API (irregular fee structure); 1.0 = standard.
    """
    if contracts <= 0:
        return [], 0.0, 0.0, 0.0, 0, 0

    fills: list[tuple[float, int, str]] = []
    remaining = contracts
    total_premium = 0.0
    fee_worst = 0.0

    for price, size, source in ladder:
        if remaining <= 0:
            break
        take = min(remaining, size)
        if take <= 0:
            continue
        fills.append((price, take, source))
        total_premium += price * take
        fee_worst += fee_per_trade(take, price, fee_mode, fee_multiplier)
        remaining -= take

    filled = contracts - remaining
    fee_best = fee_per_trade_best_case(total_premium, filled, fee_mode, fee_multiplier) if filled > 0 else 0.0
    return fills, total_premium, fee_worst, fee_best, filled, remaining

# --- Event / market parsing ---

# Kalshi market status: only live markets are tradeable for arb; exclude closed/settled/unopened/finalized
NON_TRADEABLE_STATUSES = frozenset({"closed", "settled", "unopened", "finalized"})


def filter_tradeable_markets(markets: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split markets into tradeable vs excluded (closed/settled/unopened/finalized).
    Returns (tradeable_markets, excluded_markets). Markets with no status are kept (backward compat).
    """
    tradeable: list[dict] = []
    excluded: list[dict] = []
    for m in markets:
        status = (m.get("status") or "").strip().lower()
        if status in NON_TRADEABLE_STATUSES:
            excluded.append(m)
        else:
            tradeable.append(m)
    return tradeable, excluded


def filter_markets_with_two_sided_liquidity(
    markets: list[dict],
    orderbooks_by_ticker: dict[str, dict],
    min_contracts: int = 1,
) -> tuple[list[dict], list[dict]]:
    """
    Keep only markets that have at least min_contracts on both YES and NO ladders
    (so both buy-YES and buy-NO are fillable). Excludes stale/one-sided markets.
    Returns (tradeable_markets, excluded_markets).
    """
    tradeable: list[dict] = []
    excluded: list[dict] = []
    for m in markets:
        t = m.get("ticker", "")
        ob = orderbooks_by_ticker.get(t, {})
        yes_lad = get_yes_ladder_taker(ob, t)
        no_lad = get_no_ladder_taker(ob, t)
        yes_size = sum(s for _, s, _ in yes_lad)
        no_size = sum(s for _, s, _ in no_lad)
        if yes_size >= min_contracts and no_size >= min_contracts:
            tradeable.append(m)
        else:
            excluded.append(m)
    return tradeable, excluded


def filter_markets_with_yes_liquidity(
    markets: list[dict],
    orderbooks_by_ticker: dict[str, dict],
    min_contracts: int = 1,
) -> tuple[list[dict], list[dict]]:
    """
    Keep only markets that have at least min_contracts on the YES ladder
    (so buy-YES is fillable). Returns (tradeable_markets, excluded_markets).
    """
    tradeable: list[dict] = []
    excluded: list[dict] = []
    for m in markets:
        t = m.get("ticker", "")
        ob = orderbooks_by_ticker.get(t, {})
        yes_lad = get_yes_ladder_taker(ob, t)
        yes_size = sum(s for _, s, _ in yes_lad)
        if yes_size >= min_contracts:
            tradeable.append(m)
        else:
            excluded.append(m)
    return tradeable, excluded


def filter_markets_with_no_liquidity(
    markets: list[dict],
    orderbooks_by_ticker: dict[str, dict],
    min_contracts: int = 1,
) -> tuple[list[dict], list[dict]]:
    """
    Keep only markets that have at least min_contracts on the NO ladder
    (so buy-NO is fillable). Returns (tradeable_markets, excluded_markets).
    """
    tradeable: list[dict] = []
    excluded: list[dict] = []
    for m in markets:
        t = m.get("ticker", "")
        ob = orderbooks_by_ticker.get(t, {})
        no_lad = get_no_ladder_taker(ob, t)
        no_size = sum(s for _, s, _ in no_lad)
        if no_size >= min_contracts:
            tradeable.append(m)
        else:
            excluded.append(m)
    return tradeable, excluded


def event_ticker_from_input(event_url: str | None, event_ticker: str | None) -> str | None:
    if event_ticker:
        return event_ticker.strip().upper()
    if not event_url:
        return None
    # e.g. https://kalshi.com/events/EVENT-TICKER or .../markets/...
    m = re.search(r"/(?:events?|markets?)/([A-Za-z0-9_-]+)", event_url)
    if m:
        return m.group(1).upper()
    return event_url.strip().upper()

def _resolve_path(path: str) -> str:
    """Resolve path; if not absolute and not found, try relative to script dir."""
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, path)
    return candidate if os.path.isfile(candidate) else path


def guaranteed_yes_payout(exact_yes_count: int, total_markets: int, covered_markets: int) -> float:
    """
    Worst-case payout when buying YES in covered_markets of total_markets,
    given exactly exact_yes_count markets resolve YES.
    """
    uncovered_markets = max(0, total_markets - covered_markets)
    return float(max(0, exact_yes_count - uncovered_markets))


def guaranteed_no_payout(exact_yes_count: int, covered_markets: int) -> float:
    """
    Worst-case payout when buying NO in covered_markets,
    given exactly exact_yes_count markets resolve YES.
    """
    return float(max(0, covered_markets - exact_yes_count))


# --- Kalshi API fetch (no auth required for market/orderbook data) ---

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _http_get(url: str, timeout: int = 15) -> dict[str, Any] | None:
    """GET URL and return JSON. Returns None on failure."""
    if _HAS_REQUESTS:
        try:
            r = requests.get(url, headers={"accept": "application/json"}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Request failed {url}: {e}", file=sys.stderr)
            return None
    try:
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError
        req = Request(url, headers={"accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Request failed {url}: {e}", file=sys.stderr)
        return None


def _series_fee_cache_path() -> Path:
    """Path to the series fee cache file (in same dir as this script)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return Path(script_dir) / SERIES_FEE_CACHE_FILENAME


def _load_series_fee_cache() -> dict[str, Any]:
    """Load cache from disk. Returns { \"series\": { ticker: { fee_type, fee_multiplier, cached_at } } }."""
    path = _series_fee_cache_path()
    if not path.is_file():
        return {"series": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and "series" in data else {"series": {}}
    except (json.JSONDecodeError, OSError):
        return {"series": {}}


def _save_series_fee_cache(data: dict[str, Any]) -> None:
    """Write cache to disk."""
    path = _series_fee_cache_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass  # best-effort; don't fail the run


def get_series_fee_cached(series_ticker: str, timeout: int = 8) -> dict[str, Any] | None:
    """
    Get series fee_type and fee_multiplier, using a file cache so the API is hit at most once per day per series.
    Returns { \"fee_type\": str, \"fee_multiplier\": float } or None on failure.
    """
    if not (series_ticker and str(series_ticker).strip()):
        return None
    ticker = str(series_ticker).strip()
    now = datetime.now(timezone.utc)
    max_age = timedelta(hours=SERIES_FEE_CACHE_MAX_AGE_HOURS)

    cache = _load_series_fee_cache()
    series_cache = cache.setdefault("series", {})
    if not isinstance(series_cache, dict):
        series_cache = {}
        cache["series"] = series_cache

    entry = series_cache.get(ticker)
    if isinstance(entry, dict):
        try:
            cached_at = datetime.fromisoformat(entry.get("cached_at", "").replace("Z", "+00:00"))
            if (now - cached_at) <= max_age:
                return {
                    "fee_type": entry.get("fee_type"),
                    "fee_multiplier": float(entry.get("fee_multiplier", 1.0)),
                }
        except (ValueError, TypeError):
            pass

    # Cache miss or expired: fetch from API (in thread with hard timeout so DNS/connect cannot hang)
    result_holder: list[dict[str, Any] | None] = []

    def _fetch() -> None:
        url = f"{KALSHI_API_BASE}/series/{ticker}"
        data = _http_get(url, timeout=timeout)
        result_holder.append(data)

    th = threading.Thread(target=_fetch, daemon=True)
    th.start()
    th.join(timeout=timeout + 2)  # allow slightly more than HTTP timeout
    if not result_holder:
        return None  # timed out or still running (daemon will be abandoned)
    data = result_holder[0]
    series = data.get("series") if isinstance(data, dict) else None
    if not isinstance(series, dict):
        return None

    fee_type = series.get("fee_type")
    try:
        fee_multiplier = float(series.get("fee_multiplier", 1.0))
    except (TypeError, ValueError):
        fee_multiplier = 1.0

    series_cache[ticker] = {
        "fee_type": fee_type,
        "fee_multiplier": fee_multiplier,
        "cached_at": now.isoformat(),
    }
    _save_series_fee_cache(cache)

    return {"fee_type": fee_type, "fee_multiplier": fee_multiplier}


def fetch_series_from_api(series_ticker: str, timeout: int = 15) -> dict[str, Any] | None:
    """
    Fetch series from Kalshi API (fee_type, fee_multiplier for irregular fee structure).
    Returns series dict or None. Prefer get_series_fee_cached() so the result is cached for 24h.
    """
    if not (series_ticker and str(series_ticker).strip()):
        return None
    url = f"{KALSHI_API_BASE}/series/{series_ticker}"
    data = _http_get(url, timeout=timeout)
    if data is None:
        return None
    return data.get("series") if isinstance(data.get("series"), dict) else None


def fetch_event_from_api(event_ticker: str, timeout: int = 8) -> dict[str, Any] | None:
    """
    Fetch event with nested markets from Kalshi API.
    Returns API response (event + markets) suitable as market_metadata_json, or None.
    """
    url = f"{KALSHI_API_BASE}/events/{event_ticker}?with_nested_markets=true"
    data = _http_get(url, timeout=timeout)
    if data is None:
        return None
    # API returns { "event": { ..., "markets": [...] } } - use as market metadata
    if "event" not in data:
        print("Unexpected API response: no 'event' key", file=sys.stderr)
        return None
    return data


def fetch_orderbook_from_api(ticker: str, timeout: int = 8) -> dict[str, Any] | None:
    """Fetch orderbook for one market. Returns { \"orderbook\": { \"yes\": [...], \"no\": [...] } } or None."""
    url = f"{KALSHI_API_BASE}/markets/{ticker}/orderbook"
    return _http_get(url, timeout=timeout)


def fetch_orderbooks_for_markets(
    tickers: list[str],
    delay_seconds: float = 0.1,
    timeout_per_market: int = 8,
    progress_stderr: bool = True,
) -> dict[str, dict]:
    """Fetch orderbook for each ticker. Returns { ticker: { orderbook: ... } }."""
    orderbooks_by_ticker: dict[str, dict] = {}
    n = len(tickers)
    for i, ticker in enumerate(tickers):
        if progress_stderr and n > 1:
            print(f"  Orderbook {i + 1}/{n}: {ticker}...", file=sys.stderr, flush=True)
        if delay_seconds and i > 0:
            time.sleep(delay_seconds)
        ob = fetch_orderbook_from_api(ticker, timeout=timeout_per_market)
        if ob is not None:
            orderbooks_by_ticker[ticker] = ob
        else:
            orderbooks_by_ticker[ticker] = {}  # mark as missing
    return orderbooks_by_ticker


def fetch_event_and_orderbooks_from_api(event_ticker: str) -> tuple[dict, list[dict], dict[str, dict]]:
    """
    Fetch event (with markets) and all orderbooks from Kalshi API.
    Returns (event, markets_list, orderbooks_by_ticker).
    """
    event_dict: dict = {}
    markets_list: list[dict] = []
    orderbooks_by_ticker: dict[str, dict] = {}

    raw = fetch_event_from_api(event_ticker)
    if raw is None:
        return event_dict, markets_list, orderbooks_by_ticker

    event = raw.get("event", raw)
    if isinstance(event, dict):
        event_dict = dict(event)
        if "mutually_exclusive" not in event_dict and isinstance(raw, dict):
            event_dict["mutually_exclusive"] = raw.get("mutually_exclusive", False)
        markets_list = event_dict.get("markets", raw.get("markets", []))
    if not isinstance(markets_list, list):
        markets_list = []

    markets_list, excluded = filter_tradeable_markets(markets_list)
    if excluded:
        tickers_excluded = [m.get("ticker", "") for m in excluded if m.get("ticker")]
        print(f"Excluding {len(excluded)} non-tradeable market(s) (closed/settled/unopened/finalized): {', '.join(tickers_excluded)}", file=sys.stderr)

    tickers = [m.get("ticker") for m in markets_list if m.get("ticker")]
    if not tickers:
        return event_dict, markets_list, orderbooks_by_ticker

    print(f"Fetching orderbooks for {len(tickers)} markets...", file=sys.stderr)
    orderbooks_by_ticker = fetch_orderbooks_for_markets(tickers)
    return event_dict, markets_list, orderbooks_by_ticker


def load_event_and_orderbooks(
    event_ticker: str,
    orderbooks_path: str | None,
    market_data_path: str | None,
    orderbooks_json: dict | None,
    market_metadata_json: dict | None,
) -> tuple[dict, list[dict], dict[str, dict]]:
    """
    Load event (with markets) and per-ticker orderbooks.
    Returns (event, markets_list, orderbooks_by_ticker).
    """
    event: dict = {}
    markets_list: list[dict] = []
    orderbooks_by_ticker: dict[str, dict] = {}

    if market_metadata_json is not None:
        raw_event = market_metadata_json.get("event", market_metadata_json)
        event = dict(raw_event) if isinstance(raw_event, dict) else {}
        # mutually_exclusive may be on event or at top level (API variance)
        if "mutually_exclusive" not in event and isinstance(market_metadata_json, dict):
            event["mutually_exclusive"] = market_metadata_json.get("mutually_exclusive", False)
        # Prefer event["markets"] (nested); top-level "markets" is often empty in API responses.
        if "markets" in event and isinstance(event["markets"], list) and len(event["markets"]) > 0:
            markets_list = event["markets"]
        elif "markets" in market_metadata_json and isinstance(market_metadata_json["markets"], list):
            markets_list = market_metadata_json["markets"]
        markets_list, excluded = filter_tradeable_markets(markets_list)
        if excluded:
            tickers_excluded = [m.get("ticker", "") for m in excluded if m.get("ticker")]
            print(f"Excluding {len(excluded)} non-tradeable market(s) (closed/settled/unopened/finalized): {', '.join(tickers_excluded)}", file=sys.stderr)
    elif market_data_path:
        market_data_path = _resolve_path(market_data_path)
        if os.path.isfile(market_data_path):
            with open(market_data_path, "r") as f:
                data = json.load(f)
            event = dict(data.get("event", data)) if isinstance(data.get("event", data), dict) else {}
            if not event and isinstance(data, dict):
                event = dict(data)
            markets_list = event.get("markets", data.get("markets", []))
            if "mutually_exclusive" not in event and isinstance(data, dict):
                event["mutually_exclusive"] = data.get("mutually_exclusive", False)
            markets_list, excluded = filter_tradeable_markets(markets_list)
            if excluded:
                tickers_excluded = [m.get("ticker", "") for m in excluded if m.get("ticker")]
                print(f"Excluding {len(excluded)} non-tradeable market(s) (closed/settled/unopened/finalized): {', '.join(tickers_excluded)}", file=sys.stderr)
    else:
        return event, markets_list, orderbooks_by_ticker

    if orderbooks_json is not None:
        ob_root = orderbooks_json.get("orderbooks", orderbooks_json)
        if isinstance(ob_root, dict):
            orderbooks_by_ticker = ob_root
        else:
            orderbooks_by_ticker = {}
    elif orderbooks_path:
        orderbooks_path = _resolve_path(orderbooks_path)
        if os.path.isfile(orderbooks_path):
            with open(orderbooks_path, "r") as f:
                data = json.load(f)
            orderbooks_by_ticker = data.get("orderbooks", data)
    else:
        orderbooks_by_ticker = {}

    return event, markets_list, orderbooks_by_ticker

# --- Main arb logic ---


def _run_single_binary_report(
    event_ticker: str,
    event: dict,
    markets: list[dict],
    orderbooks_by_ticker: dict[str, dict],
    fee_mode: Literal["taker", "maker"],
    contracts_per_set: int,
    result_template: dict[str, Any],
    fee_multiplier: float = 1.0,
) -> dict[str, Any]:
    """
    Single binary market (k=1, mutually_exclusive=False): report executable YES/NO
    price and fee, and optional combined "buy both sides" cost vs $1 payout.
    No partition full-set logic.
    """
    m = markets[0]
    t = m.get("ticker", "")
    ob = orderbooks_by_ticker.get(t, {})

    yes_ladder = get_yes_ladder_taker(ob, t)
    no_ladder = get_no_ladder_taker(ob, t)

    yes_fills, yes_prem, yes_fw, yes_fb, yes_filled, _ = fill_from_ladder_with_fees(
        yes_ladder, contracts_per_set, fee_mode, fee_multiplier
    )
    no_fills, no_prem, no_fw, no_fb, no_filled, _ = fill_from_ladder_with_fees(
        no_ladder, contracts_per_set, fee_mode, fee_multiplier
    )

    # Executable price: effective price for requested contracts (premium / filled)
    yes_price = (yes_prem / yes_filled) if yes_filled else None
    no_price = (no_prem / no_filled) if no_filled else None

    cost_yes_worst = (yes_prem + yes_fw) if yes_filled else None
    cost_yes_best = (yes_prem + yes_fb) if yes_filled else None
    cost_no_worst = (no_prem + no_fw) if no_filled else None
    cost_no_best = (no_prem + no_fb) if no_filled else None

    buy_both_worst = (cost_yes_worst + cost_no_worst) if (cost_yes_worst is not None and cost_no_worst is not None) else None
    buy_both_best = (cost_yes_best + cost_no_best) if (cost_yes_best is not None and cost_no_best is not None) else None
    payout_both = 1.0
    profit_both_worst = (payout_both - buy_both_worst) if buy_both_worst is not None else None
    profit_both_best = (payout_both - buy_both_best) if buy_both_best is not None else None

    out = dict(result_template)
    out["single_binary"] = True
    out["yes_side"] = {
        "ticker": t,
        "executable_price": yes_price,
        "premium": yes_prem,
        "fee_worst": yes_fw,
        "fee_best": yes_fb,
        "cost_worst": cost_yes_worst,
        "cost_best": cost_yes_best,
        "filled": yes_filled,
        "fills": [{"price": p, "size": s, "source": src} for p, s, src in yes_fills],
    }
    out["no_side"] = {
        "ticker": t,
        "executable_price": no_price,
        "premium": no_prem,
        "fee_worst": no_fw,
        "fee_best": no_fb,
        "cost_worst": cost_no_worst,
        "cost_best": cost_no_best,
        "filled": no_filled,
        "fills": [{"price": p, "size": s, "source": src} for p, s, src in no_fills],
    }
    out["buy_both_sides"] = {
        "cost_worst": buy_both_worst,
        "cost_best": buy_both_best,
        "payout": payout_both,
        "profit_worst": profit_both_worst,
        "profit_best": profit_both_best,
    }
    out["yes_arb"] = None
    out["no_arb"] = None
    out["summary"] = {
        "arb_exists_yes": False,
        "arb_exists_no": False,
        "best_side": "N/A",
        "profit_per_set_yes_worst": None,
        "profit_per_set_no_worst": None,
        "profit_per_dollar_yes": None,
        "profit_per_dollar_no": None,
        "execution_risks": [
            "Fragmentation/rounding: fees computed per-level (worst) vs single VWAP (best)",
            "Partial fills if size moves before execution",
        ],
    }
    return out


def run_full_set_arb(
    event_ticker: str,
    event: dict,
    markets: list[dict],
    orderbooks_by_ticker: dict[str, dict],
    fee_mode: Literal["taker", "maker"],
    contracts_per_set: int = 1,
    budget_dollars: float | None = None,
    n_yes_resolves: int | None = None,
) -> dict[str, Any]:
    """
    Compute YES and NO basket arbitrage.
    n_yes_resolves: exact number of markets that resolve YES. Default 1 if mutually_exclusive.
    For non-mutually-exclusive events with exactly m winners (e.g. top 10): set n_yes_resolves=m.
    Payouts are guaranteed lower bounds when some markets lack liquidity.
    """
    missing_ob: list[str] = []
    for m in markets:
        t = m.get("ticker", "")
        if t and t not in orderbooks_by_ticker:
            missing_ob.append(t)

    result: dict[str, Any] = {
        "event_ticker": event_ticker,
        "k": 0,  # set after filtering (two-sided count for backward compat)
        "k_yes": 0,
        "k_no": 0,
        "n": 0,
        "mutually_exclusive": event.get("mutually_exclusive", False),
        "fee_mode": fee_mode,
        "contracts_per_set": contracts_per_set,
        "budget_dollars": budget_dollars,
        "missing_orderbooks": missing_ob,
        "excluded_no_two_sided_liquidity": [],
        "excluded_no_yes_liquidity": [],
        "excluded_no_no_liquidity": [],
        "yes_arb": None,
        "no_arb": None,
        "summary": {},
    }

    # Irregular fee structure: use series fee_type / fee_multiplier (cached once per day in series_fee_cache.json)
    series_ticker = event.get("series_ticker") or (event.get("event") or {}).get("series_ticker")
    fee_multiplier = 1.0
    series_fee_type: str | None = None
    if series_ticker:
        cached = get_series_fee_cached(str(series_ticker).strip())
        if cached is not None:
            series_fee_type = cached.get("fee_type")
            try:
                fee_multiplier = float(cached.get("fee_multiplier", 1.0))
            except (TypeError, ValueError):
                fee_multiplier = 1.0
            result["series_ticker"] = series_ticker
            result["series_fee_type"] = series_fee_type
            result["series_fee_multiplier"] = fee_multiplier
        else:
            print(f"Note: could not fetch series {series_ticker} for fee structure; using standard fees.", file=sys.stderr)

    if missing_ob:
        result["summary"]["error"] = f"Missing orderbook data for: {missing_ob}"
        return result

    # Evaluate YES and NO tradeability separately (markets with YES liquidity vs NO liquidity)
    markets_yes, excluded_yes = filter_markets_with_yes_liquidity(markets, orderbooks_by_ticker)
    markets_no, excluded_no = filter_markets_with_no_liquidity(markets, orderbooks_by_ticker)
    k_total = len(markets)
    k_yes = len(markets_yes)
    k_no = len(markets_no)
    mutually_exclusive = result["mutually_exclusive"]
    n = n_yes_resolves if n_yes_resolves is not None else (1 if mutually_exclusive else None)
    if n is None:
        n = 1  # assume 1 YES for standard range event
    if n < 0 or n > k_total:
        result["k_total"] = k_total
        result["k_yes"] = k_yes
        result["k_no"] = k_no
        result["n"] = n
        result["summary"]["error"] = f"Invalid exact YES count {n}; event has {k_total} market(s)."
        return result
    result["k_total"] = k_total
    result["k_yes"] = k_yes
    result["k_no"] = k_no
    result["n"] = n
    # Backward compat: k = number of markets with both YES and NO liquidity
    tickers_yes_set = {m.get("ticker") for m in markets_yes if m.get("ticker")}
    tickers_no_set = {m.get("ticker") for m in markets_no if m.get("ticker")}
    two_sided_tickers = tickers_yes_set & tickers_no_set
    result["k"] = len(two_sided_tickers)
    result["excluded_no_yes_liquidity"] = [m.get("ticker", "") for m in excluded_yes if m.get("ticker")]
    result["excluded_no_no_liquidity"] = [m.get("ticker", "") for m in excluded_no if m.get("ticker")]
    all_tickers = [m.get("ticker", "") for m in markets if m.get("ticker")]
    result["excluded_no_two_sided_liquidity"] = [t for t in all_tickers if t not in two_sided_tickers]
    result["partial_cover_yes"] = k_total > k_yes
    result["partial_cover_no"] = k_total > k_no
    result["partial_cover"] = result["partial_cover_yes"] or result["partial_cover_no"]

    # Partial-cover diagnostics: payout math below uses guaranteed lower bounds, but flag high-prob excluded markets so the coverage gap is visible.
    yes_arb_high_prob_excluded = False
    no_arb_high_prob_excluded = False
    if result["partial_cover_yes"] and result["excluded_no_yes_liquidity"]:
        for t in result["excluded_no_yes_liquidity"]:
            if not t:
                continue
            ob = orderbooks_by_ticker.get(t, {})
            lad = get_yes_ladder_taker(ob, t)
            if lad:
                implied_prob = lad[0][0]  # best YES ask = implied prob for this outcome
                if implied_prob >= MIN_PROB_TO_REQUIRE_TRADEABLE:
                    yes_arb_high_prob_excluded = True
                    break
            else:
                yes_arb_high_prob_excluded = True  # no quote: assume high prob to be safe
                break
    if result["partial_cover_no"] and result["excluded_no_no_liquidity"]:
        for t in result["excluded_no_no_liquidity"]:
            if not t:
                continue
            ob = orderbooks_by_ticker.get(t, {})
            lad = get_no_ladder_taker(ob, t)
            if lad:
                no_ask = lad[0][0]  # best NO ask; outcome prob = 1 - no_ask
                outcome_implied_prob = 1.0 - no_ask
                if outcome_implied_prob >= MIN_PROB_TO_REQUIRE_TRADEABLE:
                    no_arb_high_prob_excluded = True
                    break
            else:
                no_arb_high_prob_excluded = True
                break
    result["yes_arb_high_prob_excluded"] = yes_arb_high_prob_excluded
    result["no_arb_high_prob_excluded"] = no_arb_high_prob_excluded

    if k_yes == 0 and k_no == 0:
        result["summary"]["error"] = "No tradeable markets (no YES liquidity and no NO liquidity on any market)."
        # Diagnostic: show what the API returned for the first market (helps distinguish empty API vs parsing)
        first_ticker = all_tickers[0] if all_tickers else None
        if first_ticker:
            ob = orderbooks_by_ticker.get(first_ticker, {})
            inner = ob.get("orderbook", ob)
            n_yes = len(inner.get("yes_dollars") or inner.get("yes") or [])
            n_no = len(inner.get("no_dollars") or inner.get("no") or [])
            result["orderbook_diagnostic"] = {
                "ticker": first_ticker,
                "yes_levels": n_yes,
                "no_levels": n_no,
            }
        return result

    # --- Single binary market: no partition logic; report YES/NO executable + optional buy-both ---
    if k_total == 1 and not mutually_exclusive:
        return _run_single_binary_report(
            event_ticker=event_ticker,
            event=event,
            markets=markets,
            orderbooks_by_ticker=orderbooks_by_ticker,
            fee_mode=fee_mode,
            contracts_per_set=contracts_per_set,
            result_template=result,
            fee_multiplier=fee_multiplier,
        )

    # --- YES full set (partition: mutually_exclusive and/or k_yes > 1) ---
    tickers_yes = [m["ticker"] for m in markets_yes if m.get("ticker")]
    yes_per_market: list[dict] = []
    yes_premium_total = 0.0
    yes_fee_worst_total = 0.0
    yes_fee_best_total = 0.0
    yes_ladders_for_budget: list[list[tuple[float, int, str]]] = []
    yes_infeasible = False
    yes_missing: list[str] = []

    for m in markets_yes:
        t = m.get("ticker", "")
        ob = orderbooks_by_ticker.get(t, {})
        ladder = get_yes_ladder_taker(ob, t)
        yes_ladders_for_budget.append(ladder)
        fills, prem, fee_w, fee_b, filled, unfilled = fill_from_ladder_with_fees(
            ladder, contracts_per_set, fee_mode, fee_multiplier
        )
        if filled < contracts_per_set:
            yes_infeasible = True
            yes_missing.append(t)
        yes_premium_total += prem
        yes_fee_worst_total += fee_w
        yes_fee_best_total += fee_b
        yes_per_market.append({
            "ticker": t,
            "requested": contracts_per_set,
            "filled": filled,
            "unfilled": unfilled,
            "fills": [{"price": p, "size": s, "source": src} for p, s, src in fills],
            "premium": prem,
            "fee_worst": fee_w,
            "fee_best": fee_b,
        })

    payout_per_set_yes = guaranteed_yes_payout(n, k_total, k_yes)
    cost_worst_yes = yes_premium_total + yes_fee_worst_total
    cost_best_yes = yes_premium_total + yes_fee_best_total
    profit_worst_yes = payout_per_set_yes - cost_worst_yes
    profit_best_yes = payout_per_set_yes - cost_best_yes

    result["yes_arb"] = {
        "per_market": yes_per_market,
        "total_premium": yes_premium_total,
        "fee_worst": yes_fee_worst_total,
        "fee_best": yes_fee_best_total,
        "cost_worst": cost_worst_yes,
        "cost_best": cost_best_yes,
        "payout_per_set": payout_per_set_yes,
        "profit_worst": profit_worst_yes,
        "profit_best": profit_best_yes,
        "max_sets_with_budget": None,
        "depth_consumption": None,
    }
    result["yes_arb"]["feasible"] = not yes_infeasible
    result["yes_arb"]["infeasible_tickers"] = yes_missing
    if yes_infeasible:
        result["yes_arb"]["profit_worst"] = None
        result["yes_arb"]["profit_best"] = None
        result["yes_arb"]["cost_worst"] = None
        result["yes_arb"]["cost_best"] = None
        result["yes_arb"]["payout_per_set"] = payout_per_set_yes
        profit_worst_yes = None
        profit_best_yes = None

    # Budget sizing for YES: always compute profit-maximizing amount and payout (use large cap if no --budget)
    _budget_cap = budget_dollars if (budget_dollars is not None and budget_dollars > 0) else 1e9
    m_opt, total_premium, total_fees_worst, total_cost, depth_report, bottleneck = optimal_sets_under_budget(
        yes_ladders_for_budget,
        tickers_yes,
        _budget_cap,
        payout_per_set_yes,
        fee_mode,
        contracts_per_set,
        fee_multiplier,
    )
    result["yes_arb"]["max_sets_with_budget"] = m_opt
    result["yes_arb"]["budget_total_premium"] = total_premium
    result["yes_arb"]["budget_total_fees_worst"] = total_fees_worst
    result["yes_arb"]["budget_total_cost"] = total_cost
    result["yes_arb"]["budget_guaranteed_payout"] = m_opt * payout_per_set_yes
    result["yes_arb"]["budget_guaranteed_profit"] = (m_opt * payout_per_set_yes) - total_cost
    result["yes_arb"]["depth_consumption"] = depth_report
    result["yes_arb"]["bottleneck_tickers"] = bottleneck
    result["yes_arb"]["limit_by_budget"] = not bottleneck and m_opt > 0

    # --- NO full set (only markets with NO liquidity) ---
    tickers_no = [m["ticker"] for m in markets_no if m.get("ticker")]
    no_per_market: list[dict] = []
    no_premium_total = 0.0
    no_fee_worst_total = 0.0
    no_fee_best_total = 0.0
    no_ladders_for_budget: list[list[tuple[float, int, str]]] = []
    no_infeasible = False
    no_missing: list[str] = []

    for m in markets_no:
        t = m.get("ticker", "")
        ob = orderbooks_by_ticker.get(t, {})
        ladder = get_no_ladder_taker(ob, t)
        no_ladders_for_budget.append(ladder)
        fills, prem, fee_w, fee_b, filled, unfilled = fill_from_ladder_with_fees(
            ladder, contracts_per_set, fee_mode, fee_multiplier
        )
        if filled < contracts_per_set:
            no_infeasible = True
            no_missing.append(t)
        no_premium_total += prem
        no_fee_worst_total += fee_w
        no_fee_best_total += fee_b
        no_per_market.append({
            "ticker": t,
            "requested": contracts_per_set,
            "filled": filled,
            "unfilled": unfilled,
            "fills": [{"price": p, "size": s, "source": src} for p, s, src in fills],
            "premium": prem,
            "fee_worst": fee_w,
            "fee_best": fee_b,
        })

    payout_per_set_no = guaranteed_no_payout(n, k_no)
    cost_worst_no = no_premium_total + no_fee_worst_total
    cost_best_no = no_premium_total + no_fee_best_total
    profit_worst_no = payout_per_set_no - cost_worst_no
    profit_best_no = payout_per_set_no - cost_best_no

    result["no_arb"] = {
        "per_market": no_per_market,
        "total_premium": no_premium_total,
        "fee_worst": no_fee_worst_total,
        "fee_best": no_fee_best_total,
        "cost_worst": cost_worst_no,
        "cost_best": cost_best_no,
        "payout_per_set": payout_per_set_no,
        "profit_worst": profit_worst_no,
        "profit_best": profit_best_no,
        "max_sets_with_budget": None,
        "depth_consumption": None,
    }
    result["no_arb"]["feasible"] = not no_infeasible
    result["no_arb"]["infeasible_tickers"] = no_missing
    if no_infeasible:
        result["no_arb"]["profit_worst"] = None
        result["no_arb"]["profit_best"] = None
        result["no_arb"]["cost_worst"] = None
        result["no_arb"]["cost_best"] = None
        result["no_arb"]["payout_per_set"] = payout_per_set_no
        profit_worst_no = None
        profit_best_no = None

    m_opt_no, total_premium_n, total_fees_worst_n, total_cost_n, depth_report_n, bottleneck_no = optimal_sets_under_budget(
        no_ladders_for_budget,
        tickers_no,
        _budget_cap,
        payout_per_set_no,
        fee_mode,
        contracts_per_set,
        fee_multiplier,
    )
    result["no_arb"]["max_sets_with_budget"] = m_opt_no
    result["no_arb"]["budget_total_premium"] = total_premium_n
    result["no_arb"]["budget_total_fees_worst"] = total_fees_worst_n
    result["no_arb"]["budget_total_cost"] = total_cost_n
    result["no_arb"]["budget_guaranteed_payout"] = m_opt_no * payout_per_set_no
    result["no_arb"]["budget_guaranteed_profit"] = (m_opt_no * payout_per_set_no) - total_cost_n
    result["no_arb"]["depth_consumption"] = depth_report_n
    result["no_arb"]["bottleneck_tickers"] = bottleneck_no
    result["no_arb"]["limit_by_budget"] = not bottleneck_no and m_opt_no > 0

    # --- Summary ---
    yes_arb_exists = profit_worst_yes is not None and profit_worst_yes > 0
    no_arb_exists = profit_worst_no is not None and profit_worst_no > 0
    if profit_worst_yes is not None and profit_worst_no is not None:
        best_side = "YES" if profit_worst_yes >= profit_worst_no else "NO"
    elif profit_worst_yes is not None:
        best_side = "YES"
    elif profit_worst_no is not None:
        best_side = "NO"
    else:
        best_side = "N/A"
    profit_yes = profit_worst_yes
    profit_no = profit_worst_no
    cost_yes = cost_worst_yes
    cost_no = cost_worst_no
    profit_per_dollar_yes = (profit_yes / cost_yes) if (cost_yes and cost_yes > 0 and profit_yes is not None) else (None if yes_infeasible else 0.0)
    profit_per_dollar_no = (profit_no / cost_no) if (cost_no and cost_no > 0 and profit_no is not None) else (None if no_infeasible else 0.0)
    risks = []
    if not yes_per_market and not no_per_market:
        risks.append("No orderbook data")
    else:
        k_tot = result.get("k_total", 0)
        if result.get("partial_cover_yes"):
            risks.append(f"Partial cover (YES): only {k_yes} of {k_tot} markets have YES liquidity; payout is worst-case lower bound ${payout_per_set_yes:.2f}.")
        if result.get("partial_cover_no"):
            risks.append(f"Partial cover (NO): only {k_no} of {k_tot} markets have NO liquidity; payout is worst-case lower bound ${payout_per_set_no:.2f}.")
        if result.get("yes_arb_high_prob_excluded"):
            risks.append("YES partial-cover note: an excluded market has implied probability >= 1%; guaranteed payout already uses the lower-bound payout.")
        if result.get("no_arb_high_prob_excluded"):
            risks.append("NO partial-cover note: an excluded market has implied probability >= 1%; guaranteed payout already uses the lower-bound payout.")
        if yes_missing:
            risks.append(f"YES full set infeasible (insufficient YES liquidity): {yes_missing}")
        if no_missing:
            risks.append(f"NO full set infeasible (insufficient NO liquidity): {no_missing}")
        for per_m in yes_per_market:
            if not per_m.get("fills"):
                risks.append(f"Thin/empty book: {per_m.get('ticker')}")
        if budget_dollars and result["yes_arb"].get("max_sets_with_budget") == 0:
            risks.append("Budget too small for one full set (YES)")
        risks.append("Fragmentation/rounding: fees computed per-level (worst) vs single VWAP (best)")
        risks.append("Partial fills if size moves before execution")

    result["summary"] = {
        "arb_exists_yes": yes_arb_exists,
        "arb_exists_no": no_arb_exists,
        "best_side": best_side,
        "profit_per_set_yes_worst": profit_yes,
        "profit_per_set_no_worst": profit_no,
        "profit_per_dollar_yes": profit_per_dollar_yes,
        "profit_per_dollar_no": profit_per_dollar_no,
        "execution_risks": risks,
    }

    return result

def _ladder_total_size(ladder: list[tuple[float, int, str]]) -> int:
    return sum(s for _, s, _ in ladder)


def optimal_sets_under_budget(
    ladders_per_market: list[list[tuple[float, int, str]]],
    tickers: list[str],
    budget_dollars: float,
    payout_per_set: float,
    fee_mode: Literal["taker", "maker"],
    contracts_per_set: int,
    fee_multiplier: float = 1.0,
) -> tuple[int, float, float, float, list[dict], list[str]]:
    """
    Number of complete sets that maximizes profit within budget (worst-case fees).
    Uses up to budget_dollars only until profit is maximized; may use less than full budget.
    Returns (m_opt, total_premium, total_fees_worst, total_cost, depth_report, bottleneck_tickers).
    """
    # Get max sets that fit in budget (same cap as before)
    m_max, _, _, _, _, _ = max_sets_under_budget(
        ladders_per_market,
        tickers,
        budget_dollars,
        fee_mode,
        contracts_per_set,
        fee_multiplier,
    )
    if m_max <= 0:
        return 0, 0.0, 0.0, 0.0, [], []

    depth_report: list[dict] = []
    max_contracts_per_market = [_ladder_total_size(lad) for lad in ladders_per_market]
    max_sets_by_depth_list = [c // contracts_per_set for c in max_contracts_per_market]
    depth_limit_sets = min(max_sets_by_depth_list, default=0)
    bottleneck_by_depth = [
        tickers[i]
        for i, sets_supp in enumerate(max_sets_by_depth_list)
        if sets_supp == depth_limit_sets and depth_limit_sets >= 0
    ] if tickers else []

    def cost_for_m_sets(m: int) -> tuple[float, float, list[dict], bool]:
        if m <= 0:
            return 0.0, 0.0, [], True
        total_prem = 0.0
        total_fee = 0.0
        per_market_fills: list[dict] = []
        contracts_needed = m * contracts_per_set
        feasible = True
        for i, (ladder, ticker) in enumerate(zip(ladders_per_market, tickers)):
            fills, prem, fee_w, _, filled, unfilled = fill_from_ladder_with_fees(
                ladder, contracts_needed, fee_mode, fee_multiplier
            )
            if filled < contracts_needed:
                feasible = False
            total_prem += prem
            total_fee += fee_w
            nlp, nls, nlsrc = _next_level_after_fill(ladder, contracts_needed)
            per_market_fills.append({
                "ticker": ticker,
                "requested": contracts_needed,
                "filled": filled,
                "unfilled": unfilled,
                "contracts_filled": filled,
                "max_contracts_available": max_contracts_per_market[i],
                "levels": [{"price": p, "size": s, "source": src} for p, s, src in fills],
                "premium": prem,
                "fee_worst": fee_w,
                "next_level_price": nlp,
                "next_level_size": nls,
                "next_level_source": nlsrc,
            })
        return total_prem, total_fee, per_market_fills, feasible

    def _next_level_after_fill(ladder: list[tuple[float, int, str]], contracts_needed: int) -> tuple[float | None, int, str]:
        remaining = contracts_needed
        for idx, (price, size, source) in enumerate(ladder):
            take = min(remaining, size)
            remaining -= take
            if remaining == 0:
                leftover = size - take
                if leftover > 0:
                    return price, leftover, source
                if idx + 1 < len(ladder):
                    p2, s2, src2 = ladder[idx + 1]
                    return p2, s2, src2
                return price, 0, source
        if ladder:
            return ladder[-1][0], 0, ladder[-1][2]
        return None, 0, ""

    best_m = 0
    best_profit = 0.0
    best_premium = 0.0
    best_fees = 0.0
    best_report: list[dict] = []
    best_bottleneck: list[str] = []

    for m in range(1, m_max + 1):
        prem, fee, report, feasible = cost_for_m_sets(m)
        if not feasible:
            break
        if prem + fee > budget_dollars:
            break
        profit = (payout_per_set * m) - (prem + fee)
        if profit >= best_profit:
            best_profit = profit
            best_m = m
            best_premium = prem
            best_fees = fee
            best_report = report
            if m >= depth_limit_sets and depth_limit_sets >= 0 and bottleneck_by_depth:
                best_bottleneck = bottleneck_by_depth
            else:
                best_bottleneck = []

    total_cost = best_premium + best_fees
    for dm in best_report:
        dm["is_bottleneck"] = dm["ticker"] in best_bottleneck
    return best_m, best_premium, best_fees, total_cost, best_report, best_bottleneck


def max_sets_under_budget(
    ladders_per_market: list[list[tuple[float, int, str]]],
    tickers: list[str],
    budget_dollars: float,
    fee_mode: Literal["taker", "maker"],
    contracts_per_set: int,
    fee_multiplier: float = 1.0,
) -> tuple[int, float, float, float, list[dict], list[str]]:
    """
    Maximum number of complete sets we can buy within budget (worst-case fees).
    Returns (m, total_premium, total_fees_worst, total_cost, depth_report, bottleneck_tickers).
    """
    depth_report: list[dict] = []
    # Per-market max contracts available (depth limit)
    max_contracts_per_market = [_ladder_total_size(lad) for lad in ladders_per_market]
    max_sets_by_depth_list = [c // contracts_per_set for c in max_contracts_per_market]
    depth_limit_sets = min(max_sets_by_depth_list, default=0)
    bottleneck_by_depth = [
        tickers[i]
        for i, sets_supp in enumerate(max_sets_by_depth_list)
        if sets_supp == depth_limit_sets and depth_limit_sets >= 0
    ] if tickers else []

    def _next_level_after_fill(ladder: list[tuple[float, int, str]], contracts_needed: int) -> tuple[float | None, int, str]:
        """After filling contracts_needed from ladder, return (next_price, next_size, source) for next level."""
        remaining = contracts_needed
        for idx, (price, size, source) in enumerate(ladder):
            take = min(remaining, size)
            remaining -= take
            if remaining == 0:
                leftover = size - take
                if leftover > 0:
                    return price, leftover, source
                if idx + 1 < len(ladder):
                    p2, s2, src2 = ladder[idx + 1]
                    return p2, s2, src2
                return price, 0, source
        if ladder:
            return ladder[-1][0], 0, ladder[-1][2]
        return None, 0, ""

    def cost_for_m_sets(m: int) -> tuple[float, float, list[dict], bool]:
        if m <= 0:
            return 0.0, 0.0, [], True
        total_prem = 0.0
        total_fee = 0.0
        per_market_fills: list[dict] = []
        contracts_needed = m * contracts_per_set
        feasible = True
        for i, (ladder, ticker) in enumerate(zip(ladders_per_market, tickers)):
            fills, prem, fee_w, _, filled, unfilled = fill_from_ladder_with_fees(
                ladder, contracts_needed, fee_mode, fee_multiplier
            )
            if filled < contracts_needed:
                feasible = False
            total_prem += prem
            total_fee += fee_w
            nlp, nls, nlsrc = _next_level_after_fill(ladder, contracts_needed)
            per_market_fills.append({
                "ticker": ticker,
                "requested": contracts_needed,
                "filled": filled,
                "unfilled": unfilled,
                "contracts_filled": filled,
                "max_contracts_available": max_contracts_per_market[i],
                "levels": [{"price": p, "size": s, "source": src} for p, s, src in fills],
                "premium": prem,
                "fee_worst": fee_w,
                "next_level_price": nlp,
                "next_level_size": nls,
                "next_level_source": nlsrc,
            })
        return total_prem, total_fee, per_market_fills, feasible

    m = 0
    step = 1
    max_iterations = 500  # safety cap (binary search should need O(log depth) iterations)
    for _ in range(max_iterations):
        prem, fee, report, feasible = cost_for_m_sets(m + step)
        if feasible and (prem + fee <= budget_dollars):
            m += step
            step = max(1, step * 2)
            depth_report = report
        else:
            if step == 1:
                break
            step = max(1, step // 2)
    else:
        # Hit iteration cap; use current m
        pass

    total_premium, total_fees, depth_report, feasible = cost_for_m_sets(m)
    if not feasible:
        infeasible_tickers = [dm["ticker"] for dm in depth_report if dm.get("filled", 0) < dm.get("requested", 0)]
        bottleneck_tickers = ["INFEASIBLE_MARKET_ORDER_SET"] + infeasible_tickers
        m = 0
        total_premium, total_fees, depth_report, _ = cost_for_m_sets(0)
    else:
        # Bottleneck: markets that limit m by depth (smallest max_contracts). Budget can also limit.
        prem_plus1, fee_plus1, _, _ = cost_for_m_sets(m + 1)
        budget_limited = prem_plus1 + fee_plus1 > budget_dollars
        # Depth bottleneck = markets that support exactly depth_limit_sets (they run out first)
        if m >= depth_limit_sets and depth_limit_sets >= 0 and bottleneck_by_depth:
            bottleneck_tickers = bottleneck_by_depth
        else:
            bottleneck_tickers = []  # limit is budget, not a single market's depth
    total_cost = total_premium + total_fees
    for dm in depth_report:
        dm["is_bottleneck"] = dm["ticker"] in bottleneck_tickers
    return m, total_premium, total_fees, total_cost, depth_report, bottleneck_tickers

# --- Output formatting ---


def _format_single_binary_report(data: dict[str, Any], lines: list[str]) -> str:
    """Format report for single binary market (k=1)."""
    lines.append("")
    lines.append("(Single binary market — executable prices and fees only; no full-set partition logic)")
    ys = data.get("yes_side") or {}
    ns = data.get("no_side") or {}
    both = data.get("buy_both_sides") or {}

    lines.append("")
    lines.append("--- YES (buy YES) ---")
    lines.append(f"  Market: {ys.get('ticker', '?')}")
    if ys.get("executable_price") is not None:
        lines.append(f"  Executable price: ${ys['executable_price']:.4f}")
    lines.append(f"  Premium: ${ys.get('premium', 0):.4f}  |  Fee worst: ${ys.get('fee_worst', 0):.4f}  |  Fee best: ${ys.get('fee_best', 0):.4f}")
    if ys.get("cost_worst") is not None:
        lines.append(f"  Cost: worst ${ys['cost_worst']:.4f}  best ${ys['cost_best']:.4f}")

    lines.append("")
    lines.append("--- NO (buy NO) ---")
    lines.append(f"  Market: {ns.get('ticker', '?')}")
    if ns.get("executable_price") is not None:
        lines.append(f"  Executable price: ${ns['executable_price']:.4f}")
    lines.append(f"  Premium: ${ns.get('premium', 0):.4f}  |  Fee worst: ${ns.get('fee_worst', 0):.4f}  |  Fee best: ${ns.get('fee_best', 0):.4f}")
    if ns.get("cost_worst") is not None:
        lines.append(f"  Cost: worst ${ns['cost_worst']:.4f}  best ${ns['cost_best']:.4f}")

    lines.append("")
    lines.append("--- Buy both sides (cost vs $1 payout) ---")
    if both.get("cost_worst") is not None:
        lines.append(f"  Cost: worst ${both['cost_worst']:.4f}  best ${both['cost_best']:.4f}")
        lines.append(f"  Payout: ${both.get('payout', 1):.2f}")
        if both.get("profit_worst") is not None:
            lines.append(f"  Profit: worst ${both['profit_worst']:.4f}  best ${both['profit_best']:.4f}")
    else:
        lines.append("  (One or both sides could not be filled)")

    lines.append("")
    lines.append("  Execution risks:")
    for r in data.get("summary", {}).get("execution_risks", []):
        lines.append(f"    - {r}")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_report(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("FULL-SET ARBITRAGE REPORT")
    lines.append("(Full order book depth; fees per level, worst case)")
    lines.append("=" * 60)
    k_yes = data.get("k_yes", data.get("k", 0))
    k_no = data.get("k_no", data.get("k", 0))
    k_total = data.get("k_total", max(k_yes, k_no))
    k_note = f"k_yes = {k_yes}, k_no = {k_no}" + (f" (of {k_total} total in event)" if k_total > max(k_yes, k_no) else "")
    lines.append(f"Event: {data['event_ticker']}  |  {k_note}  |  n = {data['n']}  |  mutually_exclusive = {data['mutually_exclusive']}")
    lines.append(f"Fee mode: {data['fee_mode']}  |  Contracts per set: {data['contracts_per_set']}")
    if data.get("series_ticker"):
        ft = data.get("series_fee_type", "?")
        fm = data.get("series_fee_multiplier", 1.0)
        lines.append(f"Series fee (irregular): {data['series_ticker']}  |  fee_type = {ft}  |  fee_multiplier = {fm}")
    if data.get("budget_dollars"):
        lines.append(f"Budget: ${data['budget_dollars']:.2f}")
    # Explicit list: tradeable vs excluded per side (YES / NO)
    yes_arb = data.get("yes_arb") or {}
    no_arb = data.get("no_arb") or {}
    included_yes = yes_arb.get("per_market") or []
    included_no = no_arb.get("per_market") or []
    excluded_yes = data.get("excluded_no_yes_liquidity") or []
    excluded_no = data.get("excluded_no_no_liquidity") or []
    if included_yes:
        lines.append("")
        lines.append("Tradeable for YES: " + ", ".join(pm["ticker"] for pm in included_yes))
    if excluded_yes:
        lines.append("Excluded (no YES liquidity): " + ", ".join(excluded_yes))
    if included_no:
        lines.append("Tradeable for NO: " + ", ".join(pm["ticker"] for pm in included_no))
    if excluded_no:
        lines.append("Excluded (no NO liquidity): " + ", ".join(excluded_no))
    if data.get("missing_orderbooks"):
        lines.append("")
        lines.append("MISSING DATA: orderbook ladders for tickers: " + ", ".join(data["missing_orderbooks"]))
        return "\n".join(lines)
    if k_yes == 0 and k_no == 0:
        lines.append("")
        lines.append("No tradeable markets: no YES liquidity and no NO liquidity on any market. Provide event.markets or check API.")
        diag = data.get("orderbook_diagnostic")
        if diag:
            yl, nl = diag.get("yes_levels", 0), diag.get("no_levels", 0)
            lines.append(f"Orderbook sample ({diag.get('ticker', '')}): API returned yes_levels={yl}, no_levels={nl}.")
            if yl == 0 and nl == 0:
                lines.append("(Liquidity was likely removed or the market closed since the scanner last ran—arb opportunities are time-sensitive.)")
        return "\n".join(lines)

    # Single binary market: executable YES/NO + buy both sides (no partition logic)
    if data.get("single_binary"):
        return _format_single_binary_report(data, lines)

    # YES table (partition full-set)
    lines.append("")
    lines.append("--- YES full set (buy YES in every market) ---")
    ya = data["yes_arb"]
    feasible_yes = ya.get("feasible", True)
    if not feasible_yes:
        lines.append(f"  INFEASIBLE (cannot fill full set): missing liquidity on {ya.get('infeasible_tickers', [])}")
    lines.append("Per-market executable prices (price, size, source):")
    for pm in ya["per_market"]:
        lines.append(f"  {pm['ticker']}:")
        for f in pm["fills"]:
            lines.append(f"    ${f['price']:.4f}  x {f['size']}  ({f['source']})")
        req, filled = pm.get("requested", "?"), pm.get("filled", "?")
        lines.append(f"    -> premium ${pm['premium']:.4f}  fee_worst ${pm['fee_worst']:.4f}  fee_best ${pm['fee_best']:.4f}  (filled {filled}/{req})")
    lines.append("")
    lines.append(f"  TOTAL premium: ${ya['total_premium']:.4f}")
    lines.append(f"  Fees: worst ${ya['fee_worst']:.4f}  best ${ya['fee_best']:.4f}")
    cw, cb = ya.get("cost_worst"), ya.get("cost_best")
    if cw is not None and cb is not None:
        lines.append(f"  Cost: worst ${cw:.4f}  best ${cb:.4f}")
    else:
        lines.append("  Cost: N/A (infeasible; partial-fill cost not meaningful)")
    payout_note = " (worst-case lower bound with partial coverage)" if data.get("partial_cover_yes") else ""
    lines.append(f"  Payout per set: ${ya['payout_per_set']:.2f}{payout_note}")
    pw, pb = ya.get("profit_worst"), ya.get("profit_best")
    if pw is not None and pb is not None:
        lines.append(f"  Profit per set: worst ${pw:.4f}  best ${pb:.4f}")
    else:
        lines.append("  Profit per set: N/A (infeasible as market-order full set)")

    if ya.get("max_sets_with_budget") is not None:
        lines.append("")
        cap_note = "up to budget cap" if data.get("budget_dollars") and data["budget_dollars"] > 0 else "no cap"
        lines.append(f"  Budget sizing (profit-maximizing, {cap_note}):")
        lines.append(f"    Max bet amount: ${ya.get('budget_total_cost', 0):.2f}  |  sets: {ya['max_sets_with_budget']}  |  payout: ${ya.get('budget_guaranteed_payout', 0):.2f}  |  profit: ${ya.get('budget_guaranteed_profit', 0):.2f}")
        bt = ya.get("bottleneck_tickers") or []
        if bt and bt[0] == "INFEASIBLE_MARKET_ORDER_SET":
            lines.append(f"    Reason: infeasible as market-order full set (missing liquidity: {', '.join(bt[1:])})")
        else:
            lines.append(f"    Total premium: ${ya.get('budget_total_premium', 0):.4f}")
            lines.append(f"    Total fees (worst): ${ya.get('budget_total_fees_worst', 0):.4f}")
            lines.append(f"    Total cost: ${ya.get('budget_total_cost', 0):.4f}")
            lines.append(f"    Guaranteed payout: ${ya.get('budget_guaranteed_payout', 0):.2f}")
            lines.append(f"    Guaranteed profit: ${ya.get('budget_guaranteed_profit', 0):.4f}")
        if bt and bt[0] == "INFEASIBLE_MARKET_ORDER_SET":
            lines.append(f"  Bottleneck: infeasible (cannot fill full set): {', '.join(bt[1:])}")
        elif bt:
            lines.append(f"  Bottleneck (next limiting markets): {', '.join(bt)}")
        elif ya.get("limit_by_budget"):
            lines.append("  Limit: budget (max sets constrained by budget, not by single-market depth)")
        if ya.get("depth_consumption"):
            lines.append("  Depth consumption (price levels consumed per market):")
            for dm in ya["depth_consumption"]:
                badge = " [BOTTLENECK]" if dm.get("is_bottleneck") else ""
                lines.append(f"    {dm['ticker']}{badge}: filled {dm['contracts_filled']} contracts (max available: {dm.get('max_contracts_available', '?')}), premium ${dm['premium']:.4f}, fee ${dm['fee_worst']:.4f}")
                for lev in dm["levels"][:5]:
                    lines.append(f"      ${lev['price']:.4f} x {lev['size']} ({lev['source']})")
                if len(dm["levels"]) > 5:
                    lines.append(f"      ... and {len(dm['levels']) - 5} more levels")
                np, ns, nsrc = dm.get("next_level_price"), dm.get("next_level_size"), dm.get("next_level_source", "")
                if np is not None and (ns or nsrc):
                    lines.append(f"      Next limiting level: ${np:.4f} x {ns} ({nsrc})")

    # NO table
    lines.append("")
    lines.append("--- NO full set (buy NO in every market) ---")
    na = data["no_arb"]
    feasible_no = na.get("feasible", True)
    if not feasible_no:
        lines.append(f"  INFEASIBLE (cannot fill full set): missing liquidity on {na.get('infeasible_tickers', [])}")
    lines.append("Per-market executable prices (price, size, source):")
    for pm in na["per_market"]:
        lines.append(f"  {pm['ticker']}:")
        for f in pm["fills"]:
            lines.append(f"    ${f['price']:.4f}  x {f['size']}  ({f['source']})")
        req, filled = pm.get("requested", "?"), pm.get("filled", "?")
        lines.append(f"    -> premium ${pm['premium']:.4f}  fee_worst ${pm['fee_worst']:.4f}  fee_best ${pm['fee_best']:.4f}  (filled {filled}/{req})")
    lines.append("")
    lines.append(f"  TOTAL premium: ${na['total_premium']:.4f}")
    lines.append(f"  Fees: worst ${na['fee_worst']:.4f}  best ${na['fee_best']:.4f}")
    cw_n, cb_n = na.get("cost_worst"), na.get("cost_best")
    if cw_n is not None and cb_n is not None:
        lines.append(f"  Cost: worst ${cw_n:.4f}  best ${cb_n:.4f}")
    else:
        lines.append("  Cost: N/A (infeasible; partial-fill cost not meaningful)")
    payout_note_no = " (worst-case lower bound with partial coverage)" if data.get("partial_cover_no") else ""
    lines.append(f"  Payout per set: ${na['payout_per_set']:.2f}{payout_note_no}")
    pw_n, pb_n = na.get("profit_worst"), na.get("profit_best")
    if pw_n is not None and pb_n is not None:
        lines.append(f"  Profit per set: worst ${pw_n:.4f}  best ${pb_n:.4f}")
    else:
        lines.append("  Profit per set: N/A (infeasible as market-order full set)")

    if na.get("max_sets_with_budget") is not None:
        lines.append("")
        cap_note_no = "up to budget cap" if data.get("budget_dollars") and data["budget_dollars"] > 0 else "no cap"
        lines.append(f"  Budget sizing (profit-maximizing, {cap_note_no}):")
        lines.append(f"    Max bet amount: ${na.get('budget_total_cost', 0):.2f}  |  sets: {na['max_sets_with_budget']}  |  payout: ${na.get('budget_guaranteed_payout', 0):.2f}  |  profit: ${na.get('budget_guaranteed_profit', 0):.2f}")
        bt_n = na.get("bottleneck_tickers") or []
        if bt_n and bt_n[0] == "INFEASIBLE_MARKET_ORDER_SET":
            lines.append(f"    Reason: infeasible as market-order full set (missing liquidity: {', '.join(bt_n[1:])})")
        else:
            lines.append(f"    Total cost: ${na.get('budget_total_cost', 0):.4f}")
            lines.append(f"    Guaranteed payout: ${na.get('budget_guaranteed_payout', 0):.2f}")
            lines.append(f"    Guaranteed profit: ${na.get('budget_guaranteed_profit', 0):.4f}")
        if bt_n and bt_n[0] == "INFEASIBLE_MARKET_ORDER_SET":
            lines.append(f"    Bottleneck: infeasible (cannot fill full set): {', '.join(bt_n[1:])}")
        elif bt_n:
            lines.append(f"    Bottleneck: {', '.join(bt_n)}")
        elif na.get("limit_by_budget"):
            lines.append("    Limit: budget")
        if na.get("depth_consumption"):
            for dm in na["depth_consumption"][:3]:
                lines.append(f"    {dm['ticker']}: {dm['contracts_filled']} contracts filled, next level: ${dm.get('next_level_price') or 0:.4f} x {dm.get('next_level_size', 0)}")
            if len(na["depth_consumption"]) > 3:
                lines.append(f"    ... and {len(na['depth_consumption']) - 3} more markets")

    # Summary
    lines.append("")
    lines.append("--- Summary ---")
    s = data["summary"]
    lines.append(f"  Arb exists (YES): {s.get('arb_exists_yes', False)}")
    lines.append(f"  Arb exists (NO):  {s.get('arb_exists_no', False)}")
    lines.append(f"  Best side: {s.get('best_side', 'N/A')}")
    ya, na = data.get("yes_arb") or {}, data.get("no_arb") or {}
    opt_parts = []
    if (ya.get("max_sets_with_budget") or 0) > 0 and ya.get("budget_total_cost") is not None:
        cost_yes = ya["budget_total_cost"]
        payout_yes = ya.get("budget_guaranteed_payout")
        profit_yes = ya.get("budget_guaranteed_profit")
        opt_parts.append(f"YES: max bet ${cost_yes:.2f} → payout ${payout_yes:.2f}, profit ${profit_yes:.2f}" if (payout_yes is not None and profit_yes is not None) else f"YES: max bet ${cost_yes:.2f}")
    if (na.get("max_sets_with_budget") or 0) > 0 and na.get("budget_total_cost") is not None:
        cost_no = na["budget_total_cost"]
        payout_no = na.get("budget_guaranteed_payout")
        profit_no = na.get("budget_guaranteed_profit")
        opt_parts.append(f"NO: max bet ${cost_no:.2f} → payout ${payout_no:.2f}, profit ${profit_no:.2f}" if (payout_no is not None and profit_no is not None) else f"NO: max bet ${cost_no:.2f}")
    if opt_parts:
        lines.append("  Max bet amount and payout (profit-maximizing): " + "  |  ".join(opt_parts))
    elif ya.get("max_sets_with_budget") is not None or na.get("max_sets_with_budget") is not None:
        lines.append("  Max bet amount and payout: 0 sets (no positive-profit size within depth/cap)")
    ppd_yes = s.get("profit_per_dollar_yes")
    ppd_no = s.get("profit_per_dollar_no")
    lines.append(f"  Profit per $1 deployed (YES): {ppd_yes:.4f}" if ppd_yes is not None else "  Profit per $1 deployed (YES): N/A (infeasible)")
    lines.append(f"  Profit per $1 deployed (NO):  {ppd_no:.4f}" if ppd_no is not None else "  Profit per $1 deployed (NO):  N/A (infeasible)")
    lines.append("  Execution risks:")
    for r in s.get("execution_risks", []):
        lines.append(f"    - {r}")
    lines.append("=" * 60)
    return "\n".join(lines)

# --- CLI ---

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full-set arbitrage for Kalshi range events.",
    )
    parser.add_argument("event_ticker", nargs="?", help="Event ticker (e.g. KXTRUTHSOCIAL-26FEB21)")
    parser.add_argument("--event-url", type=str, help="Event URL (alternative to event_ticker)")
    parser.add_argument("--fee-mode", choices=["taker", "maker"], default="taker")
    parser.add_argument("--budget", type=float, default=None, metavar="DOLLARS", help="Budget in USD (optional)")
    parser.add_argument("--contracts-per-set", type=int, default=1)
    parser.add_argument("--fetch", action="store_true", help="Fetch event and orderbooks from Kalshi API (default when no files given)")
    parser.add_argument("--no-fetch", action="store_true", dest="no_fetch", help="Use local files only; do not fetch (use with --orderbooks/--market-data)")
    parser.add_argument("--orderbooks", type=str, help="Path to orderbooks JSON (or use --orderbooks-stdin)")
    parser.add_argument("--market-data", type=str, help="Path to market metadata JSON")
    parser.add_argument("--orderbooks-stdin", action="store_true", help="Read orderbooks JSON from stdin")
    parser.add_argument("--market-data-stdin", action="store_true", help="Read market metadata from stdin")
    parser.add_argument("--json", action="store_true", help="Output raw JSON only")
    parser.add_argument(
        "--n-yes",
        type=int,
        default=None,
        dest="n_yes",
        help="Number of markets that resolve YES. Default 1 if mutually_exclusive. For non-mutually-exclusive with exactly m winners (e.g. top 3), use --n-yes m: YES payout=m, NO payout=k-m.",
    )
    parser.add_argument(
        "--exact-yes",
        type=int,
        default=None,
        dest="n_yes",
        metavar="M",
        help="Same as --n-yes: exactly M markets resolve YES (for 'top M' / multi-winner events).",
    )
    args = parser.parse_args()

    event_ticker = event_ticker_from_input(args.event_url, args.event_ticker)
    if not event_ticker:
        print("Error: provide event_ticker or --event-url", file=sys.stderr)
        return 1

    has_file_input = bool(args.orderbooks or args.orderbooks_stdin or args.market_data or args.market_data_stdin)
    use_fetch = (args.fetch or not has_file_input) and not getattr(args, "no_fetch", False)

    if use_fetch:
        event, markets, orderbooks_by_ticker = fetch_event_and_orderbooks_from_api(event_ticker)
        if not markets:
            print("Error: could not fetch event or event has no markets.", file=sys.stderr)
            return 1
    else:
        orderbooks_json = None
        if args.orderbooks_stdin:
            orderbooks_json = json.load(sys.stdin)
        elif args.orderbooks:
            orderbooks_path_resolved = _resolve_path(args.orderbooks)
            if os.path.isfile(orderbooks_path_resolved):
                with open(orderbooks_path_resolved) as f:
                    orderbooks_json = json.load(f)

        market_metadata_json = None
        if args.market_data_stdin:
            market_metadata_json = json.load(sys.stdin)
        elif args.market_data:
            market_data_resolved = _resolve_path(args.market_data)
            if os.path.isfile(market_data_resolved):
                with open(market_data_resolved) as f:
                    market_metadata_json = json.load(f)

        event, markets, orderbooks_by_ticker = load_event_and_orderbooks(
            event_ticker,
            args.orderbooks,
            args.market_data,
            orderbooks_json,
            market_metadata_json,
        )

    if not markets and market_metadata_json:
        print("Warning: no markets list in metadata; cannot run arb.", file=sys.stderr)
    if not orderbooks_by_ticker and not orderbooks_json:
        print("Warning: no orderbooks provided.", file=sys.stderr)

    result = run_full_set_arb(
        event_ticker=event_ticker,
        event=event,
        markets=markets,
        orderbooks_by_ticker=orderbooks_by_ticker,
        fee_mode=args.fee_mode,
        contracts_per_set=args.contracts_per_set,
        budget_dollars=args.budget,
        n_yes_resolves=args.n_yes,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result))

    return 0 if not result.get("missing_orderbooks") else 1

if __name__ == "__main__":
    sys.exit(main())
