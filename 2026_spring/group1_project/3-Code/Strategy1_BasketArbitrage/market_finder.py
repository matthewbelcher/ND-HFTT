import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

from kalshi_common import load_private_key_pem, sign_pss_base64, now_ms


API_BASE = os.getenv(
    "KALSHI_DATA_API_BASE",
    os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co"),
)
TRADE_API_V2 = "/trade-api/v2"

YES_BID_KEYS = ["yes_bid", "yes_bid_price", "best_yes_bid", "yes_bid_cents"]
YES_ASK_KEYS = ["yes_ask", "yes_ask_price", "best_yes_ask", "yes_ask_cents"]
NO_BID_KEYS = ["no_bid", "no_bid_price", "best_no_bid", "no_bid_cents"]
NO_ASK_KEYS = ["no_ask", "no_ask_price", "best_no_ask", "no_ask_cents"]

# Ordered time field schema. We use MIN timestamp for "close soon" filtering.
# Note: early_close_condition is informational only; it does not carry a timestamp.
CLOSE_TIME_KEYS = [
    "close_time",
    "close_time_ts",
    "close_ts",
    "close_time_ms",
    "closeTime",
    "closeTimeTs",
    "closeTimeMs",
]
EXPIRATION_KEYS = [
    "expected_expiration_time",
    "expectedExpirationTime",
    "latest_expiration_time",
    "latestExpirationTime",
    "event_expiration_ts",
    "expiration_time",
    "expiration_ts",
    "expiration_time_ts",
    "expirationTime",
    "expirationTimeTs",
    "expiry_time",
    "settlement_time",
    "settlement_ts",
    "settlementTime",
    "settlementTimeTs",
    "resolution_time",
    "resolve_time",
    "resolve_ts",
    "resolutionTime",
    "resolutionTimeTs",
    "end_time",
    "end_ts",
    "endTime",
    "endTimeTs",
]
PROJECTED_KEYS = [
    "projected_payout_time",
    "projected_payout_ts",
    "projectedPayoutTime",
    "projectedPayoutTs",
    "projected_settlement_time",
    "projected_settlement_ts",
    "projectedSettlementTime",
    "projectedSettlementTs",
    "projected_expiration_time",
    "projected_expiration_ts",
    "projectedExpirationTime",
    "projectedExpirationTs",
    "projected_close_time",
    "projected_close_ts",
    "projectedCloseTime",
    "projectedCloseTs",
    "projected_resolution_time",
    "projected_resolution_ts",
    "projectedResolutionTime",
    "projectedResolutionTs",
    "payout_time",
    "payout_ts",
    "payoutTime",
    "payoutTs",
]


@dataclass
class MarketQuote:
    ticker: str
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    volume: Optional[float]
    open_interest: Optional[float]


@dataclass
class EventRow:
    event_ticker: str
    series_ticker: Optional[str]
    title: Optional[str]
    n_markets: int
    mutually_exclusive: Optional[bool]
    true_count: Optional[int]
    days_out: Optional[float]
    days_out_source: Optional[str]
    trading_close_ts: Optional[int]
    expiration_ts: Optional[int]
    projected_ts: Optional[int]
    sum_yes_ask: Optional[int]
    sum_yes_bid: Optional[int]
    sum_no_ask: Optional[int]
    sum_no_bid: Optional[int]
    missing_yes_ask: int
    missing_yes_bid: int
    missing_no_ask: int
    missing_no_bid: int
    total_volume: Optional[float]
    total_open_interest: Optional[float]
    yes_ask_edge: Optional[int]
    yes_bid_edge: Optional[int]
    no_ask_edge: Optional[int]
    no_bid_edge: Optional[int]
    best_edge: Optional[int]
    alert: bool


def kalshi_rest_headers(key_id: str, private_key, method: str, path: str) -> Dict[str, str]:
    ts_ms = str(now_ms())
    path_to_sign = path.split("?", 1)[0]
    msg = ts_ms + method.upper() + path_to_sign
    sig = sign_pss_base64(private_key, msg)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }


def api_get(
    path: str,
    key_id: str,
    private_key,
    retries: int = 3,
    backoff_s: float = 0.5,
    allow_fail: bool = False,
) -> Optional[dict]:
    url = API_BASE + path
    headers = kalshi_rest_headers(key_id, private_key, "GET", path)
    req = urllib.request.Request(url, headers=headers, method="GET")
    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            return json.loads(data.decode("utf-8"))
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as e:
            last_err = e
            if attempt == retries:
                if allow_fail:
                    return None
                raise
            time.sleep(backoff_s * (2 ** attempt))
    if allow_fail:
        return None
    raise last_err  # type: ignore


def _price_cents_from_value(v) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v <= 1.0:
            return int(round(v * 100.0))
        return int(round(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f <= 1.0:
        return int(round(f * 100.0))
    return int(round(f))


def _price_cents_from_obj(obj: dict, keys: List[str]) -> Optional[int]:
    for k in keys:
        if k not in obj:
            continue
        v = obj.get(k)
        px = _price_cents_from_value(v)
        if px is not None:
            return px
    return None


def _sanitize_price(px: Optional[int]) -> Optional[int]:
    if px is None:
        return None
    if px < 0 or px > 100:
        return None
    return int(px)


def _infer_complements(
    yes_bid: Optional[int],
    yes_ask: Optional[int],
    no_bid: Optional[int],
    no_ask: Optional[int],
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    if yes_bid is None and no_ask is not None:
        yes_bid = 100 - no_ask
    if no_bid is None and yes_ask is not None:
        no_bid = 100 - yes_ask
    if yes_ask is None and no_bid is not None:
        yes_ask = 100 - no_bid
    if no_ask is None and yes_bid is not None:
        no_ask = 100 - yes_bid
    return (
        _sanitize_price(yes_bid),
        _sanitize_price(yes_ask),
        _sanitize_price(no_bid),
        _sanitize_price(no_ask),
    )


def parse_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_ts_ms(value) -> Optional[int]:
    """Parse a timestamp into milliseconds since epoch."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v <= 0:
            return None
        # Heuristic: seconds vs milliseconds.
        if v < 1e11:
            return int(round(v * 1000.0))
        return int(round(v))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.isdigit():
            return parse_ts_ms(int(s))
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    return None


def _min_ts_from_obj(obj: dict, keys: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """Return (min_ts, key) from obj for the given ordered keys."""
    best_ts: Optional[int] = None
    best_key: Optional[str] = None
    for k in keys:
        if k not in obj:
            continue
        ts = parse_ts_ms(obj.get(k))
        if ts is None:
            continue
        if best_ts is None or ts < best_ts:
            best_ts = ts
            best_key = k
    return best_ts, best_key


def _max_ts_from_obj(obj: dict, keys: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """Return (max_ts, key) from obj for the given ordered keys."""
    best_ts: Optional[int] = None
    best_key: Optional[str] = None
    for k in keys:
        if k not in obj:
            continue
        ts = parse_ts_ms(obj.get(k))
        if ts is None:
            continue
        if best_ts is None or ts > best_ts:
            best_ts = ts
            best_key = k
    return best_ts, best_key


def _min_ts_from_objects(objs: Iterable[dict], keys: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """Return (min_ts, key) across a list of objects for the given keys."""
    best_ts: Optional[int] = None
    best_key: Optional[str] = None
    for obj in objs:
        ts, key = _min_ts_from_obj(obj, keys)
        if ts is None:
            continue
        if best_ts is None or ts < best_ts:
            best_ts = ts
            best_key = key
    return best_ts, best_key


def _max_ts_from_objects(objs: Iterable[dict], keys: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """Return (max_ts, key) across a list of objects for the given keys."""
    best_ts: Optional[int] = None
    best_key: Optional[str] = None
    for obj in objs:
        ts, key = _max_ts_from_obj(obj, keys)
        if ts is None:
            continue
        if best_ts is None or ts > best_ts:
            best_ts = ts
            best_key = key
    return best_ts, best_key


def trading_close_time(
    event: dict, markets: Optional[List[dict]] = None
) -> Tuple[Optional[int], Optional[str]]:
    """
    Earliest trading close time. Prefer market.close_time; fallback to event close fields.
    MIN is used so "close soon" filtering reflects the earliest possible close.
    """
    markets_list = markets or []
    ts, key = _min_ts_from_objects(markets_list, CLOSE_TIME_KEYS)
    if ts is not None:
        return ts, f"market:{key}"
    ts, key = _min_ts_from_obj(event, CLOSE_TIME_KEYS)
    if ts is not None:
        return ts, f"event:{key}"
    return None, None


def expiration_time(
    event: dict, markets: Optional[List[dict]] = None
) -> Tuple[Optional[int], Optional[str]]:
    """Earliest expiration/settlement/resolution timestamp."""
    markets_list = markets or []
    ts, key = _min_ts_from_objects(markets_list, EXPIRATION_KEYS)
    if ts is not None:
        return ts, f"market:{key}"
    ts, key = _min_ts_from_obj(event, EXPIRATION_KEYS)
    if ts is not None:
        return ts, f"event:{key}"
    return None, None


def projected_time(
    event: dict, markets: Optional[List[dict]] = None
) -> Tuple[Optional[int], Optional[str]]:
    """Earliest projected payout/settlement time if present."""
    markets_list = markets or []
    ts, key = _min_ts_from_objects(markets_list, PROJECTED_KEYS)
    if ts is not None:
        return ts, f"market:{key}"
    ts, key = _min_ts_from_obj(event, PROJECTED_KEYS)
    if ts is not None:
        return ts, f"event:{key}"
    return None, None
    return None


def market_quote_from_snapshot(market: dict) -> Optional[MarketQuote]:
    ticker = market.get("ticker") or market.get("market_ticker")
    if not ticker:
        return None
    yes_bid = _price_cents_from_obj(market, YES_BID_KEYS)
    yes_ask = _price_cents_from_obj(market, YES_ASK_KEYS)
    no_bid = _price_cents_from_obj(market, NO_BID_KEYS)
    no_ask = _price_cents_from_obj(market, NO_ASK_KEYS)
    yes_bid, yes_ask, no_bid, no_ask = _infer_complements(yes_bid, yes_ask, no_bid, no_ask)

    volume = parse_float(market.get("volume") or market.get("volume_24h"))
    open_interest = parse_float(market.get("open_interest") or market.get("openInterest"))
    return MarketQuote(
        ticker=ticker,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        volume=volume,
        open_interest=open_interest,
    )


def parse_level(levels, price_in_dollars: bool) -> Tuple[Optional[int], Optional[float]]:
    if not levels:
        return None, None
    level = levels[0]
    if isinstance(level, (int, float, str)):
        price = level
        qty = None
    else:
        price = level[0] if len(level) > 0 else None
        qty = level[1] if len(level) > 1 else None
    if price is None:
        return None, None
    if price_in_dollars:
        price_cents = int(round(float(price) * 100))
    else:
        price_cents = int(float(price))
    qty_val = None
    if qty is not None:
        qty_val = float(qty)
    return price_cents, qty_val


def best_from_orderbook(ob_data: dict, side: str) -> Tuple[Optional[int], Optional[float]]:
    ob = ob_data.get("orderbook") or {}
    ob_fp = ob_data.get("orderbook_fp") or {}

    levels = ob.get(side)
    price, qty = parse_level(levels, price_in_dollars=False)
    if price is not None:
        return _sanitize_price(price), qty

    levels = ob_fp.get(f"{side}_dollars")
    price, qty = parse_level(levels, price_in_dollars=True)
    return _sanitize_price(price), qty


def fetch_bbo_from_orderbook(
    ticker: str, key_id: str, private_key
) -> Tuple[Optional[int], Optional[int]]:
    qs = urllib.parse.urlencode({"depth": "1"})
    path = f"{TRADE_API_V2}/markets/{urllib.parse.quote(ticker)}/orderbook?{qs}"
    ob = api_get(path, key_id, private_key, allow_fail=True)
    if not ob:
        return None, None
    yes_bid_px, _ = best_from_orderbook(ob, "yes")
    no_bid_px, _ = best_from_orderbook(ob, "no")
    return yes_bid_px, no_bid_px


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan events for arb-like basket signals")
    parser.add_argument("--series", help="Series ticker filter")
    parser.add_argument("--event", help="Event ticker substring filter")
    parser.add_argument(
        "--status",
        nargs="?",
        const="",
        default="open",
        help="Event status filter (default: open). Use --status to clear.",
    )
    parser.add_argument("--limit", type=int, default=200, help="Events per page")
    parser.add_argument("--max-pages", type=int, default=5, help="Max event pages to scan")
    parser.add_argument("--max-events", type=int, default=None, help="Hard cap on events scanned")
    parser.add_argument("--min-markets", type=int, default=2, help="Minimum markets in event")
    parser.add_argument("--near", type=int, default=5, help="Near-arb threshold in cents")
    parser.add_argument(
        "--true-count",
        type=int,
        default=None,
        help="Exact number of YES outcomes (for non-ME events)",
    )
    parser.add_argument(
        "--min-days-out",
        type=float,
        default=None,
        help="Minimum days until selected time-field (default trading close)",
    )
    parser.add_argument(
        "--max-days-out",
        type=float,
        default=None,
        help="Maximum days until selected time-field (optional)",
    )
    parser.add_argument(
        "--min-hours-to-close",
        type=float,
        default=None,
        help="Minimum hours until selected time-field (optional)",
    )
    parser.add_argument(
        "--max-hours-to-close",
        type=float,
        default=None,
        help="Maximum hours until selected time-field (optional)",
    )
    parser.add_argument(
        "--time-field",
        choices=["trading_close", "expiration", "projected"],
        default="trading_close",
        help="Time field to use for days/hours filtering (default trading_close)",
    )
    parser.add_argument(
        "--allow-unknown-time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include events with unknown resolve/close time (default false)",
    )
    parser.add_argument(
        "--me-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only include mutually exclusive events (default true)",
    )
    parser.add_argument(
        "--allow-unknown-k",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include events without a known true-count (edges blank)",
    )
    parser.add_argument(
        "--min-total-volume",
        type=float,
        default=0.0,
        help="Minimum total volume across markets",
    )
    parser.add_argument(
        "--min-total-oi",
        type=float,
        default=0.0,
        help="Minimum total open interest across markets",
    )
    parser.add_argument(
        "--probe-orderbook",
        type=int,
        default=0,
        help="Fetch orderbooks for up to N markets per event to fill missing BBO (-1 = all)",
    )
    parser.add_argument(
        "--probe-delay",
        type=float,
        default=0.05,
        help="Delay between orderbook probes in seconds",
    )
    parser.add_argument(
        "--event-delay",
        type=float,
        default=0.0,
        help="Delay between event detail requests in seconds",
    )
    parser.add_argument("--top", type=int, default=50, help="Top rows to print")
    parser.add_argument(
        "--sort",
        choices=[
            "edge",
            "yes_ask_edge",
            "yes_bid_edge",
            "no_ask_edge",
            "no_bid_edge",
            "sum_yes_ask",
            "sum_no_ask",
            "markets",
            "volume",
            "oi",
        ],
        default="edge",
        help="Sort key",
    )
    parser.add_argument("--asc", action="store_true", help="Sort ascending (default desc)")
    parser.add_argument("--alert-only", action="store_true", help="Only print near/arb rows")
    parser.add_argument(
        "--min-edge",
        type=int,
        default=None,
        help="Minimum best edge in cents (requires computed edges)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON lines")
    parser.add_argument("--include-title", action="store_true", help="Include event title column")
    parser.add_argument("--debug", action="store_true", help="Print debug counts")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal time parsing tests and exit",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Extra query param key=value to pass to /events",
    )
    return parser.parse_args()


def build_params(args: argparse.Namespace) -> Dict[str, str]:
    params: Dict[str, str] = {"limit": str(args.limit)}
    status = args.status
    if status and isinstance(status, str) and status.lower() in ("all", "any"):
        status = ""
    if status:
        params["status"] = args.status
    if args.series:
        params["series_ticker"] = args.series
    for p in args.param:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k] = v
    return params


def iter_events(
    key_id: str,
    private_key,
    params: Dict[str, str],
    max_pages: int,
    max_events: Optional[int],
) -> Iterable[dict]:
    cursor = None
    pages = 0
    yielded = 0
    while pages < max_pages:
        q = params.copy()
        if cursor:
            q["cursor"] = cursor
        qs = urllib.parse.urlencode(q)
        path = f"{TRADE_API_V2}/events?{qs}"
        resp = api_get(path, key_id, private_key, allow_fail=True)
        if not resp:
            break
        events = resp.get("events") or resp.get("data") or []
        if not isinstance(events, list):
            break
        for e in events:
            yield e
            yielded += 1
            if max_events is not None and yielded >= max_events:
                return
        cursor = resp.get("cursor") or resp.get("next_cursor") or resp.get("next_page_token")
        if not cursor:
            break
        pages += 1


def event_ticker_from_obj(obj: dict) -> Optional[str]:
    for k in ("event_ticker", "ticker", "eventTicker"):
        v = obj.get(k)
        if v:
            return str(v)
    return None


def event_from_resp(resp: dict) -> dict:
    if "event" in resp and isinstance(resp["event"], dict):
        return resp["event"]
    if "data" in resp and isinstance(resp["data"], dict):
        return resp["data"]
    return resp if isinstance(resp, dict) else {}


def markets_from_resp(resp: dict, event: dict) -> List[dict]:
    markets = resp.get("markets")
    if not markets:
        markets = event.get("markets")
    return markets if isinstance(markets, list) else []


def analyze_event(
    event_resp: dict,
    args: argparse.Namespace,
    key_id: str,
    private_key,
) -> Optional[EventRow]:
    event = event_from_resp(event_resp)
    markets = markets_from_resp(event_resp, event)
    if not markets:
        return None

    event_ticker = event_ticker_from_obj(event) or "UNKNOWN"
    series_ticker = event.get("series_ticker") or event.get("seriesTicker")
    title = event.get("title") or event.get("event_title")

    trading_ts, trading_src = trading_close_time(event, markets)
    expiration_ts, expiration_src = expiration_time(event, markets)
    projected_ts, projected_src = projected_time(event, markets)

    time_field = args.time_field
    if time_field == "trading_close":
        selected_ts = trading_ts
        selected_src = trading_src
    elif time_field == "expiration":
        selected_ts = expiration_ts
        selected_src = expiration_src
    else:
        selected_ts = projected_ts
        selected_src = projected_src

    days_out: Optional[float] = None
    days_out_source: Optional[str] = None
    if selected_ts is None:
        if not args.allow_unknown_time:
            if (
                args.min_days_out is not None
                or args.max_days_out is not None
                or args.min_hours_to_close is not None
                or args.max_hours_to_close is not None
            ):
                return None
    else:
        days_out = (selected_ts - now_ms()) / (1000.0 * 3600.0 * 24.0)
        days_out_source = selected_src
        if args.min_days_out is not None and days_out < args.min_days_out:
            return None
        if args.max_days_out is not None and days_out > args.max_days_out:
            return None
        if args.min_hours_to_close is not None and days_out * 24.0 < args.min_hours_to_close:
            return None
        if args.max_hours_to_close is not None and days_out * 24.0 > args.max_hours_to_close:
            return None

    me = event.get("mutually_exclusive")
    if me is None:
        me = event.get("mutuallyExclusive")
    me_flag = True if me is True else False

    true_count = 1 if me_flag else args.true_count
    if args.me_only and not me_flag:
        return None
    if true_count is None and not args.allow_unknown_k:
        return None

    quotes: Dict[str, MarketQuote] = {}
    for m in markets:
        q = market_quote_from_snapshot(m)
        if q:
            quotes[q.ticker] = q

    if len(quotes) < args.min_markets:
        return None

    if args.probe_orderbook != 0:
        missing = [
            q.ticker
            for q in quotes.values()
            if q.yes_bid is None
            or q.yes_ask is None
            or q.no_bid is None
            or q.no_ask is None
        ]
        limit = args.probe_orderbook
        if limit < 0:
            limit = len(missing)
        for ticker in missing[:limit]:
            yes_bid, no_bid = fetch_bbo_from_orderbook(ticker, key_id, private_key)
            q = quotes.get(ticker)
            if not q:
                continue
            if yes_bid is not None:
                q.yes_bid = yes_bid
                if q.no_ask is None:
                    q.no_ask = _sanitize_price(100 - yes_bid)
            if no_bid is not None:
                q.no_bid = no_bid
                if q.yes_ask is None:
                    q.yes_ask = _sanitize_price(100 - no_bid)
            q.yes_bid, q.yes_ask, q.no_bid, q.no_ask = _infer_complements(
                q.yes_bid, q.yes_ask, q.no_bid, q.no_ask
            )
            time.sleep(args.probe_delay)

    sum_yes_ask = 0
    sum_yes_bid = 0
    sum_no_ask = 0
    sum_no_bid = 0
    missing_yes_ask = 0
    missing_yes_bid = 0
    missing_no_ask = 0
    missing_no_bid = 0
    total_volume = 0.0
    total_oi = 0.0
    have_volume = False
    have_oi = False

    for q in quotes.values():
        if q.yes_ask is None:
            missing_yes_ask += 1
        else:
            sum_yes_ask += q.yes_ask
        if q.yes_bid is None:
            missing_yes_bid += 1
        else:
            sum_yes_bid += q.yes_bid
        if q.no_ask is None:
            missing_no_ask += 1
        else:
            sum_no_ask += q.no_ask
        if q.no_bid is None:
            missing_no_bid += 1
        else:
            sum_no_bid += q.no_bid
        if q.volume is not None:
            total_volume += q.volume
            have_volume = True
        if q.open_interest is not None:
            total_oi += q.open_interest
            have_oi = True

    total_volume_out = total_volume if have_volume else None
    total_oi_out = total_oi if have_oi else None
    if total_volume_out is not None and total_volume_out < args.min_total_volume:
        return None
    if total_oi_out is not None and total_oi_out < args.min_total_oi:
        return None

    n_markets = len(quotes)
    if true_count is not None and (true_count < 0 or true_count > n_markets):
        return None
    target_yes = true_count * 100 if true_count is not None else None
    target_no = (n_markets - true_count) * 100 if true_count is not None else None

    def edge_buy(sum_val: int, target: Optional[int], missing: int) -> Optional[int]:
        if target is None or missing > 0:
            return None
        return target - sum_val

    def edge_sell(sum_val: int, target: Optional[int], missing: int) -> Optional[int]:
        if target is None or missing > 0:
            return None
        return sum_val - target

    yes_ask_edge = edge_buy(sum_yes_ask, target_yes, missing_yes_ask)
    yes_bid_edge = edge_sell(sum_yes_bid, target_yes, missing_yes_bid)
    no_ask_edge = edge_buy(sum_no_ask, target_no, missing_no_ask)
    no_bid_edge = edge_sell(sum_no_bid, target_no, missing_no_bid)

    edges = [e for e in [yes_ask_edge, yes_bid_edge, no_ask_edge, no_bid_edge] if e is not None]
    best_edge = max(edges) if edges else None

    alert = False
    if target_yes is not None:
        if missing_yes_ask == 0 and sum_yes_ask <= target_yes + args.near:
            alert = True
        if missing_yes_bid == 0 and sum_yes_bid >= target_yes - args.near:
            alert = True
    if target_no is not None:
        if missing_no_ask == 0 and sum_no_ask <= target_no + args.near:
            alert = True
        if missing_no_bid == 0 and sum_no_bid >= target_no - args.near:
            alert = True

    if args.min_edge is not None:
        if best_edge is None or best_edge < args.min_edge:
            return None

    return EventRow(
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        title=title,
        n_markets=n_markets,
        mutually_exclusive=me_flag if me is not None else None,
        true_count=true_count,
        days_out=days_out,
        days_out_source=days_out_source,
        trading_close_ts=trading_ts,
        expiration_ts=expiration_ts,
        projected_ts=projected_ts,
        sum_yes_ask=sum_yes_ask if missing_yes_ask < n_markets else None,
        sum_yes_bid=sum_yes_bid if missing_yes_bid < n_markets else None,
        sum_no_ask=sum_no_ask if missing_no_ask < n_markets else None,
        sum_no_bid=sum_no_bid if missing_no_bid < n_markets else None,
        missing_yes_ask=missing_yes_ask,
        missing_yes_bid=missing_yes_bid,
        missing_no_ask=missing_no_ask,
        missing_no_bid=missing_no_bid,
        total_volume=total_volume_out,
        total_open_interest=total_oi_out,
        yes_ask_edge=yes_ask_edge,
        yes_bid_edge=yes_bid_edge,
        no_ask_edge=no_ask_edge,
        no_bid_edge=no_bid_edge,
        best_edge=best_edge,
        alert=alert,
    )


def fmt_px(px: Optional[int], width: int) -> str:
    if px is None:
        return "--".rjust(width)
    return f"{px:>{width}d}"


def fmt_edge(edge: Optional[int], width: int) -> str:
    if edge is None:
        return "--".rjust(width)
    return format(edge, f"+{width}d")


def fmt_num(v: Optional[float], nd: int = 0) -> str:
    if v is None:
        return "--"
    if nd == 0:
        return f"{v:.0f}"
    return f"{v:.{nd}f}"


def truncate(s: Optional[str], max_len: int) -> str:
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    return s[: max_len - 3] + "..."


def _self_test() -> None:
    # Market has close_time only (earliest trading close should be that time).
    market_close = {"close_time": "2026-02-18T12:30:00Z"}
    ts, src = trading_close_time({}, [market_close])
    assert ts == parse_ts_ms("2026-02-18T12:30:00Z")
    assert src == "market:close_time"

    # Market has latest_expiration_time only (expiration should come from market).
    market_exp = {"latest_expiration_time": "2026-07-01T00:00:00Z"}
    ts, src = expiration_time({}, [market_exp])
    assert ts == parse_ts_ms("2026-07-01T00:00:00Z")
    assert src == "market:latest_expiration_time"

    # Event has close_time, markets missing it (trading close should use event field).
    event_close = {"close_time": "2026-03-01T00:00:00Z"}
    ts, src = trading_close_time(event_close, [])
    assert ts == parse_ts_ms("2026-03-01T00:00:00Z")
    assert src == "event:close_time"


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        print("Self-test: OK")
        return
    key_id = os.environ["KALSHI_KEY_ID"]
    pem_path = os.environ["KALSHI_PRIVATE_KEY_PEM"]
    private_key = load_private_key_pem(pem_path)

    params = build_params(args)
    rows: List[EventRow] = []
    scanned = 0
    detail_fail = 0
    skipped = 0

    for ev in iter_events(key_id, private_key, params, args.max_pages, args.max_events):
        scanned += 1
        ev_ticker = event_ticker_from_obj(ev)
        if not ev_ticker:
            skipped += 1
            continue
        if args.event and args.event not in ev_ticker:
            skipped += 1
            continue
        path = f"{TRADE_API_V2}/events/{urllib.parse.quote(ev_ticker)}"
        event_resp = api_get(path, key_id, private_key, allow_fail=True)
        if not event_resp:
            detail_fail += 1
            continue
        row = analyze_event(event_resp, args, key_id, private_key)
        if row is None:
            skipped += 1
            continue
        if args.alert_only and not row.alert:
            skipped += 1
            continue
        rows.append(row)
        if args.event_delay:
            time.sleep(args.event_delay)

    def sort_key(r: EventRow):
        if args.sort == "edge":
            return r.best_edge if r.best_edge is not None else -10**9
        if args.sort == "yes_ask_edge":
            return r.yes_ask_edge if r.yes_ask_edge is not None else -10**9
        if args.sort == "yes_bid_edge":
            return r.yes_bid_edge if r.yes_bid_edge is not None else -10**9
        if args.sort == "no_ask_edge":
            return r.no_ask_edge if r.no_ask_edge is not None else -10**9
        if args.sort == "no_bid_edge":
            return r.no_bid_edge if r.no_bid_edge is not None else -10**9
        if args.sort == "sum_yes_ask":
            return r.sum_yes_ask if r.sum_yes_ask is not None else -10**9
        if args.sort == "sum_no_ask":
            return r.sum_no_ask if r.sum_no_ask is not None else -10**9
        if args.sort == "markets":
            return r.n_markets
        if args.sort == "volume":
            return r.total_volume if r.total_volume is not None else -10**9
        if args.sort == "oi":
            return r.total_open_interest if r.total_open_interest is not None else -10**9
        return 0

    rows.sort(key=sort_key, reverse=not args.asc)

    if args.debug:
        print(f"Scanned events: {scanned}")
        print(f"Detail fetch failures: {detail_fail}")
        print(f"Rows kept: {len(rows)}")
        print(f"Skipped/filtered: {skipped}")

    if not rows:
        print("No events matched filters.")
        sys.exit(0)

    if args.json:
        for r in rows[: args.top]:
            print(json.dumps(asdict(r), sort_keys=True))
        return

    event_w = 30
    sum_w = 5
    edge_w = 6
    mk_w = 2
    me_w = 2
    days_w = 7
    src_w = 5
    miss_w = 16
    vol_w = 9
    oi_w = 9
    title_w = 0
    if args.include_title:
        title_w = 40

    header = (
        f"{'Event':<{event_w}} | {'Mk':>{mk_w}} | {'ME':>{me_w}} | "
        f"{'Days':>{days_w}} | {'Src':>{src_w}} | "
        f"{'YA':>{sum_w}} | {'YAe':>{edge_w}} | "
        f"{'YB':>{sum_w}} | {'YBe':>{edge_w}} | "
        f"{'NA':>{sum_w}} | {'NAe':>{edge_w}} | "
        f"{'NB':>{sum_w}} | {'NBe':>{edge_w}} | "
        f"{'Miss(YA/YB/NA/NB)':>{miss_w}} | {'Vol':>{vol_w}} | {'OI':>{oi_w}}"
    )
    if args.include_title:
        header += f" | {'Title':<{title_w}}"
    print(header)
    print("-" * len(header))
    for r in rows[: args.top]:
        if r.mutually_exclusive is None:
            me_flag = "--"
        else:
            me_flag = "Y" if r.mutually_exclusive else "N"
        miss = f"{r.missing_yes_ask}/{r.missing_yes_bid}/{r.missing_no_ask}/{r.missing_no_bid}"
        event_disp = truncate(r.event_ticker, event_w)
        line = (
            f"{event_disp:<{event_w}} | {r.n_markets:>{mk_w}} | {me_flag:>{me_w}} | "
            f"{fmt_num(r.days_out, 0):>{days_w}} | {truncate(r.days_out_source, src_w):>{src_w}} | "
            f"{fmt_px(r.sum_yes_ask, sum_w)} | {fmt_edge(r.yes_ask_edge, edge_w)} | "
            f"{fmt_px(r.sum_yes_bid, sum_w)} | {fmt_edge(r.yes_bid_edge, edge_w)} | "
            f"{fmt_px(r.sum_no_ask, sum_w)} | {fmt_edge(r.no_ask_edge, edge_w)} | "
            f"{fmt_px(r.sum_no_bid, sum_w)} | {fmt_edge(r.no_bid_edge, edge_w)} | "
            f"{miss:>{miss_w}} | {fmt_num(r.total_volume, 0):>{vol_w}} | "
            f"{fmt_num(r.total_open_interest, 0):>{oi_w}}"
        )
        if args.include_title:
            line += f" | {truncate(r.title, title_w):<{title_w}}"
        print(line)


if __name__ == "__main__":
    main()
