import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

API_BASE = os.getenv(
    "KALSHI_DATA_API_BASE",
    os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co"),
)
TRADE_API_V2 = "/trade-api/v2"

YES_BID_KEYS = ["yes_bid", "yes_bid_price", "best_yes_bid", "yes_bid_cents"]
YES_ASK_KEYS = ["yes_ask", "yes_ask_price", "best_yes_ask", "yes_ask_cents"]
NO_BID_KEYS = ["no_bid", "no_bid_price", "best_no_bid", "no_bid_cents"]
NO_ASK_KEYS = ["no_ask", "no_ask_price", "best_no_ask", "no_ask_cents"]

SKIP_TOKENS = ["MEDAL", "TOTAL", "SPREAD"]
WINNER_TOKENS = ["CGOLD", "GOLD", "MVP", "CHAMP"]


@dataclass
class MarketQuote:
    ticker: str
    title: Optional[str]
    status: Optional[str]
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    close_ts: Optional[int]


@dataclass
class EventStats:
    event_ticker: str
    type: str
    mk: int
    sum_asks: Optional[int]
    sum_bids: Optional[int]
    sum_no_asks: Optional[int]
    sum_no_bids: Optional[int]
    buy_edge: Optional[int]
    sell_edge: Optional[int]
    buy_no_edge: Optional[int]
    sell_no_edge: Optional[int]
    best_edge: Optional[int]
    missing_yes_ask: int
    missing_yes_bid: int
    missing_no_ask: int
    missing_no_bid: int
    missing_any: int
    missing_all: int
    min_days_to_close: Optional[float]
    max_days_to_close: Optional[float]


@dataclass
class EventBundle:
    event_ticker: str
    event_type: str
    markets: List[MarketQuote]
    stats: EventStats


def open_text_auto(path: str):
    """Open a text file with BOM sniffing (UTF-16/UTF-8 BOM) and safe fallback."""
    with open(path, "rb") as fb:
        head = fb.read(4)

    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return open(path, "r", encoding="utf-16", errors="strict")
    if head.startswith(b"\xef\xbb\xbf"):
        return open(path, "r", encoding="utf-8-sig", errors="strict")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze a JSONL universe of markets grouped by event_ticker")
    p.add_argument("--infile", required=True, help="JSONL produced by olympics market finder")
    p.add_argument("--min-markets", type=int, default=2, help="Min markets per event to analyze")
    p.add_argument("--top", type=int, default=50, help="Top rows to show")
    p.add_argument("--min-days-to-close", type=float, default=None, help="Minimum days to close")
    p.add_argument("--max-days-to-close", type=float, default=None, help="Maximum days to close")
    p.add_argument(
        "--probe-orderbook",
        type=int,
        default=0,
        help="Fetch orderbooks for top N candidate events and re-rank",
    )
    p.add_argument("--json", action="store_true", help="Output JSON lines instead of text")
    p.add_argument("--debug", action="store_true", help="Print debug stats")
    return p.parse_args()


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_ts_ms(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v <= 0:
            return None
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


def price_cents_from_value(v) -> Optional[int]:
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


def price_from_keys(obj: dict, keys: List[str]) -> Optional[int]:
    for k in keys:
        if k not in obj:
            continue
        px = price_cents_from_value(obj.get(k))
        if px is not None:
            return px
    return None


def sanitize_px(px: Optional[int]) -> Optional[int]:
    if px is None:
        return None
    if px < 0 or px > 100:
        return None
    return int(px)


def infer_complements(
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
    return sanitize_px(yes_bid), sanitize_px(yes_ask), sanitize_px(no_bid), sanitize_px(no_ask)


def normalize_snapshot_quotes(m: dict) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    yb = sanitize_px(price_from_keys(m, YES_BID_KEYS))
    ya = sanitize_px(price_from_keys(m, YES_ASK_KEYS))
    nb = sanitize_px(price_from_keys(m, NO_BID_KEYS))
    na = sanitize_px(price_from_keys(m, NO_ASK_KEYS))

    if yb == 0 and ya == 0:
        yb, ya = None, None
    if nb == 100 and na == 100:
        nb, na = None, None

    yb, ya, nb, na = infer_complements(yb, ya, nb, na)
    return sanitize_px(yb), sanitize_px(ya), sanitize_px(nb), sanitize_px(na)


def event_ticker_from_obj(obj: dict) -> Optional[str]:
    for k in ("event_ticker", "eventTicker", "ticker"):
        v = obj.get(k)
        if v:
            return str(v)
    return None


def market_ticker_from_obj(obj: dict) -> Optional[str]:
    for k in ("ticker", "market_ticker", "marketTicker"):
        v = obj.get(k)
        if v:
            return str(v)
    return None


def classify_event(event_ticker: str, mk: int, title: Optional[str]) -> str:
    t = event_ticker.upper()
    if mk == 2:
        return "H2H_TWO_OUTCOME"
    title_text = title or ""
    if "MOST" in t or "most" in title_text.lower():
        return "ME_MULTI_OUTCOME"
    if "CGOLD" in t and mk > 2:
        return "ME_MULTI_OUTCOME"
    return "SKIP_MULTI_WINNER"


def compute_days_to_close(markets: List[MarketQuote], now_ts: int) -> Tuple[Optional[float], Optional[float]]:
    days_list: List[float] = []
    for m in markets:
        if m.close_ts is None:
            continue
        days_list.append((m.close_ts - now_ts) / (1000.0 * 3600.0 * 24.0))
    if not days_list:
        return None, None
    return min(days_list), max(days_list)


def compute_event_stats(
    event_ticker: str,
    event_type: str,
    markets: List[MarketQuote],
    now_ts: int,
) -> EventStats:
    mk = len(markets)
    sum_asks = 0
    sum_bids = 0
    sum_no_asks = 0
    sum_no_bids = 0
    missing_asks = 0
    missing_bids = 0
    missing_no_asks = 0
    missing_no_bids = 0
    missing_any = 0
    missing_all = 0

    for m in markets:
        if m.yes_ask is None:
            missing_asks += 1
        else:
            sum_asks += m.yes_ask
        if m.yes_bid is None:
            missing_bids += 1
        else:
            sum_bids += m.yes_bid
        if m.no_ask is None:
            missing_no_asks += 1
        else:
            sum_no_asks += m.no_ask
        if m.no_bid is None:
            missing_no_bids += 1
        else:
            sum_no_bids += m.no_bid
        any_missing = (
            m.yes_ask is None
            or m.yes_bid is None
            or m.no_ask is None
            or m.no_bid is None
        )
        all_missing = (
            m.yes_ask is None
            and m.yes_bid is None
            and m.no_ask is None
            and m.no_bid is None
        )
        if any_missing:
            missing_any += 1
        if all_missing:
            missing_all += 1

    sum_asks_out = None if missing_asks == mk else sum_asks
    sum_bids_out = None if missing_bids == mk else sum_bids
    sum_no_asks_out = None if missing_no_asks == mk else sum_no_asks
    sum_no_bids_out = None if missing_no_bids == mk else sum_no_bids

    buy_edge = None
    sell_edge = None
    buy_no_edge = None
    sell_no_edge = None
    if event_type == "ME_MULTI_OUTCOME":
        if missing_asks == 0:
            buy_edge = 100 - sum_asks
        if missing_bids == 0:
            sell_edge = sum_bids - 100
    elif event_type == "H2H_TWO_OUTCOME":
        if missing_asks == 0:
            buy_edge = 100 - sum_asks
        if missing_bids == 0:
            sell_edge = sum_bids - 100
        if missing_no_asks == 0:
            buy_no_edge = 100 - sum_no_asks
        if missing_no_bids == 0:
            sell_no_edge = sum_no_bids - 100

    edges = [e for e in (buy_edge, sell_edge, buy_no_edge, sell_no_edge) if e is not None]
    best_edge = max(edges) if edges else None

    min_days, max_days = compute_days_to_close(markets, now_ts)

    return EventStats(
        event_ticker=event_ticker,
        type=event_type,
        mk=mk,
        sum_asks=sum_asks_out,
        sum_bids=sum_bids_out,
        sum_no_asks=sum_no_asks_out,
        sum_no_bids=sum_no_bids_out,
        buy_edge=buy_edge,
        sell_edge=sell_edge,
        buy_no_edge=buy_no_edge,
        sell_no_edge=sell_no_edge,
        best_edge=best_edge,
        missing_yes_ask=missing_asks,
        missing_yes_bid=missing_bids,
        missing_no_ask=missing_no_asks,
        missing_no_bid=missing_no_bids,
        missing_any=missing_any,
        missing_all=missing_all,
        min_days_to_close=min_days,
        max_days_to_close=max_days,
    )


def kalshi_rest_headers(key_id: str, private_key, sign_fn, now_fn, method: str, path: str) -> Dict[str, str]:
    ts_ms = str(now_fn())
    path_to_sign = path.split("?", 1)[0]
    msg = ts_ms + method.upper() + path_to_sign
    sig = sign_fn(private_key, msg)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }


def api_get(
    path: str,
    key_id: str,
    private_key,
    sign_fn,
    now_fn,
    retries: int = 2,
    backoff_s: float = 0.5,
    allow_fail: bool = False,
) -> Optional[dict]:
    url = API_BASE + path
    headers = kalshi_rest_headers(key_id, private_key, sign_fn, now_fn, "GET", path)
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
        return sanitize_px(price), qty

    levels = ob_fp.get(f"{side}_dollars")
    price, qty = parse_level(levels, price_in_dollars=True)
    return sanitize_px(price), qty


def fetch_orderbook_quotes(
    ticker: str,
    key_id: str,
    private_key,
    sign_fn,
    now_fn,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    qs = urllib.parse.urlencode({"depth": "1"})
    path = f"{TRADE_API_V2}/markets/{urllib.parse.quote(ticker)}/orderbook?{qs}"
    ob = api_get(path, key_id, private_key, sign_fn, now_fn, allow_fail=True)
    if not ob:
        return None, None, None, None
    yes_bid_px, _ = best_from_orderbook(ob, "yes")
    no_bid_px, _ = best_from_orderbook(ob, "no")
    yes_ask_px = 100 - no_bid_px if no_bid_px is not None else None
    no_ask_px = 100 - yes_bid_px if yes_bid_px is not None else None
    return (
        sanitize_px(yes_bid_px),
        sanitize_px(yes_ask_px),
        sanitize_px(no_bid_px),
        sanitize_px(no_ask_px),
    )


def update_bundle_with_orderbooks(
    bundle: EventBundle,
    key_id: str,
    private_key,
    sign_fn,
    now_fn,
) -> None:
    for m in bundle.markets:
        if m.yes_bid is not None and m.yes_ask is not None:
            continue
        yb, ya, nb, na = fetch_orderbook_quotes(m.ticker, key_id, private_key, sign_fn, now_fn)
        if yb is not None:
            m.yes_bid = yb
        if ya is not None:
            m.yes_ask = ya
        if nb is not None:
            m.no_bid = nb
        if na is not None:
            m.no_ask = na
        m.yes_bid, m.yes_ask, m.no_bid, m.no_ask = infer_complements(
            m.yes_bid, m.yes_ask, m.no_bid, m.no_ask
        )


def best_edge_sort(stats: EventStats) -> int:
    if stats.best_edge is None:
        return -10**9
    return stats.best_edge - 2 * stats.missing_any


def fmt_int(v: Optional[int], width: int) -> str:
    if v is None:
        return "--".rjust(width)
    return f"{v:>{width}d}"


def fmt_edge(v: Optional[int], width: int) -> str:
    if v is None:
        return "--".rjust(width)
    return format(v, f"+{width}d")


def fmt_float(v: Optional[float], width: int, nd: int = 1) -> str:
    if v is None:
        return "--".rjust(width)
    return f"{v:>{width}.{nd}f}"


def main() -> None:
    args = parse_args()

    by_event: Dict[str, List[MarketQuote]] = defaultdict(list)
    skipped_nonjson = 0
    skipped_badjson = 0
    missing_ticker = 0
    kept = 0

    with open_text_auto(args.infile) as f:
        for _, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            if not s.startswith("{"):
                skipped_nonjson += 1
                continue
            try:
                m = json.loads(s)
            except json.JSONDecodeError:
                skipped_badjson += 1
                continue

            ev = event_ticker_from_obj(m)
            if not ev:
                continue
            ticker = market_ticker_from_obj(m)
            if not ticker:
                missing_ticker += 1
                continue

            yb, ya, nb, na = normalize_snapshot_quotes(m)
            close_ts = parse_ts_ms(m.get("close_time") or m.get("closeTime"))
            title = m.get("title") or m.get("event_title")
            status = m.get("status") or m.get("event_status")

            by_event[ev].append(
                MarketQuote(
                    ticker=ticker,
                    title=title,
                    status=status,
                    yes_bid=yb,
                    yes_ask=ya,
                    no_bid=nb,
                    no_ask=na,
                    close_ts=close_ts,
                )
            )
            kept += 1

    now_ts = now_ms()
    bundles: List[EventBundle] = []

    for ev, markets in by_event.items():
        if len(markets) < args.min_markets:
            continue
        title = next((m.title for m in markets if m.title), None)
        event_type = classify_event(ev, len(markets), title)
        stats = compute_event_stats(ev, event_type, markets, now_ts)
        if args.min_days_to_close is not None:
            if stats.min_days_to_close is None or stats.min_days_to_close < args.min_days_to_close:
                continue
        if args.max_days_to_close is not None:
            if stats.max_days_to_close is None or stats.max_days_to_close > args.max_days_to_close:
                continue
        bundles.append(EventBundle(event_ticker=ev, event_type=event_type, markets=markets, stats=stats))

    bundles.sort(key=lambda b: best_edge_sort(b.stats), reverse=True)

    if args.probe_orderbook and args.probe_orderbook > 0:
        try:
            from kalshi_common import load_private_key_pem, sign_pss_base64, now_ms as kalshi_now_ms
        except Exception as exc:
            raise SystemExit(
                f"Orderbook probing requires kalshi_common and cryptography: {exc}"
            )
        key_id = os.environ.get("KALSHI_KEY_ID")
        pem_path = os.environ.get("KALSHI_PRIVATE_KEY_PEM")
        if not key_id or not pem_path:
            raise SystemExit("Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PEM to probe orderbooks.")
        private_key = load_private_key_pem(pem_path)

        candidates = [
            b for b in bundles if b.stats.best_edge is not None and b.event_type != "SKIP_MULTI_WINNER"
        ]
        probe_n = min(args.probe_orderbook, len(candidates))
        for b in candidates[:probe_n]:
            update_bundle_with_orderbooks(b, key_id, private_key, sign_pss_base64, kalshi_now_ms)
            b.stats = compute_event_stats(b.event_ticker, b.event_type, b.markets, now_ts)

        bundles.sort(key=lambda b: best_edge_sort(b.stats), reverse=True)

    rows = [b.stats for b in bundles]
    out = rows[: args.top]

    if args.debug:
        print(f"Lines kept as markets: {kept}")
        print(f"Skipped non-JSON lines: {skipped_nonjson}")
        print(f"Skipped invalid JSON lines: {skipped_badjson}")
        print(f"Skipped missing ticker: {missing_ticker}")
        print(f"Events analyzed (after min-markets): {len(rows)}")

    if args.json:
        for r in out:
            print(json.dumps(asdict(r), sort_keys=True))
        return

    event_w = 35
    type_w = 18
    mk_w = 3
    sum_w = 6
    edge_w = 7
    miss_w = 4
    days_w = 9

    header = (
        f"{'event_ticker':<{event_w}} | {'type':<{type_w}} | {'mk':>{mk_w}} | "
        f"{'sum_asks':>{sum_w}} | {'sum_bids':>{sum_w}} | "
        f"{'sum_no_asks':>{sum_w}} | {'sum_no_bids':>{sum_w}} | "
        f"{'buy_edge':>{edge_w}} | {'sell_edge':>{edge_w}} | "
        f"{'buy_no_edge':>{edge_w}} | {'sell_no_edge':>{edge_w}} | "
        f"{'best_edge':>{edge_w}} | "
        f"{'mYA':>{miss_w}} | {'mYB':>{miss_w}} | {'mNA':>{miss_w}} | {'mNB':>{miss_w}} | "
        f"{'mAny':>{miss_w}} | {'mAll':>{miss_w}} | "
        f"{'min_days_to_close':>{days_w}} | {'max_days_to_close':>{days_w}}"
    )
    print(header)
    print("-" * len(header))

    for r in out:
        line = (
            f"{r.event_ticker:<{event_w}} | {r.type:<{type_w}} | {r.mk:>{mk_w}} | "
            f"{fmt_int(r.sum_asks, sum_w)} | {fmt_int(r.sum_bids, sum_w)} | "
            f"{fmt_int(r.sum_no_asks, sum_w)} | {fmt_int(r.sum_no_bids, sum_w)} | "
            f"{fmt_edge(r.buy_edge, edge_w)} | {fmt_edge(r.sell_edge, edge_w)} | "
            f"{fmt_edge(r.buy_no_edge, edge_w)} | {fmt_edge(r.sell_no_edge, edge_w)} | "
            f"{fmt_edge(r.best_edge, edge_w)} | "
            f"{fmt_int(r.missing_yes_ask, miss_w)} | {fmt_int(r.missing_yes_bid, miss_w)} | "
            f"{fmt_int(r.missing_no_ask, miss_w)} | {fmt_int(r.missing_no_bid, miss_w)} | "
            f"{fmt_int(r.missing_any, miss_w)} | {fmt_int(r.missing_all, miss_w)} | "
            f"{fmt_float(r.min_days_to_close, days_w)} | "
            f"{fmt_float(r.max_days_to_close, days_w)}"
        )
        print(line)


if __name__ == "__main__":
    main()
