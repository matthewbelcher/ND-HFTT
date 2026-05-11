import argparse
import json
import math
import os
import sys
import time
import uuid
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from kalshi_common import load_private_key_pem, sign_pss_base64, now_ms


DATA_API_BASE = os.getenv(
    "KALSHI_DATA_API_BASE",
    os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co"),
)
TRADE_API_BASE = os.getenv(
    "KALSHI_TRADE_API_BASE",
    os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co"),
)
TRADE_API_V2 = "/trade-api/v2"


@dataclass
class MarketTop:
    ticker: str
    yes_bid_px: Optional[int]
    yes_bid_qty: Optional[float]
    yes_ask_px: Optional[int]
    yes_ask_qty: Optional[float]
    yes_ask_qty_depth_taker_proxy: Optional[float]
    yes_mid: Optional[float]
    no_bid_px: Optional[int]
    no_bid_qty: Optional[float]
    no_ask_px: Optional[int]
    no_ask_qty: Optional[float]
    no_ask_qty_depth_taker_proxy: Optional[float]
    no_mid: Optional[float]
    yes_bids: List[Tuple[int, float]]
    no_bids: List[Tuple[int, float]]


@dataclass
class FeePolicy:
    series_ticker: str
    taker_fee_coef: float
    maker_fee_rate: float
    source: str


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
    base_url: str = DATA_API_BASE,
) -> Optional[dict]:
    url = base_url + path
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


def api_get_trade(
    path: str,
    key_id: str,
    private_key,
    retries: int = 2,
    backoff_s: float = 0.5,
    allow_fail: bool = False,
) -> Optional[dict]:
    return api_get(
        path,
        key_id,
        private_key,
        retries=retries,
        backoff_s=backoff_s,
        allow_fail=allow_fail,
        base_url=TRADE_API_BASE,
    )


def trade_get_paginated(
    path: str,
    list_key: str,
    key_id: str,
    private_key,
    limit: int = 200,
) -> List[dict]:
    out: List[dict] = []
    cursor = None
    while True:
        params = {"limit": str(limit)}
        if cursor:
            params["cursor"] = cursor
        qs = urllib.parse.urlencode(params)
        sep = "&" if "?" in path else "?"
        resp = api_get_trade(f"{path}{sep}{qs}", key_id, private_key, allow_fail=True)
        if not resp:
            break
        items = resp.get(list_key) or []
        if isinstance(items, list):
            out.extend(items)
        cursor = resp.get("cursor")
        if not cursor:
            break
    return out


def _price_cents_from_obj(obj: dict, keys: List[str]) -> Optional[int]:
    for k in keys:
        if k not in obj:
            continue
        v = obj.get(k)
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            if v <= 1.0:
                return int(round(v * 100.0))
            return int(round(v))
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f <= 1.0:
            return int(round(f * 100.0))
        return int(round(f))
    return None


def _price_cents_from_side(
    obj: dict,
    side: Optional[str],
    *,
    yes_keys: List[str],
    no_keys: List[str],
    fallback_price_keys: Optional[List[str]] = None,
) -> Optional[int]:
    side_norm = (side or "").lower()
    if side_norm == "yes":
        px = _price_cents_from_obj(obj, yes_keys)
        if px is not None:
            return px
    if side_norm == "no":
        px = _price_cents_from_obj(obj, no_keys)
        if px is not None:
            return px
    if fallback_price_keys:
        px = _price_cents_from_obj(obj, fallback_price_keys)
        if px is not None:
            if side_norm == "no":
                return 100 - px
            return px
    return None


def _fee_cents_from_fill(fill: dict) -> int:
    fee = fill.get("fee_cost")
    if fee is None:
        return 0
    try:
        return int(round(float(fee) * 100.0))
    except (TypeError, ValueError):
        return 0


def sync_trade_state(
    event_ticker: str,
    tickers: List[str],
    trade_key_id: str,
    trade_private_key,
    fills_lookback_hours: Optional[float],
    subaccount: Optional[int],
) -> dict:
    params = {"event_ticker": event_ticker, "count_filter": "position"}
    if subaccount is not None:
        params["subaccount"] = str(subaccount)
    qs = urllib.parse.urlencode(params)
    positions = trade_get_paginated(
        f"{TRADE_API_V2}/portfolio/positions?{qs}",
        "market_positions",
        trade_key_id,
        trade_private_key,
        limit=200,
    )
    pos_yes: Dict[str, int] = {}
    for p in positions:
        ticker = p.get("ticker")
        if not ticker:
            continue
        try:
            pos_yes[ticker] = int(p.get("position") or 0)
        except (TypeError, ValueError):
            continue
    pos_yes_total = sum(v for v in pos_yes.values() if v > 0)
    pos_no_total = sum(-v for v in pos_yes.values() if v < 0)

    params = {"event_ticker": event_ticker, "status": "resting"}
    if subaccount is not None:
        params["subaccount"] = str(subaccount)
    qs = urllib.parse.urlencode(params)
    orders = trade_get_paginated(
        f"{TRADE_API_V2}/portfolio/orders?{qs}",
        "orders",
        trade_key_id,
        trade_private_key,
        limit=200,
    )
    pending_no_qty: Dict[str, int] = {}
    pending_no_cost: Dict[str, int] = {}
    pending_yes_qty: Dict[str, int] = {}
    pending_yes_cost: Dict[str, int] = {}
    resting_orders_total = len(orders)
    resting_no_buy_orders = 0
    resting_no_buy_qty_total = 0
    resting_yes_buy_orders = 0
    resting_yes_buy_qty_total = 0
    for o in orders:
        if o.get("action") != "buy":
            continue
        side = o.get("side")
        if side not in ("no", "yes"):
            continue
        ticker = o.get("ticker")
        if not ticker:
            continue
        remaining = o.get("remaining_count")
        if remaining is None:
            remaining = o.get("remaining_count_fp") or o.get("initial_count") or 0
        try:
            remaining_int = int(float(remaining))
        except (TypeError, ValueError):
            remaining_int = 0
        if remaining_int <= 0:
            continue
        px = _price_cents_from_side(
            o,
            side,
            yes_keys=["yes_price", "yes_price_fixed", "yes_price_dollars"],
            no_keys=["no_price", "no_price_fixed", "no_price_dollars"],
            fallback_price_keys=["price"],
        )
        if px is None:
            continue
        if side == "no":
            resting_no_buy_orders += 1
            resting_no_buy_qty_total += remaining_int
            pending_no_qty[ticker] = pending_no_qty.get(ticker, 0) + remaining_int
            pending_no_cost[ticker] = pending_no_cost.get(ticker, 0) + (px * remaining_int)
        else:
            resting_yes_buy_orders += 1
            resting_yes_buy_qty_total += remaining_int
            pending_yes_qty[ticker] = pending_yes_qty.get(ticker, 0) + remaining_int
            pending_yes_cost[ticker] = pending_yes_cost.get(ticker, 0) + (px * remaining_int)

    fills_no_qty: Dict[str, int] = {}
    fills_no_cost: Dict[str, int] = {}
    fills_yes_qty: Dict[str, int] = {}
    fills_yes_cost: Dict[str, int] = {}
    min_ts = None
    if fills_lookback_hours is not None and fills_lookback_hours > 0:
        min_ts = now_ms() - int(fills_lookback_hours * 3600 * 1000)
    if fills_lookback_hours is not None and fills_lookback_hours <= 0:
        return {
            "pos_yes": pos_yes,
            "pos_yes_total": pos_yes_total,
            "pos_no_total": pos_no_total,
            "pending_no_qty": pending_no_qty,
            "pending_no_cost": pending_no_cost,
            "pending_yes_qty": pending_yes_qty,
            "pending_yes_cost": pending_yes_cost,
            "resting_orders_total": resting_orders_total,
            "resting_no_buy_orders": resting_no_buy_orders,
            "resting_no_buy_qty_total": resting_no_buy_qty_total,
            "resting_yes_buy_orders": resting_yes_buy_orders,
            "resting_yes_buy_qty_total": resting_yes_buy_qty_total,
            "fills_no_qty": fills_no_qty,
            "fills_no_cost": fills_no_cost,
            "fills_yes_qty": fills_yes_qty,
            "fills_yes_cost": fills_yes_cost,
            "fills_lookback_hours": fills_lookback_hours,
        }
    for ticker in tickers:
        params = {"ticker": ticker}
        if min_ts is not None:
            params["min_ts"] = str(min_ts)
        if subaccount is not None:
            params["subaccount"] = str(subaccount)
        qs = urllib.parse.urlencode(params)
        fills = trade_get_paginated(
            f"{TRADE_API_V2}/portfolio/fills?{qs}",
            "fills",
            trade_key_id,
            trade_private_key,
            limit=200,
        )
        for f in fills:
            if f.get("action") != "buy":
                continue
            side = f.get("side")
            if side not in ("no", "yes"):
                continue
            count = f.get("count") or f.get("count_fp") or 0
            try:
                count_int = int(float(count))
            except (TypeError, ValueError):
                count_int = 0
            if count_int <= 0:
                continue
            px = _price_cents_from_side(
                f,
                side,
                yes_keys=["yes_price", "yes_price_fixed", "yes_price_dollars"],
                no_keys=["no_price", "no_price_fixed", "no_price_dollars"],
                fallback_price_keys=["price"],
            )
            if px is None:
                continue
            fee = _fee_cents_from_fill(f)
            if side == "no":
                fills_no_qty[ticker] = fills_no_qty.get(ticker, 0) + count_int
                fills_no_cost[ticker] = fills_no_cost.get(ticker, 0) + (px * count_int + fee)
            else:
                fills_yes_qty[ticker] = fills_yes_qty.get(ticker, 0) + count_int
                fills_yes_cost[ticker] = fills_yes_cost.get(ticker, 0) + (px * count_int + fee)

    return {
        "pos_yes": pos_yes,
        "pos_yes_total": pos_yes_total,
        "pos_no_total": pos_no_total,
        "pending_no_qty": pending_no_qty,
        "pending_no_cost": pending_no_cost,
        "pending_yes_qty": pending_yes_qty,
        "pending_yes_cost": pending_yes_cost,
        "resting_orders_total": resting_orders_total,
        "resting_no_buy_orders": resting_no_buy_orders,
        "resting_no_buy_qty_total": resting_no_buy_qty_total,
        "resting_yes_buy_orders": resting_yes_buy_orders,
        "resting_yes_buy_qty_total": resting_yes_buy_qty_total,
        "fills_no_qty": fills_no_qty,
        "fills_no_cost": fills_no_cost,
        "fills_yes_qty": fills_yes_qty,
        "fills_yes_cost": fills_yes_cost,
        "fills_lookback_hours": fills_lookback_hours,
    }
def api_post(
    path: str,
    body: dict,
    key_id: str,
    private_key,
    retries: int = 1,
    backoff_s: float = 0.5,
    allow_fail: bool = False,
    base_url: str = TRADE_API_BASE,
) -> Optional[dict]:
    url = base_url + path
    headers = kalshi_rest_headers(key_id, private_key, "POST", path)
    headers["Content-Type"] = "application/json"
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, headers=headers, data=payload, method="POST")
    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if not data:
                return {}
            return json.loads(data.decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", "replace")
            except Exception:
                body_text = ""
            print(f"HTTP {e.code} on POST {path}: {body_text}", file=sys.stderr)
            if attempt == retries:
                if allow_fail:
                    return None
                raise
            time.sleep(backoff_s * (2 ** attempt))
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


def place_orders_batch(
    orders: List[dict],
    key_id: str,
    private_key,
    allow_fail: bool = False,
    base_url: str = TRADE_API_BASE,
) -> Tuple[Optional[dict], List[dict]]:
    if not orders:
        return None, []
    path = f"{TRADE_API_V2}/portfolio/orders/batched"
    resp = api_post(
        path,
        {"orders": orders},
        key_id,
        private_key,
        allow_fail=allow_fail,
        base_url=base_url,
    )
    results: List[dict] = []
    if not resp:
        return resp, results
    raw_results = resp.get("orders") or resp.get("results") or resp.get("order_results") or []
    if not isinstance(raw_results, list):
        print("Warning: batch response missing per-order results.", file=sys.stderr)
        return resp, results
    success = 0
    fail = 0
    for idx, item in enumerate(raw_results):
        err = item.get("error") or item.get("errors") or item.get("message")
        ok = err is None and item.get("order") is not None
        if ok:
            success += 1
        else:
            fail += 1
        results.append({"index": idx, "success": ok, "error": err})
    print(f"Batch create orders: {success} succeeded, {fail} failed (total {len(raw_results)})")
    for r in results:
        if not r["success"] and r["error"]:
            print(f"Order {r['index']} error: {r['error']}", file=sys.stderr)
    return resp, results


def warn_price_units(price: float, price_in_dollars: bool) -> None:
    if price_in_dollars and price > 1.0 + 1e-6:
        print(
            f"Warning: price {price} looks like cents but marked dollars",
            file=sys.stderr,
        )
    if (not price_in_dollars) and 0 < price < 1.0 - 1e-9:
        print(
            f"Warning: price {price} looks like dollars but marked cents",
            file=sys.stderr,
        )


def warn_price_bounds(price_cents: int, context: str) -> None:
    if price_cents < 0 or price_cents > 100:
        print(
            f"Warning: price out of bounds ({price_cents}) in {context}",
            file=sys.stderr,
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
    try:
        warn_price_units(float(price), price_in_dollars)
    except (TypeError, ValueError):
        pass
    if price_in_dollars:
        price_cents = int(round(float(price) * 100))
    else:
        price_cents = int(float(price))
    warn_price_bounds(price_cents, "parse_level")
    qty_val = None
    if qty is not None:
        qty_val = float(qty)
    return price_cents, qty_val


def parse_levels(levels, price_in_dollars: bool) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    if not levels:
        return out
    for level in levels:
        if isinstance(level, (int, float, str)):
            price = level
            qty = None
        else:
            price = level[0] if len(level) > 0 else None
            qty = level[1] if len(level) > 1 else None
        if price is None:
            continue
        try:
            warn_price_units(float(price), price_in_dollars)
        except (TypeError, ValueError):
            pass
        if price_in_dollars:
            price_cents = int(round(float(price) * 100))
        else:
            price_cents = int(float(price))
        warn_price_bounds(price_cents, "parse_levels")
        if qty is None:
            continue
        qty_val = float(qty)
        out.append((price_cents, qty_val))
    return out


def normalize_bids(
    levels: List[Tuple[int, float]],
    debug: bool = False,
) -> List[Tuple[int, float]]:
    agg: Dict[int, float] = {}
    for price, qty in levels:
        if qty is None or qty <= 0:
            continue
        if abs(qty - round(qty)) > 1e-6 and debug:
            print(f"Non-integer qty seen at {price}: {qty}")
        if price < 0 or price > 100:
            if debug:
                print(f"Skipping invalid price: {price}")
            continue
        agg[price] = agg.get(price, 0.0) + float(qty)
    bids = sorted(agg.items(), key=lambda x: x[0], reverse=True)
    return bids


def best_bid(bids: List[Tuple[int, float]]) -> Tuple[Optional[int], Optional[float]]:
    if not bids:
        return None, None
    price, qty = bids[0]
    return price, qty


def implied_asks_from_bids(bids: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    asks = []
    for price, qty in bids:
        ask_px = 100 - price
        if ask_px < 0 or ask_px > 100 or qty <= 0:
            continue
        asks.append((ask_px, qty))
    asks.sort(key=lambda x: x[0])
    return asks


def qty_to_int(qty: Optional[float], debug: bool = False) -> int:
    if qty is None:
        return 0
    if abs(qty - round(qty)) < 1e-6:
        return int(round(qty))
    q_int = int(math.floor(qty + 1e-9))
    if debug:
        print(f"Warning: non-integer qty {qty} -> {q_int}", file=sys.stderr)
    return q_int


def rec_qty_estimate(
    ask_qty_top: Optional[int],
    ask_qty_depth_proxy: Optional[int],
    fraction: float,
    cap: Optional[int],
) -> Optional[int]:
    # For taker fallback sizing, prioritize depth proxy (marketable across levels).
    proxy = ask_qty_depth_proxy if ask_qty_depth_proxy is not None else ask_qty_top
    if proxy is None or proxy <= 0:
        return None
    frac = max(0.0, min(1.0, fraction))
    est = int(math.floor(proxy * frac))
    if cap is not None and cap > 0:
        est = min(est, cap)
    if est <= 0:
        return None
    return est


def taker_max_size_with_slippage(
    levels: List[Tuple[int, float]],
    max_slippage_c: int,
) -> int:
    if not levels:
        return 0
    best_ask = levels[0][0]
    if best_ask is None:
        return 0
    limit_px = best_ask + max(0, max_slippage_c)
    total = 0
    for price, qty in levels:
        if price > limit_px:
            break
        total += qty_to_int(qty)
    return total


def net_buy_cost(
    levels: List[Tuple[int, float]],
    size: int,
    fee_rate: float,
    debug: bool = False,
) -> Tuple[Optional[float], int]:
    remaining = size
    cost = 0.0
    for price, qty in levels:
        if remaining <= 0:
            break
        if price < 0 or price > 100:
            if debug:
                print(f"Warning: invalid price in buy levels: {price}", file=sys.stderr)
            continue
        q_int = qty_to_int(qty, debug=debug)
        take = min(remaining, q_int)
        if take <= 0:
            continue
        cost += (price * take) + fee_cents(fee_rate, price, take)
        remaining -= take
    filled = size - remaining
    if filled <= 0:
        return None, 0
    return cost, filled


def net_sell_proceeds(
    levels: List[Tuple[int, float]],
    size: int,
    fee_rate: float,
    debug: bool = False,
) -> Tuple[Optional[float], int]:
    remaining = size
    proceeds = 0.0
    for price, qty in levels:
        if remaining <= 0:
            break
        if price < 0 or price > 100:
            if debug:
                print(f"Warning: invalid price in sell levels: {price}", file=sys.stderr)
            continue
        q_int = qty_to_int(qty, debug=debug)
        take = min(remaining, q_int)
        if take <= 0:
            continue
        proceeds += (price * take) - fee_cents(fee_rate, price, take)
        remaining -= take
    filled = size - remaining
    if filled <= 0:
        return None, 0
    return proceeds, filled


def normalize_probs_k(raw_probs: List[Optional[float]], k: int) -> Tuple[List[Optional[float]], bool]:
    probs = raw_probs[:]
    indices = [i for i, p in enumerate(probs) if p is not None]
    if not indices:
        return probs, False
    remaining_k = float(k)
    active = set(indices)
    changed = True
    for _ in range(20):
        if not active:
            break
        total = sum(probs[i] for i in active if probs[i] is not None)
        if total <= 0:
            break
        scale = remaining_k / total
        exceeded = []
        for i in list(active):
            p = probs[i]
            if p is None:
                active.discard(i)
                continue
            p_scaled = p * scale
            if p_scaled >= 1.0:
                probs[i] = 1.0
                exceeded.append(i)
            else:
                probs[i] = p_scaled
        if not exceeded:
            return probs, True
        for i in exceeded:
            active.discard(i)
        remaining_k = float(k) - sum(probs[i] for i in indices if probs[i] is not None)
        if remaining_k < 0:
            break
    return probs, False


def basket_buy_cost(
    level_sets: List[List[Tuple[int, float]]],
    size: int,
    fee_rates: List[float],
) -> Tuple[Optional[float], bool]:
    total = 0.0
    if len(level_sets) != len(fee_rates):
        return None, False
    for levels, fee_rate in zip(level_sets, fee_rates):
        cost, filled = net_buy_cost(levels, size, fee_rate)
        if filled < size:
            return None, False
        total += cost
    return total, True


def basket_sell_proceeds(
    level_sets: List[List[Tuple[int, float]]],
    size: int,
    fee_rates: List[float],
) -> Tuple[Optional[float], bool]:
    total = 0.0
    if len(level_sets) != len(fee_rates):
        return None, False
    for levels, fee_rate in zip(level_sets, fee_rates):
        proceeds, filled = net_sell_proceeds(levels, size, fee_rate)
        if filled < size:
            return None, False
        total += proceeds
    return total, True


def best_from_orderbook(ob_data: dict, side: str) -> Tuple[Optional[int], Optional[float]]:
    ob = ob_data.get("orderbook") or {}
    ob_fp = ob_data.get("orderbook_fp") or {}

    levels = ob.get(side)
    price, qty = parse_level(levels, price_in_dollars=False)
    if price is not None:
        return price, qty

    levels = ob_fp.get(f"{side}_dollars")
    return parse_level(levels, price_in_dollars=True)


def fmt_px(px: Optional[int]) -> str:
    return "--" if px is None else f"{px:02d}"


def fmt_qty(qty: Optional[float]) -> str:
    if qty is None:
        return "--"
    if abs(qty - int(qty)) < 1e-9:
        return f"{int(qty)}"
    return f"{qty:.2f}"


def fmt_levels(levels: List[Tuple[int, float]]) -> str:
    parts = []
    for px, qty in levels:
        q = qty_to_int(qty)
        if q <= 0:
            continue
        parts.append(f"{px}c@{q}")
    return ", ".join(parts) if parts else "--"


def load_fee_whitelist(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("fee whitelist must be a JSON object")
    out: Dict[str, float] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        try:
            coef = float(v)
        except Exception as e:
            raise ValueError(f"Invalid fee coef for {k}: {v}") from e
        out[k.upper()] = coef
    return out


def fee_policy_for_series(
    series_ticker: Optional[str],
    whitelist: Dict[str, float],
    default_taker: float,
    maker_fee_rate: float,
) -> FeePolicy:
    st = (series_ticker or "").upper()
    if st in whitelist:
        return FeePolicy(
            series_ticker=st,
            taker_fee_coef=whitelist[st],
            maker_fee_rate=maker_fee_rate,
            source="whitelist",
        )
    return FeePolicy(
        series_ticker=st,
        taker_fee_coef=default_taker,
        maker_fee_rate=maker_fee_rate,
        source="default",
    )


def is_arb_safe(
    edge: float,
    require_taker_fallback: bool,
    taker_fallback_worst: Optional[float],
) -> bool:
    if edge <= 0:
        return False
    if not require_taker_fallback:
        return True
    return taker_fallback_worst is not None and taker_fallback_worst >= 0


def fee_cents(rate: float, price_cents: int, contracts: int) -> int:
    if rate <= 0 or price_cents <= 0 or contracts <= 0:
        return 0
    p = price_cents / 100.0
    fee_dollars = rate * contracts * p * (1.0 - p)
    return int(math.ceil(fee_dollars * 100.0 - 1e-9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze event for probability sum and arb edges")
    parser.add_argument("--event", required=True, help="Event ticker (e.g., KXWOFREESKI-MBA26CGOLD)")
    parser.add_argument("--depth", type=int, default=1, help="Orderbook depth to request (API)")
    parser.add_argument("--size", type=int, default=1, help="Contracts per leg to size baskets")
    parser.add_argument("--near", type=int, default=5, help="Near-arb threshold in cents")
    parser.add_argument(
        "--true-count",
        type=int,
        default=None,
        help="Exact number of outcomes that can resolve YES (e.g., 3 for medal winners)",
    )
    parser.add_argument("--watch", action="store_true", help="Continuously monitor the event")
    parser.add_argument("--interval", type=float, default=15.0, help="Watch interval in seconds")
    parser.add_argument("--alert-only", action="store_true", help="Only print when near/arb is detected")
    parser.add_argument("--brief", action="store_true", help="Skip per-market tables")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Show basket options (keep all, drop negative EV, middle ground)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug/progress logs",
    )
    parser.add_argument(
        "--fee-kind",
        choices=["maker", "taker", "none"],
        default="taker",
        help="Fee side to apply for net edges (default: taker)",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=None,
        help="Override fee rate (e.g., 0.0175). If set, used for fee-kind.",
    )
    parser.add_argument(
        "--fee-rate-maker",
        type=float,
        default=0.0,
        help="Maker fee rate (default 0.0)",
    )
    parser.add_argument(
        "--fee-rate-taker",
        type=float,
        default=0.07,
        help="Taker fee rate (default 0.07)",
    )
    parser.add_argument(
        "--limit-mode",
        choices=["none", "mid", "improve"],
        default="none",
        help="Limit order price mode for NO basket (default: none)",
    )
    parser.add_argument(
        "--limit-improve-c",
        type=int,
        default=1,
        help="Cents to improve over bid for limit-mode=improve",
    )
    parser.add_argument(
        "--limit-fee-kind",
        choices=["maker", "taker", "none"],
        default="maker",
        help="Fee kind to apply for limit orders (default: maker)",
    )
    parser.add_argument(
        "--limit-fee-rate",
        type=float,
        default=None,
        help="Override fee rate for limit orders",
    )
    parser.add_argument(
        "--fee-whitelist",
        default=None,
        help="JSON file mapping series_ticker to taker fee coef (e.g., 0.035)",
    )
    parser.add_argument(
        "--limit-min-qty",
        type=int,
        default=1,
        help="Minimum ask qty to include in limit basket",
    )
    parser.add_argument(
        "--limit-min-spread",
        type=int,
        default=1,
        help="Minimum spread (cents) to include in limit basket",
    )
    parser.add_argument(
        "--max-slippage-c",
        type=int,
        default=2,
        help="Max taker slippage (cents) for recommended basket sizing (default 2)",
    )
    parser.add_argument(
        "--taker-max-cap",
        type=int,
        default=None,
        help="Cap for taker slippage-based basket size (optional)",
    )
    parser.add_argument(
        "--exec-max-dollars",
        type=float,
        default=None,
        help="Hard dollar cap for auto-exec basket cost (optional)",
    )
    parser.add_argument(
        "--sync-trade-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sync positions and resting orders from trade API (default false)",
    )
    parser.add_argument(
        "--fills-lookback-hours",
        type=float,
        default=168.0,
        help="Lookback window for fills when syncing trade state (default 168)",
    )
    parser.add_argument(
        "--subaccount",
        type=int,
        default=None,
        help="Optional subaccount ID for trade sync",
    )
    parser.add_argument(
        "--auto-exec",
        action="store_true",
        help="Auto-submit limit orders when limit basket is SAFE",
    )
    parser.add_argument(
        "--rec-qty-fraction",
        type=float,
        default=0.5,
        help="Recommended qty fraction of liquidity proxy (default 0.5)",
    )
    parser.add_argument(
        "--rec-qty-cap",
        type=int,
        default=None,
        help="Cap for recommended qty estimate (optional)",
    )
    parser.add_argument(
        "--limit-out",
        default=None,
        help="Write limit basket report to this file",
    )
    parser.add_argument(
        "--fallback-taker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include taker-fallback worst-case checks for limit baskets",
    )
    parser.add_argument(
        "--assumed-fill-ratio",
        type=float,
        default=None,
        help="Assumed limit fill ratio for blended worst-case (0-1)",
    )
    parser.add_argument(
        "--require-taker-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require taker-fallback worst-case >= 0 to label ARB/SAFE",
    )
    parser.add_argument(
        "--allow-default-fees",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow default fee policy when series not in whitelist (default false)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal sanity tests and exit",
    )
    return parser.parse_args()


def fetch_event_snapshot(
    event_ticker: str,
    depth: int,
    key_id: str,
    private_key,
    debug: bool,
) -> Tuple[dict, List[MarketTop], List[str], List[str], List[str], List[str]]:
    event_path = f"{TRADE_API_V2}/events/{urllib.parse.quote(event_ticker)}"
    if debug:
        print(f"Fetching event: {event_ticker}")
    event_resp = api_get(event_path, key_id, private_key)
    if not event_resp:
        print("Failed to fetch event (no response).")
        sys.exit(1)

    event = event_resp.get("event", {})
    markets = event_resp.get("markets") or event.get("markets") or []
    if not markets:
        print("No markets found for event.")
        sys.exit(1)
    if debug:
        print(f"Markets to fetch: {len(markets)}")

    market_info: Dict[str, dict] = {m.get("ticker"): m for m in markets if m.get("ticker")}
    tickers = [m for m in market_info.keys()]

    tops: List[MarketTop] = []
    missing_yes_bid = []
    missing_yes_ask = []
    missing_no_bid = []
    missing_no_ask = []

    for ticker in tickers:
        qs = urllib.parse.urlencode({"depth": str(depth)})
        ob_path = f"{TRADE_API_V2}/markets/{urllib.parse.quote(ticker)}/orderbook?{qs}"
        ob = api_get(ob_path, key_id, private_key, allow_fail=True) or {}
        if debug and (len(tops) + 1) % 10 == 0:
            print(f"Fetched orderbooks: {len(tops) + 1}/{len(tickers)}")

        ob_book = ob.get("orderbook") or {}
        ob_fp = ob.get("orderbook_fp") or {}
        yes_levels = parse_levels(ob_book.get("yes"), price_in_dollars=False)
        no_levels = parse_levels(ob_book.get("no"), price_in_dollars=False)
        if not yes_levels and ob_fp.get("yes_dollars"):
            yes_levels = parse_levels(ob_fp.get("yes_dollars"), price_in_dollars=True)
        if not no_levels and ob_fp.get("no_dollars"):
            no_levels = parse_levels(ob_fp.get("no_dollars"), price_in_dollars=True)

        yes_bids = normalize_bids(yes_levels, debug=debug)
        no_bids = normalize_bids(no_levels, debug=debug)

        yes_bid_px, yes_bid_qty = best_bid(yes_bids)
        no_bid_px, no_bid_qty = best_bid(no_bids)

        yes_ask_px = None
        yes_ask_qty = None
        yes_ask_qty_depth_taker_proxy = None
        if no_bid_px is not None:
            yes_ask_px = 100 - no_bid_px
            yes_ask_qty = no_bid_qty
        if no_bids:
            yes_ask_qty_depth_taker_proxy = sum(q for _, q in no_bids[: max(1, depth)])

        no_ask_px = None
        no_ask_qty = None
        no_ask_qty_depth_taker_proxy = None
        if yes_bid_px is not None:
            no_ask_px = 100 - yes_bid_px
            no_ask_qty = yes_bid_qty
        if yes_bids:
            no_ask_qty_depth_taker_proxy = sum(q for _, q in yes_bids[: max(1, depth)])

        # Fallback to event market snapshot for prices if orderbook lacks them.
        info = market_info.get(ticker, {})
        if yes_bid_px is None and info.get("yes_bid") is not None:
            yes_bid_px = int(info["yes_bid"])
        if yes_ask_px is None and info.get("yes_ask") is not None:
            yes_ask_px = int(info["yes_ask"])
        if no_bid_px is None and info.get("no_bid") is not None:
            no_bid_px = int(info["no_bid"])
        if no_ask_px is None and info.get("no_ask") is not None:
            no_ask_px = int(info["no_ask"])
        if yes_ask_px is None and no_bid_px is not None:
            yes_ask_px = 100 - no_bid_px
        if no_ask_px is None and yes_bid_px is not None:
            no_ask_px = 100 - yes_bid_px

        if yes_bid_px is None:
            missing_yes_bid.append(ticker)
        if yes_ask_px is None:
            missing_yes_ask.append(ticker)
        if no_bid_px is None:
            missing_no_bid.append(ticker)
        if no_ask_px is None:
            missing_no_ask.append(ticker)

        yes_mid = None
        if yes_bid_px is not None and yes_ask_px is not None:
            if yes_ask_px < yes_bid_px:
                print(
                    f"Warning: negative YES spread for {ticker}: {yes_bid_px}/{yes_ask_px}",
                    file=sys.stderr,
                )
            yes_mid = (yes_bid_px + yes_ask_px) / 2.0

        no_mid = None
        if no_bid_px is not None and no_ask_px is not None:
            if no_ask_px < no_bid_px:
                print(
                    f"Warning: negative NO spread for {ticker}: {no_bid_px}/{no_ask_px}",
                    file=sys.stderr,
                )
            no_mid = (no_bid_px + no_ask_px) / 2.0

        tops.append(
            MarketTop(
                ticker=ticker,
                yes_bid_px=yes_bid_px,
                yes_bid_qty=yes_bid_qty,
                yes_ask_px=yes_ask_px,
                yes_ask_qty=yes_ask_qty,
                yes_ask_qty_depth_taker_proxy=yes_ask_qty_depth_taker_proxy,
                yes_mid=yes_mid,
                no_bid_px=no_bid_px,
                no_bid_qty=no_bid_qty,
                no_ask_px=no_ask_px,
                no_ask_qty=no_ask_qty,
                no_ask_qty_depth_taker_proxy=no_ask_qty_depth_taker_proxy,
                no_mid=no_mid,
                yes_bids=yes_bids,
                no_bids=no_bids,
            )
        )

        time.sleep(0.05)

    return event_resp, tops, missing_yes_bid, missing_yes_ask, missing_no_bid, missing_no_ask


def has_alert(
    sum_yes_ask,
    sum_yes_bid,
    sum_no_ask,
    sum_no_bid,
    target_yes,
    target_no,
    near: int,
) -> bool:
    if target_yes is not None and sum_yes_ask is not None and sum_yes_ask <= target_yes + near:
        return True
    if target_yes is not None and sum_yes_bid is not None and sum_yes_bid >= target_yes - near:
        return True
    if target_no is not None and sum_no_ask is not None and sum_no_ask <= target_no + near:
        return True
    if target_no is not None and sum_no_bid is not None and sum_no_bid >= target_no - near:
        return True
    return False


def sum_or_none(values) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals)


def basket_stats(
    legs: List[dict],
    k: int,
    size: int,
    *,
    full_count: Optional[int] = None,
    side: str = "no",
    prob_key: str = "p_no",
) -> Optional[dict]:
    if not legs:
        return None
    m = len(legs)
    expected_payout = sum((leg.get(prob_key) or 0.0) * 100.0 * size for leg in legs)
    cost = sum(leg["cost_per_contract"] * size for leg in legs)
    payout_best = m * 100.0 * size
    payout_worst = max(0.0, (m - k) * 100.0 * size)
    full_set = full_count is not None and m == full_count
    if full_count is not None:
        n = full_count
        max_yes = min(k, m)
        min_yes = max(0, k - (n - m))
        side_norm = side.lower()
        if side_norm == "yes":
            winners_best = max_yes
            winners_worst = min_yes
        else:
            winners_best = m - min_yes
            winners_worst = m - max_yes
        payout_best = winners_best * 100.0 * size
        payout_worst = winners_worst * 100.0 * size
    edge_best = payout_best - cost
    edge_worst = payout_worst - cost
    if abs(edge_best - (payout_best - cost)) > 1e-6:
        print("Warning: edge_best inconsistency", file=sys.stderr)
    if abs(edge_worst - (payout_worst - cost)) > 1e-6:
        print("Warning: edge_worst inconsistency", file=sys.stderr)
    ev = expected_payout - cost
    qty_vals = [leg["qty"] for leg in legs if leg.get("qty") is not None]
    min_qty = min(qty_vals) if qty_vals else None
    max_size = int(min_qty) if min_qty is not None else 0
    max_cost = None
    if max_size > 0:
        max_cost = 0.0
        for leg in legs:
            leg_rate = leg.get("fee_rate", 0.0)
            max_cost += (leg["price"] * max_size) + fee_cents(
                leg_rate, leg["price"], max_size
            )
    return {
        "count": m,
        "cost": cost,
        "expected_payout": expected_payout,
        "payout_best": payout_best,
        "payout_worst": payout_worst,
        "edge_best": edge_best,
        "edge_worst": edge_worst,
        "ev": ev,
        "full_set": full_set,
        "max_size": max_size,
        "max_cost": max_cost,
    }


def best_middle_subset(
    legs: List[dict],
    k: int,
    size: int,
    *,
    full_count: Optional[int] = None,
    side: str = "no",
    prob_key: str = "p_no",
) -> Optional[Tuple[List[dict], dict]]:
    if not legs:
        return None
    legs_sorted = sorted(legs, key=lambda x: x["ev"], reverse=True)
    best = None
    for i in range(1, len(legs_sorted) + 1):
        subset = legs_sorted[:i]
        stats = basket_stats(
            subset, k, size, full_count=full_count, side=side, prob_key=prob_key
        )
        if stats is None:
            continue
        if stats["edge_worst"] >= 0:
            if best is None or stats["ev"] > best[1]["ev"]:
                best = (subset, stats)
    if best is None:
        for i in range(1, len(legs_sorted) + 1):
            subset = legs_sorted[:i]
            stats = basket_stats(
                subset, k, size, full_count=full_count, side=side, prob_key=prob_key
            )
            if stats is None:
                continue
            if best is None or stats["ev"] > best[1]["ev"]:
                best = (subset, stats)
    return best


def print_report(
    event_resp: dict,
    tops: List[MarketTop],
    missing_yes_bid: List[str],
    missing_yes_ask: List[str],
    missing_no_bid: List[str],
    missing_no_ask: List[str],
    near: int,
    show_tables: bool,
    print_header: bool,
    fee_policy: FeePolicy,
    require_taker_fallback: bool,
    optimize: bool,
    limit_mode: str,
    limit_improve_c: int,
    limit_min_qty: int,
    limit_min_spread: int,
    max_slippage_c: int,
    taker_max_cap: Optional[int],
    rec_qty_fraction: float,
    rec_qty_cap: Optional[int],
    trade_state: Optional[dict],
    depth: int,
    limit_out: Optional[str],
    fallback_taker: bool,
    assumed_fill_ratio: Optional[float],
    size: int,
) -> Optional[dict]:
    event = event_resp.get("event", {})
    limit_state = None
    if print_header:
        print(f"Event: {event.get('event_ticker', '--')}")
        print(f"Title: {event.get('title', '--')}")
        print(f"Series: {event.get('series_ticker', '--')}")
        print(f"Mutually exclusive: {event.get('mutually_exclusive', '--')}")
        print(f"Markets: {len(tops)}")
        print(f"Contracts per leg: {size}")
        fee_policy_desc = (
            "taker-only (maker fee assumed 0.0)"
            if abs(fee_policy.maker_fee_rate) < 1e-9
            else "series-based"
        )
        print(f"Fee policy: {fee_policy_desc}")
        print(f"Taker fee coef: {fee_policy.taker_fee_coef}")
        print(f"Maker fee: {fee_policy.maker_fee_rate}")
        print(f"Fee source: {fee_policy.source}")
        if limit_mode != "none":
            print(
                f"Limit mode: {limit_mode}, improve={limit_improve_c}c, "
                f"fee=maker({fee_policy.maker_fee_rate}), "
                f"min_qty={limit_min_qty} (depth={depth}), min_spread={limit_min_spread}c"
            )
            print(f"Taker slippage cap (for sizing): {max_slippage_c}c")
            if taker_max_cap is not None and taker_max_cap > 0:
                print(f"Taker slippage size cap: {taker_max_cap}")
            print(
                "Recommended qty estimate: "
                f"fraction={rec_qty_fraction}, cap={rec_qty_cap if rec_qty_cap is not None else '--'} "
                "(taker depth proxy)"
            )
            if trade_state is not None:
                lookback = trade_state.get("fills_lookback_hours")
                pos_yes_total = trade_state.get("pos_yes_total", 0)
                pos_no_total = trade_state.get("pos_no_total", 0)
                resting_total = trade_state.get("resting_orders_total", 0)
                resting_no_orders = trade_state.get("resting_no_buy_orders", 0)
                resting_no_qty = trade_state.get("resting_no_buy_qty_total", 0)
                resting_yes_orders = trade_state.get("resting_yes_buy_orders", 0)
                resting_yes_qty = trade_state.get("resting_yes_buy_qty_total", 0)
                print(
                    "Trade sync: positions+resting orders"
                    f"{f', fills lookback {lookback:.1f}h' if lookback is not None else ''}"
                )
                print(
                    f"Trade sync totals: YES pos {pos_yes_total}, NO pos {pos_no_total}, "
                    f"resting orders {resting_total} "
                    f"(NO buy orders {resting_no_orders}, NO buy qty {resting_no_qty}; "
                    f"YES buy orders {resting_yes_orders}, YES buy qty {resting_yes_qty})"
                )
        print("")

    if show_tables:
        ticker_w = 28
        print(f"{'Ticker':<{ticker_w}} | YES bid (qty) | YES ask (qty) |  Mid | Spread")
        print("-" * (ticker_w + 38))
        for t in tops:
            spread = None
            if t.yes_bid_px is not None and t.yes_ask_px is not None:
                spread = t.yes_ask_px - t.yes_bid_px
            spread_s = "--" if spread is None else f"{spread}c"
            mid_s = "--" if t.yes_mid is None else f"{t.yes_mid:.1f}"
            print(
                f"{t.ticker:<{ticker_w}} | "
                f"{fmt_px(t.yes_bid_px):>3} ({fmt_qty(t.yes_bid_qty):>5}) | "
                f"{fmt_px(t.yes_ask_px):>3} ({fmt_qty(t.yes_ask_qty):>5}) | "
                f"{mid_s:>5} | {spread_s:>6}"
            )

        print("")
        print(f"{'Ticker':<{ticker_w}} |  NO bid (qty) |  NO ask (qty) |  Mid | Spread")
        print("-" * (ticker_w + 38))
        for t in tops:
            spread = None
            if t.no_bid_px is not None and t.no_ask_px is not None:
                spread = t.no_ask_px - t.no_bid_px
            spread_s = "--" if spread is None else f"{spread}c"
            mid_s = "--" if t.no_mid is None else f"{t.no_mid:.1f}"
            print(
                f"{t.ticker:<{ticker_w}} | "
                f"{fmt_px(t.no_bid_px):>3} ({fmt_qty(t.no_bid_qty):>5}) | "
                f"{fmt_px(t.no_ask_px):>3} ({fmt_qty(t.no_ask_qty):>5}) | "
                f"{mid_s:>5} | {spread_s:>6}"
            )

    sum_yes_mid = sum_or_none(t.yes_mid for t in tops)
    sum_no_mid = sum_or_none(t.no_mid for t in tops)

    if missing_yes_bid:
        print(f"Missing YES bid for: {', '.join(missing_yes_bid)}")
    if missing_yes_ask:
        print(f"Missing YES ask for: {', '.join(missing_yes_ask)}")
    if missing_no_bid:
        print(f"Missing NO bid for: {', '.join(missing_no_bid)}")
    if missing_no_ask:
        print(f"Missing NO ask for: {', '.join(missing_no_ask)}")

    me = event.get("mutually_exclusive") is True
    true_count = event.get("_true_count")
    target_yes = true_count * 100 * size if true_count is not None else None
    target_no = (len(tops) - true_count) * 100 * size if true_count is not None else None

    yes_ask_levels = [implied_asks_from_bids(t.no_bids) for t in tops]
    no_ask_levels = [implied_asks_from_bids(t.yes_bids) for t in tops]
    yes_bid_levels = [t.yes_bids for t in tops]
    no_bid_levels = [t.no_bids for t in tops]
    fee_rates = [fee_policy.taker_fee_coef for _ in tops]

    yes_ask_best = [levels[0][0] if levels else None for levels in yes_ask_levels]
    no_ask_best = [levels[0][0] if levels else None for levels in no_ask_levels]
    yes_bid_best = [levels[0][0] if levels else None for levels in yes_bid_levels]
    no_bid_best = [levels[0][0] if levels else None for levels in no_bid_levels]

    print("")
    print(f"Sum YES mids: {sum_yes_mid:.1f}" if sum_yes_mid is not None else "Sum YES mids: n/a")
    print(f"Sum NO mids: {sum_no_mid:.1f}" if sum_no_mid is not None else "Sum NO mids: n/a")

    print("")
    print("Quoted theoretical (no fees/liquidity):")
    if target_yes is not None and all(p is not None for p in yes_ask_best):
        cost_yes_buy = sum(p * size for p in yes_ask_best)
        print(f"BUY-ALL YES edge: {target_yes - cost_yes_buy:.1f}c")
    else:
        print("BUY-ALL YES edge: n/a")
    if target_yes is not None and all(p is not None for p in yes_bid_best):
        proceeds_yes_sell = sum(p * size for p in yes_bid_best)
        print(f"SELL-ALL YES edge: {proceeds_yes_sell - target_yes:.1f}c")
    else:
        print("SELL-ALL YES edge: n/a")
    if target_no is not None and all(p is not None for p in no_ask_best):
        cost_no_buy = sum(p * size for p in no_ask_best)
        print(f"BUY-ALL NO edge: {target_no - cost_no_buy:.1f}c")
    else:
        print("BUY-ALL NO edge: n/a")
    if target_no is not None and all(p is not None for p in no_bid_best):
        proceeds_no_sell = sum(p * size for p in no_bid_best)
        print(f"SELL-ALL NO edge: {proceeds_no_sell - target_no:.1f}c")
    else:
        print("SELL-ALL NO edge: n/a")

    print("")
    print(f"Net executable (fees + slippage, size={size}, depth={depth}):")
    if target_yes is not None:
        buy_yes_cost, buy_yes_ok = basket_buy_cost(yes_ask_levels, size, fee_rates)
        sell_yes_proceeds, sell_yes_ok = basket_sell_proceeds(yes_bid_levels, size, fee_rates)
        if buy_yes_ok and buy_yes_cost is not None:
            edge = target_yes - buy_yes_cost
            status = "ARB" if edge > 0 else "no arb"
            print(f"BUY-ALL YES edge: {edge:.1f}c ({status})")
        else:
            print("BUY-ALL YES edge: not executable")
        if sell_yes_ok and sell_yes_proceeds is not None:
            edge = sell_yes_proceeds - target_yes
            status = "ARB" if edge > 0 else "no arb"
            print(f"SELL-ALL YES edge: {edge:.1f}c ({status})")
        else:
            print("SELL-ALL YES edge: not executable")
    else:
        print("YES arb checks require a known true-count.")

    if target_no is not None:
        buy_no_cost, buy_no_ok = basket_buy_cost(no_ask_levels, size, fee_rates)
        sell_no_proceeds, sell_no_ok = basket_sell_proceeds(no_bid_levels, size, fee_rates)
        if buy_no_ok and buy_no_cost is not None:
            edge = target_no - buy_no_cost
            status = "ARB" if edge > 0 else "no arb"
            print(f"BUY-ALL NO edge: {edge:.1f}c ({status})")
        else:
            print("BUY-ALL NO edge: not executable")
        if sell_no_ok and sell_no_proceeds is not None:
            edge = sell_no_proceeds - target_no
            status = "ARB" if edge > 0 else "no arb"
            print(f"SELL-ALL NO edge: {edge:.1f}c ({status})")
        else:
            print("SELL-ALL NO edge: not executable")
    else:
        print("NO arb checks require a known true-count.")

    # Mid-implied expected value for BUY-ALL YES/NO on quoted legs (k-of-n adjusted).
    true_count = event.get("_true_count")
    if true_count is not None:
        implied_probs = []
        for t in tops:
            p_yes = None
            if t.yes_mid is not None:
                p_yes = t.yes_mid / 100.0
            elif t.no_mid is not None:
                p_yes = 1.0 - (t.no_mid / 100.0)
            implied_probs.append(p_yes)

        norm_probs, ok = normalize_probs_k(implied_probs, true_count)
        if not ok:
            print("Warning: probability normalization did not converge to k exactly.")
        if any(p is not None for p in norm_probs):
            k = true_count
            limit_out_lines = None
            if limit_out:
                lines = []
                lines.append(f"Event: {event.get('event_ticker', '--')}")
                lines.append(f"Title: {event.get('title', '--')}")
                lines.append(f"Series: {event.get('series_ticker', '--')}")
                lines.append(f"True count: {k}")
                lines.append(f"Size per leg: {size}")
                fee_policy_desc = (
                    "taker-only (maker fee assumed 0.0)"
                    if abs(fee_policy.maker_fee_rate) < 1e-9
                    else "series-based"
                )
                lines.append(f"Fee policy: {fee_policy_desc}")
                lines.append(f"Taker fee coef: {fee_policy.taker_fee_coef}")
                lines.append(f"Maker fee: {fee_policy.maker_fee_rate}")
                lines.append(f"Fee source: {fee_policy.source}")
                lines.append("")
                lines.append(
                    f"Limit mode: {limit_mode}, improve={limit_improve_c}c, "
                    f"min_qty={limit_min_qty} (depth={depth}), min_spread={limit_min_spread}c"
                )
                lines.append(f"Taker slippage cap (for sizing): {max_slippage_c}c")
                if taker_max_cap is not None and taker_max_cap > 0:
                    lines.append(f"Taker slippage size cap: {taker_max_cap}")
                lines.append(
                    "Recommended qty estimate: "
                    f"fraction={rec_qty_fraction}, "
                    f"cap={rec_qty_cap if rec_qty_cap is not None else '--'} "
                    "(taker depth proxy)"
                )
                if trade_state is not None:
                    lookback = trade_state.get("fills_lookback_hours")
                    pos_yes_total = trade_state.get("pos_yes_total", 0)
                    pos_no_total = trade_state.get("pos_no_total", 0)
                    resting_total = trade_state.get("resting_orders_total", 0)
                    resting_no_orders = trade_state.get("resting_no_buy_orders", 0)
                    resting_no_qty = trade_state.get("resting_no_buy_qty_total", 0)
                    resting_yes_orders = trade_state.get("resting_yes_buy_orders", 0)
                    resting_yes_qty = trade_state.get("resting_yes_buy_qty_total", 0)
                    lines.append(
                        "Trade sync: positions+resting orders"
                        f"{f', fills lookback {lookback:.1f}h' if lookback is not None else ''}"
                    )
                    lines.append(
                        f"Trade sync totals: YES pos {pos_yes_total}, NO pos {pos_no_total}, "
                        f"resting orders {resting_total} "
                        f"(NO buy orders {resting_no_orders}, NO buy qty {resting_no_qty}; "
                        f"YES buy orders {resting_yes_orders}, YES buy qty {resting_yes_qty})"
                    )
                lines.append(
                    f"Limit fees: maker={fee_policy.maker_fee_rate}; "
                    f"fallback taker={fee_policy.taker_fee_coef} (source={fee_policy.source})"
                )
                lines.append("")
                limit_out_lines = lines
            expected_payout = 0.0
            expected_cost = 0.0
            missing_payout = 0.0
            missing_prob = 0
            missing_exec = 0
            quoted_count = 0
            yes_legs = []
            for t, p, levels in zip(tops, norm_probs, yes_ask_levels):
                if p is None:
                    missing_prob += 1
                    continue
                p_yes = max(0.0, min(1.0, p))
                if levels:
                    leg_fee_rate = fee_policy.taker_fee_coef
                    cost_total, filled = net_buy_cost(levels, size, leg_fee_rate)
                    if filled < size or cost_total is None:
                        missing_exec += 1
                        missing_payout += p_yes * 100.0 * size
                        continue
                    cost_per_contract = cost_total / size
                    ask_px = levels[0][0] if levels else None
                    fee = fee_cents(leg_fee_rate, ask_px, 1) if ask_px is not None else 0
                    expected_payout += p_yes * 100.0 * size
                    expected_cost += cost_total
                    quoted_count += 1
                    yes_legs.append(
                        {
                            "ticker": t.ticker,
                            "p_yes": p_yes,
                            "price": ask_px if ask_px is not None else 0,
                            "fee": fee,
                            "fee_rate": leg_fee_rate,
                            "cost_per_contract": cost_per_contract,
                            "qty": qty_to_int(t.yes_ask_qty),
                            "ev": (p_yes * 100.0) - cost_per_contract,
                        }
                    )
                else:
                    missing_payout += p_yes * 100.0 * size

            ev = expected_payout - expected_cost
            print("")
            print(
                "BUY-ALL YES expected value (mid-implied, quoted legs): "
                f"{ev:.1f}c (payout {expected_payout:.1f}c - cost {expected_cost:.1f}c, size={size})"
            )
            if missing_payout > 0:
                print(
                    "Missing YES asks implied payout (mid-implied): "
                    f"{missing_payout:.1f}c"
                )
            if missing_prob:
                print(f"Missing implied probs: {missing_prob} legs")
                if missing_prob > len(tops) // 2:
                    print("Warning: many markets missing mid-implied probabilities.")
            if missing_exec:
                print(f"Missing executable YES asks (insufficient depth): {missing_exec} legs")

            # Partial-basket bounds using quoted legs only.
            k = true_count
            stats_all_yes = basket_stats(
                yes_legs,
                k,
                size,
                full_count=len(tops),
                side="yes",
                prob_key="p_yes",
            )
            if stats_all_yes is not None:
                payout_worst = stats_all_yes["payout_worst"]
                print(
                    f"Partial basket (YES) legs: {stats_all_yes['count']}, "
                    f"cost: {stats_all_yes['cost']:.1f}c"
                )
                print(
                    "Partial basket edges: "
                    f"best {stats_all_yes['edge_best']:.1f}c, "
                    f"worst {stats_all_yes['edge_worst']:.1f}c, EV {stats_all_yes['ev']:.1f}c"
                )
                print(
                    "Worst-case payout (k-of-n) "
                    f"(n={len(tops)}, k={k}): {payout_worst:.1f}c"
                )
                if stats_all_yes["max_size"] > 0:
                    print(
                        f"Basket max size (min YES ask qty): {stats_all_yes['max_size']}"
                    )
                    print(
                        "Basket cost range: "
                        f"min {stats_all_yes['cost']:.1f}c (1x), "
                        f"max {stats_all_yes['max_cost']:.1f}c ({stats_all_yes['max_size']}x)"
                    )
                elif limit_mode != "none":
                    print("Basket size: unknown for limit orders (resting fill)")

            # Per-leg EV contributions (mid-implied).
            if yes_legs:
                contribs = sorted(yes_legs, key=lambda x: x["ev"], reverse=True)
                print("")
                print("Top YES legs by EV (mid-implied):")
                ticker_w = 28
                print(f"{'Ticker':<{ticker_w}} |   EV | ask | fee | p_yes")
                print("-" * (ticker_w + 28))
                for leg in contribs[:10]:
                    print(
                        f"{leg['ticker']:<{ticker_w}} | "
                        f"{leg['ev']:>4.1f}c | "
                        f"{leg['price']:>3}c | "
                        f"{leg['fee']:>3}c | "
                        f"{leg['p_yes']:.3f}"
                    )

            if optimize and yes_legs:
                print("")
                print("Optimize baskets (YES, quoted legs):")
                drop_neg = [leg for leg in yes_legs if leg["ev"] >= 0]
                mid = best_middle_subset(
                    yes_legs,
                    k,
                    size,
                    full_count=len(tops),
                    side="yes",
                    prob_key="p_yes",
                )

                def show_option(name: str, subset: List[dict]) -> None:
                    stats = basket_stats(
                        subset,
                        k,
                        size,
                        full_count=len(tops),
                        side="yes",
                        prob_key="p_yes",
                    )
                    if stats is None:
                        print(f"{name}: n/a")
                        return
                    print(
                        f"{name}: legs {stats['count']}, cost {stats['cost']:.1f}c, "
                        f"best {stats['edge_best']:.1f}c, worst {stats['edge_worst']:.1f}c, "
                        f"EV {stats['ev']:.1f}c"
                    )

                show_option("Keep all", yes_legs)
                show_option("Drop negative EV", drop_neg)
                if mid is not None:
                    show_option("Middle ground", mid[0])

            if limit_mode != "none":
                limit_yes_legs = []
                total_improve_yes = 0
                total_ask_yes = 0
                for t, p, levels in zip(tops, norm_probs, yes_ask_levels):
                    if p is None:
                        continue
                    if t.yes_bid_px is None or t.yes_ask_px is None:
                        continue
                    qty_depth = (
                        t.yes_ask_qty_depth_taker_proxy
                        if t.yes_ask_qty_depth_taker_proxy is not None
                        else t.yes_ask_qty
                    )
                    qty_depth_int = qty_to_int(qty_depth)
                    if qty_depth_int <= 0:
                        continue
                    if qty_depth_int < limit_min_qty:
                        continue
                    if qty_depth_int < size:
                        continue
                    spread = t.yes_ask_px - t.yes_bid_px
                    if spread < limit_min_spread:
                        continue
                    if limit_mode == "mid":
                        if spread <= 1:
                            limit_px = t.yes_bid_px
                        else:
                            limit_px = t.yes_bid_px + (spread // 2)
                    else:
                        if spread <= 1:
                            limit_px = t.yes_bid_px
                        else:
                            improve = min(max(spread - 1, 0), max(limit_improve_c, 0))
                            limit_px = t.yes_bid_px + improve
                    warn_price_bounds(limit_px, "limit_px_yes")
                    limit_fee_rate = fee_policy.maker_fee_rate
                    fee = fee_cents(limit_fee_rate, limit_px, 1)
                    cost = limit_px + fee
                    taker_rate = fee_policy.taker_fee_coef
                    taker_cost_total, taker_filled = net_buy_cost(levels, size, taker_rate)
                    taker_cost_1, taker_fill_1 = net_buy_cost(levels, 1, taker_rate)
                    taker_cost_per_contract = (
                        (taker_cost_1 / 1.0)
                        if taker_fill_1 >= 1 and taker_cost_1 is not None
                        else None
                    )
                    ask_qty_top = qty_to_int(levels[0][1]) if levels else None
                    ask_qty_depth_proxy = (
                        sum(qty_to_int(q) for _, q in levels) if levels else None
                    )
                    immediate_fillable_taker = (
                        sum(qty_to_int(q) for px, q in levels if px <= limit_px)
                        if levels
                        else 0
                    )
                    taker_max_size_slip = taker_max_size_with_slippage(levels, max_slippage_c)
                    rec_qty_est = rec_qty_estimate(
                        ask_qty_top, ask_qty_depth_proxy, rec_qty_fraction, rec_qty_cap
                    )
                    p_yes = max(0.0, min(1.0, p))
                    limit_yes_legs.append(
                        {
                            "ticker": t.ticker,
                            "p_yes": p_yes,
                            "price": limit_px,
                            "fee": fee,
                            "fee_rate": limit_fee_rate,
                            "cost_per_contract": cost,
                            "ask_px": t.yes_ask_px,
                            "spread": spread,
                            "ask_qty_top": ask_qty_top,
                            "ask_qty_depth_proxy": ask_qty_depth_proxy,
                            "limit_qty_estimate": None,
                            "immediate_fillable_taker": immediate_fillable_taker,
                            "taker_max_size_slip": taker_max_size_slip,
                            "rec_qty_est": rec_qty_est,
                            "ev": (p_yes * 100.0) - cost,
                            "taker_cost_total": taker_cost_total
                            if taker_filled >= size
                            else None,
                            "taker_cost_per_contract": taker_cost_per_contract,
                            "taker_rate": taker_rate,
                        }
                    )
                    total_improve_yes += t.yes_ask_px - limit_px
                    total_ask_yes += t.yes_ask_px

                if trade_state is not None and limit_yes_legs:
                    pos_yes = trade_state.get("pos_yes", {})
                    pending_yes_qty = trade_state.get("pending_yes_qty", {})
                    pending_yes_cost = trade_state.get("pending_yes_cost", {})
                    fills_yes_qty = trade_state.get("fills_yes_qty", {})
                    fills_yes_cost = trade_state.get("fills_yes_cost", {})
                    for leg in limit_yes_legs:
                        ticker = leg.get("ticker")
                        if not ticker:
                            continue
                        pos = pos_yes.get(ticker, 0)
                        pos_yes_qty = max(0, int(pos))
                        pending_qty = int(pending_yes_qty.get(ticker, 0))
                        pending_cost = int(pending_yes_cost.get(ticker, 0))
                        fill_qty = int(fills_yes_qty.get(ticker, 0))
                        fill_cost = int(fills_yes_cost.get(ticker, 0))
                        fill_avg = (fill_cost / fill_qty) if fill_qty > 0 else None
                        est_cost_pc = None
                        taker_pc = leg.get("taker_cost_per_contract")
                        if taker_pc is not None:
                            est_cost_pc = float(taker_pc)
                        elif leg.get("cost_per_contract") is not None:
                            est_cost_pc = float(leg["cost_per_contract"])
                        if pos_yes_qty > 0 and fill_avg is not None:
                            filled_cost = int(round(fill_avg * pos_yes_qty))
                            filled_cost_source = "fills"
                        elif pos_yes_qty > 0 and est_cost_pc is not None:
                            filled_cost = int(round(est_cost_pc * pos_yes_qty))
                            filled_cost_source = "estimated"
                        else:
                            filled_cost = 0
                            filled_cost_source = "unknown" if pos_yes_qty > 0 else "none"
                        pos_avg_cost = None
                        if pos_yes_qty > 0 and filled_cost > 0:
                            pos_avg_cost = filled_cost / float(pos_yes_qty)
                        remaining_qty = max(0, size - pos_yes_qty - pending_qty)
                        plan_cost_per_contract = leg["cost_per_contract"]
                        planned_remaining_cost = int(round(plan_cost_per_contract * remaining_qty))
                        leg["pos_yes_qty"] = pos_yes_qty
                        leg["pending_yes_qty"] = pending_qty
                        leg["pending_yes_cost"] = pending_cost
                        leg["filled_yes_cost"] = filled_cost
                        leg["filled_cost_source"] = filled_cost_source
                        leg["pos_yes_avg_cost"] = pos_avg_cost
                        leg["remaining_qty"] = remaining_qty
                        leg["planned_remaining_cost"] = planned_remaining_cost

                stats_limit_yes = basket_stats(
                    limit_yes_legs,
                    k,
                    size,
                    full_count=len(tops),
                    side="yes",
                    prob_key="p_yes",
                )
                if stats_limit_yes is not None:
                    payout_best = stats_limit_yes["payout_best"]
                    payout_worst = stats_limit_yes["payout_worst"]
                    limit_expected_payout = sum(
                        leg["p_yes"] * 100.0 * size for leg in limit_yes_legs
                    )
                    maker_worst = stats_limit_yes["edge_worst"]
                    maker_best = stats_limit_yes["edge_best"]
                    maker_ev = stats_limit_yes["ev"]

                    yes_taker_ok = fallback_taker and all(
                        leg.get("taker_cost_total") is not None for leg in limit_yes_legs
                    )
                    yes_taker_cost_total = None
                    yes_ask_edge_worst = None
                    if yes_taker_ok:
                        yes_taker_cost_total = sum(
                            leg["taker_cost_total"] for leg in limit_yes_legs  # type: ignore
                        )
                    print("")
                    print("Limit basket (YES, quoted bid/ask legs):")
                    print(
                        f"Legs {stats_limit_yes['count']}, cost {stats_limit_yes['cost']:.1f}c, "
                        f"best {maker_best:.1f}c, worst {maker_worst:.1f}c, "
                        f"EV {maker_ev:.1f}c"
                    )
                    print(
                        "Worst-case payout (k-of-n) "
                        f"(n={len(tops)}, k={k}): {payout_worst:.1f}c"
                    )
                    if yes_taker_ok and yes_taker_cost_total is not None:
                        yes_ask_edge_best = payout_best - yes_taker_cost_total
                        yes_ask_edge_worst = payout_worst - yes_taker_cost_total
                        yes_ask_ev = limit_expected_payout - yes_taker_cost_total
                        print(
                            "Fallback at ask (taker, slippage+fees): "
                            f"cost {yes_taker_cost_total:.1f}c, best {yes_ask_edge_best:.1f}c, "
                            f"worst {yes_ask_edge_worst:.1f}c, EV {yes_ask_ev:.1f}c"
                        )
                        if assumed_fill_ratio is not None:
                            ratio = max(0.0, min(1.0, assumed_fill_ratio))
                            blended_cost = (ratio * stats_limit_yes["cost"]) + (
                                (1.0 - ratio) * yes_taker_cost_total
                            )
                            blended_worst = payout_worst - blended_cost
                            print(
                                f"Blended worst-case (fill_ratio={ratio:.2f}): {blended_worst:.1f}c"
                            )
                    else:
                        print("Fallback at ask (taker): not executable")
                        yes_ask_edge_worst = None
                    if total_ask_yes > 0:
                        print(
                            "Limit price improvement: "
                            f"{total_improve_yes}c total vs ask "
                            f"(avg {total_improve_yes / stats_limit_yes['count']:.2f}c)"
                        )
                    if abs(maker_best - (payout_best - stats_limit_yes["cost"])) > 1e-6:
                        print("Warning: maker edge_best inconsistency", file=sys.stderr)
                    if abs(maker_worst - (payout_worst - stats_limit_yes["cost"])) > 1e-6:
                        print("Warning: maker edge_worst inconsistency", file=sys.stderr)
                    immediate_sizes = [
                        leg.get("immediate_fillable_taker")
                        for leg in limit_yes_legs
                        if leg.get("immediate_fillable_taker") is not None
                    ]
                    immediate_min = min(immediate_sizes) if immediate_sizes else None
                    if immediate_min is None:
                        print("Immediate fillable (taker proxy, at limit): unknown")
                    elif immediate_min <= 0:
                        print("Immediate fillable (taker proxy, at limit): 0")
                    else:
                        print(f"Immediate fillable (taker proxy, at limit): {immediate_min}")
                    rec_sizes = [
                        leg.get("rec_qty_est")
                        for leg in limit_yes_legs
                        if leg.get("rec_qty_est") is not None
                    ]
                    rec_min = min(rec_sizes) if rec_sizes else None
                    if rec_sizes:
                        print(
                            "Recommended qty estimate (taker depth proxy, not fillable): "
                            f"{rec_min}"
                        )
                    taker_slip_sizes = [
                        leg.get("taker_max_size_slip")
                        for leg in limit_yes_legs
                        if leg.get("taker_max_size_slip") is not None
                    ]
                    taker_slip_min = min(taker_slip_sizes) if taker_slip_sizes else None
                    if taker_slip_min is not None and taker_max_cap is not None and taker_max_cap > 0:
                        taker_slip_min = min(taker_slip_min, taker_max_cap)
                    if taker_slip_min is not None:
                        cap_label = taker_max_cap if taker_max_cap is not None else "--"
                        print(
                            "Taker slippage max size (cap "
                            f"{cap_label}, max_slippage {max_slippage_c}c): {taker_slip_min}"
                        )
                    remaining_qty_total = None
                    remaining_budget_per_contract = None
                    remaining_qty_max = None
                    if trade_state is not None and limit_yes_legs:
                        filled_cost_total = sum(
                            int(leg.get("filled_yes_cost") or 0) for leg in limit_yes_legs
                        )
                        pending_cost_total = sum(
                            int(leg.get("pending_yes_cost") or 0) for leg in limit_yes_legs
                        )
                        remaining_cost_total = sum(
                            int(leg.get("planned_remaining_cost") or 0) for leg in limit_yes_legs
                        )
                        remaining_qty_total = sum(
                            int(leg.get("remaining_qty") or 0) for leg in limit_yes_legs
                        )
                        remaining_vals = [int(leg.get("remaining_qty") or 0) for leg in limit_yes_legs]
                        if remaining_vals:
                            remaining_qty_min = min(remaining_vals)
                            remaining_qty_max = max(remaining_vals)
                            print(
                                "Remaining qty across legs: "
                                f"min {remaining_qty_min}, max {remaining_qty_max}"
                            )
                        cost_est_total = filled_cost_total + pending_cost_total + remaining_cost_total
                        edge_worst_est = payout_worst - cost_est_total
                        print(
                            "Adjusted (filled+resting+planned) worst-case edge: "
                            f"{edge_worst_est:.1f}c"
                        )
                        if remaining_qty_total > 0:
                            budget_remaining = payout_worst - filled_cost_total - pending_cost_total
                            remaining_budget_per_contract = budget_remaining / remaining_qty_total
                            print(
                                "Max all-in cost per remaining contract (to keep worst >= 0): "
                                f"{remaining_budget_per_contract:.1f}c"
                            )
                    if trade_state is not None:
                        print("")
                        print("Action guidance:")
                        pos_yes_total = trade_state.get("pos_yes_total", 0)
                        pending_yes_qty_total = trade_state.get("resting_yes_buy_qty_total", 0)
                        if remaining_qty_max == 0:
                            if pending_yes_qty_total > 0:
                                print(
                                    "- Fully covered for target size. Consider canceling remaining "
                                    "resting orders if you don't want more exposure."
                                )
                            else:
                                print("- Fully covered for target size. No action needed.")
                        else:
                            if (
                                yes_taker_ok
                                and yes_ask_edge_worst is not None
                                and yes_ask_edge_worst >= 0
                            ):
                                if remaining_budget_per_contract is not None:
                                    over_budget = []
                                    for leg in limit_yes_legs:
                                        cost_pc = leg.get("taker_cost_per_contract")
                                        if cost_pc is None:
                                            continue
                                        if cost_pc > remaining_budget_per_contract + 1e-9:
                                            over_budget.append(leg["ticker"])
                                    if not over_budget:
                                        print(
                                            "- SAFE with taker fallback. Taking remaining at taker "
                                            "should keep worst-case non-negative."
                                        )
                                    else:
                                        print(
                                            "- SAFE, but some legs exceed remaining budget at taker: "
                                            + ", ".join(over_budget)
                                        )
                                else:
                                    print(
                                        "- SAFE with taker fallback. Taking remaining at taker should be OK."
                                    )
                            else:
                                print(
                                    "- UNSAFE taker fallback. Consider canceling resting orders and "
                                    "reducing exposure if you want to exit."
                                )
                        if pending_yes_qty_total > 0:
                            print("")
                            print("Resting order price check (YES buy):")
                            ticker_w = 28
                            print(f"{'Ticker':<{ticker_w}} | limit | best ask | delta | status")
                            print("-" * (ticker_w + 36))
                            for leg in limit_yes_legs:
                                pending_qty = int(leg.get("pending_yes_qty") or 0)
                                if pending_qty <= 0:
                                    continue
                                limit_px = leg.get("price")
                                ask_px = leg.get("ask_px")
                                if limit_px is None or ask_px is None:
                                    continue
                                delta = ask_px - limit_px
                                status = "good" if delta >= 0 else "overpay"
                                print(
                                    f"{leg['ticker']:<{ticker_w}} | "
                                    f"{limit_px:>5}c | "
                                    f"{ask_px:>8}c | "
                                    f"{delta:>5}c | "
                                    f"{status}"
                                )
                        if limit_yes_legs:
                            has_estimated = any(
                                leg.get("filled_cost_source") == "estimated"
                                for leg in limit_yes_legs
                            )
                            has_missing = any(
                                leg.get("filled_cost_source") == "unknown"
                                for leg in limit_yes_legs
                            )
                            payout_a = 0.0
                            cost_a = 0.0
                            for leg in limit_yes_legs:
                                qty = int(leg.get("pos_yes_qty") or 0) + int(
                                    leg.get("pending_yes_qty") or 0
                                )
                                if qty <= 0:
                                    continue
                                payout_a += leg["p_yes"] * 100.0 * qty
                                cost_a += float(leg.get("filled_yes_cost") or 0)
                                cost_a += float(leg.get("pending_yes_cost") or 0)
                            if payout_a > 0 or cost_a > 0:
                                ev_a = payout_a - cost_a
                                note_a = ""
                                if has_missing:
                                    note_a = " (filled cost missing)"
                                elif has_estimated:
                                    note_a = " (filled cost estimated)"
                                print(
                                    "EV if all resting orders fill (no new orders): "
                                    f"{ev_a:.1f}c (payout {payout_a:.1f}c - cost {cost_a:.1f}c){note_a}"
                                )
                            payout_c = 0.0
                            cost_c = 0.0
                            pos_qtys = []
                            ev_c = None
                            worst_edge_c = None
                            for leg in limit_yes_legs:
                                pos_qty = int(leg.get("pos_yes_qty") or 0)
                                if pos_qty <= 0:
                                    continue
                                payout_c += leg["p_yes"] * 100.0 * pos_qty
                                cost_c += float(leg.get("filled_yes_cost") or 0)
                                pos_qtys.append(pos_qty)
                            if pos_qtys:
                                ev_c = payout_c - cost_c
                                note_c = ""
                                if has_missing:
                                    note_c = " (filled cost missing)"
                                elif has_estimated:
                                    note_c = " (filled cost estimated)"
                                print(
                                    "EV if cancel resting and do nothing (positions only): "
                                    f"{ev_c:.1f}c (payout {payout_c:.1f}c - cost {cost_c:.1f}c){note_c}"
                                )
                                pos_qtys_sorted = sorted(pos_qtys)
                                n = len(tops)
                                m = len(pos_qtys_sorted)
                                min_yes = max(0, k - (n - m))
                                min_yes = min(min_yes, m)
                                worst_qty = sum(pos_qtys_sorted[:min_yes]) if min_yes > 0 else 0
                                payout_worst_c = worst_qty * 100.0
                                worst_edge_c = payout_worst_c - cost_c
                                print(
                                    "Worst-case (k-of-n) if cancel resting (positions only): "
                                    f"{worst_edge_c:.1f}c (payout {payout_worst_c:.1f}c - cost {cost_c:.1f}c){note_c}"
                                )
                            payout_b = 0.0
                            cost_b = 0.0
                            ok_b = True
                            for leg in limit_yes_legs:
                                pos_qty = int(leg.get("pos_yes_qty") or 0)
                                remaining = max(0, size - pos_qty)
                                payout_b += leg["p_yes"] * 100.0 * size
                                cost_b += float(leg.get("filled_yes_cost") or 0)
                                cost_pc = leg.get("taker_cost_per_contract")
                                if remaining > 0:
                                    if cost_pc is None:
                                        ok_b = False
                                        break
                                    cost_b += float(cost_pc) * remaining
                            if ok_b:
                                ev_b = payout_b - cost_b
                                note_b = ""
                                if has_missing:
                                    note_b = " (filled cost missing)"
                                elif has_estimated:
                                    note_b = " (filled cost estimated)"
                                print(
                                    "EV if cancel resting and take remaining at taker now: "
                                    f"{ev_b:.1f}c (payout {payout_b:.1f}c - cost {cost_b:.1f}c){note_b}"
                                )
                                payout_worst_b = payout_worst
                                worst_edge_b = payout_worst_b - cost_b
                                print(
                                    "Worst-case (k-of-n) if cancel resting and take remaining at taker: "
                                    f"{worst_edge_b:.1f}c (payout {payout_worst_b:.1f}c - cost {cost_b:.1f}c){note_b}"
                                )
                                if ev_c is not None and worst_edge_c is not None:
                                    delta_ev = ev_b - ev_c
                                    delta_worst = worst_edge_b - worst_edge_c
                                    print(
                                        "Delta vs positions-only (take remaining at taker): "
                                        f"EV {delta_ev:+.1f}c, worst-case {delta_worst:+.1f}c"
                                    )
                    if limit_out_lines is not None:
                        lines = limit_out_lines
                        lines.append("Limit basket summary (YES):")
                        lines.append(
                            f"Legs {stats_limit_yes['count']}, cost {stats_limit_yes['cost']:.1f}c, "
                            f"best {maker_best:.1f}c, worst {maker_worst:.1f}c, "
                            f"EV {maker_ev:.1f}c"
                        )
                        lines.append(
                            "Worst-case payout (k-of-n) "
                            f"(n={len(tops)}, k={k}): {payout_worst:.1f}c"
                        )
                        if yes_taker_ok and yes_taker_cost_total is not None:
                            yes_ask_edge_best = payout_best - yes_taker_cost_total
                            yes_ask_edge_worst = payout_worst - yes_taker_cost_total
                            yes_ask_ev = limit_expected_payout - yes_taker_cost_total
                            lines.append(
                                "Fallback at ask (taker, slippage+fees): "
                                f"cost {yes_taker_cost_total:.1f}c, best {yes_ask_edge_best:.1f}c, "
                                f"worst {yes_ask_edge_worst:.1f}c, EV {yes_ask_ev:.1f}c"
                            )
                            if assumed_fill_ratio is not None:
                                ratio = max(0.0, min(1.0, assumed_fill_ratio))
                                blended_cost = (ratio * stats_limit_yes["cost"]) + (
                                    (1.0 - ratio) * yes_taker_cost_total
                                )
                                blended_worst = payout_worst - blended_cost
                                lines.append(
                                    f"Blended worst-case (fill_ratio={ratio:.2f}): {blended_worst:.1f}c"
                                )
                        else:
                            lines.append("Fallback at ask (taker): not executable")
                            yes_ask_edge_worst = None
                        if require_taker_fallback:
                            if yes_taker_ok and yes_ask_edge_worst is not None:
                                safe = yes_ask_edge_worst >= 0
                                safety_label = "SAFE" if safe else "UNSAFE (taker fallback negative)"
                            else:
                                safety_label = "UNSAFE (taker fallback not executable)"
                            lines.append(f"Safety: {safety_label}")
                        lines.append(
                            f"Limit price improvement: {total_improve_yes}c total vs ask "
                            f"(avg {total_improve_yes / stats_limit_yes['count']:.2f}c)"
                        )
                        if immediate_min is None:
                            lines.append("Immediate fillable (taker proxy, at limit): unknown")
                        elif immediate_min <= 0:
                            lines.append("Immediate fillable (taker proxy, at limit): 0")
                        else:
                            lines.append(
                                f"Immediate fillable (taker proxy, at limit): {immediate_min}"
                            )
                        if rec_sizes:
                            lines.append(
                                "Recommended qty estimate (taker depth proxy, not fillable): "
                                f"{rec_min}"
                            )
                        if taker_slip_min is not None:
                            cap_label = taker_max_cap if taker_max_cap is not None else "--"
                            lines.append(
                                "Taker slippage max size (cap "
                                f"{cap_label}, max_slippage {max_slippage_c}c): {taker_slip_min}"
                            )
                        if trade_state is not None and remaining_qty_total is not None:
                            lines.append(
                                "Adjusted (filled+resting+planned) worst-case edge: "
                                f"{edge_worst_est:.1f}c"
                            )
                            if remaining_budget_per_contract is not None:
                                lines.append(
                                    "Max all-in cost per remaining contract "
                                    "(to keep worst >= 0): "
                                    f"{remaining_budget_per_contract:.1f}c"
                                )
                        pending_yes_qty_total = (
                            trade_state.get("resting_yes_buy_qty_total", 0)
                            if trade_state is not None
                            else 0
                        )
                        if trade_state is not None:
                            lines.append("")
                            lines.append("Action guidance:")
                            if remaining_qty_max == 0:
                                if pending_yes_qty_total > 0:
                                    lines.append(
                                        "- Fully covered for target size. Consider canceling remaining "
                                        "resting orders if you don't want more exposure."
                                    )
                                else:
                                    lines.append("- Fully covered for target size. No action needed.")
                            else:
                                if (
                                    yes_taker_ok
                                    and yes_ask_edge_worst is not None
                                    and yes_ask_edge_worst >= 0
                                ):
                                    if remaining_budget_per_contract is not None:
                                        over_budget = []
                                        for leg in limit_yes_legs:
                                            cost_pc = leg.get("taker_cost_per_contract")
                                            if cost_pc is None:
                                                continue
                                            if cost_pc > remaining_budget_per_contract + 1e-9:
                                                over_budget.append(leg["ticker"])
                                        if not over_budget:
                                            lines.append(
                                                "- SAFE with taker fallback. Taking remaining at taker "
                                                "should keep worst-case non-negative."
                                            )
                                        else:
                                            lines.append(
                                                "- SAFE, but some legs exceed remaining budget at taker: "
                                                + ", ".join(over_budget)
                                            )
                                    else:
                                        lines.append(
                                            "- SAFE with taker fallback. Taking remaining at taker should be OK."
                                        )
                                else:
                                    lines.append(
                                        "- UNSAFE taker fallback. Consider canceling resting orders and "
                                        "reducing exposure if you want to exit."
                                    )
                            lines.append("")
                            lines.append("Positions (YES) by market:")
                            ticker_w = 28
                            lines.append(f"{'Ticker':<{ticker_w}} | pos | avg_cost | source")
                            lines.append("-" * (ticker_w + 27))
                            any_pos = False
                            for leg in limit_yes_legs:
                                pos_qty = int(leg.get("pos_yes_qty") or 0)
                                if pos_qty <= 0:
                                    continue
                                any_pos = True
                                avg_cost = leg.get("pos_yes_avg_cost")
                                avg_s = "--" if avg_cost is None else f"{avg_cost:.1f}c"
                                src = leg.get("filled_cost_source") or "--"
                                lines.append(
                                    f"{leg['ticker']:<{ticker_w}} | "
                                    f"{pos_qty:>3} | {avg_s:>7} | {src}"
                                )
                            if not any_pos:
                                lines.append("(none)")
                            no_pos_no_pending = [
                                leg["ticker"]
                                for leg in limit_yes_legs
                                if int(leg.get("pos_yes_qty") or 0) <= 0
                                and int(leg.get("pending_yes_qty") or 0) <= 0
                                and int(leg.get("remaining_qty") or 0) > 0
                            ]
                            if no_pos_no_pending:
                                lines.append("")
                                lines.append(
                                    "Recommended markets with no position or resting orders:"
                                )
                                for t in no_pos_no_pending:
                                    lines.append(f"- {t}")
                        if pending_yes_qty_total > 0:
                            lines.append("")
                            lines.append("Resting order price check (YES buy):")
                            ticker_w = 28
                            lines.append(f"{'Ticker':<{ticker_w}} | limit | best ask | delta | status")
                            lines.append("-" * (ticker_w + 36))
                            for leg in limit_yes_legs:
                                pending_qty = int(leg.get("pending_yes_qty") or 0)
                                if pending_qty <= 0:
                                    continue
                                limit_px = leg.get("price")
                                ask_px = leg.get("ask_px")
                                if limit_px is None or ask_px is None:
                                    continue
                                delta = ask_px - limit_px
                                status = "good" if delta >= 0 else "overpay"
                                lines.append(
                                    f"{leg['ticker']:<{ticker_w}} | "
                                    f"{limit_px:>5}c | "
                                    f"{ask_px:>8}c | "
                                    f"{delta:>5}c | "
                                    f"{status}"
                                )
                        if limit_yes_legs:
                            has_estimated = any(
                                leg.get("filled_cost_source") == "estimated"
                                for leg in limit_yes_legs
                            )
                            has_missing = any(
                                leg.get("filled_cost_source") == "unknown"
                                for leg in limit_yes_legs
                            )
                            payout_a = 0.0
                            cost_a = 0.0
                            for leg in limit_yes_legs:
                                qty = int(leg.get("pos_yes_qty") or 0) + int(
                                    leg.get("pending_yes_qty") or 0
                                )
                                if qty <= 0:
                                    continue
                                payout_a += leg["p_yes"] * 100.0 * qty
                                cost_a += float(leg.get("filled_yes_cost") or 0)
                                cost_a += float(leg.get("pending_yes_cost") or 0)
                            if payout_a > 0 or cost_a > 0:
                                ev_a = payout_a - cost_a
                                note_a = ""
                                if has_missing:
                                    note_a = " (filled cost missing)"
                                elif has_estimated:
                                    note_a = " (filled cost estimated)"
                                lines.append(
                                    "EV if all resting orders fill (no new orders): "
                                    f"{ev_a:.1f}c (payout {payout_a:.1f}c - cost {cost_a:.1f}c){note_a}"
                                )
                            payout_c = 0.0
                            cost_c = 0.0
                            pos_qtys = []
                            ev_c = None
                            worst_edge_c = None
                            for leg in limit_yes_legs:
                                pos_qty = int(leg.get("pos_yes_qty") or 0)
                                if pos_qty <= 0:
                                    continue
                                payout_c += leg["p_yes"] * 100.0 * pos_qty
                                cost_c += float(leg.get("filled_yes_cost") or 0)
                                pos_qtys.append(pos_qty)
                            if pos_qtys:
                                ev_c = payout_c - cost_c
                                note_c = ""
                                if has_missing:
                                    note_c = " (filled cost missing)"
                                elif has_estimated:
                                    note_c = " (filled cost estimated)"
                                lines.append(
                                    "EV if cancel resting and do nothing (positions only): "
                                    f"{ev_c:.1f}c (payout {payout_c:.1f}c - cost {cost_c:.1f}c){note_c}"
                                )
                                pos_qtys_sorted = sorted(pos_qtys)
                                n = len(tops)
                                m = len(pos_qtys_sorted)
                                min_yes = max(0, k - (n - m))
                                min_yes = min(min_yes, m)
                                worst_qty = sum(pos_qtys_sorted[:min_yes]) if min_yes > 0 else 0
                                payout_worst_c = worst_qty * 100.0
                                worst_edge_c = payout_worst_c - cost_c
                                lines.append(
                                    "Worst-case (k-of-n) if cancel resting (positions only): "
                                    f"{worst_edge_c:.1f}c (payout {payout_worst_c:.1f}c - cost {cost_c:.1f}c){note_c}"
                                )
                            payout_b = 0.0
                            cost_b = 0.0
                            ok_b = True
                            for leg in limit_yes_legs:
                                pos_qty = int(leg.get("pos_yes_qty") or 0)
                                remaining = max(0, size - pos_qty)
                                payout_b += leg["p_yes"] * 100.0 * size
                                cost_b += float(leg.get("filled_yes_cost") or 0)
                                cost_pc = leg.get("taker_cost_per_contract")
                                if remaining > 0:
                                    if cost_pc is None:
                                        ok_b = False
                                        break
                                    cost_b += float(cost_pc) * remaining
                            if ok_b:
                                ev_b = payout_b - cost_b
                                note_b = ""
                                if has_missing:
                                    note_b = " (filled cost missing)"
                                elif has_estimated:
                                    note_b = " (filled cost estimated)"
                                lines.append(
                                    "EV if cancel resting and take remaining at taker now: "
                                    f"{ev_b:.1f}c (payout {payout_b:.1f}c - cost {cost_b:.1f}c){note_b}"
                                )
                                payout_worst_b = payout_worst
                                worst_edge_b = payout_worst_b - cost_b
                                lines.append(
                                    "Worst-case (k-of-n) if cancel resting and take remaining at taker: "
                                    f"{worst_edge_b:.1f}c (payout {payout_worst_b:.1f}c - cost {cost_b:.1f}c){note_b}"
                                )
                                if ev_c is not None and worst_edge_c is not None:
                                    delta_ev = ev_b - ev_c
                                    delta_worst = worst_edge_b - worst_edge_c
                                    lines.append(
                                        "Delta vs positions-only (take remaining at taker): "
                                        f"EV {delta_ev:+.1f}c, worst-case {delta_worst:+.1f}c"
                                    )
                        lines.append("")
                        lines.append("YES taker ask levels (qty > 0):")
                        ticker_w = 28
                        lines.append(f"{'Ticker':<{ticker_w}} | Levels")
                        lines.append("-" * (ticker_w + 9))
                        for t, levels in zip(tops, yes_ask_levels):
                            levels_s = fmt_levels(levels)
                            if levels_s == "--":
                                continue
                            lines.append(f"{t.ticker:<{ticker_w}} | {levels_s}")
                        lines.append("")
                        lines.append("Order list (YES limit buys):")
                        for leg in sorted(limit_yes_legs, key=lambda x: x["ev"], reverse=True):
                            ask_top = leg.get("ask_qty_top")
                            ask_depth = leg.get("ask_qty_depth_proxy")
                            fillable_leg = leg.get("immediate_fillable_taker")
                            rec_leg = leg.get("rec_qty_est")
                            pos_yes = leg.get("pos_yes_qty")
                            pending_yes = leg.get("pending_yes_qty")
                            remaining_yes = leg.get("remaining_qty")
                            max_all_in = (
                                remaining_budget_per_contract
                                if remaining_budget_per_contract is not None and remaining_yes
                                else None
                            )
                            lines.append(
                                f"{leg['ticker']:<30} | "
                                f"limit {leg['price']:>3}c | "
                                f"ask {leg['ask_px']:>3}c | "
                                f"spread {leg['spread']:>2}c | "
                                f"fee {leg['fee']:>2}c | "
                                f"p_yes {leg['p_yes']:.3f} | "
                                f"ask_qty_top {ask_top if ask_top is not None else '--':>4} | "
                                f"ask_qty_depth_proxy {ask_depth if ask_depth is not None else '--':>4} | "
                                f"immediate_fillable_taker {fillable_leg if fillable_leg is not None else '--':>3} | "
                                f"rec_qty_est {rec_leg if rec_leg is not None else '--':>3} | "
                                f"pos_yes {pos_yes if pos_yes is not None else '--':>3} | "
                                f"pending_yes {pending_yes if pending_yes is not None else '--':>3} | "
                                f"remaining {remaining_yes if remaining_yes is not None else '--':>3} | "
                                f"max_all_in {f'{max_all_in:.1f}' if max_all_in is not None else '--':>5} | "
                                f"EV {leg['ev']:.1f}c"
                            )
                        lines.append("")
                        lines.append("Notes:")
                        lines.append(
                            "Expected payout uses mid-implied probabilities normalized to k-of-n."
                        )
                        lines.append(
                            "Worst case assumes as few YES outcomes as possible are inside your basket."
                        )
                        lines.append("")
                print("")
                print("YES taker ask levels (qty > 0):")
                ticker_w = 28
                print(f"{'Ticker':<{ticker_w}} | Levels")
                print("-" * (ticker_w + 9))
                for t, levels in zip(tops, yes_ask_levels):
                    levels_s = fmt_levels(levels)
                    if levels_s == "--":
                        continue
                    print(f"{t.ticker:<{ticker_w}} | {levels_s}")

            expected_payout = 0.0
            expected_cost = 0.0
            missing_payout = 0.0
            missing_prob = 0
            missing_exec = 0
            quoted_count = 0
            legs = []
            for t, p, levels in zip(tops, norm_probs, no_ask_levels):
                if p is None:
                    missing_prob += 1
                    continue
                p_yes = max(0.0, min(1.0, p))
                p_no = 1.0 - p_yes
                if levels:
                    leg_fee_rate = fee_policy.taker_fee_coef
                    cost_total, filled = net_buy_cost(levels, size, leg_fee_rate)
                    if filled < size or cost_total is None:
                        missing_exec += 1
                        missing_payout += p_no * 100.0 * size
                        continue
                    cost_per_contract = cost_total / size
                    ask_px = levels[0][0] if levels else None
                    fee = fee_cents(leg_fee_rate, ask_px, 1) if ask_px is not None else 0
                    expected_payout += p_no * 100.0 * size
                    expected_cost += cost_total
                    quoted_count += 1
                    legs.append(
                        {
                            "ticker": t.ticker,
                            "p_no": p_no,
                            "price": ask_px if ask_px is not None else 0,
                            "fee": fee,
                            "fee_rate": leg_fee_rate,
                            "cost_per_contract": cost_per_contract,
                            "qty": qty_to_int(t.no_ask_qty),
                            "ev": (p_no * 100.0) - cost_per_contract,
                        }
                    )
                else:
                    missing_payout += p_no * 100.0 * size

            ev = expected_payout - expected_cost
            print("")
            print(
                "BUY-ALL NO expected value (mid-implied, quoted legs): "
                f"{ev:.1f}c (payout {expected_payout:.1f}c - cost {expected_cost:.1f}c, size={size})"
            )
            if missing_payout > 0:
                print(
                    "Missing NO asks implied payout (mid-implied): "
                    f"{missing_payout:.1f}c"
                )
            if missing_prob:
                print(f"Missing implied probs: {missing_prob} legs")
                if missing_prob > len(tops) // 2:
                    print("Warning: many markets missing mid-implied probabilities.")
            if missing_exec:
                print(f"Missing executable NO asks (insufficient depth): {missing_exec} legs")

            # Partial-basket bounds using quoted legs only.
            k = true_count
            stats_all = basket_stats(legs, k, size, full_count=len(tops))
            if stats_all is not None:
                payout_worst = stats_all["payout_worst"]
                print(
                    f"Partial basket (NO) legs: {stats_all['count']}, cost: {stats_all['cost']:.1f}c"
                )
                print(
                    "Partial basket edges: "
                    f"best {stats_all['edge_best']:.1f}c, "
                    f"worst {stats_all['edge_worst']:.1f}c, EV {stats_all['ev']:.1f}c"
                )
                print(
                    "Worst-case payout (n-k) "
                    f"(n={stats_all['count']}, k={k}): {payout_worst:.1f}c"
                )
                if stats_all["max_size"] > 0:
                    print(
                        f"Basket max size (min NO ask qty): {stats_all['max_size']}"
                    )
                    print(
                        "Basket cost range: "
                        f"min {stats_all['cost']:.1f}c (1x), "
                        f"max {stats_all['max_cost']:.1f}c ({stats_all['max_size']}x)"
                    )
                elif limit_mode != "none":
                    print("Basket size: unknown for limit orders (resting fill)")

            # Per-leg EV contributions (mid-implied).
            if legs:
                contribs = sorted(legs, key=lambda x: x["ev"], reverse=True)
                print("")
                print("Top NO legs by EV (mid-implied):")
                ticker_w = 28
                print(f"{'Ticker':<{ticker_w}} |   EV | ask | fee |  p_no")
                print("-" * (ticker_w + 27))
                for leg in contribs[:10]:
                    print(
                        f"{leg['ticker']:<{ticker_w}} | "
                        f"{leg['ev']:>4.1f}c | "
                        f"{leg['price']:>3}c | "
                        f"{leg['fee']:>3}c | "
                        f"{leg['p_no']:.3f}"
                    )

            if optimize and legs:
                print("")
                print("Optimize baskets (NO, quoted legs):")
                drop_neg = [leg for leg in legs if leg["ev"] >= 0]
                mid = best_middle_subset(legs, k, size, full_count=len(tops))

                def show_option(name: str, subset: List[dict]) -> None:
                    stats = basket_stats(subset, k, size, full_count=len(tops))
                    if stats is None:
                        print(f"{name}: n/a")
                        return
                    print(
                        f"{name}: legs {stats['count']}, cost {stats['cost']:.1f}c, "
                        f"best {stats['edge_best']:.1f}c, worst {stats['edge_worst']:.1f}c, "
                        f"EV {stats['ev']:.1f}c"
                    )

                show_option("Keep all", legs)
                show_option("Drop negative EV", drop_neg)
                if mid is not None:
                    show_option("Middle ground", mid[0])

            if limit_mode != "none":
                limit_legs = []
                total_improve = 0
                total_ask = 0
                for t, p, levels in zip(tops, norm_probs, no_ask_levels):
                    if p is None:
                        continue
                    if t.no_bid_px is None or t.no_ask_px is None:
                        continue
                    qty_depth = (
                        t.no_ask_qty_depth_taker_proxy
                        if t.no_ask_qty_depth_taker_proxy is not None
                        else t.no_ask_qty
                    )
                    qty_depth_int = qty_to_int(qty_depth)
                    if qty_depth_int <= 0:
                        continue
                    if qty_depth_int < limit_min_qty:
                        continue
                    if qty_depth_int < size:
                        continue
                    spread = t.no_ask_px - t.no_bid_px
                    if spread < limit_min_spread:
                        continue
                    if limit_mode == "mid":
                        if spread <= 1:
                            limit_px = t.no_bid_px
                        else:
                            limit_px = t.no_bid_px + (spread // 2)
                    else:
                        if spread <= 1:
                            limit_px = t.no_bid_px
                        else:
                            improve = min(max(spread - 1, 0), max(limit_improve_c, 0))
                            limit_px = t.no_bid_px + improve
                    warn_price_bounds(limit_px, "limit_px")
                    limit_fee_rate = fee_policy.maker_fee_rate
                    fee = fee_cents(limit_fee_rate, limit_px, 1)
                    cost = limit_px + fee
                    taker_rate = fee_policy.taker_fee_coef
                    taker_cost_total, taker_filled = net_buy_cost(levels, size, taker_rate)
                    taker_cost_1, taker_fill_1 = net_buy_cost(levels, 1, taker_rate)
                    taker_cost_per_contract = (
                        (taker_cost_1 / 1.0) if taker_fill_1 >= 1 and taker_cost_1 is not None else None
                    )
                    ask_qty_top = qty_to_int(levels[0][1]) if levels else None
                    ask_qty_depth_proxy = (
                        sum(qty_to_int(q) for _, q in levels) if levels else None
                    )
                    immediate_fillable_taker = (
                        sum(qty_to_int(q) for px, q in levels if px <= limit_px) if levels else 0
                    )
                    taker_max_size_slip = taker_max_size_with_slippage(levels, max_slippage_c)
                    rec_qty_est = rec_qty_estimate(
                        ask_qty_top, ask_qty_depth_proxy, rec_qty_fraction, rec_qty_cap
                    )
                    p_yes = max(0.0, min(1.0, p))
                    p_no = 1.0 - p_yes
                    limit_legs.append(
                        {
                            "ticker": t.ticker,
                            "p_no": p_no,
                            "price": limit_px,
                            "fee": fee,
                            "fee_rate": limit_fee_rate,
                            "cost_per_contract": cost,
                            "ask_px": t.no_ask_px,
                            "spread": spread,
                            "ask_qty_top": ask_qty_top,
                            "ask_qty_depth_proxy": ask_qty_depth_proxy,
                            "limit_qty_estimate": None,
                            "immediate_fillable_taker": immediate_fillable_taker,
                            "taker_max_size_slip": taker_max_size_slip,
                            "rec_qty_est": rec_qty_est,
                            "ev": (p_no * 100.0) - cost,
                            "taker_cost_total": taker_cost_total if taker_filled >= size else None,
                            "taker_cost_per_contract": taker_cost_per_contract,
                            "taker_rate": taker_rate,
                        }
                    )
                    total_improve += t.no_ask_px - limit_px
                    total_ask += t.no_ask_px

                if trade_state is not None and limit_legs:
                    pos_yes = trade_state.get("pos_yes", {})
                    pending_no_qty = trade_state.get("pending_no_qty", {})
                    pending_no_cost = trade_state.get("pending_no_cost", {})
                    fills_no_qty = trade_state.get("fills_no_qty", {})
                    fills_no_cost = trade_state.get("fills_no_cost", {})
                    for leg in limit_legs:
                        ticker = leg.get("ticker")
                        if not ticker:
                            continue
                        pos = pos_yes.get(ticker, 0)
                        pos_no_qty = max(0, -int(pos))
                        pending_qty = int(pending_no_qty.get(ticker, 0))
                        pending_cost = int(pending_no_cost.get(ticker, 0))
                        fill_qty = int(fills_no_qty.get(ticker, 0))
                        fill_cost = int(fills_no_cost.get(ticker, 0))
                        fill_avg = (fill_cost / fill_qty) if fill_qty > 0 else None
                        est_cost_pc = None
                        taker_pc = leg.get("taker_cost_per_contract")
                        if taker_pc is not None:
                            est_cost_pc = float(taker_pc)
                        elif leg.get("cost_per_contract") is not None:
                            est_cost_pc = float(leg["cost_per_contract"])
                        if pos_no_qty > 0 and fill_avg is not None:
                            filled_cost = int(round(fill_avg * pos_no_qty))
                            filled_cost_source = "fills"
                        elif pos_no_qty > 0 and est_cost_pc is not None:
                            filled_cost = int(round(est_cost_pc * pos_no_qty))
                            filled_cost_source = "estimated"
                        else:
                            filled_cost = 0
                            filled_cost_source = "unknown" if pos_no_qty > 0 else "none"
                        pos_avg_cost = None
                        if pos_no_qty > 0 and filled_cost > 0:
                            pos_avg_cost = filled_cost / float(pos_no_qty)
                        remaining_qty = max(0, size - pos_no_qty - pending_qty)
                        plan_cost_per_contract = leg["cost_per_contract"]
                        planned_remaining_cost = int(round(plan_cost_per_contract * remaining_qty))
                        leg["pos_no_qty"] = pos_no_qty
                        leg["pending_no_qty"] = pending_qty
                        leg["pending_no_cost"] = pending_cost
                        leg["filled_no_cost"] = filled_cost
                        leg["filled_cost_source"] = filled_cost_source
                        leg["pos_no_avg_cost"] = pos_avg_cost
                        leg["remaining_qty"] = remaining_qty
                        leg["planned_remaining_cost"] = planned_remaining_cost

                stats_limit = basket_stats(limit_legs, k, size, full_count=len(tops))
                if stats_limit is not None:
                    payout_best = stats_limit["payout_best"]
                    payout_worst = stats_limit["payout_worst"]
                    limit_expected_payout = sum(leg["p_no"] * 100.0 * size for leg in limit_legs)
                    maker_worst = stats_limit["edge_worst"]
                    maker_best = stats_limit["edge_best"]
                    maker_ev = stats_limit["ev"]

                    taker_ok = fallback_taker and all(
                        leg.get("taker_cost_total") is not None for leg in limit_legs
                    )
                    taker_cost_total = None
                    ask_edge_worst = None
                    if taker_ok:
                        taker_cost_total = sum(leg["taker_cost_total"] for leg in limit_legs)  # type: ignore
                    print("")
                    print("Limit basket (NO, quoted bid/ask legs):")
                    print(
                        f"Legs {stats_limit['count']}, cost {stats_limit['cost']:.1f}c, "
                        f"best {maker_best:.1f}c, worst {maker_worst:.1f}c, "
                        f"EV {maker_ev:.1f}c"
                    )
                    print(
                        "Worst-case payout (n-k) "
                        f"(n={stats_limit['count']}, k={k}): {payout_worst:.1f}c"
                    )
                    if taker_ok and taker_cost_total is not None:
                        ask_edge_best = payout_best - taker_cost_total
                        ask_edge_worst = payout_worst - taker_cost_total
                        ask_ev = limit_expected_payout - taker_cost_total
                        print(
                            "Fallback at ask (taker, slippage+fees): "
                            f"cost {taker_cost_total:.1f}c, best {ask_edge_best:.1f}c, "
                            f"worst {ask_edge_worst:.1f}c, EV {ask_ev:.1f}c"
                        )
                        if assumed_fill_ratio is not None:
                            ratio = max(0.0, min(1.0, assumed_fill_ratio))
                            blended_cost = (ratio * stats_limit["cost"]) + ((1.0 - ratio) * taker_cost_total)
                            blended_worst = payout_worst - blended_cost
                            print(
                                f"Blended worst-case (fill_ratio={ratio:.2f}): {blended_worst:.1f}c"
                            )
                    else:
                        print("Fallback at ask (taker): not executable")
                        ask_edge_worst = None
                    if require_taker_fallback:
                        if taker_ok and ask_edge_worst is not None:
                            safe = ask_edge_worst >= 0
                            safety_label = "SAFE" if safe else "UNSAFE (taker fallback negative)"
                        else:
                            safety_label = "UNSAFE (taker fallback not executable)"
                        print(f"Safety: {safety_label}")
                    if total_ask > 0:
                        print(
                            "Limit price improvement: "
                            f"{total_improve}c total vs ask (avg {total_improve / stats_limit['count']:.2f}c)"
                        )
                    # Internal consistency check
                    if abs(maker_best - (payout_best - stats_limit["cost"])) > 1e-6:
                        print("Warning: maker edge_best inconsistency", file=sys.stderr)
                    if abs(maker_worst - (payout_worst - stats_limit["cost"])) > 1e-6:
                        print("Warning: maker edge_worst inconsistency", file=sys.stderr)
                    immediate_sizes = [
                        leg.get("immediate_fillable_taker")
                        for leg in limit_legs
                        if leg.get("immediate_fillable_taker") is not None
                    ]
                    immediate_min = min(immediate_sizes) if immediate_sizes else None
                    if immediate_min is None:
                        print("Immediate fillable (taker proxy, at limit): unknown")
                    elif immediate_min <= 0:
                        print("Immediate fillable (taker proxy, at limit): 0")
                    else:
                        print(f"Immediate fillable (taker proxy, at limit): {immediate_min}")
                    rec_sizes = [
                        leg.get("rec_qty_est")
                        for leg in limit_legs
                        if leg.get("rec_qty_est") is not None
                    ]
                    rec_min = min(rec_sizes) if rec_sizes else None
                    if rec_sizes:
                        print(
                            "Recommended qty estimate (taker depth proxy, not fillable): "
                            f"{rec_min}"
                        )
                    taker_slip_sizes = [
                        leg.get("taker_max_size_slip")
                        for leg in limit_legs
                        if leg.get("taker_max_size_slip") is not None
                    ]
                    taker_slip_min = min(taker_slip_sizes) if taker_slip_sizes else None
                    if taker_slip_min is not None and taker_max_cap is not None and taker_max_cap > 0:
                        taker_slip_min = min(taker_slip_min, taker_max_cap)
                    if taker_slip_min is not None:
                        cap_label = taker_max_cap if taker_max_cap is not None else "--"
                        print(
                            "Taker slippage max size (cap "
                            f"{cap_label}, max_slippage {max_slippage_c}c): {taker_slip_min}"
                        )
                    remaining_qty_total = None
                    remaining_budget_per_contract = None
                    if trade_state is not None and limit_legs:
                        filled_cost_total = sum(
                            int(leg.get("filled_no_cost") or 0) for leg in limit_legs
                        )
                        pending_cost_total = sum(
                            int(leg.get("pending_no_cost") or 0) for leg in limit_legs
                        )
                        remaining_cost_total = sum(
                            int(leg.get("planned_remaining_cost") or 0) for leg in limit_legs
                        )
                        remaining_qty_total = sum(
                            int(leg.get("remaining_qty") or 0) for leg in limit_legs
                        )
                        remaining_qty_min = None
                        remaining_qty_max = None
                        remaining_vals = [int(leg.get("remaining_qty") or 0) for leg in limit_legs]
                        if remaining_vals:
                            remaining_qty_min = min(remaining_vals)
                            remaining_qty_max = max(remaining_vals)
                            print(
                                "Remaining qty across legs: "
                                f"min {remaining_qty_min}, max {remaining_qty_max}"
                            )
                        cost_est_total = filled_cost_total + pending_cost_total + remaining_cost_total
                        edge_worst_est = payout_worst - cost_est_total
                        print(
                            "Adjusted (filled+resting+planned) worst-case edge: "
                            f"{edge_worst_est:.1f}c"
                        )
                        if remaining_qty_total > 0:
                            budget_remaining = payout_worst - filled_cost_total - pending_cost_total
                            remaining_budget_per_contract = budget_remaining / remaining_qty_total
                            print(
                                "Max all-in cost per remaining contract (to keep worst >= 0): "
                                f"{remaining_budget_per_contract:.1f}c"
                            )
                    if trade_state is not None:
                        print("")
                        print("Action guidance:")
                        pos_no_total = trade_state.get("pos_no_total", 0)
                        pending_no_qty_total = trade_state.get("resting_no_buy_qty_total", 0)
                        if remaining_qty_max == 0:
                            if pending_no_qty_total > 0:
                                print(
                                    "- Fully covered for target size. Consider canceling remaining "
                                    "resting orders if you don't want more exposure."
                                )
                            else:
                                print("- Fully covered for target size. No action needed.")
                        else:
                            if taker_ok and ask_edge_worst is not None and ask_edge_worst >= 0:
                                if remaining_budget_per_contract is not None:
                                    over_budget = []
                                    for leg in limit_legs:
                                        cost_pc = leg.get("taker_cost_per_contract")
                                        if cost_pc is None:
                                            continue
                                        if cost_pc > remaining_budget_per_contract + 1e-9:
                                            over_budget.append(leg["ticker"])
                                    if not over_budget:
                                        print(
                                            "- SAFE with taker fallback. Taking remaining at taker "
                                            "should keep worst-case non-negative."
                                        )
                                    else:
                                        print(
                                            "- SAFE, but some legs exceed remaining budget at taker: "
                                            + ", ".join(over_budget)
                                        )
                                else:
                                    print(
                                        "- SAFE with taker fallback. Taking remaining at taker should be OK."
                                    )
                            else:
                                print(
                                    "- UNSAFE taker fallback. Consider canceling resting orders and "
                                    "reducing exposure if you want to exit."
                                )
                        # Resting order price check vs current ask
                        if pending_no_qty_total > 0:
                            print("")
                            print("Resting order price check (NO buy):")
                            ticker_w = 28
                            print(f"{'Ticker':<{ticker_w}} | limit | best ask | delta | status")
                            print("-" * (ticker_w + 36))
                            for leg in limit_legs:
                                pending_qty = int(leg.get("pending_no_qty") or 0)
                                if pending_qty <= 0:
                                    continue
                                limit_px = leg.get("price")
                                ask_px = leg.get("ask_px")
                                if limit_px is None or ask_px is None:
                                    continue
                                delta = ask_px - limit_px
                                status = "good" if delta >= 0 else "overpay"
                                print(
                                    f"{leg['ticker']:<{ticker_w}} | "
                                    f"{limit_px:>5}c | "
                                    f"{ask_px:>8}c | "
                                    f"{delta:>5}c | "
                                    f"{status}"
                                )
                        # Scenario EVs using current positions/resting orders
                        if limit_legs:
                            has_estimated = any(
                                leg.get("filled_cost_source") == "estimated"
                                for leg in limit_legs
                            )
                            has_missing = any(
                                leg.get("filled_cost_source") == "unknown"
                                for leg in limit_legs
                            )
                            # Scenario A: all resting orders fill, no new orders
                            payout_a = 0.0
                            cost_a = 0.0
                            for leg in limit_legs:
                                qty = int(leg.get("pos_no_qty") or 0) + int(
                                    leg.get("pending_no_qty") or 0
                                )
                                if qty <= 0:
                                    continue
                                payout_a += leg["p_no"] * 100.0 * qty
                                cost_a += float(leg.get("filled_no_cost") or 0)
                                cost_a += float(leg.get("pending_no_cost") or 0)
                            if payout_a > 0 or cost_a > 0:
                                ev_a = payout_a - cost_a
                                note_a = ""
                                if has_missing:
                                    note_a = " (filled cost missing)"
                                elif has_estimated:
                                    note_a = " (filled cost estimated)"
                                print(
                                    "EV if all resting orders fill (no new orders): "
                                    f"{ev_a:.1f}c (payout {payout_a:.1f}c - cost {cost_a:.1f}c){note_a}"
                                )
                            # Scenario C: cancel resting, keep positions only
                            payout_c = 0.0
                            cost_c = 0.0
                            pos_qtys = []
                            ev_c = None
                            worst_edge_c = None
                            for leg in limit_legs:
                                pos_qty = int(leg.get("pos_no_qty") or 0)
                                if pos_qty <= 0:
                                    continue
                                payout_c += leg["p_no"] * 100.0 * pos_qty
                                cost_c += float(leg.get("filled_no_cost") or 0)
                                pos_qtys.append(pos_qty)
                            if pos_qtys:
                                ev_c = payout_c - cost_c
                                note_c = ""
                                if has_missing:
                                    note_c = " (filled cost missing)"
                                elif has_estimated:
                                    note_c = " (filled cost estimated)"
                                print(
                                    "EV if cancel resting and do nothing (positions only): "
                                    f"{ev_c:.1f}c (payout {payout_c:.1f}c - cost {cost_c:.1f}c){note_c}"
                                )
                                pos_qtys_sorted = sorted(pos_qtys, reverse=True)
                                worst_qty = max(0, sum(pos_qtys) - sum(pos_qtys_sorted[:k]))
                                payout_worst_c = worst_qty * 100.0
                                worst_edge_c = payout_worst_c - cost_c
                                print(
                                    "Worst-case (n-k) if cancel resting (positions only): "
                                    f"{worst_edge_c:.1f}c (payout {payout_worst_c:.1f}c - cost {cost_c:.1f}c){note_c}"
                                )
                            # Scenario B: cancel resting, take remaining to target at taker
                            payout_b = 0.0
                            cost_b = 0.0
                            ok_b = True
                            for leg in limit_legs:
                                pos_qty = int(leg.get("pos_no_qty") or 0)
                                remaining = max(0, size - pos_qty)
                                payout_b += leg["p_no"] * 100.0 * size
                                cost_b += float(leg.get("filled_no_cost") or 0)
                                cost_pc = leg.get("taker_cost_per_contract")
                                if remaining > 0:
                                    if cost_pc is None:
                                        ok_b = False
                                        break
                                    cost_b += float(cost_pc) * remaining
                            if ok_b:
                                ev_b = payout_b - cost_b
                                note_b = ""
                                if has_missing:
                                    note_b = " (filled cost missing)"
                                elif has_estimated:
                                    note_b = " (filled cost estimated)"
                                print(
                                    "EV if cancel resting and take remaining at taker now: "
                                    f"{ev_b:.1f}c (payout {payout_b:.1f}c - cost {cost_b:.1f}c){note_b}"
                                )
                                payout_worst_b = max(
                                    0.0, (len(limit_legs) - k) * 100.0 * size
                                )
                                worst_edge_b = payout_worst_b - cost_b
                                print(
                                    "Worst-case (n-k) if cancel resting and take remaining at taker: "
                                    f"{worst_edge_b:.1f}c (payout {payout_worst_b:.1f}c - cost {cost_b:.1f}c){note_b}"
                                )
                                if ev_c is not None and worst_edge_c is not None:
                                    delta_ev = ev_b - ev_c
                                    delta_worst = worst_edge_b - worst_edge_c
                                    print(
                                        "Delta vs positions-only (take remaining at taker): "
                                        f"EV {delta_ev:+.1f}c, worst-case {delta_worst:+.1f}c"
                                    )
                    print("")
                    print("NO taker ask levels (qty > 0):")
                    ticker_w = 28
                    print(f"{'Ticker':<{ticker_w}} | Levels")
                    print("-" * (ticker_w + 9))
                    for t, levels in zip(tops, no_ask_levels):
                        levels_s = fmt_levels(levels)
                        if levels_s == "--":
                            continue
                        print(f"{t.ticker:<{ticker_w}} | {levels_s}")
                    if limit_out_lines is not None:
                        lines = limit_out_lines
                        pending_no_qty_total = (
                            trade_state.get("resting_no_buy_qty_total", 0)
                            if trade_state is not None
                            else 0
                        )
                        lines.append("Limit basket summary (NO):")
                        lines.append(
                            f"Legs {stats_limit['count']}, cost {stats_limit['cost']:.1f}c, "
                            f"best {maker_best:.1f}c, worst {maker_worst:.1f}c, "
                            f"EV {maker_ev:.1f}c"
                        )
                        lines.append(
                            "Worst-case payout (n-k) "
                            f"(n={stats_limit['count']}, k={k}): {payout_worst:.1f}c"
                        )
                        if taker_ok and taker_cost_total is not None:
                            ask_edge_best = payout_best - taker_cost_total
                            ask_edge_worst = payout_worst - taker_cost_total
                            ask_ev = limit_expected_payout - taker_cost_total
                            lines.append(
                                "Fallback at ask (taker, slippage+fees): "
                                f"cost {taker_cost_total:.1f}c, best {ask_edge_best:.1f}c, "
                                f"worst {ask_edge_worst:.1f}c, EV {ask_ev:.1f}c"
                            )
                            if assumed_fill_ratio is not None:
                                ratio = max(0.0, min(1.0, assumed_fill_ratio))
                                blended_cost = (ratio * stats_limit["cost"]) + (
                                    (1.0 - ratio) * taker_cost_total
                                )
                                blended_worst = payout_worst - blended_cost
                                lines.append(
                                    f"Blended worst-case (fill_ratio={ratio:.2f}): {blended_worst:.1f}c"
                                )
                        else:
                            lines.append("Fallback at ask (taker): not executable")
                            ask_edge_worst = None
                        if require_taker_fallback:
                            if taker_ok and ask_edge_worst is not None:
                                safe = ask_edge_worst >= 0
                                safety_label = (
                                    "SAFE" if safe else "UNSAFE (taker fallback negative)"
                                )
                            else:
                                safety_label = "UNSAFE (taker fallback not executable)"
                            lines.append(f"Safety: {safety_label}")
                        lines.append(
                            f"Limit price improvement: {total_improve}c total vs ask "
                            f"(avg {total_improve / stats_limit['count']:.2f}c)"
                        )
                        if immediate_min is None:
                            lines.append("Immediate fillable (taker proxy, at limit): unknown")
                        elif immediate_min <= 0:
                            lines.append("Immediate fillable (taker proxy, at limit): 0")
                        else:
                            lines.append(
                                f"Immediate fillable (taker proxy, at limit): {immediate_min}"
                            )
                        if rec_sizes:
                            lines.append(
                                "Recommended qty estimate (taker depth proxy, not fillable): "
                                f"{rec_min}"
                            )
                        if taker_slip_min is not None:
                            cap_label = taker_max_cap if taker_max_cap is not None else "--"
                            lines.append(
                                "Taker slippage max size (cap "
                                f"{cap_label}, max_slippage {max_slippage_c}c): {taker_slip_min}"
                            )
                        if trade_state is not None and remaining_qty_total is not None:
                            lines.append(
                                "Adjusted (filled+resting+planned) worst-case edge: "
                                f"{edge_worst_est:.1f}c"
                            )
                            if remaining_budget_per_contract is not None:
                                lines.append(
                                    "Max all-in cost per remaining contract "
                                    "(to keep worst >= 0): "
                                    f"{remaining_budget_per_contract:.1f}c"
                                )
                        if trade_state is not None:
                            lines.append("")
                            lines.append("Action guidance:")
                            pos_no_total = trade_state.get("pos_no_total", 0)
                            pending_no_qty_total = trade_state.get("resting_no_buy_qty_total", 0)
                            if remaining_qty_max == 0:
                                if pending_no_qty_total > 0:
                                    lines.append(
                                        "- Fully covered for target size. Consider canceling remaining "
                                        "resting orders if you don't want more exposure."
                                    )
                                else:
                                    lines.append("- Fully covered for target size. No action needed.")
                            else:
                                if taker_ok and ask_edge_worst is not None and ask_edge_worst >= 0:
                                    if remaining_budget_per_contract is not None:
                                        over_budget = []
                                        for leg in limit_legs:
                                            cost_pc = leg.get("taker_cost_per_contract")
                                            if cost_pc is None:
                                                continue
                                            if cost_pc > remaining_budget_per_contract + 1e-9:
                                                over_budget.append(leg["ticker"])
                                        if not over_budget:
                                            lines.append(
                                                "- SAFE with taker fallback. Taking remaining at taker "
                                                "should keep worst-case non-negative."
                                            )
                                        else:
                                            lines.append(
                                                "- SAFE, but some legs exceed remaining budget at taker: "
                                                + ", ".join(over_budget)
                                            )
                                    else:
                                        lines.append(
                                            "- SAFE with taker fallback. Taking remaining at taker should be OK."
                                        )
                                else:
                                    lines.append(
                                        "- UNSAFE taker fallback. Consider canceling resting orders and "
                                        "reducing exposure if you want to exit."
                                    )
                            lines.append("")
                            lines.append("Positions (NO) by market:")
                            ticker_w = 28
                            lines.append(
                                f"{'Ticker':<{ticker_w}} | pos | avg_cost | source"
                            )
                            lines.append("-" * (ticker_w + 27))
                            any_pos = False
                            for leg in limit_legs:
                                pos_qty = int(leg.get("pos_no_qty") or 0)
                                if pos_qty <= 0:
                                    continue
                                any_pos = True
                                avg_cost = leg.get("pos_no_avg_cost")
                                avg_s = "--" if avg_cost is None else f"{avg_cost:.1f}c"
                                src = leg.get("filled_cost_source") or "--"
                                lines.append(
                                    f"{leg['ticker']:<{ticker_w}} | "
                                    f"{pos_qty:>3} | {avg_s:>7} | {src}"
                                )
                            if not any_pos:
                                lines.append("(none)")
                            no_pos_no_pending = [
                                leg["ticker"]
                                for leg in limit_legs
                                if int(leg.get("pos_no_qty") or 0) <= 0
                                and int(leg.get("pending_no_qty") or 0) <= 0
                                and int(leg.get("remaining_qty") or 0) > 0
                            ]
                            if no_pos_no_pending:
                                lines.append("")
                                lines.append(
                                    "Recommended markets with no position or resting orders:"
                                )
                                for t in no_pos_no_pending:
                                    lines.append(f"- {t}")
                        if pending_no_qty_total > 0:
                            lines.append("")
                            lines.append("Resting order price check (NO buy):")
                            ticker_w = 28
                            lines.append(f"{'Ticker':<{ticker_w}} | limit | best ask | delta | status")
                            lines.append("-" * (ticker_w + 36))
                            for leg in limit_legs:
                                pending_qty = int(leg.get("pending_no_qty") or 0)
                                if pending_qty <= 0:
                                    continue
                                limit_px = leg.get("price")
                                ask_px = leg.get("ask_px")
                                if limit_px is None or ask_px is None:
                                    continue
                                delta = ask_px - limit_px
                                status = "good" if delta >= 0 else "overpay"
                                lines.append(
                                    f"{leg['ticker']:<{ticker_w}} | "
                                    f"{limit_px:>5}c | "
                                    f"{ask_px:>8}c | "
                                    f"{delta:>5}c | "
                                    f"{status}"
                                )
                            if limit_legs:
                                has_estimated = any(
                                    leg.get("filled_cost_source") == "estimated"
                                    for leg in limit_legs
                                )
                                has_missing = any(
                                    leg.get("filled_cost_source") == "unknown"
                                    for leg in limit_legs
                                )
                                payout_a = 0.0
                                cost_a = 0.0
                                for leg in limit_legs:
                                    qty = int(leg.get("pos_no_qty") or 0) + int(
                                        leg.get("pending_no_qty") or 0
                                    )
                                    if qty <= 0:
                                        continue
                                    payout_a += leg["p_no"] * 100.0 * qty
                                    cost_a += float(leg.get("filled_no_cost") or 0)
                                    cost_a += float(leg.get("pending_no_cost") or 0)
                                if payout_a > 0 or cost_a > 0:
                                    ev_a = payout_a - cost_a
                                    note_a = ""
                                    if has_missing:
                                        note_a = " (filled cost missing)"
                                    elif has_estimated:
                                        note_a = " (filled cost estimated)"
                                    lines.append(
                                        "EV if all resting orders fill (no new orders): "
                                        f"{ev_a:.1f}c (payout {payout_a:.1f}c - cost {cost_a:.1f}c){note_a}"
                                    )
                                payout_c = 0.0
                                cost_c = 0.0
                                pos_qtys = []
                                ev_c = None
                                worst_edge_c = None
                                for leg in limit_legs:
                                    pos_qty = int(leg.get("pos_no_qty") or 0)
                                    if pos_qty <= 0:
                                        continue
                                    payout_c += leg["p_no"] * 100.0 * pos_qty
                                    cost_c += float(leg.get("filled_no_cost") or 0)
                                    pos_qtys.append(pos_qty)
                                if pos_qtys:
                                    ev_c = payout_c - cost_c
                                    note_c = ""
                                    if has_missing:
                                        note_c = " (filled cost missing)"
                                    elif has_estimated:
                                        note_c = " (filled cost estimated)"
                                    lines.append(
                                        "EV if cancel resting and do nothing (positions only): "
                                        f"{ev_c:.1f}c (payout {payout_c:.1f}c - cost {cost_c:.1f}c){note_c}"
                                    )
                                    pos_qtys_sorted = sorted(pos_qtys, reverse=True)
                                    worst_qty = max(0, sum(pos_qtys) - sum(pos_qtys_sorted[:k]))
                                    payout_worst_c = worst_qty * 100.0
                                    worst_edge_c = payout_worst_c - cost_c
                                    lines.append(
                                        "Worst-case (n-k) if cancel resting (positions only): "
                                        f"{worst_edge_c:.1f}c (payout {payout_worst_c:.1f}c - cost {cost_c:.1f}c){note_c}"
                                    )
                                payout_b = 0.0
                                cost_b = 0.0
                                ok_b = True
                                for leg in limit_legs:
                                    pos_qty = int(leg.get("pos_no_qty") or 0)
                                    remaining = max(0, size - pos_qty)
                                    payout_b += leg["p_no"] * 100.0 * size
                                    cost_b += float(leg.get("filled_no_cost") or 0)
                                    cost_pc = leg.get("taker_cost_per_contract")
                                    if remaining > 0:
                                        if cost_pc is None:
                                            ok_b = False
                                            break
                                        cost_b += float(cost_pc) * remaining
                                if ok_b:
                                    ev_b = payout_b - cost_b
                                    note_b = ""
                                    if has_missing:
                                        note_b = " (filled cost missing)"
                                    elif has_estimated:
                                        note_b = " (filled cost estimated)"
                                    lines.append(
                                        "EV if cancel resting and take remaining at taker now: "
                                        f"{ev_b:.1f}c (payout {payout_b:.1f}c - cost {cost_b:.1f}c){note_b}"
                                    )
                                    payout_worst_b = max(
                                        0.0, (len(limit_legs) - k) * 100.0 * size
                                    )
                                    worst_edge_b = payout_worst_b - cost_b
                                    lines.append(
                                        "Worst-case (n-k) if cancel resting and take remaining at taker: "
                                        f"{worst_edge_b:.1f}c (payout {payout_worst_b:.1f}c - cost {cost_b:.1f}c){note_b}"
                                    )
                                    if ev_c is not None and worst_edge_c is not None:
                                        delta_ev = ev_b - ev_c
                                        delta_worst = worst_edge_b - worst_edge_c
                                        lines.append(
                                            "Delta vs positions-only (take remaining at taker): "
                                            f"EV {delta_ev:+.1f}c, worst-case {delta_worst:+.1f}c"
                                        )
                        lines.append("")
                        lines.append("NO taker ask levels (qty > 0):")
                        ticker_w = 28
                        lines.append(f"{'Ticker':<{ticker_w}} | Levels")
                        lines.append("-" * (ticker_w + 9))
                        for t, levels in zip(tops, no_ask_levels):
                            levels_s = fmt_levels(levels)
                            if levels_s == "--":
                                continue
                            lines.append(f"{t.ticker:<{ticker_w}} | {levels_s}")
                        lines.append("")
                        lines.append("Order list (NO limit buys):")
                        for leg in sorted(limit_legs, key=lambda x: x["ev"], reverse=True):
                            ask_top = leg.get("ask_qty_top")
                            ask_depth = leg.get("ask_qty_depth_proxy")
                            fillable_leg = leg.get("immediate_fillable_taker")
                            rec_leg = leg.get("rec_qty_est")
                            pos_no = leg.get("pos_no_qty")
                            pending_no = leg.get("pending_no_qty")
                            remaining_no = leg.get("remaining_qty")
                            max_all_in = (
                                remaining_budget_per_contract
                                if remaining_budget_per_contract is not None and remaining_no
                                else None
                            )
                            lines.append(
                                f"{leg['ticker']:<30} | "
                                f"limit {leg['price']:>3}c | "
                                f"ask {leg['ask_px']:>3}c | "
                                f"spread {leg['spread']:>2}c | "
                                f"fee {leg['fee']:>2}c | "
                                f"p_no {leg['p_no']:.3f} | "
                                f"ask_qty_top {ask_top if ask_top is not None else '--':>4} | "
                                f"ask_qty_depth_proxy {ask_depth if ask_depth is not None else '--':>4} | "
                                f"immediate_fillable_taker {fillable_leg if fillable_leg is not None else '--':>3} | "
                                f"rec_qty_est {rec_leg if rec_leg is not None else '--':>3} | "
                                f"pos_no {pos_no if pos_no is not None else '--':>3} | "
                                f"pending_no {pending_no if pending_no is not None else '--':>3} | "
                                f"remaining {remaining_no if remaining_no is not None else '--':>3} | "
                                f"max_all_in {f'{max_all_in:.1f}' if max_all_in is not None else '--':>5} | "
                                f"EV {leg['ev']:.1f}c"
                            )
                        lines.append("")
                        lines.append("Notes:")
                        lines.append(
                            "Expected payout uses mid-implied probabilities normalized to k-of-n."
                        )
                        lines.append(
                            "Worst case assumes all k YES outcomes are inside your basket."
                        )
                    basket_cost_per_unit = (
                        sum(leg["cost_per_contract"] for leg in limit_legs) if limit_legs else None
                    )
                    safe_flag = (
                        taker_ok and ask_edge_worst is not None and ask_edge_worst >= 0
                    )
                    remaining_qty_max = None
                    if trade_state is not None and limit_legs:
                        remaining_vals = [
                            int(leg.get("remaining_qty") or 0) for leg in limit_legs
                        ]
                        remaining_qty_max = max(remaining_vals) if remaining_vals else None
                    limit_state = {
                        "limit_legs": limit_legs,
                        "safe": safe_flag,
                        "rec_min": rec_min,
                        "taker_slip_min": taker_slip_min,
                        "basket_cost_per_unit": basket_cost_per_unit,
                        "remaining_qty_max": remaining_qty_max,
                        "remaining_budget_per_contract": remaining_budget_per_contract,
                    }
                    if limit_out_lines is not None and limit_out:
                        out_path = Path(limit_out)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text("\n".join(limit_out_lines), encoding="utf-8")

    return limit_state


def _self_test() -> None:
    # Fee rounding
    assert fee_cents(0.07, 50, 1) == 2
    assert fee_cents(0.07, 50, 100) == 175
    # Fee policy default / whitelist
    policy_default = fee_policy_for_series("KXFOO", {}, 0.07, 0.0)
    assert policy_default.taker_fee_coef == 0.07
    assert policy_default.maker_fee_rate == 0.0
    policy_white = fee_policy_for_series("INX", {"INX": 0.035}, 0.07, 0.0)
    assert policy_white.taker_fee_coef == 0.035
    # Best bid selection on unsorted levels
    bids = normalize_bids([(10, 1), (20, 1), (15, 1)])
    assert best_bid(bids)[0] == 20
    # Implied ask consistency
    asks = implied_asks_from_bids([(70, 5)])
    assert asks[0][0] == 30
    # Qty rounding
    assert qty_to_int(1.0000001) == 1
    # Normalization sums to k
    probs, ok = normalize_probs_k([0.6, 0.4, 0.1], 1)
    assert ok
    assert abs(sum(p for p in probs if p is not None) - 1.0) < 1e-6
    # Basket payoff logic
    legs = [{"p_no": 1.0, "cost_per_contract": 0.0, "qty": 1, "price": 1}]
    stats = basket_stats(legs * 2, 1, 1)
    assert stats is not None
    assert stats["edge_best"] == 200.0
    assert stats["edge_worst"] == 100.0
    # ARB safety gating
    assert is_arb_safe(1.0, True, 0.1) is True
    assert is_arb_safe(1.0, True, -0.1) is False


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        print("Self-test: OK")
        return
    data_key_id = os.getenv("KALSHI_DATA_KEY_ID") or os.environ["KALSHI_KEY_ID"]
    data_pem_path = os.getenv("KALSHI_DATA_PRIVATE_KEY_PEM") or os.environ[
        "KALSHI_PRIVATE_KEY_PEM"
    ]
    private_key = load_private_key_pem(data_pem_path)
    trade_key_id = os.getenv("KALSHI_TRADE_KEY_ID")
    trade_pem_path = os.getenv("KALSHI_TRADE_PRIVATE_KEY_PEM")
    trade_private_key = None
    if args.auto_exec or args.sync_trade_state:
        if not trade_key_id or not trade_pem_path:
            print(
                "Trade sync/auto-exec requires trade credentials. Set KALSHI_TRADE_KEY_ID and "
                "KALSHI_TRADE_PRIVATE_KEY_PEM.",
                file=sys.stderr,
            )
            sys.exit(1)
        trade_private_key = load_private_key_pem(trade_pem_path)
    if args.fee_kind != "taker" or args.fee_rate is not None or args.limit_fee_rate is not None:
        print(
            "Warning: fee-kind/fee-rate overrides are ignored; using series-based fee policy.",
            file=sys.stderr,
        )
    fee_rate_maker = args.fee_rate_maker
    fee_rate_taker = args.fee_rate_taker
    whitelist = load_fee_whitelist(args.fee_whitelist)
    fallback_taker = args.fallback_taker
    assumed_fill_ratio = args.assumed_fill_ratio

    first = True
    auto_exec_done = False
    while True:
        event_resp, tops, myb, mya, mnb, mna = fetch_event_snapshot(
            args.event, args.depth, data_key_id, private_key, args.debug
        )
        trade_state = None
        if args.sync_trade_state and trade_private_key is not None:
            tickers = [t.ticker for t in tops]
            trade_state = sync_trade_state(
                args.event,
                tickers,
                trade_key_id,
                trade_private_key,
                args.fills_lookback_hours,
                args.subaccount,
            )

        event = event_resp.get("event", {})
        me = event.get("mutually_exclusive") is True
        true_count = 1 if me else args.true_count
        event["_true_count"] = true_count
        fee_policy = fee_policy_for_series(
            event.get("series_ticker"),
            whitelist,
            fee_rate_taker,
            fee_rate_maker,
        )
        target_yes = true_count * 100 if true_count is not None else None
        target_no = (len(tops) - true_count) * 100 if true_count is not None else None
        yes_ask_levels = [implied_asks_from_bids(t.no_bids) for t in tops]
        no_ask_levels = [implied_asks_from_bids(t.yes_bids) for t in tops]
        yes_bid_levels = [t.yes_bids for t in tops]
        no_bid_levels = [t.no_bids for t in tops]
        fee_rates = [fee_policy.taker_fee_coef for _ in tops]

        buy_yes_cost, buy_yes_ok = basket_buy_cost(yes_ask_levels, args.size, fee_rates)
        buy_no_cost, buy_no_ok = basket_buy_cost(no_ask_levels, args.size, fee_rates)
        sell_yes_proceeds, sell_yes_ok = basket_sell_proceeds(
            yes_bid_levels, args.size, fee_rates
        )
        sell_no_proceeds, sell_no_ok = basket_sell_proceeds(
            no_bid_levels, args.size, fee_rates
        )

        alert = False
        if target_yes is not None:
            if buy_yes_ok and buy_yes_cost is not None:
                if target_yes - buy_yes_cost >= -args.near:
                    alert = True
            if sell_yes_ok and sell_yes_proceeds is not None:
                if sell_yes_proceeds - target_yes >= -args.near:
                    alert = True
        if target_no is not None:
            if buy_no_ok and buy_no_cost is not None:
                if target_no - buy_no_cost >= -args.near:
                    alert = True
            if sell_no_ok and sell_no_proceeds is not None:
                if sell_no_proceeds - target_no >= -args.near:
                    alert = True
        if (
            args.limit_mode != "none"
            and args.fallback_taker
            and true_count is not None
            and not alert
        ):
            implied_probs = []
            for t in tops:
                p_yes = None
                if t.yes_mid is not None:
                    p_yes = t.yes_mid / 100.0
                elif t.no_mid is not None:
                    p_yes = 1.0 - (t.no_mid / 100.0)
                implied_probs.append(p_yes)
            norm_probs, ok = normalize_probs_k(implied_probs, true_count)
            if any(p is not None for p in norm_probs):
                taker_costs = []
                for t, p, levels in zip(tops, norm_probs, no_ask_levels):
                    if p is None:
                        continue
                    if t.no_bid_px is None or t.no_ask_px is None:
                        continue
                    qty_depth = (
                        t.no_ask_qty_depth_taker_proxy
                        if t.no_ask_qty_depth_taker_proxy is not None
                        else t.no_ask_qty
                    )
                    qty_depth_int = qty_to_int(qty_depth)
                    if qty_depth_int < args.limit_min_qty or qty_depth_int < args.size:
                        continue
                    spread = t.no_ask_px - t.no_bid_px
                    if spread < args.limit_min_spread:
                        continue
                    taker_rate = fee_policy.taker_fee_coef
                    cost_total, filled = net_buy_cost(levels, args.size, taker_rate)
                    if filled < args.size or cost_total is None:
                        taker_costs = []
                        break
                    taker_costs.append(cost_total)
                if taker_costs:
                    payout_worst = max(0.0, (len(taker_costs) - true_count) * 100.0 * args.size)
                    edge_worst = payout_worst - sum(taker_costs)
                    if edge_worst >= -args.near:
                        alert = True
        if (not args.alert_only) or alert:
            if args.watch:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}]")
            if args.debug:
                print("Computing report...")
            limit_state = None
            try:
                limit_state = print_report(
                    event_resp,
                    tops,
                    myb,
                    mya,
                    mnb,
                    mna,
                    args.near,
                    show_tables=not args.brief,
                    print_header=first,
                    fee_policy=fee_policy,
                    require_taker_fallback=args.require_taker_fallback,
                    optimize=args.optimize,
                    limit_mode=args.limit_mode,
                    limit_improve_c=args.limit_improve_c,
                    limit_min_qty=args.limit_min_qty,
                    limit_min_spread=args.limit_min_spread,
                    max_slippage_c=args.max_slippage_c,
                    taker_max_cap=args.taker_max_cap,
                    rec_qty_fraction=args.rec_qty_fraction,
                    rec_qty_cap=args.rec_qty_cap,
                    trade_state=trade_state,
                    depth=args.depth,
                    limit_out=args.limit_out,
                    fallback_taker=fallback_taker,
                    assumed_fill_ratio=assumed_fill_ratio,
                    size=args.size,
                )
                print("")
                if args.auto_exec and not auto_exec_done:
                    if limit_state is None:
                        print("Auto-exec: no limit basket computed; skipping.")
                    elif not limit_state.get("safe"):
                        print("Auto-exec: limit basket not SAFE; skipping.")
                    else:
                        rec_min = limit_state.get("rec_min")
                        slip_min = limit_state.get("taker_slip_min")
                        exec_size = args.size
                        recommended = slip_min if slip_min is not None else rec_min
                        if recommended is not None:
                            exec_size = min(exec_size, int(recommended))
                        # exec_size is capped per-leg below; no global cap needed
                        if args.exec_max_dollars is not None:
                            basket_cost = limit_state.get("basket_cost_per_unit")
                            if basket_cost is None or basket_cost <= 0:
                                print(
                                    "Auto-exec: missing basket cost; cannot apply dollar cap."
                                )
                            else:
                                budget_cents = int(math.floor(args.exec_max_dollars * 100.0 + 1e-6))
                                max_by_budget = int(
                                    math.floor(budget_cents / float(basket_cost) + 1e-9)
                                )
                                exec_size = min(exec_size, max_by_budget)
                        if exec_size <= 0:
                            print("Auto-exec: size resolved to 0; skipping.")
                        else:
                            orders = []
                            client_prefix = f"auto-{int(time.time())}-{uuid.uuid4().hex[:6]}"
                            for idx, leg in enumerate(limit_state.get("limit_legs", [])):
                                price = leg.get("price")
                                ticker = leg.get("ticker")
                                if price is None or ticker is None:
                                    continue
                                remaining_qty = leg.get("remaining_qty")
                                if remaining_qty is not None:
                                    remaining_qty = int(remaining_qty)
                                    if remaining_qty <= 0:
                                        continue
                                    count = min(exec_size, remaining_qty)
                                else:
                                    count = exec_size
                                if count <= 0:
                                    continue
                                order = {
                                    "ticker": ticker,
                                    "side": "no",
                                    "action": "buy",
                                    "count": count,
                                    "type": "limit",
                                    "client_order_id": f"{client_prefix}-{idx}",
                                    "no_price": price,
                                }
                                orders.append(order)
                            if not orders:
                                print("Auto-exec: no orders to submit.")
                            else:
                                est_cost = limit_state.get("basket_cost_per_unit")
                                if est_cost is not None:
                                    print(
                                        "Auto-exec: submitting "
                                        f"{len(orders)} orders up to x{exec_size} "
                                        f"(est cost ${(est_cost * exec_size) / 100.0:.2f})"
                                    )
                                else:
                                    print(
                                        f"Auto-exec: submitting {len(orders)} orders up to x{exec_size}"
                                    )
                                place_orders_batch(orders, trade_key_id, trade_private_key)
                                auto_exec_done = True
            except Exception as e:
                print(f"Error during report: {e}")
                raise

        if not args.watch:
            break
        first = False
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
