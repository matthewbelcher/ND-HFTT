import argparse
import json
import os
import re
import sys
import time
import uuid
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kalshi_common import load_private_key_pem, sign_pss_base64, now_ms


TRADE_API_BASE = os.getenv(
    "KALSHI_TRADE_API_BASE",
    os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co"),
)
TRADE_API_V2 = "/trade-api/v2"


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


def api_post(
    path: str,
    body: dict,
    key_id: str,
    private_key,
    base_url: str = TRADE_API_BASE,
) -> Optional[dict]:
    url = base_url + path
    headers = kalshi_rest_headers(key_id, private_key, "POST", path)
    headers["Content-Type"] = "application/json"
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, headers=headers, data=payload, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", "replace")
        except Exception:
            body_text = ""
        print(f"HTTP {e.code} on POST {path}: {body_text}", file=sys.stderr)
        return None


def place_orders_batch(
    orders: List[dict],
    key_id: str,
    private_key,
    base_url: str = TRADE_API_BASE,
) -> Tuple[Optional[dict], List[dict]]:
    if not orders:
        return None, []
    path = f"{TRADE_API_V2}/portfolio/orders/batched"
    resp = api_post(path, {"orders": orders}, key_id, private_key, base_url=base_url)
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


def parse_limit_report(text: str) -> Tuple[str, Optional[int], Optional[bool], List[Tuple[str, int]]]:
    side = "no"
    basket_max = None
    safety = None
    orders: List[Tuple[str, int]] = []
    order_section = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Order list"):
            order_section = True
            if "YES" in line.upper():
                side = "yes"
            else:
                side = "no"
            continue
        if line.startswith("Safety:"):
            safety = "SAFE" in line.upper()
            continue
        if line.startswith("Basket max size at limit:"):
            m = re.search(r":\s*(\d+)", line)
            if m:
                basket_max = int(m.group(1))
            continue
        if order_section:
            m = re.match(r"^(?P<ticker>\S+)\s+\|\s+limit\s+(?P<px>\d+)c", line)
            if not m:
                continue
            orders.append((m.group("ticker"), int(m.group("px"))))
    return side, basket_max, safety, orders


def build_orders(
    side: str,
    orders_in: List[Tuple[str, int]],
    size: int,
    post_only: bool,
    client_prefix: str,
) -> List[dict]:
    out: List[dict] = []
    action = "buy"
    for idx, (ticker, px) in enumerate(orders_in):
        if px < 0 or px > 100:
            print(f"Skipping {ticker}: invalid price {px}", file=sys.stderr)
            continue
        order = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": size,
            "type": "limit",
            "client_order_id": f"{client_prefix}-{idx}",
        }
        if side == "yes":
            order["yes_price"] = px
        else:
            order["no_price"] = px
        if post_only:
            order["post_only"] = True
        out.append(order)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Place limit orders from a limit_orders.txt report")
    parser.add_argument("--file", required=True, help="Path to limit_orders.txt")
    parser.add_argument("--size", type=int, default=1, help="Contracts per order")
    parser.add_argument(
        "--use-basket-max",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap size by Basket max size at limit (default true)",
    )
    parser.add_argument(
        "--require-safe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require Safety: SAFE in report before placing orders",
    )
    parser.add_argument(
        "--post-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set post_only=true on orders (default false)",
    )
    parser.add_argument(
        "--max-orders",
        type=int,
        default=20,
        help="Max orders to submit in one batch (default 20)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit orders (otherwise dry-run)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info (key id and key path)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_text = Path(args.file).read_text(encoding="utf-8")
    side, basket_max, safety, orders_in = parse_limit_report(report_text)

    if not orders_in:
        print("No orders found in report.")
        return

    size = args.size
    if args.use_basket_max and basket_max is not None:
        size = min(size, basket_max)
    if size <= 0:
        print("Size resolved to 0; no orders will be placed.")
        return

    if args.require_safe and safety is not True:
        print("Safety is not SAFE; refusing to place orders.")
        return

    client_prefix = f"arb-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    orders = build_orders(side, orders_in, size, args.post_only, client_prefix)
    if args.max_orders is not None and args.max_orders > 0:
        orders = orders[: args.max_orders]

    print(f"Parsed {len(orders)} orders (side={side}, size={size}, post_only={args.post_only})")
    print(f"Trade API base: {TRADE_API_BASE}")

    if not args.execute:
        print("Dry run. Orders payload:")
        print(json.dumps({"orders": orders}, indent=2))
        return

    trade_key_id = os.getenv("KALSHI_TRADE_KEY_ID") or os.getenv("KALSHI_KEY_ID")
    trade_pem_path = os.getenv("KALSHI_TRADE_PRIVATE_KEY_PEM") or os.getenv(
        "KALSHI_PRIVATE_KEY_PEM"
    )
    if not trade_key_id or not trade_pem_path:
        print(
            "Missing trade credentials. Set KALSHI_TRADE_KEY_ID and "
            "KALSHI_TRADE_PRIVATE_KEY_PEM.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.debug:
        print(f"Trade key id: {trade_key_id}")
        print(f"Trade key path: {trade_pem_path}")
    trade_private_key = load_private_key_pem(trade_pem_path)

    place_orders_batch(orders, trade_key_id, trade_private_key)


if __name__ == "__main__":
    main()
