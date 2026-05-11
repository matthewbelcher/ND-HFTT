import argparse
import asyncio
import csv
import json
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import websockets

from kalshi_common import Book, WS_URL, kalshi_ws_headers, now_ms


@dataclass
class BboState:
    bid_px: int
    bid_qty: int
    ask_px: int
    ask_qty: int
    spread: int
    ts_ms: int


def fmt_ts_hms(ts_ms: int) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts_ms / 1000))


def parse_ts_ms(value) -> int:
    if value is None:
        return now_ms()
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return now_ms()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    return now_ms()

class CsvLogger:
    def __init__(self, path: Path, header: Tuple[str, ...]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._lock = asyncio.Lock()
        if self.path.stat().st_size == 0:
            self._writer.writerow(header)
            self._file.flush()

    async def write_row(self, row: Tuple) -> None:
        async with self._lock:
            self._writer.writerow(row)
            self._file.flush()

    def close(self) -> None:
        self._file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalshi BBO tracker + logger")
    parser.add_argument(
        "--markets",
        nargs="+",
        required=True,
        help="Market tickers to subscribe to",
    )
    parser.add_argument(
        "--out",
        default="logs",
        help="Output directory for CSV logs",
    )
    parser.add_argument(
        "--trades",
        action="store_true",
        help="Subscribe to trades channel (if available) and log to trades.csv",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=5.0,
        help="Heartbeat print interval in seconds",
    )
    return parser.parse_args()


async def heartbeat_loop(
    markets,
    last_bbo: Dict[str, BboState],
    last_update_ms: Dict[str, int],
    start_ms: int,
    interval_s: float,
) -> None:
    while True:
        await asyncio.sleep(interval_s)
        now = now_ms()
        for m in markets:
            age_ms = now - last_update_ms.get(m, start_ms)
            age_s = age_ms / 1000.0
            if m in last_bbo:
                bbo = last_bbo[m]
                ts = fmt_ts_hms(now)
                print(
                    f"[{ts}] {m} YES {bbo.bid_qty} @ {bbo.bid_px:02d} | "
                    f"{bbo.ask_qty} @ {bbo.ask_px:02d} spread={bbo.spread}c "
                    f"age={age_s:.1f}s"
                )
            else:
                ts = fmt_ts_hms(now)
                print(f"[{ts}] {m} NO BOOK age={age_s:.1f}s")


def extract_trade_fields(msg_type: str, payload: dict) -> Optional[Tuple[int, str, int, int, str, str]]:
    market = payload.get("market_ticker")
    if not market:
        return None
    ts_ms = parse_ts_ms(payload.get("ts") or payload.get("timestamp_ms") or payload.get("time_ms"))
    price = payload.get("price") or payload.get("price_cents") or payload.get("yes_price")
    qty = payload.get("qty") or payload.get("quantity") or payload.get("size")
    side = payload.get("side") or payload.get("taker_side") or ""
    if price is None or qty is None:
        return None
    return int(ts_ms), market, int(price), int(qty), str(side), msg_type


async def run() -> None:
    args = parse_args()
    markets = list(dict.fromkeys(args.markets))

    key_id = os.environ["KALSHI_KEY_ID"]
    pem_path = os.environ["KALSHI_PRIVATE_KEY_PEM"]
    headers = kalshi_ws_headers(key_id, pem_path)

    out_dir = Path(args.out)
    bbo_logger = CsvLogger(
        out_dir / "bbo.csv",
        ("timestamp_ms", "market_ticker", "bid_px", "bid_qty", "ask_px", "ask_qty", "spread"),
    )
    trade_logger = None
    if args.trades:
        trade_logger = CsvLogger(
            out_dir / "trades.csv",
            ("timestamp_ms", "market_ticker", "price", "qty", "side", "msg_type"),
        )

    books: Dict[str, Book] = {m: Book() for m in markets}
    last_bbo: Dict[str, BboState] = {}
    last_update_ms: Dict[str, int] = {}
    start_ms = now_ms()

    hb_task = asyncio.create_task(
        heartbeat_loop(markets, last_bbo, last_update_ms, start_ms, args.heartbeat)
    )

    channels = ["orderbook_delta"]
    if args.trades:
        channels.append("trades")

    try:
        async with websockets.connect(
            WS_URL,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**23,
        ) as ws:
            sub = {
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": channels,
                    "market_tickers": markets,
                },
            }
            await ws.send(json.dumps(sub))
            print(f"Subscribed: {', '.join(markets)}")

            while True:
                msg = json.loads(await ws.recv())
                mtype = msg.get("type")
                payload = msg.get("msg", {})
                market = payload.get("market_ticker")
                if market not in books:
                    continue

                if mtype == "orderbook_snapshot":
                    books[market].apply_snapshot(payload.get("yes", []), payload.get("no", []))
                elif mtype == "orderbook_delta":
                    side = payload["side"]
                    price = int(payload["price"])
                    delta = int(payload["delta"])
                    books[market].apply_delta(side, price, delta)
                elif args.trades and "trade" in str(mtype).lower():
                    trade = extract_trade_fields(mtype, payload)
                    if trade_logger and trade:
                        await trade_logger.write_row(trade)
                    continue
                else:
                    continue

                bbo = books[market].yes_bbo()
                if not bbo:
                    continue

                bid_px, bid_qty, ask_px, ask_qty = bbo
                spread = ask_px - bid_px
                ts_ms = parse_ts_ms(payload.get("ts") or payload.get("timestamp_ms"))
                last_update_ms[market] = int(ts_ms)
                bbo_state = BboState(bid_px, bid_qty, ask_px, ask_qty, spread, int(ts_ms))
                last_bbo[market] = bbo_state

                await bbo_logger.write_row(
                    (int(ts_ms), market, bid_px, bid_qty, ask_px, ask_qty, spread)
                )

                ts = fmt_ts_hms(int(ts_ms))
                print(
                    f"[{ts}] {market} YES {bid_qty} @ {bid_px:02d} | "
                    f"{ask_qty} @ {ask_px:02d} spread={spread}c"
                )
    finally:
        hb_task.cancel()
        bbo_logger.close()
        if trade_logger:
            trade_logger.close()


if __name__ == "__main__":
    asyncio.run(run())
