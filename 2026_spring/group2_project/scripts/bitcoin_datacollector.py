#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK


COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"

CSV_COLUMNS = [
    "trade_id",
    "time",
    "price",
    "size",
    "side",
]

MAX_BACKOFF_SECONDS = 60
DEFAULT_BUFFER_SIZE = 500
FLUSH_INTERVAL_SECONDS = 5.0
STATS_WINDOW_SECONDS = 5.0


class TradeBuffer:
    """Owns the CSV file handle and an in-memory row buffer.

    Not thread-safe by design — intended for single-event-loop use only.
    """

    def __init__(self, filepath: Path, buffer_size: int) -> None:
        self._filepath = filepath
        self._buffer_size = buffer_size
        self._rows: list[dict] = []
        self._file = None
        self._writer: csv.DictWriter | None = None
        self._total_written: int = 0

    def open(self) -> None:
        """Create the output file and write the CSV header."""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._file.flush()

    def write(self, row: dict) -> bool:
        """Append a row to the buffer. Returns True if an auto-flush occurred."""
        self._rows.append(row)
        if len(self._rows) >= self._buffer_size:
            self.flush()
            return True
        return False

    def flush(self) -> int:
        """Write buffered rows to disk. Returns the number of rows flushed."""
        if not self._rows:
            return 0
        count = len(self._rows)
        self._writer.writerows(self._rows)
        self._file.flush()
        self._rows = []
        self._total_written += count
        return count

    def close(self) -> None:
        """Flush any remaining rows and close the file handle."""
        if self._file is not None:
            self.flush()
            self._file.close()
            self._file = None

    @property
    def total_written(self) -> int:
        return self._total_written

    @property
    def pending(self) -> int:
        return len(self._rows)


class StatsTracker:
    """Tracks a rolling-window trades/sec rate and the latest price."""

    def __init__(self, window_seconds: float = STATS_WINDOW_SECONDS) -> None:
        self._window = window_seconds
        self._timestamps: collections.deque[float] = collections.deque()
        self._last_price: str = "N/A"
        self._total: int = 0

    def record(self, price: str) -> None:
        """Record a trade arrival."""
        now = time.monotonic()
        self._timestamps.append(now)
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        self._last_price = price
        self._total += 1

    def trades_per_second(self) -> float:
        """Return the rolling-window rate."""
        now = time.monotonic()
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        return len(self._timestamps) / self._window

    @property
    def total(self) -> int:
        return self._total

    @property
    def last_price(self) -> str:
        return self._last_price


def parse_trades(msg: str) -> list[dict]:
    """Parse a raw Coinbase WebSocket message into a list of CSV row dicts.

    Coinbase messages can contain multiple trades per message. Returns an
    empty list for non-trade messages (subscription acks, heartbeats, etc.)
    and on any parse error.

    Only 'update' events are returned — the initial 'snapshot' of recent
    trades is skipped so the CSV contains only live trades from connection time.
    """
    try:
        data = json.loads(msg)
        if data.get("channel") != "market_trades":
            return []
        rows = []
        for event in data.get("events", []):
            if event.get("type") != "update":
                continue
            for trade in event.get("trades", []):
                rows.append(
                    {
                        "trade_id": trade["trade_id"],
                        "time": trade["time"],
                        "price": trade["price"],
                        "size": trade["size"],
                        "side": trade["side"],
                    }
                )
        return rows
    except (KeyError, json.JSONDecodeError, TypeError):
        return []


def build_output_path(output_dir: str, symbol: str) -> Path:
    """Return a timestamped CSV path inside output_dir."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.lower().replace("-", "")
    filename = f"{safe_symbol}_trades_{ts}.csv"
    return Path(output_dir) / filename


async def consume_stream(
    symbol: str,
    buffer: TradeBuffer,
    stats: StatsTracker,
    shutdown_event: asyncio.Event,
) -> None:
    """Open one Coinbase WebSocket session and consume messages until shutdown or disconnect."""
    async with websockets.connect(
        COINBASE_WS_URL,
        ping_interval=20,
        ping_timeout=10,
        close_timeout=5,
    ) as ws:
        subscribe_msg = json.dumps(
            {
                "type": "subscribe",
                "product_ids": [symbol.upper()],
                "channel": "market_trades",
            }
        )
        await ws.send(subscribe_msg)

        async for message in ws:
            if shutdown_event.is_set():
                break
            for row in parse_trades(message):
                buffer.write(row)
                stats.record(row["price"])


async def run_with_reconnect(
    symbol: str,
    buffer: TradeBuffer,
    stats: StatsTracker,
    shutdown_event: asyncio.Event,
) -> None:
    """Run consume_stream with exponential-backoff reconnection."""
    logger = logging.getLogger("trade_collector")
    backoff = 1

    while not shutdown_event.is_set():
        try:
            logger.info(
                "Connecting to %s (symbol: %s)", COINBASE_WS_URL, symbol.upper()
            )
            await consume_stream(symbol, buffer, stats, shutdown_event)
            backoff = 1  # clean exit only happens during shutdown
        except (ConnectionClosedError, ConnectionClosedOK) as exc:
            if shutdown_event.is_set():
                break
            logger.warning(
                "Connection closed (%s). Reconnecting in %ds...", exc, backoff
            )
        except OSError as exc:
            if shutdown_event.is_set():
                break
            logger.warning("Network error (%s). Reconnecting in %ds...", exc, backoff)
        except Exception as exc:  # noqa: BLE001
            if shutdown_event.is_set():
                break
            logger.error("Unexpected error (%s). Reconnecting in %ds...", exc, backoff)

        if not shutdown_event.is_set():
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)


async def periodic_flush(
    buffer: TradeBuffer,
    stats: StatsTracker,
    interval: float,
    shutdown_event: asyncio.Event,
) -> None:
    """Flush the buffer and print live stats every `interval` seconds."""
    while not shutdown_event.is_set():
        await asyncio.sleep(interval)
        buffer.flush()
        rate = stats.trades_per_second()
        print(
            f"\r[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] "
            f"Total: {stats.total:,} | "
            f"Rate: {rate:.1f} trades/s | "
            f"Price: ${stats.last_price} | "
            f"Pending: {buffer.pending}   ",
            end="",
            flush=True,
        )


async def duration_watchdog(seconds: float, shutdown_event: asyncio.Event) -> None:
    """Set shutdown_event after `seconds` have elapsed."""
    logger = logging.getLogger("trade_collector")
    await asyncio.sleep(seconds)
    logger.info("Duration limit reached (%.0fs). Shutting down.", seconds)
    shutdown_event.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Coinbase tick-by-tick trade data to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        default="BTC-USD",
        help="Trading pair symbol (default: BTC-USD)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.cwd()),
        help="Directory to write CSV files (default: current directory)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Stop collecting after N seconds (default: run forever)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_SIZE,
        metavar="N",
        help=f"Flush to disk after N rows (default: {DEFAULT_BUFFER_SIZE})",
    )
    return parser.parse_args()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger("trade_collector")

    args = parse_args()
    symbol = args.symbol.upper()

    output_path = build_output_path(args.output_dir, symbol)
    logger.info("Output file: %s", output_path)

    buffer = TradeBuffer(output_path, args.buffer_size)
    stats = StatsTracker()
    shutdown_event = asyncio.Event()

    # Signal handling — add_signal_handler is not available on Windows
    loop = asyncio.get_running_loop()

    def _handle_sigint() -> None:
        if not shutdown_event.is_set():
            print("\nShutdown requested. Flushing buffer...")
            shutdown_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _handle_sigint)
    except NotImplementedError:
        logger.debug(
            "SIGINT handler unavailable on this platform; using KeyboardInterrupt fallback."
        )

    buffer.open()
    try:
        tasks = [
            asyncio.create_task(
                run_with_reconnect(symbol, buffer, stats, shutdown_event)
            ),
            asyncio.create_task(
                periodic_flush(buffer, stats, FLUSH_INTERVAL_SECONDS, shutdown_event)
            ),
        ]
        if args.duration is not None:
            tasks.append(
                asyncio.create_task(duration_watchdog(args.duration, shutdown_event))
            )

        await asyncio.gather(*tasks, return_exceptions=True)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught. Flushing buffer...")
        shutdown_event.set()
    finally:
        buffer.close()
        print(f"\nDone. Total trades written: {buffer.total_written:,}")
        print(f"File: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
