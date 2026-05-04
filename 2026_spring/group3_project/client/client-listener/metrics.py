"""Minimal in-process Prometheus-compatible metrics for the listener."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any


_DEFAULT_BUCKETS_MS: tuple[float, ...] = (
    1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0,
)


class Metrics:
    def __init__(self, bucket_upper_bounds_ms: tuple[float, ...] = _DEFAULT_BUCKETS_MS):
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._buckets_ms = bucket_upper_bounds_ms
        self._hist_buckets: dict[str, list[int]] = {}
        self._hist_count: dict[str, int] = {}
        self._hist_sum_ms: dict[str, float] = {}

    def incr(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe_ms(self, name: str, value_ms: float) -> None:
        with self._lock:
            buckets = self._hist_buckets.get(name)
            if buckets is None:
                buckets = [0] * len(self._buckets_ms)
                self._hist_buckets[name] = buckets
                self._hist_count[name] = 0
                self._hist_sum_ms[name] = 0.0
            for i, upper in enumerate(self._buckets_ms):
                if value_ms <= upper:
                    buckets[i] += 1
                    break
            self._hist_count[name] += 1
            self._hist_sum_ms[name] += value_ms

    def render_prometheus(self) -> str:
        with self._lock:
            lines: list[str] = []
            for name, val in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {val}")
            for name, val in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {val}")
            for name in sorted(self._hist_buckets.keys()):
                buckets = self._hist_buckets[name]
                count = self._hist_count[name]
                sum_ms = self._hist_sum_ms[name]
                lines.append(f"# TYPE {name} histogram")
                cumulative = 0
                for i, upper in enumerate(self._buckets_ms):
                    cumulative += buckets[i]
                    lines.append(f'{name}_bucket{{le="{upper}"}} {cumulative}')
                lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                lines.append(f"{name}_sum {sum_ms}")
                lines.append(f"{name}_count {count}")
            return "\n".join(lines) + "\n"


METRICS = Metrics()


class Timer:
    __slots__ = ("metrics", "name", "_start_ns")

    def __init__(self, metrics: Metrics, name: str):
        self.metrics = metrics
        self.name = name
        self._start_ns = 0

    def __enter__(self) -> "Timer":
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000.0
        self.metrics.observe_ms(self.name, elapsed_ms)


async def _handle_metrics_http(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        try:
            await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=2.0)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            pass
        body = METRICS.render_prometheus().encode("utf-8")
        head = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain; version=0.0.4\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
            b"Connection: close\r\n"
            b"\r\n"
        )
        writer.write(head + body)
        await writer.drain()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def serve_metrics(host: str, port: int) -> Any:
    return await asyncio.start_server(_handle_metrics_http, host, port, reuse_port=True)
