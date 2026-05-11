#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ClientResult:
    sent: int = 0
    acked: int = 0
    ok: int = 0
    errors: int = 0
    latency_ms: list[float] | None = None

    def __post_init__(self) -> None:
        if self.latency_ms is None:
            self.latency_ms = []


def _json_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _parse_clients(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(v < 1 for v in values):
        raise ValueError("clients must be a comma-separated list of positive integers")
    return values


async def _send_tcp_json(host: str, port: int, payload: dict[str, Any]) -> dict[str, Any]:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(_json_line(payload))
        await writer.drain()
        raw = await reader.readline()
        return json.loads(raw.decode("utf-8"))
    finally:
        writer.close()
        await writer.wait_closed()


async def _wait_for_listener(host: str, port: int, timeout_s: float = 30.0) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        try:
            resp = await _send_tcp_json(host, port, {"action": "ping"})
            if resp.get("ok") and resp.get("pong"):
                return
        except Exception:
            pass
        await asyncio.sleep(0.25)
    raise RuntimeError(f"listener on {host}:{port} did not become ready")


async def _run_client(
    *,
    host: str,
    port: int,
    action: str,
    requests_per_client: int,
    rate_per_second: int,
) -> ClientResult:
    if rate_per_second <= 0:
        raise ValueError("rate_per_second must be positive")

    result = ClientResult()
    interval = 1.0 / float(rate_per_second)
    pending: deque[int] = deque()
    reader, writer = await asyncio.open_connection(host, port)

    async def reader_task() -> None:
        while result.acked < requests_per_client:
            raw = await reader.readline()
            if not raw:
                raise RuntimeError("listener connection closed before all responses were received")
            ack_ns = time.perf_counter_ns()
            sent_ns = pending.popleft()
            payload = json.loads(raw.decode("utf-8"))
            result.acked += 1
            result.latency_ms.append((ack_ns - sent_ns) / 1_000_000.0)
            if payload.get("ok"):
                result.ok += 1
            else:
                result.errors += 1

    reader_fut = asyncio.create_task(reader_task())
    start = time.perf_counter()
    try:
        for idx in range(requests_per_client):
            target = start + (idx * interval)
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

            pending.append(time.perf_counter_ns())
            writer.write(_json_line({"action": action}))
            result.sent += 1
            if (idx + 1) % 64 == 0:
                await writer.drain()

        await writer.drain()
        await reader_fut
    finally:
        if not reader_fut.done():
            reader_fut.cancel()
            try:
                await reader_fut
            except asyncio.CancelledError:
                pass
        writer.close()
        await writer.wait_closed()
    return result


async def _run_point(
    *,
    host: str,
    port: int,
    action: str,
    client_count: int,
    requests_per_client: int,
    rate_per_second: int,
) -> dict[str, Any]:
    point_start = time.perf_counter()
    results = await asyncio.gather(
        *[
            _run_client(
                host=host,
                port=port,
                action=action,
                requests_per_client=requests_per_client,
                rate_per_second=rate_per_second,
            )
            for _ in range(client_count)
        ]
    )
    point_end = time.perf_counter()

    latencies = sorted(lat for item in results for lat in item.latency_ms)
    sent = sum(item.sent for item in results)
    acked = sum(item.acked for item in results)
    ok = sum(item.ok for item in results)
    errors = sum(item.errors for item in results)
    wall_time = point_end - point_start
    throughput = ok / wall_time if wall_time > 0 else 0.0

    return {
        "action": action,
        "clients": client_count,
        "target_requests_per_second_per_client": rate_per_second,
        "requests_per_client": requests_per_client,
        "total_requests_sent": sent,
        "total_requests_acked": acked,
        "ok_requests": ok,
        "error_count": errors,
        "wall_time_sec": round(wall_time, 6),
        "avg_throughput_ops": round(throughput, 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "p50_latency_ms": round(_percentile(latencies, 0.50), 3) if latencies else 0.0,
        "p95_latency_ms": round(_percentile(latencies, 0.95), 3) if latencies else 0.0,
        "max_latency_ms": round(latencies[-1], 3) if latencies else 0.0,
    }


async def _run(args: argparse.Namespace) -> int:
    output_path = Path(args.output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clients = _parse_clients(args.clients)
    await _wait_for_listener(args.listener_host, args.listener_port)

    fieldnames = [
        "action",
        "clients",
        "target_requests_per_second_per_client",
        "requests_per_client",
        "total_requests_sent",
        "total_requests_acked",
        "ok_requests",
        "error_count",
        "wall_time_sec",
        "avg_throughput_ops",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "max_latency_ms",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for client_count in clients:
            row = await _run_point(
                host=args.listener_host,
                port=args.listener_port,
                action=args.action,
                client_count=client_count,
                requests_per_client=args.requests_per_client,
                rate_per_second=args.client_rate,
            )
            writer.writerow(row)
            csv_file.flush()
            print(
                f"action={row['action']} clients={row['clients']} "
                f"ok_requests={row['ok_requests']} "
                f"throughput={row['avg_throughput_ops']} ops/s "
                f"avg_latency={row['avg_latency_ms']} ms "
                f"p95={row['p95_latency_ms']} ms errors={row['error_count']}"
            )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark lightweight TCP RPC latency and throughput")
    parser.add_argument("--listener-host", required=True)
    parser.add_argument("--listener-port", type=int, required=True)
    parser.add_argument("--action", default="ping", choices=["ping", "health", "ready"])
    parser.add_argument("--clients", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--requests-per-client", type=int, default=1000)
    parser.add_argument("--client-rate", type=int, default=1000)
    parser.add_argument("--output-csv", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_build_parser().parse_args())))
