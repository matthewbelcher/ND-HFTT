#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg


@dataclass(frozen=True)
class TestAccount:
    account_id: str
    username: str
    password: str
    session_token: str


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


def _uuid() -> str:
    return str(uuid.uuid4())


def _uuid_csv(values: list[str]) -> str:
    return ", ".join(f"'{value}'::uuid" for value in values)


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[idx: idx + size] for idx in range(0, len(values), size)]


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


def _safe_rate_interval(rate: int) -> float:
    if rate <= 0:
        raise ValueError("rate must be positive")
    return 1.0 / float(rate)


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


async def _create_pool(dsn: str) -> asyncpg.Pool:
    async def _dsql_safe_reset(_conn: asyncpg.Connection) -> None:
        return None

    return await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=12, reset=_dsql_safe_reset)


async def _create_and_auth_account(host: str, port: int, auth_token: str, username: str, password: str) -> TestAccount:
    create_resp = await _send_tcp_json(
        host,
        port,
        {
            "action": "create_account",
            "auth_token": auth_token,
            "request": {
                "username": username,
                "password": password,
                "external_user_id": f"bench-{username}",
            },
        },
    )
    if not create_resp.get("ok"):
        raise RuntimeError(f"create_account failed for {username}: {create_resp}")

    auth_resp = await _send_tcp_json(
        host,
        port,
        {
            "action": "authenticate_account",
            "auth_token": auth_token,
            "request": {"username": username, "password": password},
        },
    )
    if not auth_resp.get("ok"):
        raise RuntimeError(f"authenticate_account failed for {username}: {auth_resp}")

    return TestAccount(
        account_id=create_resp["result"]["account_id"],
        username=username,
        password=password,
        session_token=auth_resp["result"]["account_session_token"],
    )


async def _create_market(conn: asyncpg.Connection, run_id: str, point_label: str) -> str:
    market_id = _uuid()
    await conn.execute(
        """
        INSERT INTO markets (
            market_id, slug, title, description, status,
            tick_size_cents, min_price_cents, max_price_cents,
            close_time, resolve_time, created_by
        )
        VALUES (
            $1, $2, $3, $4, 'ACTIVE',
            1, 1, 99,
            now() + interval '1 day',
            now() + interval '2 days',
            NULL
        )
        """,
        market_id,
        f"{run_id}-{point_label}",
        f"{run_id} {point_label}",
        "benchmark submit-only market",
    )
    return market_id


ACCOUNT_CASH_BUCKETS = 16


def _split_amount(amount_cents: int, n: int) -> list[int]:
    base, rem = divmod(amount_cents, n)
    out = [base] * n
    out[0] += rem
    return out


async def _fund_accounts(conn: asyncpg.Connection, account_ids: list[str], amount_cents: int) -> None:
    if not account_ids:
        return
    account_id_sql = _uuid_csv(account_ids)
    await conn.execute(
        f"""
        UPDATE accounts
        SET available_cash_cents = {int(amount_cents)},
            locked_cash_cents = 0,
            updated_at = now()
        WHERE account_id IN ({account_id_sql})
        """
    )
    avail_split = _split_amount(int(amount_cents), ACCOUNT_CASH_BUCKETS)
    for account_id in account_ids:
        for b in range(ACCOUNT_CASH_BUCKETS):
            await conn.execute(
                """
                INSERT INTO account_cash_buckets (
                    account_id, bucket_id, available_cash_cents, locked_cash_cents
                )
                VALUES ($1, $2, $3, 0)
                ON CONFLICT (account_id, bucket_id) DO UPDATE
                SET available_cash_cents = EXCLUDED.available_cash_cents,
                    locked_cash_cents    = 0,
                    updated_at = now()
                """,
                account_id,
                b,
                avail_split[b],
            )


async def _delete_orders_for_markets(conn: asyncpg.Connection, market_ids: list[str]) -> None:
    if not market_ids:
        return
    for batch in _chunks(market_ids, 4):
        market_id_sql = _uuid_csv(batch)
        order_rows = await conn.fetch(
            f"SELECT request_id FROM orders WHERE market_id IN ({market_id_sql})",
        )
        order_ids = [str(row["request_id"]) for row in order_rows]
        for order_batch in _chunks(order_ids, 100):
            order_id_sql = _uuid_csv(order_batch)
            async with conn.transaction():
                await conn.execute(
                    f"DELETE FROM ledger_entries WHERE order_id IN ({order_id_sql})",
                )
                await conn.execute(
                    f"DELETE FROM orders WHERE request_id IN ({order_id_sql})",
                )
        async with conn.transaction():
            await conn.execute(
                f"DELETE FROM ledger_entries WHERE market_id IN ({market_id_sql})",
            )
            await conn.execute(
                f"DELETE FROM match_work_queue WHERE market_id IN ({market_id_sql})",
            )
            await conn.execute(
                f"DELETE FROM markets WHERE market_id IN ({market_id_sql})",
            )


async def _cleanup(pool: asyncpg.Pool, market_ids: list[str], account_ids: list[str]) -> None:
    async with pool.acquire() as conn:
        if market_ids:
            for batch in _chunks(market_ids, 4):
                await _delete_orders_for_markets(conn, batch)
        if account_ids:
            for batch in _chunks(account_ids, 25):
                account_id_sql = _uuid_csv(batch)
                async with conn.transaction():
                    await conn.execute(
                        f"DELETE FROM account_auth_sessions WHERE account_id IN ({account_id_sql})",
                    )
                    await conn.execute(
                        f"DELETE FROM account_cash_buckets WHERE account_id IN ({account_id_sql})",
                    )
                    await conn.execute(
                        f"DELETE FROM accounts WHERE account_id IN ({account_id_sql})",
                    )


def _start_listener(repo: Path, args: argparse.Namespace) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(
        {
            "CLIENT_LISTENER_HOST": args.listener_host,
            "CLIENT_LISTENER_PORT": str(args.listener_port),
            "CLIENT_LISTENER_DB_DSN": args.db_dsn,
            "CLIENT_LISTENER_AUTH_MODE": "static-token",
            "CLIENT_LISTENER_AUTH_TOKEN": args.auth_token,
        }
    )
    listener_dir = repo / "client" / "client-listener"
    log_dir = repo / "test" / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "submit-scaling-listener.log"
    log_file = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=str(listener_dir),
            env=env,
            stdout=log_file,
            stderr=log_file,
            text=True,
        )
    except Exception:
        log_file.close()
        raise


async def _run_client(
    *,
    host: str,
    port: int,
    auth_token: str,
    account: TestAccount,
    market_id: str,
    price_cents: int,
    qty: int,
    time_in_force: str,
    orders_per_client: int,
    rate_per_second: int,
) -> ClientResult:
    result = ClientResult()
    interval = _safe_rate_interval(rate_per_second)
    pending: deque[tuple[str, int]] = deque()
    reader, writer = await asyncio.open_connection(host, port)

    async def reader_task() -> None:
        while result.acked < orders_per_client:
            raw = await reader.readline()
            if not raw:
                raise RuntimeError("listener connection closed before all responses were received")
            ack_ns = time.perf_counter_ns()
            payload = json.loads(raw.decode("utf-8"))
            req_id, sent_ns = pending.popleft()
            response_request_id = payload.get("order", {}).get("request_id")
            if response_request_id and response_request_id != req_id:
                raise RuntimeError("response/request ordering mismatch on a single connection")
            result.acked += 1
            result.latency_ms.append((ack_ns - sent_ns) / 1_000_000.0)
            if payload.get("ok"):
                result.ok += 1
            else:
                result.errors += 1

    reader_fut = asyncio.create_task(reader_task())
    start = time.perf_counter()
    try:
        for idx in range(orders_per_client):
            target = start + (idx * interval)
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

            request_id = _uuid()
            sent_ns = time.perf_counter_ns()
            payload = {
                "action": "submit_order",
                "auth_token": auth_token,
                "request": {
                    "request_id": request_id,
                    "account_id": account.account_id,
                    "account_session_token": account.session_token,
                    "market_id": market_id,
                    "side": "YES",
                    "qty": qty,
                    "price_cents": price_cents,
                    "time_in_force": time_in_force,
                    "ingress_ts_ns": time.time_ns(),
                },
            }
            writer.write(_json_line(payload))
            pending.append((request_id, sent_ns))
            result.sent += 1
            if (idx + 1) % 32 == 0:
                await writer.drain()

        await writer.drain()
        await reader_fut
    finally:
        if not reader_fut.done():
            reader_fut.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_fut
        writer.close()
        await writer.wait_closed()
    return result


async def _run_point(
    *,
    host: str,
    port: int,
    auth_token: str,
    accounts: list[TestAccount],
    market_id: str,
    orders_per_client: int,
    rate_per_second: int,
    price_cents: int,
    qty: int,
    time_in_force: str,
) -> dict[str, Any]:
    point_start = time.perf_counter()
    results = await asyncio.gather(
        *[
            _run_client(
                host=host,
                port=port,
                auth_token=auth_token,
                account=account,
                market_id=market_id,
                price_cents=price_cents,
                qty=qty,
                time_in_force=time_in_force,
                orders_per_client=orders_per_client,
                rate_per_second=rate_per_second,
            )
            for account in accounts
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
        "clients": len(accounts),
        "target_orders_per_second_per_client": rate_per_second,
        "orders_per_client": orders_per_client,
        "total_orders_sent": sent,
        "total_orders_acked": acked,
        "ok_orders": ok,
        "error_count": errors,
        "wall_time_sec": round(wall_time, 6),
        "avg_throughput_ops": round(throughput, 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "p50_latency_ms": round(_percentile(latencies, 0.50), 3) if latencies else 0.0,
        "p95_latency_ms": round(_percentile(latencies, 0.95), 3) if latencies else 0.0,
        "max_latency_ms": round(latencies[-1], 3) if latencies else 0.0,
    }


def _parse_clients(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value < 1:
            raise ValueError("client counts must be positive")
        values.append(value)
    if not values:
        raise ValueError("at least one client count is required")
    return values


async def _run(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[1]
    output_path = Path(args.output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clients = _parse_clients(args.clients)
    max_clients = max(clients)
    listener_proc: subprocess.Popen[str] | None = None
    pool = await _create_pool(args.db_dsn)
    run_id = f"submit-bench-{uuid.uuid4().hex[:10]}"
    created_markets: list[str] = []
    created_accounts: list[TestAccount] = []

    try:
        if args.start_listener:
            listener_proc = _start_listener(repo, args)
        await _wait_for_listener(args.listener_host, args.listener_port)

        for idx in range(max_clients):
            username = f"{run_id}-user-{idx + 1}"
            password = f"BenchPass!{idx + 1:04d}"
            created_accounts.append(
                await _create_and_auth_account(
                    args.listener_host,
                    args.listener_port,
                    args.auth_token,
                    username,
                    password,
                )
            )

        async with pool.acquire() as conn:
            await _fund_accounts(
                conn,
                [account.account_id for account in created_accounts],
                args.account_cash_cents,
            )

        fieldnames = [
            "clients",
            "target_orders_per_second_per_client",
            "orders_per_client",
            "total_orders_sent",
            "total_orders_acked",
            "ok_orders",
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
                async with pool.acquire() as conn:
                    market_id = await _create_market(conn, run_id, f"c{client_count}")
                created_markets.append(market_id)

                row = await _run_point(
                    host=args.listener_host,
                    port=args.listener_port,
                    auth_token=args.auth_token,
                    accounts=created_accounts[:client_count],
                    market_id=market_id,
                    orders_per_client=args.orders_per_client,
                    rate_per_second=args.client_rate,
                    price_cents=args.price_cents,
                    qty=args.qty,
                    time_in_force=args.time_in_force,
                )
                writer.writerow(row)
                csv_file.flush()
                print(
                    f"clients={row['clients']} ok_orders={row['ok_orders']} "
                    f"throughput={row['avg_throughput_ops']} ops/s avg_latency={row['avg_latency_ms']} ms "
                    f"p95={row['p95_latency_ms']} ms errors={row['error_count']}"
                )

        return 0
    finally:
        await _cleanup(
            pool,
            created_markets,
            [account.account_id for account in created_accounts],
        )
        await pool.close()
        if listener_proc is not None:
            if listener_proc.poll() is None:
                listener_proc.send_signal(signal.SIGTERM)
                try:
                    listener_proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    listener_proc.kill()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TCP order submit throughput and latency while scaling clients")
    parser.add_argument("--db-dsn", required=True)
    parser.add_argument("--auth-token", default="dev-shared-token")
    parser.add_argument("--listener-host", default="127.0.0.1")
    parser.add_argument("--listener-port", type=int, default=9501)
    parser.add_argument("--clients", default="1,2,4,8,16,32,48,64,96,128")
    parser.add_argument("--orders-per-client", type=int, default=1000)
    parser.add_argument("--client-rate", type=int, default=1000)
    parser.add_argument("--price-cents", type=int, default=60)
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--time-in-force", default="GTC")
    parser.add_argument("--account-cash-cents", type=int, default=100_000_000)
    parser.add_argument("--output-csv", default="test/results/submit-scaling.csv")
    parser.add_argument("--start-listener", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_build_parser().parse_args())))
