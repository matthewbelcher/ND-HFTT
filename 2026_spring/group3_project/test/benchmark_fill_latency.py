#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg


ACCOUNT_CASH_BUCKETS = 16


@dataclass(frozen=True)
class TestAccount:
    account_id: str
    username: str
    password: str
    session_token: str


@dataclass
class ClientResult:
    submitted_pairs: int = 0
    filled_pairs: int = 0
    submit_errors: int = 0
    timeout_errors: int = 0
    latency_ms: list[float] | None = None

    def __post_init__(self) -> None:
        if self.latency_ms is None:
            self.latency_ms = []


class ListenerSession:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def __aenter__(self) -> "ListenerSession":
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.reader is None or self.writer is None:
            raise RuntimeError("session is not open")
        self.writer.write(_json_line(payload))
        await self.writer.drain()
        raw = await self.reader.readline()
        if not raw:
            raise RuntimeError("listener closed connection")
        return json.loads(raw.decode("utf-8"))


def _json_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def _uuid() -> str:
    return str(uuid.uuid4())


def _uuid_csv(values: list[str]) -> str:
    return ", ".join(f"'{value}'::uuid" for value in values)


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _split_amount(amount_cents: int, n: int) -> list[int]:
    base, rem = divmod(amount_cents, n)
    out = [base] * n
    out[0] += rem
    return out


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


async def _create_pool(dsn: str) -> asyncpg.Pool:
    async def _dsql_safe_reset(_conn: asyncpg.Connection) -> None:
        return None

    return await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=16, reset=_dsql_safe_reset)


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
                "external_user_id": f"fill-bench-{username}",
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


async def _create_market(conn: asyncpg.Connection, run_id: str, client_idx: int) -> str:
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
        f"fill-bench-{run_id}-c{client_idx}",
        f"fill bench {run_id} client {client_idx}",
        "fill latency benchmark market",
    )
    return market_id


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
                    locked_cash_cents = 0,
                    updated_at = now()
                """,
                account_id,
                b,
                avail_split[b],
            )


async def _submit_order(
    session: ListenerSession,
    *,
    auth_token: str,
    account: TestAccount,
    market_id: str,
    side: str,
    price_cents: int,
    qty: int,
) -> str:
    request_id = _uuid()
    resp = await session.call(
        {
            "action": "submit_order",
            "auth_token": auth_token,
            "request": {
                "request_id": request_id,
                "account_id": account.account_id,
                "account_session_token": account.session_token,
                "market_id": market_id,
                "side": side,
                "qty": qty,
                "price_cents": price_cents,
                "time_in_force": "GTC",
                "ingress_ts_ns": time.time_ns(),
            },
        }
    )
    if not resp.get("ok"):
        raise RuntimeError(f"submit_order failed: {resp}")
    return request_id


async def _wait_for_trade(
    pool: asyncpg.Pool,
    *,
    market_id: str,
    yes_order_id: str,
    no_order_id: str,
    timeout_s: float,
    poll_interval_ms: int,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        async with pool.acquire() as conn:
            trade_id = await conn.fetchval(
                """
                SELECT trade_id
                FROM trades
                WHERE market_id = $1
                  AND (
                    (resting_order_id = $2 AND aggressing_order_id = $3)
                    OR
                    (resting_order_id = $3 AND aggressing_order_id = $2)
                  )
                LIMIT 1
                """,
                market_id,
                yes_order_id,
                no_order_id,
            )
        if trade_id is not None:
            return True
        await asyncio.sleep(poll_interval_ms / 1000)
    return False


async def _run_fill_client(
    *,
    pool: asyncpg.Pool,
    host: str,
    port: int,
    auth_token: str,
    yes_account: TestAccount,
    no_account: TestAccount,
    market_id: str,
    pairs_per_client: int,
    rate_per_second: int,
    timeout_s: float,
    poll_interval_ms: int,
) -> ClientResult:
    result = ClientResult()
    interval = 1.0 / float(rate_per_second)
    start = time.perf_counter()
    async with ListenerSession(host, port) as session:
        for idx in range(pairs_per_client):
            target = start + (idx * interval)
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                yes_order_id = await _submit_order(
                    session,
                    auth_token=auth_token,
                    account=yes_account,
                    market_id=market_id,
                    side="YES",
                    price_cents=60,
                    qty=1,
                )
                fill_start_ns = time.perf_counter_ns()
                no_order_id = await _submit_order(
                    session,
                    auth_token=auth_token,
                    account=no_account,
                    market_id=market_id,
                    side="NO",
                    price_cents=40,
                    qty=1,
                )
                result.submitted_pairs += 1
            except Exception:
                result.submit_errors += 1
                continue

            matched = await _wait_for_trade(
                pool,
                market_id=market_id,
                yes_order_id=yes_order_id,
                no_order_id=no_order_id,
                timeout_s=timeout_s,
                poll_interval_ms=poll_interval_ms,
            )
            if matched:
                result.filled_pairs += 1
                result.latency_ms.append((time.perf_counter_ns() - fill_start_ns) / 1_000_000.0)
            else:
                result.timeout_errors += 1
    return result


async def _run_point(
    *,
    pool: asyncpg.Pool,
    host: str,
    port: int,
    auth_token: str,
    accounts: list[tuple[TestAccount, TestAccount, str]],
    client_count: int,
    pairs_per_client: int,
    rate_per_second: int,
    timeout_s: float,
    poll_interval_ms: int,
) -> dict[str, Any]:
    point_start = time.perf_counter()
    results = await asyncio.gather(
        *[
            _run_fill_client(
                pool=pool,
                host=host,
                port=port,
                auth_token=auth_token,
                yes_account=accounts[idx][0],
                no_account=accounts[idx][1],
                market_id=accounts[idx][2],
                pairs_per_client=pairs_per_client,
                rate_per_second=rate_per_second,
                timeout_s=timeout_s,
                poll_interval_ms=poll_interval_ms,
            )
            for idx in range(client_count)
        ]
    )
    wall_time = time.perf_counter() - point_start
    latencies = sorted(lat for item in results for lat in item.latency_ms)
    submitted_pairs = sum(item.submitted_pairs for item in results)
    filled_pairs = sum(item.filled_pairs for item in results)
    submit_errors = sum(item.submit_errors for item in results)
    timeout_errors = sum(item.timeout_errors for item in results)
    throughput = filled_pairs / wall_time if wall_time > 0 else 0.0
    return {
        "clients": client_count,
        "pairs_per_client": pairs_per_client,
        "target_pairs_per_second_per_client": rate_per_second,
        "submitted_pairs": submitted_pairs,
        "filled_pairs": filled_pairs,
        "submit_errors": submit_errors,
        "timeout_errors": timeout_errors,
        "wall_time_sec": round(wall_time, 6),
        "fill_throughput_pairs_sec": round(throughput, 3),
        "avg_fill_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "p50_fill_latency_ms": round(_percentile(latencies, 0.50), 3) if latencies else 0.0,
        "p95_fill_latency_ms": round(_percentile(latencies, 0.95), 3) if latencies else 0.0,
        "max_fill_latency_ms": round(latencies[-1], 3) if latencies else 0.0,
    }


async def _delete_rows(conn: asyncpg.Connection, sql: str, ids: list[str], batch_size: int = 100) -> None:
    for batch in _chunks(ids, batch_size):
        id_sql = _uuid_csv(batch)
        async with conn.transaction():
            await conn.execute(sql.format(id_sql=id_sql))


async def _cleanup(pool: asyncpg.Pool, market_ids: list[str], account_ids: list[str]) -> None:
    async with pool.acquire() as conn:
        for market_batch in _chunks(market_ids, 4):
            m_sql = _uuid_csv(market_batch)
            trade_rows = await conn.fetch(f"SELECT trade_id FROM trades WHERE market_id IN ({m_sql})")
            trade_ids = [str(row["trade_id"]) for row in trade_rows]
            if trade_ids:
                await _delete_rows(conn, "DELETE FROM trade_parties WHERE trade_id IN ({id_sql})", trade_ids)
            order_rows = await conn.fetch(f"SELECT request_id FROM orders WHERE market_id IN ({m_sql})")
            order_ids = [str(row["request_id"]) for row in order_rows]
            if order_ids:
                await _delete_rows(conn, "DELETE FROM ledger_entries WHERE order_id IN ({id_sql})", order_ids)
                await _delete_rows(conn, "DELETE FROM order_cancels WHERE order_id IN ({id_sql})", order_ids)
                await _delete_rows(conn, "DELETE FROM orders WHERE request_id IN ({id_sql})", order_ids)
            if trade_ids:
                await _delete_rows(conn, "DELETE FROM ledger_entries WHERE trade_id IN ({id_sql})", trade_ids)
                await _delete_rows(conn, "DELETE FROM trades WHERE trade_id IN ({id_sql})", trade_ids)
            async with conn.transaction():
                await conn.execute(f"DELETE FROM ledger_entries WHERE market_id IN ({m_sql})")
                await conn.execute(f"DELETE FROM positions WHERE market_id IN ({m_sql})")
                await conn.execute(f"DELETE FROM match_work_queue WHERE market_id IN ({m_sql})")
                await conn.execute(f"DELETE FROM matchmaker_market_leases WHERE market_id IN ({m_sql})")
                await conn.execute(f"DELETE FROM markets WHERE market_id IN ({m_sql})")

        for account_batch in _chunks(account_ids, 25):
            a_sql = _uuid_csv(account_batch)
            async with conn.transaction():
                await conn.execute(f"DELETE FROM account_auth_sessions WHERE account_id IN ({a_sql})")
                await conn.execute(f"DELETE FROM cash_transactions WHERE account_id IN ({a_sql})")
                await conn.execute(f"DELETE FROM account_cash_buckets WHERE account_id IN ({a_sql})")
                await conn.execute(f"DELETE FROM ledger_entries WHERE account_id IN ({a_sql})")
                await conn.execute(f"DELETE FROM accounts WHERE account_id IN ({a_sql})")


async def _run(args: argparse.Namespace) -> int:
    output_path = Path(args.output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clients = _parse_clients(args.clients)
    max_clients = max(clients)
    await _wait_for_listener(args.listener_host, args.listener_port)
    pool = await _create_pool(args.db_dsn)
    run_id = uuid.uuid4().hex[:10]
    created_accounts: list[TestAccount] = []
    market_ids: list[str] = []
    account_sets: list[tuple[TestAccount, TestAccount, str]] = []
    try:
        for idx in range(max_clients):
            yes_account = await _create_and_auth_account(
                args.listener_host,
                args.listener_port,
                args.auth_token,
                f"fill-bench-{run_id}-yes-{idx + 1}",
                f"FillPass!Y{idx + 1:04d}",
            )
            no_account = await _create_and_auth_account(
                args.listener_host,
                args.listener_port,
                args.auth_token,
                f"fill-bench-{run_id}-no-{idx + 1}",
                f"FillPass!N{idx + 1:04d}",
            )
            created_accounts.extend([yes_account, no_account])
            async with pool.acquire() as conn:
                market_id = await _create_market(conn, run_id, idx + 1)
            market_ids.append(market_id)
            account_sets.append((yes_account, no_account, market_id))
        async with pool.acquire() as conn:
            await _fund_accounts(conn, [account.account_id for account in created_accounts], args.account_cash_cents)

        fieldnames = [
            "clients",
            "pairs_per_client",
            "target_pairs_per_second_per_client",
            "submitted_pairs",
            "filled_pairs",
            "submit_errors",
            "timeout_errors",
            "wall_time_sec",
            "fill_throughput_pairs_sec",
            "avg_fill_latency_ms",
            "p50_fill_latency_ms",
            "p95_fill_latency_ms",
            "max_fill_latency_ms",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for client_count in clients:
                row = await _run_point(
                    pool=pool,
                    host=args.listener_host,
                    port=args.listener_port,
                    auth_token=args.auth_token,
                    accounts=account_sets,
                    client_count=client_count,
                    pairs_per_client=args.pairs_per_client,
                    rate_per_second=args.client_rate,
                    timeout_s=args.fill_timeout_seconds,
                    poll_interval_ms=args.poll_interval_ms,
                )
                writer.writerow(row)
                csv_file.flush()
                print(
                    f"clients={row['clients']} filled_pairs={row['filled_pairs']} "
                    f"throughput={row['fill_throughput_pairs_sec']} pairs/s "
                    f"avg_fill_latency={row['avg_fill_latency_ms']} ms "
                    f"p95={row['p95_fill_latency_ms']} ms "
                    f"submit_errors={row['submit_errors']} timeout_errors={row['timeout_errors']}"
                )
        return 0
    finally:
        try:
            await _cleanup(pool, market_ids, [account.account_id for account in created_accounts])
        finally:
            await pool.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark crossing-order submit-to-fill latency")
    parser.add_argument("--db-dsn", required=True)
    parser.add_argument("--auth-token", default="dev-shared-token")
    parser.add_argument("--listener-host", required=True)
    parser.add_argument("--listener-port", type=int, required=True)
    parser.add_argument("--clients", default="1,2,4,8")
    parser.add_argument("--pairs-per-client", type=int, default=50)
    parser.add_argument("--client-rate", type=int, default=20)
    parser.add_argument("--fill-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--poll-interval-ms", type=int, default=50)
    parser.add_argument("--account-cash-cents", type=int, default=100_000_000)
    parser.add_argument("--output-csv", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_build_parser().parse_args())))
