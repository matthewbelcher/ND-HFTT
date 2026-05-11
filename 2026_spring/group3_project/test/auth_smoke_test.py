#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import asyncpg


def _uuid() -> str:
    return str(uuid.uuid4())


def _json_line(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


async def _send_tcp_json(host: str, port: int, payload: dict) -> dict:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(_json_line(payload))
        await writer.drain()
        line = await reader.readline()
        return json.loads(line.decode("utf-8"))
    finally:
        writer.close()
        await writer.wait_closed()


async def _wait_for_listener(host: str, port: int, timeout_s: float = 20.0) -> None:
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

    return await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5, reset=_dsql_safe_reset)


async def _run(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    listener_dir = repo / "client" / "client-listener"
    env = os.environ.copy()
    env.update(
        {
            "CLIENT_LISTENER_HOST": args.listener_host,
            "CLIENT_LISTENER_PORT": str(args.listener_port),
            "CLIENT_LISTENER_DB_DSN": args.db_dsn,
            "CLIENT_LISTENER_AUTH_MODE": "static-token",
            "CLIENT_LISTENER_AUTH_TOKEN": args.auth_token,
            "CLIENT_LISTENER_ACCOUNT_SESSION_TTL_SECONDS": "43200",
        }
    )

    log_path = repo / "test" / ".logs" / "auth-smoke-listener.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=str(listener_dir),
        env=env,
        stdout=log_file,
        stderr=log_file,
    )

    pool = await _create_pool(args.db_dsn)
    run_id = f"authsmoke-{uuid.uuid4().hex[:10]}"
    market_id = _uuid()
    account_ids: list[str] = []

    try:
        await _wait_for_listener(args.listener_host, args.listener_port)

        async with pool.acquire() as conn:
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
                f"{run_id}-market",
                f"{run_id} market",
                "auth smoke test market",
            )

        create_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port,
            {
                "action": "create_account",
                "auth_token": args.auth_token,
                "request": {
                    "username": f"{run_id}-user",
                    "password": "AuthPass-1234",
                    "external_user_id": f"{run_id}-external",
                },
            },
        )
        if not create_resp.get("ok"):
            raise AssertionError(f"create_account failed: {create_resp}")
        account_id = create_resp["result"]["account_id"]
        account_ids.append(account_id)

        second_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port,
            {
                "action": "create_account",
                "auth_token": args.auth_token,
                "request": {
                    "username": f"{run_id}-user-2",
                    "password": "AuthPass-5678",
                    "external_user_id": f"{run_id}-external-2",
                },
            },
        )
        if not second_resp.get("ok"):
            raise AssertionError(f"second create_account failed: {second_resp}")
        other_account_id = second_resp["result"]["account_id"]
        account_ids.append(other_account_id)

        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE accounts SET available_cash_cents = 5000, updated_at = now() WHERE account_id = $1",
                account_id,
            )
            await conn.execute(
                "UPDATE accounts SET available_cash_cents = 5000, updated_at = now() WHERE account_id = $1",
                other_account_id,
            )

        auth_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port,
            {
                "action": "authenticate_account",
                "auth_token": args.auth_token,
                "request": {
                    "username": f"{run_id}-user",
                    "password": "AuthPass-1234",
                },
            },
        )
        if not auth_resp.get("ok"):
            raise AssertionError(f"authenticate_account failed: {auth_resp}")
        account_session_token = auth_resp["result"]["account_session_token"]

        success_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port,
            {
                "action": "submit_order",
                "auth_token": args.auth_token,
                "request": {
                    "request_id": _uuid(),
                    "account_id": account_id,
                    "account_session_token": account_session_token,
                    "market_id": market_id,
                    "side": "YES",
                    "qty": 2,
                    "price_cents": 40,
                    "time_in_force": "GTC",
                    "ingress_ts_ns": time.time_ns(),
                },
            },
        )
        if not success_resp.get("ok"):
            raise AssertionError(f"authenticated submit_order failed: {success_resp}")

        wrong_account_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port,
            {
                "action": "submit_order",
                "auth_token": args.auth_token,
                "request": {
                    "request_id": _uuid(),
                    "account_id": other_account_id,
                    "account_session_token": account_session_token,
                    "market_id": market_id,
                    "side": "NO",
                    "qty": 1,
                    "price_cents": 55,
                    "time_in_force": "GTC",
                    "ingress_ts_ns": time.time_ns(),
                },
            },
        )
        if wrong_account_resp.get("ok"):
            raise AssertionError(f"cross-account submit_order unexpectedly succeeded: {wrong_account_resp}")
        if "account authentication failed" not in str(wrong_account_resp.get("error", "")):
            raise AssertionError(f"unexpected cross-account error: {wrong_account_resp}")

        bad_session_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port,
            {
                "action": "submit_order",
                "auth_token": args.auth_token,
                "request": {
                    "request_id": _uuid(),
                    "account_id": account_id,
                    "account_session_token": "bad-token",
                    "market_id": market_id,
                    "side": "YES",
                    "qty": 1,
                    "price_cents": 25,
                    "time_in_force": "GTC",
                    "ingress_ts_ns": time.time_ns(),
                },
            },
        )
        if bad_session_resp.get("ok"):
            raise AssertionError(f"bad-session submit_order unexpectedly succeeded: {bad_session_resp}")
        if "account authentication failed" not in str(bad_session_resp.get("error", "")):
            raise AssertionError(f"unexpected bad-session error: {bad_session_resp}")

        print("AUTH SMOKE TEST PASSED")
    finally:
        async with pool.acquire() as conn:
            if account_ids:
                account_csv = ", ".join(f"'{account_id}'::uuid" for account_id in account_ids)
                await conn.execute(f"DELETE FROM account_auth_sessions WHERE account_id IN ({account_csv})")
                await conn.execute(f"DELETE FROM ledger_entries WHERE account_id IN ({account_csv})")
                await conn.execute(f"DELETE FROM account_cash_buckets WHERE account_id IN ({account_csv})")
                await conn.execute(f"DELETE FROM accounts WHERE account_id IN ({account_csv})")
            await conn.execute("DELETE FROM orders WHERE market_id = $1", market_id)
            await conn.execute("DELETE FROM markets WHERE market_id = $1", market_id)
        await pool.close()
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_file.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for password-authenticated account order submission")
    p.add_argument("--db-dsn", required=True)
    p.add_argument("--auth-token", default="dev-shared-token")
    p.add_argument("--listener-host", default="127.0.0.1")
    p.add_argument("--listener-port", type=int, default=9201)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_run(parse_args()))
