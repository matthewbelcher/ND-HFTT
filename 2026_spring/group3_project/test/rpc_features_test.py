#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

import asyncpg


@dataclass
class ManagedProc:
    name: str
    proc: subprocess.Popen
    log_path: Path
    log_file: object


@dataclass
class TestAccount:
    account_id: str
    username: str
    password: str
    session_token: str


def _uuid() -> str:
    return str(uuid.uuid4())


def _uuid_csv(values: list[str]) -> str:
    return ", ".join(f"'{value}'::uuid" for value in values)


# Must agree with the listener / matchmaker / executor ACCOUNT_CASH_BUCKETS.
ACCOUNT_CASH_BUCKETS = 16


def _split_amount(amount_cents: int, n: int) -> list[int]:
    base, rem = divmod(amount_cents, n)
    out = [base] * n
    out[0] += rem
    return out


async def _reseed_account_cash_buckets(
    conn: asyncpg.Connection, account_id: str, available_cents: int, locked_cents: int = 0
) -> None:
    avail_split = _split_amount(available_cents, ACCOUNT_CASH_BUCKETS)
    locked_split = _split_amount(locked_cents, ACCOUNT_CASH_BUCKETS)
    for b in range(ACCOUNT_CASH_BUCKETS):
        await conn.execute(
            """
            INSERT INTO account_cash_buckets (
                account_id, bucket_id, available_cash_cents, locked_cash_cents
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (account_id, bucket_id) DO UPDATE
            SET available_cash_cents = EXCLUDED.available_cash_cents,
                locked_cash_cents    = EXCLUDED.locked_cash_cents,
                updated_at = now()
            """,
            account_id,
            b,
            avail_split[b],
            locked_split[b],
        )


async def _send_tcp_json(host: str, port: int, payload: dict) -> dict:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
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


def _start_proc(name: str, cmd: list[str], cwd: Path, env: dict[str, str], log_dir: Path) -> ManagedProc:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=log_file, stderr=log_file)
    return ManagedProc(name=name, proc=proc, log_path=log_path, log_file=log_file)


def _tail(path: Path, n: int = 80) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]) if path.exists() else "<no log file>"


def _stop_procs(procs: list[ManagedProc]) -> None:
    for proc in procs:
        if proc.proc.poll() is None:
            proc.proc.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 8
    for proc in procs:
        if proc.proc.poll() is None:
            try:
                proc.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.proc.kill()
    for proc in procs:
        try:
            proc.log_file.close()
        except Exception:
            pass


async def _create_pool(dsn: str) -> asyncpg.Pool:
    async def _dsql_safe_reset(_conn: asyncpg.Connection) -> None:
        return None

    return await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5, reset=_dsql_safe_reset)


async def _create_and_auth_account(host: str, port: int, auth_token: str, username: str, password: str) -> TestAccount:
    create_resp = await _send_tcp_json(
        host,
        port,
        {
            "action": "create_account",
            "auth_token": auth_token,
            "request": {"username": username, "password": password, "external_user_id": f"ext-{username}"},
        },
    )
    if not create_resp.get("ok"):
        raise AssertionError(f"create_account failed for {username}: {create_resp}")
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
        raise AssertionError(f"authenticate_account failed for {username}: {auth_resp}")
    return TestAccount(create_resp["result"]["account_id"], username, password, auth_resp["result"]["account_session_token"])


async def _cleanup(conn: asyncpg.Connection, market_ids: list[str], account_ids: list[str]) -> None:
    if market_ids:
        m_ids = _uuid_csv(market_ids)
        await conn.execute(f"DELETE FROM market_settlements WHERE market_id IN ({m_ids})")
        await conn.execute(f"DELETE FROM market_settlement_runs WHERE market_id IN ({m_ids})")
        await conn.execute(f"DELETE FROM market_resolutions WHERE market_id IN ({m_ids})")
        trade_rows = await conn.fetch(f"SELECT trade_id FROM trades WHERE market_id IN ({m_ids})")
        trade_ids = [str(row['trade_id']) for row in trade_rows]
        if trade_ids:
            await conn.execute(f"DELETE FROM trade_parties WHERE trade_id IN ({_uuid_csv(trade_ids)})")
        await conn.execute(f"DELETE FROM ledger_entries WHERE market_id IN ({m_ids})")
        await conn.execute(f"DELETE FROM order_cancels WHERE order_id IN (SELECT request_id FROM orders WHERE market_id IN ({m_ids}))")
        await conn.execute(f"DELETE FROM trades WHERE market_id IN ({m_ids})")
        await conn.execute(f"DELETE FROM positions WHERE market_id IN ({m_ids})")
        await conn.execute(f"DELETE FROM orders WHERE market_id IN ({m_ids})")
        await conn.execute(f"DELETE FROM execution_market_leases WHERE market_id IN ({m_ids})")
        await conn.execute(f"DELETE FROM matchmaker_market_leases WHERE market_id IN ({m_ids})")
        await conn.execute(f"DELETE FROM match_work_queue WHERE market_id IN ({m_ids})")
        await conn.execute(f"DELETE FROM markets WHERE market_id IN ({m_ids})")
    if account_ids:
        a_ids = _uuid_csv(account_ids)
        await conn.execute(f"DELETE FROM ledger_entries WHERE account_id IN ({a_ids})")
        await conn.execute(f"DELETE FROM cash_transactions WHERE account_id IN ({a_ids})")
        await conn.execute(f"DELETE FROM account_auth_sessions WHERE account_id IN ({a_ids})")
        await conn.execute(f"DELETE FROM account_cash_buckets WHERE account_id IN ({a_ids})")
        await conn.execute(f"DELETE FROM accounts WHERE account_id IN ({a_ids})")


async def _run_test(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    listener_dir = repo / "client" / "client-listener"
    matchmaker_dir = repo / "matchmaker"
    executor_dir = repo / "executor"
    log_dir = repo / "test" / ".logs"

    procs: list[ManagedProc] = []
    pool = await _create_pool(args.db_dsn)
    run_id = f"rpc-{uuid.uuid4().hex[:10]}"
    accounts: dict[str, TestAccount] = {}
    market_ids: list[str] = []

    try:
        base_env = os.environ.copy()

        listener_env = base_env.copy()
        listener_env.update(
            {
                "CLIENT_LISTENER_HOST": args.listener_host,
                "CLIENT_LISTENER_PORT": str(args.listener_port),
                "CLIENT_LISTENER_DB_DSN": args.db_dsn,
                "CLIENT_LISTENER_AUTH_MODE": "static-token",
                "CLIENT_LISTENER_AUTH_TOKEN": args.auth_token,
            }
        )
        procs.append(_start_proc("rpc-listener", [sys.executable, "server.py"], listener_dir, listener_env, log_dir))

        mm_env = base_env.copy()
        mm_env.update({"MATCHMAKER_DB_DSN": args.db_dsn, "MATCHMAKER_INSTANCE_ID": f"{run_id}-mm", "MATCHMAKER_POLL_INTERVAL_MS": "200"})
        procs.append(_start_proc("rpc-matchmaker", [sys.executable, "matchmaker.py"], matchmaker_dir, mm_env, log_dir))

        ex_env = base_env.copy()
        ex_env.update({"EXECUTOR_DB_DSN": args.db_dsn, "EXECUTOR_INSTANCE_ID": f"{run_id}-ex", "EXECUTOR_POLL_INTERVAL_MS": "300"})
        procs.append(_start_proc("rpc-executor", [sys.executable, "executor.py"], executor_dir, ex_env, log_dir))

        await _wait_for_listener(args.listener_host, args.listener_port)

        accounts["admin"] = await _create_and_auth_account(args.listener_host, args.listener_port, args.auth_token, f"{run_id}-admin", "Admin-Pass-1234")
        accounts["user1"] = await _create_and_auth_account(args.listener_host, args.listener_port, args.auth_token, f"{run_id}-user1", "Trader-Pass-1234")
        accounts["user2"] = await _create_and_auth_account(args.listener_host, args.listener_port, args.auth_token, f"{run_id}-user2", "Flow-Pass-1234")

        m_active = _uuid()
        market_ids.append(m_active)

        async with pool.acquire() as conn:
            await conn.execute("UPDATE accounts SET is_admin = TRUE, available_cash_cents = 300000, updated_at = now() WHERE account_id = $1", accounts["admin"].account_id)
            await _reseed_account_cash_buckets(conn, accounts["admin"].account_id, 300000)
            for name in ["user1", "user2"]:
                await conn.execute("UPDATE accounts SET available_cash_cents = 300000, updated_at = now() WHERE account_id = $1", accounts[name].account_id)
                await _reseed_account_cash_buckets(conn, accounts[name].account_id, 300000)
            await conn.execute(
                """
                INSERT INTO markets (
                    market_id, slug, title, description, status,
                    tick_size_cents, min_price_cents, max_price_cents,
                    close_time, resolve_time, created_by
                )
                VALUES ($1, $2, $3, $4, 'ACTIVE', 1, 1, 99, now() + interval '1 day', now() + interval '2 days', $5)
                """,
                m_active,
                f"{run_id}-active",
                f"{run_id} active",
                "rpc active market",
                accounts["admin"].account_id,
            )

        for action in ("health", "ready"):
            resp = await _send_tcp_json(args.listener_host, args.listener_port, {"action": action})
            if not resp.get("ok"):
                raise AssertionError(f"{action} failed: {resp}")

        m_new = _uuid()
        market_ids.append(m_new)
        create_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port,
            {
                "action": "create_market",
                "auth_token": args.auth_token,
                "request": {
                    "market_id": m_new,
                    "slug": f"{run_id}-rpc-market",
                    "title": f"{run_id} rpc market",
                    "description": "rpc test market",
                    "status": "DRAFT",
                    "account_id": accounts["admin"].account_id,
                    "account_session_token": accounts["admin"].session_token,
                },
            },
        )
        if not create_resp.get("ok"):
            raise AssertionError(f"create_market failed: {create_resp}")

        status_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port,
            {
                "action": "update_market_status",
                "auth_token": args.auth_token,
                "request": {
                    "market_id": m_new,
                    "new_status": "ACTIVE",
                    "account_id": accounts["admin"].account_id,
                    "account_session_token": accounts["admin"].session_token,
                },
            },
        )
        if not status_resp.get("ok"):
            raise AssertionError(f"update_market_status failed: {status_resp}")

        order1 = _uuid()
        order2 = _uuid()
        for req, account in [
            ({"request_id": order1, "account_id": accounts["user1"].account_id, "account_session_token": accounts["user1"].session_token, "market_id": m_new, "side": "YES", "qty": 2, "price_cents": 55, "time_in_force": "GTC", "ingress_ts_ns": time.time_ns()}, accounts["user1"]),
            ({"request_id": order2, "account_id": accounts["user2"].account_id, "account_session_token": accounts["user2"].session_token, "market_id": m_new, "side": "NO", "qty": 1, "price_cents": 30, "time_in_force": "GTC", "ingress_ts_ns": time.time_ns()}, accounts["user2"]),
        ]:
            resp = await _send_tcp_json(args.listener_host, args.listener_port, {"action": "submit_order", "auth_token": args.auth_token, "request": req})
            if not resp.get("ok"):
                raise AssertionError(f"submit_order failed: {resp}")

        get_order_resp = await _send_tcp_json(args.listener_host, args.listener_port, {"action": "get_order", "auth_token": args.auth_token, "request": {"order_id": order1}})
        if not get_order_resp.get("ok") or get_order_resp["order"]["request_id"] != order1:
            raise AssertionError(f"get_order failed: {get_order_resp}")

        list_orders_resp = await _send_tcp_json(args.listener_host, args.listener_port, {"action": "list_orders", "auth_token": args.auth_token, "request": {"market_id": m_new, "limit": 20}})
        if not list_orders_resp.get("ok") or len(list_orders_resp["result"]["orders"]) < 2:
            raise AssertionError(f"list_orders failed: {list_orders_resp}")

        open_orders_resp = await _send_tcp_json(args.listener_host, args.listener_port, {"action": "list_open_orders", "auth_token": args.auth_token, "request": {"account_id": accounts['user1'].account_id, "market_id": m_new}})
        if not open_orders_resp.get("ok") or not open_orders_resp["result"]["orders"]:
            raise AssertionError(f"list_open_orders failed: {open_orders_resp}")

        replaced_order = _uuid()
        replace_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port,
            {
                "action": "replace_order",
                "auth_token": args.auth_token,
                "request": {
                    "cancel_id": _uuid(),
                    "old_order_id": order1,
                    "account_id": accounts["user1"].account_id,
                    "new_order": {
                        "request_id": replaced_order,
                        "account_id": accounts["user1"].account_id,
                        "account_session_token": accounts["user1"].session_token,
                        "market_id": m_new,
                        "side": "YES",
                        "qty": 1,
                        "price_cents": 20,
                        "time_in_force": "GTC",
                        "ingress_ts_ns": time.time_ns(),
                    },
                },
            },
        )
        if not replace_resp.get("ok"):
            raise AssertionError(f"replace_order failed: {replace_resp}")

        cancel_all_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port,
            {
                "action": "cancel_all_orders",
                "auth_token": args.auth_token,
                "request": {
                    "account_id": accounts["user2"].account_id,
                    "account_session_token": accounts["user2"].session_token,
                    "market_id": m_new,
                    "reason": "rpc-test",
                },
            },
        )
        if not cancel_all_resp.get("ok") or int(cancel_all_resp["result"]["cancelled_count"]) < 1:
            raise AssertionError(f"cancel_all_orders failed: {cancel_all_resp}")

        for action, request in [
            ("deposit_cash", {"cash_txn_id": _uuid(), "account_id": accounts['user2'].account_id, "account_session_token": accounts['user2'].session_token, "amount_cents": 5000, "notes": "rpc deposit"}),
            ("withdraw_cash", {"cash_txn_id": _uuid(), "account_id": accounts['user2'].account_id, "account_session_token": accounts['user2'].session_token, "amount_cents": 1200, "notes": "rpc withdrawal"}),
        ]:
            resp = await _send_tcp_json(args.listener_host, args.listener_port, {"action": action, "auth_token": args.auth_token, "request": request})
            if not resp.get("ok"):
                raise AssertionError(f"{action} failed: {resp}")

        balances_resp = await _send_tcp_json(args.listener_host, args.listener_port, {"action": "get_account_balances", "auth_token": args.auth_token, "request": {"account_id": accounts['user2'].account_id}})
        if not balances_resp.get("ok"):
            raise AssertionError(f"get_account_balances failed: {balances_resp}")
        wallet_resp = await _send_tcp_json(args.listener_host, args.listener_port, {"action": "get_wallet_history", "auth_token": args.auth_token, "request": {"account_id": accounts['user2'].account_id}})
        if not wallet_resp.get("ok") or not wallet_resp["result"]["points"]:
            raise AssertionError(f"get_wallet_history failed: {wallet_resp}")

        cross_yes = _uuid()
        cross_no = _uuid()
        for req in [
            {"request_id": cross_yes, "account_id": accounts['user1'].account_id, "account_session_token": accounts['user1'].session_token, "market_id": m_active, "side": "YES", "qty": 1, "price_cents": 60, "time_in_force": "GTC", "ingress_ts_ns": time.time_ns()},
            {"request_id": cross_no, "account_id": accounts['user2'].account_id, "account_session_token": accounts['user2'].session_token, "market_id": m_active, "side": "NO", "qty": 1, "price_cents": 45, "time_in_force": "GTC", "ingress_ts_ns": time.time_ns()},
        ]:
            resp = await _send_tcp_json(args.listener_host, args.listener_port, {"action": "submit_order", "auth_token": args.auth_token, "request": req})
            if not resp.get("ok"):
                raise AssertionError(f"cross submit failed: {resp}")

        await asyncio.sleep(args.match_wait_seconds)

        for action, request in [
            ("get_order_book", {"market_id": m_new, "depth": 5}),
            ("get_market_history", {"market_id": m_active, "limit": 20}),
            ("get_trades", {"market_id": m_active, "limit": 20}),
            ("get_positions", {"account_id": accounts['user1'].account_id, "market_id": m_active}),
        ]:
            resp = await _send_tcp_json(args.listener_host, args.listener_port, {"action": action, "auth_token": args.auth_token, "request": request})
            if not resp.get("ok"):
                raise AssertionError(f"{action} failed: {resp}")

        resolve_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port,
            {
                "action": "resolve_market",
                "auth_token": args.auth_token,
                "request": {
                    "resolution_id": _uuid(),
                    "market_id": m_new,
                    "outcome": "CANCELLED",
                    "notes": "rpc resolve",
                    "account_id": accounts['admin'].account_id,
                    "account_session_token": accounts['admin'].session_token,
                },
            },
        )
        if not resolve_resp.get("ok"):
            raise AssertionError(f"resolve_market failed: {resolve_resp}")

        print('RPC FEATURES TEST PASSED')
    except Exception:
        print('\n--- rpc component logs (tail) ---')
        for proc in procs:
            print(f"\n[{proc.name}] {proc.log_path}")
            print(_tail(proc.log_path))
        raise
    finally:
        try:
            async with pool.acquire() as conn:
                await _cleanup(conn, market_ids, [account.account_id for account in accounts.values()])
        finally:
            await pool.close()
            _stop_procs(procs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Authenticated RPC feature coverage test')
    parser.add_argument('--db-dsn', required=True)
    parser.add_argument('--auth-token', default='dev-shared-token')
    parser.add_argument('--listener-host', default='127.0.0.1')
    parser.add_argument('--listener-port', type=int, default=9201)
    parser.add_argument('--match-wait-seconds', type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    try:
        asyncio.run(_run_test(parse_args()))
        return 0
    except Exception as exc:
        print(f'RPC FEATURES TEST FAILED: {exc}')
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
