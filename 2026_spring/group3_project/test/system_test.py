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


def _uuid() -> str:
    return str(uuid.uuid4())


def _uuid_csv(values: list[str]) -> str:
    return ", ".join(f"'{value}'::uuid" for value in values)


def _is_retryable_occ(exc: Exception) -> bool:
    return isinstance(exc, asyncpg.PostgresError) and exc.sqlstate in {"40001", "40P01"} or "OC000" in str(exc)


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


def _start_proc(name: str, cmd: list[str], cwd: Path, env: dict[str, str], log_dir: Path) -> ManagedProc:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=log_file, stderr=log_file)
    return ManagedProc(name=name, proc=proc, log_path=log_path, log_file=log_file)


def _tail(path: Path, n: int = 80) -> str:
    if not path.exists():
        return "<no log file>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


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
            "request": {
                "username": username,
                "password": password,
                "external_user_id": f"ext-{username}",
            },
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

    return TestAccount(
        account_id=create_resp["result"]["account_id"],
        username=username,
        password=password,
        session_token=auth_resp["result"]["account_session_token"],
    )


async def _seed_markets(conn: asyncpg.Connection, run_id: str) -> dict[str, str]:
    markets = {
        "m1": _uuid(),
        "m2": _uuid(),
        "m3": _uuid(),
        "m4": _uuid(),
        "m5": _uuid(),
    }
    for key, title, desc in [
        ("m1", "partial cross", "partial cross market"),
        ("m2", "full cross", "full cross market"),
        ("m3", "persistent open", "unmatched market"),
        ("m4", "fairness cross one", "must not starve"),
        ("m5", "fairness cross two", "must not starve"),
    ]:
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
            markets[key],
            f"{run_id}-{key}",
            f"{run_id} {title}",
            desc,
        )
    return markets


async def _cleanup(conn: asyncpg.Connection, market_ids: list[str], account_ids: list[str]) -> None:
    m_ids = _uuid_csv(market_ids)
    a_ids = _uuid_csv(account_ids)
    await conn.execute(f"DELETE FROM market_settlements WHERE market_id IN ({m_ids})")
    await conn.execute(f"DELETE FROM market_settlement_runs WHERE market_id IN ({m_ids})")
    await conn.execute(f"DELETE FROM market_resolutions WHERE market_id IN ({m_ids})")
    trade_rows = await conn.fetch(f"SELECT trade_id FROM trades WHERE market_id IN ({m_ids})")
    trade_ids = [str(row['trade_id']) for row in trade_rows]
    if trade_ids:
        await conn.execute(f"DELETE FROM trade_parties WHERE trade_id IN ({_uuid_csv(trade_ids)})")
    await conn.execute(f"DELETE FROM ledger_entries WHERE market_id IN ({m_ids}) OR account_id IN ({a_ids})")
    await conn.execute(f"DELETE FROM order_cancels WHERE order_id IN (SELECT request_id FROM orders WHERE market_id IN ({m_ids}))")
    await conn.execute(f"DELETE FROM trades WHERE market_id IN ({m_ids})")
    await conn.execute(f"DELETE FROM positions WHERE market_id IN ({m_ids})")
    await conn.execute(f"DELETE FROM orders WHERE market_id IN ({m_ids})")
    await conn.execute(f"DELETE FROM execution_market_leases WHERE market_id IN ({m_ids})")
    await conn.execute(f"DELETE FROM matchmaker_market_leases WHERE market_id IN ({m_ids})")
    await conn.execute(f"DELETE FROM match_work_queue WHERE market_id IN ({m_ids})")
    await conn.execute(f"DELETE FROM account_auth_sessions WHERE account_id IN ({a_ids})")
    await conn.execute(f"DELETE FROM cash_transactions WHERE account_id IN ({a_ids})")
    await conn.execute(f"DELETE FROM account_cash_buckets WHERE account_id IN ({a_ids})")
    await conn.execute(f"DELETE FROM markets WHERE market_id IN ({m_ids})")
    await conn.execute(f"DELETE FROM accounts WHERE account_id IN ({a_ids})")


async def _cleanup_with_retries(conn: asyncpg.Connection, market_ids: list[str], account_ids: list[str]) -> None:
    for attempt in range(6):
        try:
            await _cleanup(conn, market_ids, account_ids)
            return
        except Exception as exc:
            if not _is_retryable_occ(exc) or attempt == 5:
                raise
            await asyncio.sleep(0.15 * (attempt + 1))


async def _run_test(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    listener_dir = repo / "client" / "client-listener"
    matchmaker_dir = repo / "matchmaker"
    executor_dir = repo / "executor"
    log_dir = repo / "test" / ".logs"

    procs: list[ManagedProc] = []
    pool = await _create_pool(args.db_dsn)
    run_id = f"system-{uuid.uuid4().hex[:10]}"
    markets: dict[str, str] = {}
    accounts: dict[str, TestAccount] = {}

    try:
        base_env = os.environ.copy()

        for idx, port in enumerate([args.listener_port_a, args.listener_port_b], start=1):
            env = base_env.copy()
            env.update(
                {
                    "CLIENT_LISTENER_HOST": args.listener_host,
                    "CLIENT_LISTENER_PORT": str(port),
                    "CLIENT_LISTENER_DB_DSN": args.db_dsn,
                    "CLIENT_LISTENER_AUTH_MODE": "static-token",
                    "CLIENT_LISTENER_AUTH_TOKEN": args.auth_token,
                }
            )
            procs.append(_start_proc(f"listener-{idx}", [sys.executable, "server.py"], listener_dir, env, log_dir))

        for idx in range(1, 3):
            env = base_env.copy()
            env.update(
                {
                    "MATCHMAKER_DB_DSN": args.db_dsn,
                    "MATCHMAKER_INSTANCE_ID": f"{run_id}-mm-{idx}",
                    "MATCHMAKER_POLL_INTERVAL_MS": "200",
                    "MATCHMAKER_MARKET_SCAN_LIMIT": "2",
                }
            )
            procs.append(_start_proc(f"matchmaker-{idx}", [sys.executable, "matchmaker.py"], matchmaker_dir, env, log_dir))

        for idx in range(1, 3):
            env = base_env.copy()
            env.update(
                {
                    "EXECUTOR_DB_DSN": args.db_dsn,
                    "EXECUTOR_INSTANCE_ID": f"{run_id}-ex-{idx}",
                    "EXECUTOR_POLL_INTERVAL_MS": "300",
                    "EXECUTOR_MARKET_SCAN_LIMIT": "2",
                }
            )
            procs.append(_start_proc(f"executor-{idx}", [sys.executable, "executor.py"], executor_dir, env, log_dir))

        await _wait_for_listener(args.listener_host, args.listener_port_a)
        await _wait_for_listener(args.listener_host, args.listener_port_b)

        async with pool.acquire() as conn:
            markets = await _seed_markets(conn, run_id)

        accounts["yes"] = await _create_and_auth_account(args.listener_host, args.listener_port_a, args.auth_token, f"{run_id}-yes", "Pass-yes-1234")
        accounts["no"] = await _create_and_auth_account(args.listener_host, args.listener_port_b, args.auth_token, f"{run_id}-no", "Pass-no-1234")
        accounts["third"] = await _create_and_auth_account(args.listener_host, args.listener_port_a, args.auth_token, f"{run_id}-third", "Pass-third-1234")

        async with pool.acquire() as conn:
            for account in accounts.values():
                await conn.execute(
                    "UPDATE accounts SET available_cash_cents = 200000, updated_at = now() WHERE account_id = $1",
                    account.account_id,
                )
                await _reseed_account_cash_buckets(conn, account.account_id, 200000, 0)

        submissions = [
            (args.listener_port_a, _uuid(), accounts["yes"], markets["m1"], "YES", 2, 60),
            (args.listener_port_b, _uuid(), accounts["no"], markets["m1"], "NO", 1, 40),
            (args.listener_port_a, _uuid(), accounts["third"], markets["m1"], "NO", 1, 35),
            (args.listener_port_b, _uuid(), accounts["yes"], markets["m2"], "YES", 1, 45),
            (args.listener_port_a, _uuid(), accounts["no"], markets["m2"], "NO", 1, 60),
            (args.listener_port_b, _uuid(), accounts["yes"], markets["m3"], "YES", 1, 30),
            (args.listener_port_a, _uuid(), accounts["no"], markets["m3"], "NO", 1, 20),
            (args.listener_port_b, _uuid(), accounts["yes"], markets["m4"], "YES", 1, 52),
            (args.listener_port_a, _uuid(), accounts["third"], markets["m4"], "NO", 1, 48),
            (args.listener_port_b, _uuid(), accounts["third"], markets["m5"], "YES", 1, 46),
            (args.listener_port_a, _uuid(), accounts["no"], markets["m5"], "NO", 1, 54),
        ]

        order_ids: dict[str, str] = {
            "m1_yes": submissions[0][1],
            "m1_no_cross": submissions[1][1],
            "m1_no_open": submissions[2][1],
            "m2_yes_cross": submissions[3][1],
            "m2_no_cross": submissions[4][1],
            "m3_yes_open": submissions[5][1],
            "m3_no_open": submissions[6][1],
            "m4_yes_cross": submissions[7][1],
            "m4_no_cross": submissions[8][1],
            "m5_yes_cross": submissions[9][1],
            "m5_no_cross": submissions[10][1],
        }

        for port, request_id, account, market_id, side, qty, price in submissions:
            resp = await _send_tcp_json(
                args.listener_host,
                port,
                {
                    "action": "submit_order",
                    "auth_token": args.auth_token,
                    "request": {
                        "request_id": request_id,
                        "account_id": account.account_id,
                        "account_session_token": account.session_token,
                        "market_id": market_id,
                        "side": side,
                        "qty": qty,
                        "price_cents": price,
                        "time_in_force": "GTC",
                        "ingress_ts_ns": time.time_ns(),
                    },
                },
            )
            if not resp.get("ok"):
                raise AssertionError(f"submit_order failed for {request_id}: {resp}")

        await asyncio.sleep(args.match_wait_seconds)

        cancel_id = _uuid()
        cancel_resp = await _send_tcp_json(
            args.listener_host,
            args.listener_port_b,
            {
                "action": "cancel_order",
                "auth_token": args.auth_token,
                "request": {
                    "cancel_id": cancel_id,
                    "order_id": order_ids["m3_no_open"],
                    "account_id": accounts["no"].account_id,
                    "account_session_token": accounts["no"].session_token,
                    "reason": "system-test-cancel-open-order",
                },
            },
        )
        if not cancel_resp.get("ok"):
            raise AssertionError(f"cancel failed: {cancel_resp}")
        if cancel_resp["cancel"]["status"] != "CANCELLED":
            raise AssertionError(f"unexpected cancel status: {cancel_resp}")

        async with pool.acquire() as conn:
            all_order_ids = list(order_ids.values())
            rows = await conn.fetch(
                f"SELECT request_id, remaining_qty, status FROM orders WHERE request_id IN ({_uuid_csv(all_order_ids)})"
            )
            state_map = {str(row['request_id']): (int(row['remaining_qty']), str(row['status'])) for row in rows}
            expected = {
                order_ids["m1_yes"]: (1, "PARTIALLY_FILLED"),
                order_ids["m1_no_cross"]: (0, "FILLED"),
                order_ids["m1_no_open"]: (1, "ACCEPTED"),
                order_ids["m2_yes_cross"]: (0, "FILLED"),
                order_ids["m2_no_cross"]: (0, "FILLED"),
                order_ids["m3_yes_open"]: (1, "ACCEPTED"),
                order_ids["m3_no_open"]: (0, "CANCELLED"),
                order_ids["m4_yes_cross"]: (0, "FILLED"),
                order_ids["m4_no_cross"]: (0, "FILLED"),
                order_ids["m5_yes_cross"]: (0, "FILLED"),
                order_ids["m5_no_cross"]: (0, "FILLED"),
            }
            for request_id, exp in expected.items():
                if state_map.get(request_id) != exp:
                    raise AssertionError(f"unexpected order state for {request_id}: got={state_map.get(request_id)} expected={exp}")

            trade_count = await conn.fetchval(
                f"SELECT count(*) FROM trades WHERE market_id IN ({_uuid_csv([markets['m1'], markets['m2'], markets['m3'], markets['m4'], markets['m5']])})"
            )
            if int(trade_count) != 4:
                raise AssertionError(f"expected 4 trades, got {trade_count}")

            open_count = await conn.fetchval(
                f"""
                SELECT count(*)
                FROM orders
                WHERE request_id IN ({_uuid_csv([order_ids['m1_yes'], order_ids['m1_no_open'], order_ids['m3_yes_open']])})
                  AND status IN ('ACCEPTED', 'OPEN', 'PARTIALLY_FILLED')
                  AND remaining_qty > 0
                """
            )
            if int(open_count) != 3:
                raise AssertionError(f"expected 3 open orders, got {open_count}")

            await conn.execute(
                f"UPDATE markets SET status = 'RESOLVED', resolve_time = now() - interval '1 minute' WHERE market_id IN ({_uuid_csv([markets['m1'], markets['m2'], markets['m4'], markets['m5']])})"
            )
            await conn.execute(
                """
                INSERT INTO market_resolutions (resolution_id, market_id, outcome, resolved_by, notes)
                VALUES
                  ($1, $2, 'YES', $3, 'm1 yes'),
                  ($4, $5, 'NO', $6, 'm2 no'),
                  ($7, $8, 'YES', $9, 'm4 yes'),
                  ($10, $11, 'NO', $12, 'm5 no')
                """,
                _uuid(), markets['m1'], accounts['yes'].account_id,
                _uuid(), markets['m2'], accounts['no'].account_id,
                _uuid(), markets['m4'], accounts['yes'].account_id,
                _uuid(), markets['m5'], accounts['no'].account_id,
            )

        await asyncio.sleep(args.execute_wait_seconds)

        async with pool.acquire() as conn:
            runs = await conn.fetch(
                f"SELECT market_id, status FROM market_settlement_runs WHERE market_id IN ({_uuid_csv([markets['m1'], markets['m2'], markets['m4'], markets['m5']])})"
            )
            run_map = {str(row['market_id']): str(row['status']) for row in runs}
            for market_id in [markets['m1'], markets['m2'], markets['m4'], markets['m5']]:
                if run_map.get(market_id) != 'COMPLETED':
                    raise AssertionError(f"settlement run not completed for {market_id}: {run_map.get(market_id)}")

            settlements = await conn.fetch(
                f"SELECT market_id, account_id, payout_cents FROM market_settlements WHERE market_id IN ({_uuid_csv([markets['m1'], markets['m2'], markets['m4'], markets['m5']])})"
            )
            payout_map = {(str(row['market_id']), str(row['account_id'])): int(row['payout_cents']) for row in settlements}
            expected_payouts = {
                (markets['m1'], accounts['yes'].account_id): 100,
                (markets['m1'], accounts['no'].account_id): 0,
                (markets['m2'], accounts['yes'].account_id): 0,
                (markets['m2'], accounts['no'].account_id): 100,
                (markets['m4'], accounts['yes'].account_id): 100,
                (markets['m4'], accounts['third'].account_id): 0,
                (markets['m5'], accounts['third'].account_id): 0,
                (markets['m5'], accounts['no'].account_id): 100,
            }
            for key, value in expected_payouts.items():
                if payout_map.get(key) != value:
                    raise AssertionError(f"unexpected payout for {key}: got={payout_map.get(key)} expected={value}")

            m3_open = await conn.fetchval(
                """
                SELECT count(*)
                FROM orders
                WHERE market_id = $1
                  AND status IN ('ACCEPTED', 'OPEN', 'PARTIALLY_FILLED')
                  AND remaining_qty > 0
                """,
                markets['m3'],
            )
            if int(m3_open) != 1:
                raise AssertionError(f"expected 1 open order in m3, got {m3_open}")

        print('SYSTEM TEST PASSED')
    except Exception:
        print('\n--- component logs (tail) ---')
        for proc in procs:
            print(f"\n[{proc.name}] {proc.log_path}")
            print(_tail(proc.log_path))
        raise
    finally:
        try:
            if markets or accounts:
                async with pool.acquire() as conn:
                    await _cleanup_with_retries(conn, list(markets.values()), [account.account_id for account in accounts.values()])
        finally:
            await pool.close()
            _stop_procs(procs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='End-to-end authenticated system test with concurrent listeners, matchmakers, and executors')
    parser.add_argument('--db-dsn', required=True)
    parser.add_argument('--auth-token', default='dev-shared-token')
    parser.add_argument('--listener-host', default='127.0.0.1')
    parser.add_argument('--listener-port-a', type=int, default=9101)
    parser.add_argument('--listener-port-b', type=int, default=9102)
    parser.add_argument('--match-wait-seconds', type=float, default=5.0)
    parser.add_argument('--execute-wait-seconds', type=float, default=6.0)
    return parser.parse_args()


def main() -> int:
    try:
        asyncio.run(_run_test(parse_args()))
        return 0
    except Exception as exc:
        print(f'SYSTEM TEST FAILED: {exc}')
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
