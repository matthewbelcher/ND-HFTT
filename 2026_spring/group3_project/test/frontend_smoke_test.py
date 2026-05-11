#!/usr/bin/env python3
import argparse
import asyncio
import http.cookiejar
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
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


def _uuid() -> str:
    return str(uuid.uuid4())


def _uuid_csv(values: list[str]) -> str:
    return ", ".join(f"'{value}'::uuid" for value in values)


async def _create_pool(dsn: str) -> asyncpg.Pool:
    async def _dsql_safe_reset(_conn: asyncpg.Connection) -> None:
        return None

    return await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5, reset=_dsql_safe_reset)


def _start_proc(name: str, cmd: list[str], cwd: Path, env: dict[str, str], log_dir: Path) -> ManagedProc:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=log_file, stderr=log_file)
    return ManagedProc(name=name, proc=proc, log_path=log_path, log_file=log_file)


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


def _tail(path: Path, n: int = 80) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]) if path.exists() else "<no log file>"


def _http_json(opener, method: str, url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    with opener.open(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def _wait_for_http(url: str, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("ok"):
                    return
        except Exception:
            pass
        await asyncio.sleep(0.25)
    raise RuntimeError(f"frontend did not become ready at {url}")


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
        await conn.execute(f"DELETE FROM markets WHERE market_id IN ({m_ids})")
    if account_ids:
        a_ids = _uuid_csv(account_ids)
        await conn.execute(f"DELETE FROM ledger_entries WHERE account_id IN ({a_ids})")
        await conn.execute(f"DELETE FROM cash_transactions WHERE account_id IN ({a_ids})")
        await conn.execute(f"DELETE FROM account_auth_sessions WHERE account_id IN ({a_ids})")
        await conn.execute(f"DELETE FROM account_cash_buckets WHERE account_id IN ({a_ids})")
        await conn.execute(f"DELETE FROM accounts WHERE account_id IN ({a_ids})")


async def _run(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    listener_dir = repo / 'client' / 'client-listener'
    frontend_dir = repo / 'frontend'
    log_dir = repo / 'test' / '.logs'
    base_env = os.environ.copy()
    procs: list[ManagedProc] = []
    pool = await _create_pool(args.db_dsn)
    run_id = f"front-{uuid.uuid4().hex[:10]}"
    market_ids: list[str] = []
    account_ids: list[str] = []

    try:
        listener_env = base_env.copy()
        listener_env.update(
            {
                'CLIENT_LISTENER_HOST': args.listener_host,
                'CLIENT_LISTENER_PORT': str(args.listener_port),
                'CLIENT_LISTENER_DB_DSN': args.db_dsn,
                'CLIENT_LISTENER_AUTH_MODE': 'static-token',
                'CLIENT_LISTENER_AUTH_TOKEN': args.auth_token,
            }
        )
        procs.append(_start_proc('frontend-smoke-listener', [sys.executable, 'server.py'], listener_dir, listener_env, log_dir))

        frontend_env = base_env.copy()
        frontend_env.update(
            {
                'FRONTEND_HOST': args.frontend_host,
                'FRONTEND_PORT': str(args.frontend_port),
                'FRONTEND_LISTENER_HOST': args.listener_host,
                'FRONTEND_LISTENER_PORT': str(args.listener_port),
                'FRONTEND_LISTENER_AUTH_TOKEN': args.auth_token,
                'FRONTEND_DB_DSN': args.db_dsn,
                'FRONTEND_BOOTSTRAP_ADMIN_USERNAME': 'admin',
                'FRONTEND_BOOTSTRAP_ADMIN_PASSWORD': 'MarketAdmin!2026Forever',
            }
        )
        procs.append(_start_proc('frontend-smoke-ui', [sys.executable, 'server.py'], frontend_dir, frontend_env, log_dir))

        async with pool.acquire() as conn:
            market_id = _uuid()
            market_ids.append(market_id)
            await conn.execute(
                """
                INSERT INTO markets (
                    market_id, slug, title, description, status,
                    tick_size_cents, min_price_cents, max_price_cents,
                    close_time, resolve_time, created_by
                )
                VALUES ($1, $2, $3, $4, 'ACTIVE', 1, 1, 99, now() + interval '1 day', now() + interval '2 days', NULL)
                """,
                market_id,
                f"{run_id}-seed-market",
                f"{run_id} seeded market",
                'frontend smoke seeded market',
            )

        await _wait_for_http(f"http://{args.frontend_host}:{args.frontend_port}/api/health")

        user_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        admin_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        base_url = f"http://{args.frontend_host}:{args.frontend_port}"

        markets_resp = _http_json(user_opener, 'GET', f"{base_url}/api/markets")
        if not markets_resp.get('ok') or not markets_resp.get('markets'):
            raise AssertionError(f"frontend markets failed: {markets_resp}")

        signup_resp = _http_json(
            user_opener,
            'POST',
            f"{base_url}/api/signup",
            {'username': f'{run_id}-user', 'password': 'Front-Pass-1234'},
        )
        if not signup_resp.get('ok'):
            raise AssertionError(f"frontend signup failed: {signup_resp}")
        account_ids.append(signup_resp['account']['account_id'])

        session_resp = _http_json(user_opener, 'GET', f"{base_url}/api/session")
        if not session_resp.get('ok') or not session_resp.get('session'):
            raise AssertionError(f"frontend session failed: {session_resp}")

        portfolio_resp = _http_json(user_opener, 'GET', f"{base_url}/api/portfolio")
        if not portfolio_resp.get('ok') or 'wallet_history' not in portfolio_resp:
            raise AssertionError(f"frontend portfolio failed: {portfolio_resp}")

        selected_market_id = markets_resp['markets'][0]['market_id']
        order_resp = _http_json(
            user_opener,
            'POST',
            f"{base_url}/api/orders",
            {'market_id': selected_market_id, 'side': 'YES', 'qty': 1, 'price_cents': 40, 'time_in_force': 'GTC'},
        )
        if not order_resp.get('ok'):
            raise AssertionError(f"frontend order failed: {order_resp}")

        snapshot_resp = _http_json(user_opener, 'GET', f"{base_url}/api/markets/{selected_market_id}/snapshot")
        if not snapshot_resp.get('ok'):
            raise AssertionError(f"frontend snapshot failed: {snapshot_resp}")

        admin_login_resp = _http_json(
            admin_opener,
            'POST',
            f"{base_url}/api/login",
            {'username': 'admin', 'password': 'MarketAdmin!2026Forever'},
        )
        if not admin_login_resp.get('ok'):
            raise AssertionError(f"admin login failed: {admin_login_resp}")

        admin_session = _http_json(admin_opener, 'GET', f"{base_url}/api/session")
        if not admin_session.get('ok') or not admin_session['session'] or not admin_session['session']['is_admin']:
            raise AssertionError(f"admin session missing: {admin_session}")
        account_ids.append(admin_session['session']['account_id'])

        create_market_resp = _http_json(
            admin_opener,
            'POST',
            f"{base_url}/api/admin/markets",
            {'title': f'{run_id} admin market', 'slug': f'{run_id}-admin-market', 'description': 'created from frontend smoke', 'status': 'DRAFT'},
        )
        if not create_market_resp.get('ok'):
            raise AssertionError(f"frontend create market failed: {create_market_resp}")
        new_market_id = create_market_resp['market']['market_id']
        market_ids.append(new_market_id)

        status_resp = _http_json(
            admin_opener,
            'POST',
            f"{base_url}/api/admin/markets/status",
            {'market_id': new_market_id, 'new_status': 'ACTIVE'},
        )
        if not status_resp.get('ok'):
            raise AssertionError(f"frontend market status failed: {status_resp}")

        print('FRONTEND SMOKE TEST PASSED')
    except Exception:
        print('\n--- frontend smoke logs (tail) ---')
        for proc in procs:
            print(f"\n[{proc.name}] {proc.log_path}")
            print(_tail(proc.log_path))
        raise
    finally:
        try:
            async with pool.acquire() as conn:
                await _cleanup(conn, market_ids, account_ids)
        finally:
            await pool.close()
            _stop_procs(procs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Frontend smoke test')
    parser.add_argument('--db-dsn', required=True)
    parser.add_argument('--auth-token', default='dev-shared-token')
    parser.add_argument('--listener-host', default='127.0.0.1')
    parser.add_argument('--listener-port', type=int, default=9301)
    parser.add_argument('--frontend-host', default='127.0.0.1')
    parser.add_argument('--frontend-port', type=int, default=8081)
    return parser.parse_args()


if __name__ == '__main__':
    try:
        asyncio.run(_run(parse_args()))
    except Exception as exc:
        print(f'FRONTEND SMOKE TEST FAILED: {exc}')
        traceback.print_exc()
        raise SystemExit(1)
