import asyncio
import json
import multiprocessing
import os
import signal
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

from auth import AuthError, build_authenticator
from config import load_settings
from db import OrderRepository, SubmissionError, create_pool
from metrics import METRICS, serve_metrics
from models import CancelOrderRequest, SubmitOrderRequest, ValidationError


def _json_response(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(',', ':')) + '\n').encode('utf-8')


async def _ensure_auth(auth, auth_token: str) -> None:
    if not await auth.validate(auth_token):
        raise SubmissionError('unauthorized')


async def _handle_submit_order(message: dict[str, Any], auth_token: str, auth, repo: OrderRepository) -> dict[str, Any]:
    await _ensure_auth(auth, auth_token)
    request_payload = message.get('request')
    if not isinstance(request_payload, dict):
        raise SubmissionError('request payload must be an object')
    req = SubmitOrderRequest.from_dict(request_payload)
    order = await repo.submit_order(req)
    return {'ok': True, 'order': order}


async def _handle_cancel_order(message: dict[str, Any], auth_token: str, auth, repo: OrderRepository) -> dict[str, Any]:
    await _ensure_auth(auth, auth_token)
    request_payload = message.get('request')
    if not isinstance(request_payload, dict):
        raise SubmissionError('request payload must be an object')
    req = CancelOrderRequest.from_dict(request_payload)
    cancel = await repo.cancel_order(req)
    return {'ok': True, 'cancel': cancel}


async def _handle_repo_call(
    *,
    key: str,
    fn,
    message: dict[str, Any],
    auth_token: str,
    auth,
) -> dict[str, Any]:
    await _ensure_auth(auth, auth_token)
    request_payload = message.get('request')
    if request_payload is None:
        request_payload = {}
    if not isinstance(request_payload, dict):
        raise SubmissionError('request payload must be an object')
    result = await fn(request_payload)
    return {'ok': True, key: result}


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    auth,
    repo: OrderRepository,
) -> None:
    peer = writer.get_extra_info('peername')
    print(f'[{datetime.now(timezone.utc).isoformat()}] connected: {peer}')

    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            try:
                message = json.loads(line.decode('utf-8'))
                action = str(message.get('action', '')).lower()
                auth_token = str(message.get('auth_token', ''))

                if action == 'ping':
                    response = {'ok': True, 'pong': True}
                elif action == 'health':
                    response = {'ok': True, 'status': 'healthy'}
                elif action == 'ready':
                    response = {'ok': True, 'ready': await repo.ready()}
                elif action == 'create_account':
                    response = await _handle_repo_call(key='result', fn=repo.create_account, message=message, auth_token=auth_token, auth=auth)
                elif action == 'authenticate_account':
                    response = await _handle_repo_call(key='result', fn=repo.authenticate_account, message=message, auth_token=auth_token, auth=auth)
                elif action == 'submit_order':
                    response = await _handle_submit_order(message, auth_token, auth, repo)
                elif action == 'cancel_order':
                    response = await _handle_cancel_order(message, auth_token, auth, repo)
                elif action == 'get_order':
                    response = await _handle_repo_call(key='order', fn=repo.get_order, message=message, auth_token=auth_token, auth=auth)
                elif action == 'list_orders':
                    response = await _handle_repo_call(key='result', fn=repo.list_orders, message=message, auth_token=auth_token, auth=auth)
                elif action == 'list_open_orders':
                    response = await _handle_repo_call(key='result', fn=repo.list_open_orders, message=message, auth_token=auth_token, auth=auth)
                elif action == 'cancel_all_orders':
                    response = await _handle_repo_call(key='result', fn=repo.cancel_all_orders, message=message, auth_token=auth_token, auth=auth)
                elif action == 'replace_order':
                    response = await _handle_repo_call(key='result', fn=repo.replace_order, message=message, auth_token=auth_token, auth=auth)
                elif action == 'get_trades':
                    response = await _handle_repo_call(key='result', fn=repo.get_trades, message=message, auth_token=auth_token, auth=auth)
                elif action == 'get_positions':
                    response = await _handle_repo_call(key='result', fn=repo.get_positions, message=message, auth_token=auth_token, auth=auth)
                elif action == 'get_account_balances':
                    response = await _handle_repo_call(key='result', fn=repo.get_account_balances, message=message, auth_token=auth_token, auth=auth)
                elif action == 'get_wallet_history':
                    response = await _handle_repo_call(key='result', fn=repo.get_wallet_history, message=message, auth_token=auth_token, auth=auth)
                elif action == 'deposit_cash':
                    response = await _handle_repo_call(key='result', fn=repo.deposit_cash, message=message, auth_token=auth_token, auth=auth)
                elif action == 'withdraw_cash':
                    response = await _handle_repo_call(key='result', fn=repo.withdraw_cash, message=message, auth_token=auth_token, auth=auth)
                elif action == 'list_markets':
                    response = await _handle_repo_call(key='result', fn=repo.list_markets, message=message, auth_token=auth_token, auth=auth)
                elif action == 'get_market_history':
                    response = await _handle_repo_call(key='result', fn=repo.get_market_history, message=message, auth_token=auth_token, auth=auth)
                elif action == 'create_market':
                    response = await _handle_repo_call(key='result', fn=repo.create_market, message=message, auth_token=auth_token, auth=auth)
                elif action == 'update_market_status':
                    response = await _handle_repo_call(key='result', fn=repo.update_market_status, message=message, auth_token=auth_token, auth=auth)
                elif action == 'resolve_market':
                    response = await _handle_repo_call(key='result', fn=repo.resolve_market, message=message, auth_token=auth_token, auth=auth)
                elif action == 'get_order_book':
                    response = await _handle_repo_call(key='result', fn=repo.get_order_book, message=message, auth_token=auth_token, auth=auth)
                else:
                    response = {'ok': False, 'error': f'unsupported action: {action}'}
            except (json.JSONDecodeError, UnicodeDecodeError):
                response = {'ok': False, 'error': 'invalid JSON line'}
            except ValidationError as exc:
                response = {'ok': False, 'error': f'validation error: {exc}'}
            except SubmissionError as exc:
                response = {'ok': False, 'error': f'submission failed: {exc}'}
            except Exception as exc:
                print(f'unexpected handler error: {exc}')
                traceback.print_exc()
                response = {'ok': False, 'error': 'internal server error'}

            writer.write(_json_response(response))
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()
        print(f'[{datetime.now(timezone.utc).isoformat()}] disconnected: {peer}')


async def _serve_one_worker(worker_index: int) -> None:
    settings = load_settings()
    try:
        auth = build_authenticator(settings)
    except AuthError as exc:
        raise SystemExit(f'auth configuration error: {exc}') from exc

    pool = await create_pool(
        dsn=settings.db_dsn,
        min_size=settings.db_min_pool_size,
        max_size=settings.db_max_pool_size,
    )
    repo = OrderRepository(
        pool,
        account_session_ttl_seconds=settings.account_session_ttl_seconds,
    )

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, auth=auth, repo=repo),
        settings.host,
        settings.port,
        reuse_port=settings.workers > 1,
    )
    sockets = ', '.join(str(sock.getsockname()) for sock in (server.sockets or []))
    pid = os.getpid()
    print(f'client-listener worker pid={pid} index={worker_index} running on {sockets}')

    metrics_server = None
    if settings.metrics_port > 0:
        metrics_port = settings.metrics_port + worker_index
        try:
            metrics_server = await serve_metrics(settings.metrics_host, metrics_port)
            print(f'client-listener metrics worker={worker_index} on {settings.metrics_host}:{metrics_port}')
        except OSError as exc:
            print(f'client-listener metrics failed to bind worker={worker_index} port={metrics_port}: {exc}')

    async def _pool_gauge_loop():
        while True:
            try:
                if hasattr(pool, "get_size") and hasattr(pool, "get_idle_size"):
                    METRICS.set_gauge("listener_pool_size", float(pool.get_size()))
                    METRICS.set_gauge("listener_pool_idle", float(pool.get_idle_size()))
            except Exception:
                pass
            await asyncio.sleep(2.0)

    pool_gauge_task = asyncio.create_task(_pool_gauge_loop())

    try:
        async with server:
            await server.serve_forever()
    finally:
        pool_gauge_task.cancel()
        if metrics_server is not None:
            metrics_server.close()
            try:
                await metrics_server.wait_closed()
            except Exception:
                pass


def _worker_entry(worker_index: int) -> None:
    try:
        asyncio.run(_serve_one_worker(worker_index))
    except KeyboardInterrupt:
        pass


async def main() -> None:
    await _serve_one_worker(0)


def _run_multi_worker(settings) -> None:
    ctx = multiprocessing.get_context('spawn')
    procs: list[multiprocessing.Process] = []

    def _shutdown(signum, _frame) -> None:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _shutdown)

    for i in range(settings.workers):
        proc = ctx.Process(target=_worker_entry, args=(i,), name=f'listener-worker-{i}')
        proc.daemon = False
        proc.start()
        procs.append(proc)

    print(f'client-listener parent pid={os.getpid()} forked {settings.workers} workers')

    exit_code = 0
    try:
        while procs:
            for proc in list(procs):
                proc.join(timeout=0.5)
                if not proc.is_alive():
                    if proc.exitcode and proc.exitcode != 0:
                        exit_code = proc.exitcode
                    procs.remove(proc)
                    # Tear down siblings; supervisor will restart the group.
                    for other in procs:
                        if other.is_alive():
                            other.terminate()
    except KeyboardInterrupt:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()

    sys.exit(exit_code)


if __name__ == '__main__':
    settings = load_settings()
    if settings.workers > 1:
        _run_multi_worker(settings)
    else:
        asyncio.run(main())
