#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import secrets
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SESSION_STORE: dict[str, dict[str, Any]] = {}
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "MarketAdmin!2026Forever"
PBKDF2_ITERATIONS = 600_000
PASSWORD_SALT_BYTES = 16


@dataclass(frozen=True)
class FrontendConfig:
    host: str
    port: int
    listener_host: str
    listener_port: int
    listener_auth_token: str
    session_cookie_name: str
    session_ttl_seconds: int
    session_secure_cookie: bool
    bootstrap_admin_username: str
    bootstrap_admin_password: str
    db_dsn: str


class FrontendError(Exception):
    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.message = message
        self.status = int(status)


class ListenerClient:
    _MAX_IDLE_SOCKETS = 16
    _CONNECT_TIMEOUT_S = 10.0

    def __init__(self, host: str, port: int, auth_token: str):
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self._idle: list[tuple[socket.socket, Any]] = []
        self._lock = threading.Lock()

    def _checkout(self) -> tuple[socket.socket, Any]:
        with self._lock:
            if self._idle:
                return self._idle.pop()
        sock = socket.create_connection((self.host, self.port), timeout=self._CONNECT_TIMEOUT_S)
        sock.settimeout(self._CONNECT_TIMEOUT_S)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        reader = sock.makefile("rb")
        return sock, reader

    def _release(self, sock: socket.socket, reader: Any, healthy: bool) -> None:
        if not healthy:
            try:
                sock.close()
            except OSError:
                pass
            return
        with self._lock:
            if len(self._idle) >= self._MAX_IDLE_SOCKETS:
                try:
                    sock.close()
                except OSError:
                    pass
                return
            self._idle.append((sock, reader))

    def call(self, action: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        message = {
            "action": action,
            "auth_token": self.auth_token,
            "request": request or {},
        }
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")

        # On error, retry once with a fresh connection in case the listener
        # half-closed the pooled socket since we last used it.
        for attempt in range(2):
            sock, reader = self._checkout()
            try:
                sock.sendall(payload)
                raw = reader.readline()
            except (OSError, socket.timeout):
                self._release(sock, reader, healthy=False)
                if attempt == 0:
                    continue
                raise FrontendError("listener did not respond", HTTPStatus.BAD_GATEWAY)

            if not raw:
                self._release(sock, reader, healthy=False)
                if attempt == 0:
                    continue
                raise FrontendError("listener did not respond", HTTPStatus.BAD_GATEWAY)

            self._release(sock, reader, healthy=True)
            try:
                response = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise FrontendError("listener returned invalid JSON", HTTPStatus.BAD_GATEWAY) from exc
            if not response.get("ok"):
                raise FrontendError(str(response.get("error") or "listener error"), HTTPStatus.BAD_REQUEST)
            return response

        raise FrontendError("listener did not respond", HTTPStatus.BAD_GATEWAY)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> FrontendConfig:
    listener_auth_token = os.getenv("FRONTEND_LISTENER_AUTH_TOKEN") or os.getenv("CLIENT_LISTENER_AUTH_TOKEN", "")
    config = FrontendConfig(
        host=os.getenv("FRONTEND_HOST", "0.0.0.0"),
        port=int(os.getenv("FRONTEND_PORT", "8080")),
        listener_host=os.getenv("FRONTEND_LISTENER_HOST", os.getenv("CLIENT_LISTENER_HOST", "127.0.0.1")),
        listener_port=int(os.getenv("FRONTEND_LISTENER_PORT", os.getenv("CLIENT_LISTENER_PORT", "9001"))),
        listener_auth_token=listener_auth_token,
        session_cookie_name=os.getenv("FRONTEND_SESSION_COOKIE_NAME", "market_session"),
        session_ttl_seconds=int(os.getenv("FRONTEND_SESSION_TTL_SECONDS", "86400")),
        session_secure_cookie=_env_bool("FRONTEND_SESSION_SECURE_COOKIE", False),
        bootstrap_admin_username=os.getenv("FRONTEND_BOOTSTRAP_ADMIN_USERNAME", DEFAULT_ADMIN_USERNAME).strip(),
        bootstrap_admin_password=os.getenv("FRONTEND_BOOTSTRAP_ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD).strip(),
        db_dsn=os.getenv("FRONTEND_DB_DSN", os.getenv("DB_DSN", "")).strip(),
    )
    if not config.listener_auth_token:
        raise SystemExit("FRONTEND_LISTENER_AUTH_TOKEN or CLIENT_LISTENER_AUTH_TOKEN is required")
    return config


CONFIG = load_config()
LISTENER = ListenerClient(CONFIG.listener_host, CONFIG.listener_port, CONFIG.listener_auth_token)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _prune_sessions() -> None:
    now = time.time()
    expired = [token for token, session in SESSION_STORE.items() if session.get("expires_at_epoch", 0) <= now]
    for token in expired:
        SESSION_STORE.pop(token, None)


def _new_session_cookie(handler: "FrontendHandler", session_payload: dict[str, Any]) -> None:
    session_id = secrets.token_urlsafe(32)
    expires_at_epoch = time.time() + CONFIG.session_ttl_seconds
    SESSION_STORE[session_id] = {
        **session_payload,
        "created_at": _iso_now(),
        "last_seen_at": _iso_now(),
        "expires_at_epoch": expires_at_epoch,
    }
    handler._set_cookie(
        CONFIG.session_cookie_name,
        session_id,
        max_age=CONFIG.session_ttl_seconds,
        http_only=True,
        same_site="Lax",
        secure=CONFIG.session_secure_cookie,
        path="/",
    )


def _drop_session(handler: "FrontendHandler") -> None:
    session_id = handler._read_cookie(CONFIG.session_cookie_name)
    if session_id:
        SESSION_STORE.pop(session_id, None)
    handler._set_cookie(
        CONFIG.session_cookie_name,
        "",
        max_age=0,
        http_only=True,
        same_site="Lax",
        secure=CONFIG.session_secure_cookie,
        path="/",
    )


def _get_session(handler: "FrontendHandler", required: bool = False) -> dict[str, Any] | None:
    _prune_sessions()
    session_id = handler._read_cookie(CONFIG.session_cookie_name)
    session = SESSION_STORE.get(session_id or "")
    if session and session.get("expires_at_epoch", 0) > time.time():
        session["last_seen_at"] = _iso_now()
        return session
    if session_id:
        SESSION_STORE.pop(session_id, None)
    if required:
        raise FrontendError("authentication required", HTTPStatus.UNAUTHORIZED)
    return None


def _require_admin(handler: "FrontendHandler") -> dict[str, Any]:
    session = _get_session(handler, required=True)
    if not session or not session.get("is_admin"):
        raise FrontendError("admin access required", HTTPStatus.FORBIDDEN)
    return session


def _listener_payload_for_session(session: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = dict(payload or {})
    body.setdefault("account_id", session["account_id"])
    body.setdefault("account_session_token", session["account_session_token"])
    return body


def _hash_password(password: str) -> tuple[str, str, int]:
    salt_bytes = secrets.token_bytes(PASSWORD_SALT_BYTES)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        PBKDF2_ITERATIONS,
    ).hex()
    return password_hash, salt_bytes.hex(), PBKDF2_ITERATIONS


async def _ensure_bootstrap_admin(username: str, password: str, db_dsn: str) -> str:
    import asyncpg

    password_hash, password_salt, password_iterations = _hash_password(password)
    conn = await asyncpg.connect(db_dsn)
    try:
        existing = await conn.fetchrow(
            "SELECT account_id FROM accounts WHERE username = $1",
            username,
        )
        if existing:
            account_id = str(existing["account_id"])
            await conn.execute(
                """
                UPDATE accounts
                SET status = 'ACTIVE',
                    password_hash = $2,
                    password_salt = $3,
                    password_iterations = $4,
                    is_admin = TRUE,
                    updated_at = now()
                WHERE account_id = $1
                """,
                account_id,
                password_hash,
                password_salt,
                password_iterations,
            )
            n_buckets = 16
            for b in range(n_buckets):
                await conn.execute(
                    """
                    INSERT INTO account_cash_buckets (
                        account_id, bucket_id, available_cash_cents, locked_cash_cents
                    )
                    VALUES ($1, $2, 0, 0)
                    ON CONFLICT (account_id, bucket_id) DO NOTHING
                    """,
                    account_id,
                    b,
                )
            return account_id

        account_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO accounts (
                account_id,
                username,
                status,
                available_cash_cents,
                locked_cash_cents,
                password_hash,
                password_salt,
                password_iterations,
                is_admin
            )
            VALUES ($1, $2, 'ACTIVE', 5000, 0, $3, $4, $5, TRUE)
            """,
            account_id,
            username,
            password_hash,
            password_salt,
            password_iterations,
        )
        n_buckets = 16
        base, rem = divmod(5000, n_buckets)
        for b in range(n_buckets):
            avail = base + (rem if b == 0 else 0)
            await conn.execute(
                """
                INSERT INTO account_cash_buckets (
                    account_id, bucket_id, available_cash_cents, locked_cash_cents
                )
                VALUES ($1, $2, $3, 0)
                ON CONFLICT (account_id, bucket_id) DO NOTHING
                """,
                account_id,
                b,
                avail,
            )
        return account_id
    finally:
        await conn.close()


def maybe_bootstrap_admin() -> None:
    username = CONFIG.bootstrap_admin_username
    password = CONFIG.bootstrap_admin_password
    db_dsn = CONFIG.db_dsn
    if not username or not password or not db_dsn:
        return

    account_id = asyncio.run(_ensure_bootstrap_admin(username, password, db_dsn))
    print(
        f"ensured bootstrap admin account {username} "
        f"with password {password} and account_id {account_id}"
    )


class FrontendHandler(BaseHTTPRequestHandler):
    server_version = "PredictionMarketFrontend/1.0"

    def __init__(self, *args, **kwargs):
        self._response_cookies: list[str] = []
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{_iso_now()}] {self.address_string()} {fmt % args}")

    def _set_cookie(
        self,
        name: str,
        value: str,
        *,
        max_age: int,
        http_only: bool,
        same_site: str,
        secure: bool,
        path: str,
    ) -> None:
        cookie = SimpleCookie()
        cookie[name] = value
        morsel = cookie[name]
        morsel["max-age"] = str(max_age)
        morsel["path"] = path
        morsel["samesite"] = same_site
        if http_only:
            morsel["httponly"] = True
        if secure:
            morsel["secure"] = True
        self._response_cookies.append(cookie.output(header="").strip())

    def _read_cookie(self, name: str) -> str | None:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        cookie = SimpleCookie()
        cookie.load(raw)
        morsel = cookie.get(name)
        return morsel.value if morsel else None

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for cookie in self._response_cookies:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise FrontendError("not found", HTTPStatus.NOT_FOUND)
        body = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type or 'application/octet-stream'}")
        self.send_header("Content-Length", str(len(body)))
        if path.suffix in {".html", ".js", ".css"}:
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise FrontendError("invalid Content-Length") from exc
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise FrontendError("invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise FrontendError("request body must be a JSON object")
        return payload

    def _session_snapshot(self, session: dict[str, Any]) -> dict[str, Any]:
        balances = LISTENER.call(
            "get_account_balances",
            {
                "account_id": session["account_id"],
            },
        )["result"]
        session["username"] = balances["username"]
        session["is_admin"] = bool(balances["is_admin"])
        return {
            "account_id": session["account_id"],
            "username": balances["username"],
            "is_admin": bool(balances["is_admin"]),
            "status": balances["status"],
            "available_cash_cents": balances["available_cash_cents"],
            "locked_cash_cents": balances["locked_cash_cents"],
            "updated_at": balances["updated_at"],
            "web_session_created_at": session["created_at"],
        }

    def _market_snapshot(self, market_id: str) -> dict[str, Any]:
        markets = LISTENER.call("list_markets", {"limit": 500})["result"]["markets"]
        market = next((item for item in markets if item["market_id"] == market_id), None)
        if market is None:
            raise FrontendError("market not found", HTTPStatus.NOT_FOUND)
        order_book = LISTENER.call("get_order_book", {"market_id": market_id, "depth": 12})["result"]
        history = LISTENER.call("get_market_history", {"market_id": market_id, "limit": 500})["result"]
        return {
            "market": market,
            "order_book": order_book,
            "history": history,
        }

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self._handle_api_get(parsed)
                return
            self._handle_static(parsed.path)
        except FrontendError as exc:
            self._send_json({"ok": False, "error": exc.message}, status=exc.status)
        except Exception as exc:
            print(f"unexpected GET error: {exc}")
            self._send_json({"ok": False, "error": "internal server error"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/"):
                raise FrontendError("not found", HTTPStatus.NOT_FOUND)
            self._handle_api_post(parsed)
        except FrontendError as exc:
            self._send_json({"ok": False, "error": exc.message}, status=exc.status)
        except Exception as exc:
            print(f"unexpected POST error: {exc}")
            self._send_json({"ok": False, "error": "internal server error"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_static(self, request_path: str) -> None:
        relative = Path((request_path or "/").lstrip("/") or "index.html")
        path = (STATIC_DIR / relative).resolve()
        static_root = STATIC_DIR.resolve()
        if path != static_root and static_root not in path.parents:
            raise FrontendError("not found", HTTPStatus.NOT_FOUND)
        self._send_file(path)

    def _handle_api_get(self, parsed) -> None:
        query = parse_qs(parsed.query)
        if parsed.path == "/api/health":
            self._send_json({"ok": True, "status": "healthy", "time": _iso_now()})
            return
        if parsed.path == "/api/session":
            session = _get_session(self, required=False)
            if not session:
                self._send_json({"ok": True, "session": None})
                return
            self._send_json({"ok": True, "session": self._session_snapshot(session)})
            return
        if parsed.path == "/api/markets":
            search = (query.get("search") or [""])[0].strip()
            statuses_raw = (query.get("statuses") or [""])[0].strip()
            payload: dict[str, Any] = {"limit": 500}
            if search:
                payload["search"] = search
            if statuses_raw:
                payload["statuses"] = [item.strip().upper() for item in statuses_raw.split(",") if item.strip()]
            result = LISTENER.call("list_markets", payload)["result"]
            self._send_json({"ok": True, **result})
            return
        if parsed.path == "/api/portfolio":
            session = _get_session(self, required=True)
            balances = LISTENER.call("get_account_balances", {"account_id": session["account_id"]})["result"]
            wallet_history = LISTENER.call("get_wallet_history", {"account_id": session["account_id"], "limit": 1000})["result"]
            positions = LISTENER.call("get_positions", {"account_id": session["account_id"]})["result"]["positions"]
            open_orders = LISTENER.call("list_open_orders", {"account_id": session["account_id"], "limit": 100})["result"]["orders"]
            trades = LISTENER.call("get_trades", {"account_id": session["account_id"], "limit": 100})["result"]["trades"]
            markets = LISTENER.call("list_markets", {"limit": 500})["result"]["markets"]
            market_map = {item["market_id"]: item for item in markets}
            for row in positions:
                market = market_map.get(row["market_id"], {})
                row["market_title"] = market.get("title")
                row["live_yes_price"] = market.get("live_yes_price")
                row["live_no_price"] = market.get("live_no_price")
            for row in open_orders:
                market = market_map.get(row["market_id"], {})
                row["market_title"] = market.get("title")
            for row in trades:
                market = market_map.get(row["market_id"], {})
                row["market_title"] = market.get("title")
            self._send_json(
                {
                    "ok": True,
                    "balances": balances,
                    "wallet_history": wallet_history,
                    "positions": positions,
                    "open_orders": open_orders,
                    "recent_trades": trades,
                }
            )
            return

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 4 and parts[:2] == ["api", "markets"] and parts[3] in {"book", "history", "snapshot"}:
            market_id = parts[2]
            if parts[3] == "book":
                result = LISTENER.call("get_order_book", {"market_id": market_id, "depth": 12})["result"]
                self._send_json({"ok": True, **result})
                return
            if parts[3] == "history":
                result = LISTENER.call("get_market_history", {"market_id": market_id, "limit": 500})["result"]
                self._send_json({"ok": True, **result})
                return
            snapshot = self._market_snapshot(market_id)
            self._send_json({"ok": True, **snapshot})
            return

        raise FrontendError("not found", HTTPStatus.NOT_FOUND)

    def _handle_api_post(self, parsed) -> None:
        payload = self._read_json_body()
        if parsed.path == "/api/signup":
            username = str(payload.get("username", "")).strip()
            password = str(payload.get("password", ""))
            if not username or not password:
                raise FrontendError("username and password are required")
            create_result = LISTENER.call(
                "create_account",
                {
                    "username": username,
                    "password": password,
                },
            )["result"]
            auth_result = LISTENER.call(
                "authenticate_account",
                {
                    "username": username,
                    "password": password,
                },
            )["result"]
            _new_session_cookie(
                self,
                {
                    "account_id": auth_result["account_id"],
                    "username": auth_result["username"],
                    "is_admin": bool(auth_result.get("is_admin")),
                    "account_session_token": auth_result["account_session_token"],
                    "account_session_expires_at": auth_result["expires_at"],
                },
            )
            self._send_json(
                {
                    "ok": True,
                    "account": create_result,
                    "session": {
                        "account_id": auth_result["account_id"],
                        "username": auth_result["username"],
                        "is_admin": bool(auth_result.get("is_admin")),
                        "available_cash_cents": create_result["available_cash_cents"],
                        "bonus_cents": create_result["bonus_cents"],
                    },
                },
                status=HTTPStatus.CREATED,
            )
            return
        if parsed.path == "/api/login":
            username = str(payload.get("username", "")).strip()
            password = str(payload.get("password", ""))
            auth_result = LISTENER.call(
                "authenticate_account",
                {
                    "username": username,
                    "password": password,
                },
            )["result"]
            _new_session_cookie(
                self,
                {
                    "account_id": auth_result["account_id"],
                    "username": auth_result["username"],
                    "is_admin": bool(auth_result.get("is_admin")),
                    "account_session_token": auth_result["account_session_token"],
                    "account_session_expires_at": auth_result["expires_at"],
                },
            )
            self._send_json(
                {
                    "ok": True,
                    "session": {
                        "account_id": auth_result["account_id"],
                        "username": auth_result["username"],
                        "is_admin": bool(auth_result.get("is_admin")),
                        "account_session_expires_at": auth_result["expires_at"],
                    },
                }
            )
            return
        if parsed.path == "/api/logout":
            _drop_session(self)
            self._send_json({"ok": True, "logged_out": True})
            return
        if parsed.path == "/api/orders":
            session = _get_session(self, required=True)
            market_id = str(payload.get("market_id", "")).strip()
            side = str(payload.get("side", "")).upper().strip()
            time_in_force = str(payload.get("time_in_force", "GTC")).upper().strip()
            try:
                qty = int(payload.get("qty"))
                price_cents = int(payload.get("price_cents"))
            except Exception as exc:
                raise FrontendError("qty and price_cents must be integers") from exc
            result = LISTENER.call(
                "submit_order",
                {
                    "request_id": str(uuid.uuid4()),
                    "account_id": session["account_id"],
                    "account_session_token": session["account_session_token"],
                    "market_id": market_id,
                    "side": side,
                    "qty": qty,
                    "price_cents": price_cents,
                    "time_in_force": time_in_force,
                    "ingress_ts_ns": time.time_ns(),
                },
            )["order"]
            self._send_json({"ok": True, "order": result})
            return
        if parsed.path == "/api/orders/cancel":
            session = _get_session(self, required=True)
            order_id = str(payload.get("order_id", "")).strip()
            result = LISTENER.call(
                "cancel_order",
                {
                    "cancel_id": str(uuid.uuid4()),
                    "order_id": order_id,
                    "account_id": session["account_id"],
                    "account_session_token": session["account_session_token"],
                    "reason": str(payload.get("reason", "cancelled from frontend")).strip() or "cancelled from frontend",
                },
            )["cancel"]
            self._send_json({"ok": True, "cancel": result})
            return
        if parsed.path == "/api/admin/markets":
            session = _require_admin(self)
            title = str(payload.get("title", "")).strip()
            slug = str(payload.get("slug", "")).strip()
            if not slug:
                slug = "-".join(title.lower().split())[:80]
            create_payload = _listener_payload_for_session(
                session,
                {
                    "market_id": str(uuid.uuid4()),
                    "slug": slug,
                    "title": title,
                    "description": str(payload.get("description", "")).strip() or None,
                    "status": str(payload.get("status", "DRAFT")).upper().strip() or "DRAFT",
                    "tick_size_cents": int(payload.get("tick_size_cents", 1)),
                    "min_price_cents": int(payload.get("min_price_cents", 1)),
                    "max_price_cents": int(payload.get("max_price_cents", 99)),
                    "close_time": str(payload.get("close_time", "")).strip() or None,
                    "resolve_time": str(payload.get("resolve_time", "")).strip() or None,
                },
            )
            result = LISTENER.call("create_market", create_payload)["result"]
            self._send_json({"ok": True, "market": result}, status=HTTPStatus.CREATED)
            return
        if parsed.path == "/api/admin/markets/status":
            session = _require_admin(self)
            result = LISTENER.call(
                "update_market_status",
                _listener_payload_for_session(
                    session,
                    {
                        "market_id": str(payload.get("market_id", "")).strip(),
                        "new_status": str(payload.get("new_status", "")).upper().strip(),
                    },
                ),
            )["result"]
            self._send_json({"ok": True, "market": result})
            return
        if parsed.path == "/api/admin/markets/resolve":
            session = _require_admin(self)
            result = LISTENER.call(
                "resolve_market",
                _listener_payload_for_session(
                    session,
                    {
                        "resolution_id": str(uuid.uuid4()),
                        "market_id": str(payload.get("market_id", "")).strip(),
                        "outcome": str(payload.get("outcome", "")).upper().strip(),
                        "notes": str(payload.get("notes", "")).strip() or None,
                    },
                ),
            )["result"]
            self._send_json({"ok": True, "resolution": result})
            return
        raise FrontendError("not found", HTTPStatus.NOT_FOUND)


def main() -> None:
    maybe_bootstrap_admin()
    server = ThreadingHTTPServer((CONFIG.host, CONFIG.port), FrontendHandler)
    server.daemon_threads = True
    print(
        f"frontend running on http://{CONFIG.host}:{CONFIG.port} "
        f"-> listener {CONFIG.listener_host}:{CONFIG.listener_port}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
