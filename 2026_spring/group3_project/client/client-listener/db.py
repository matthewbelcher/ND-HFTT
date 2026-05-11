import asyncio
import random
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg

from metrics import METRICS, Timer
from models import CancelOrderRequest, SubmitOrderRequest
from security import generate_session_token, hash_password, hash_session_token, verify_password


class SubmissionError(Exception):
    pass


class RetryableOCCError(Exception):
    pass


OPEN_ORDER_STATUSES = ("ACCEPTED", "OPEN", "PARTIALLY_FILLED")

# Must agree with the matcher and executor processes.
ACCOUNT_CASH_BUCKETS = 16


def _bucket_for_uuid(value: str) -> int:
    try:
        return int(uuid.UUID(value).int) % ACCOUNT_CASH_BUCKETS
    except Exception:
        return abs(hash(value)) % ACCOUNT_CASH_BUCKETS


def _split_amount(amount_cents: int, n_buckets: int) -> list[int]:
    base, rem = divmod(amount_cents, n_buckets)
    out = [base] * n_buckets
    out[0] += rem
    return out


def _is_occ_error(exc: Exception) -> bool:
    if not isinstance(exc, asyncpg.PostgresError):
        return False
    return exc.sqlstate in {"40001", "40P01"}


def _require_uuid(value: Any, name: str) -> str:
    try:
        out = str(value)
        uuid.UUID(out)
        return out
    except Exception as exc:
        raise SubmissionError(f"{name} must be a valid UUID") from exc


def _safe_limit(value: Any, default: int, max_limit: int = 500) -> int:
    try:
        n = int(value) if value is not None else default
    except Exception as exc:
        raise SubmissionError("limit must be an integer") from exc
    if n < 1 or n > max_limit:
        raise SubmissionError(f"limit must be between 1 and {max_limit}")
    return n


def _require_non_empty_str(value: Any, name: str) -> str:
    out = str(value or "").strip()
    if not out:
        raise SubmissionError(f"{name} is required")
    return out


def _wallet_cash_effect(entry: Any) -> int:
    reason = str(entry["reason"])
    cash_delta = int(entry["cash_delta_cents"])
    locked_delta = int(entry["locked_cash_delta_cents"])

    if reason in {"ORDER_LOCK", "ORDER_UNLOCK"}:
        return 0
    if reason == "TRADE_EXECUTION":
        return cash_delta + locked_delta
    return cash_delta


_MARKET_CACHE_MISS = object()


class MarketCache:
    def __init__(self, pool: asyncpg.Pool, ttl_seconds: float = 5.0) -> None:
        self.pool = pool
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, tuple[float, dict[str, Any] | None]] = {}
        self._lock = asyncio.Lock()

    def _lookup(self, market_id: str) -> Any:
        entry = self._entries.get(market_id)
        if entry is None:
            return _MARKET_CACHE_MISS
        ts, data = entry
        if (time.monotonic() - ts) > self.ttl_seconds:
            return _MARKET_CACHE_MISS
        return data

    def invalidate(self, market_id: str) -> None:
        self._entries.pop(market_id, None)

    async def validate(self, market_id: str, price_cents: int) -> None:
        cached = self._lookup(market_id)
        if cached is _MARKET_CACHE_MISS:
            async with self._lock:
                cached = self._lookup(market_id)
                if cached is _MARKET_CACHE_MISS:
                    async with self.pool.acquire() as conn:
                        row = await conn.fetchrow(
                            """
                            SELECT status, min_price_cents, max_price_cents
                            FROM markets
                            WHERE market_id = $1
                            """,
                            market_id,
                        )
                    cached = (
                        None
                        if row is None
                        else {
                            "status": str(row["status"]),
                            "min_price_cents": int(row["min_price_cents"]),
                            "max_price_cents": int(row["max_price_cents"]),
                        }
                    )
                    self._entries[market_id] = (time.monotonic(), cached)
        if cached is None:
            raise SubmissionError("market not found")
        if cached["status"] != "ACTIVE":
            raise SubmissionError(f"market status is {cached['status']}, not ACTIVE")
        if price_cents < cached["min_price_cents"] or price_cents > cached["max_price_cents"]:
            raise SubmissionError("price outside market bounds")


class SessionCache:
    def __init__(self, ttl_seconds: float = 5.0, max_entries: int = 50_000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[tuple[str, str], float] = {}

    def is_valid(self, account_id: str, token_hash: str) -> bool:
        key = (account_id, token_hash)
        ts = self._entries.get(key)
        if ts is None:
            return False
        if (time.monotonic() - ts) > self.ttl_seconds:
            self._entries.pop(key, None)
            return False
        return True

    def remember(self, account_id: str, token_hash: str) -> None:
        if len(self._entries) >= self.max_entries:
            cutoff = sorted(self._entries.values())[len(self._entries) // 20]
            self._entries = {k: v for k, v in self._entries.items() if v >= cutoff}
        self._entries[(account_id, token_hash)] = time.monotonic()

    def forget(self, account_id: str, token_hash: str | None = None) -> None:
        if token_hash is None:
            for key in [k for k in self._entries if k[0] == account_id]:
                self._entries.pop(key, None)
        else:
            self._entries.pop((account_id, token_hash), None)


class OrderRepository:
    def __init__(self, pool: asyncpg.Pool, account_session_ttl_seconds: int = 43200):
        self.pool = pool
        self.account_session_ttl_seconds = account_session_ttl_seconds
        self.market_cache = MarketCache(pool)
        self.session_cache = SessionCache()

    async def _with_occ_retry(self, fn):
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                return await fn()
            except RetryableOCCError:
                METRICS.incr("listener_occ_retries_total")
                if attempt == max_attempts:
                    raise SubmissionError("database conflict after retries")
            except asyncpg.PostgresError as exc:
                if not _is_occ_error(exc):
                    raise SubmissionError(f"database error: {exc}") from exc
                METRICS.incr("listener_occ_retries_total")
                if attempt == max_attempts:
                    raise SubmissionError("database conflict after retries") from exc
            delay = (0.02 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.02)
            await asyncio.sleep(delay)

    async def ready(self) -> bool:
        async with self.pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
            return int(val) == 1

    async def create_account(self, payload: dict[str, Any]) -> dict[str, Any]:
        username = _require_non_empty_str(payload.get("username"), "username")
        password = _require_non_empty_str(payload.get("password"), "password")
        external_user_id = payload.get("external_user_id")
        if external_user_id is not None:
            external_user_id = str(external_user_id).strip() or None

        if len(username) > 128:
            raise SubmissionError("username must be at most 128 characters")
        if len(password) < 8:
            raise SubmissionError("password must be at least 8 characters")
        if len(password) > 1024:
            raise SubmissionError("password must be at most 1024 characters")
        if external_user_id is not None and len(external_user_id) > 256:
            raise SubmissionError("external_user_id must be at most 256 characters")

        account_id = str(uuid.uuid4())
        password_hash, password_salt, password_iterations = hash_password(password)

        initial_cash_cents = 5000
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(
                        """
                        INSERT INTO accounts (
                            account_id,
                            external_user_id,
                            username,
                            status,
                            available_cash_cents,
                            locked_cash_cents,
                            password_hash,
                            password_salt,
                            password_iterations,
                            is_admin
                        )
                        VALUES ($1, $2, $3, 'ACTIVE', $7, 0, $4, $5, $6, FALSE)
                        RETURNING account_id, username, external_user_id, status, available_cash_cents, is_admin
                        """,
                        account_id,
                        external_user_id,
                        username,
                        password_hash,
                        password_salt,
                        password_iterations,
                        initial_cash_cents,
                    )
                    await self._seed_account_cash_buckets(conn, account_id, initial_cash_cents)
        except asyncpg.PostgresError as exc:
            if exc.sqlstate == "23505":
                raise SubmissionError("username or external_user_id already exists") from exc
            raise SubmissionError(f"database error: {exc}") from exc

        return {
            "account_id": str(row["account_id"]),
            "username": str(row["username"]),
            "external_user_id": row["external_user_id"],
            "status": str(row["status"]),
            "available_cash_cents": int(row["available_cash_cents"]),
            "bonus_cents": initial_cash_cents,
            "is_admin": bool(row["is_admin"]),
        }

    async def _seed_account_cash_buckets(
        self,
        conn: asyncpg.Connection,
        account_id: str,
        available_cash_cents: int,
        locked_cash_cents: int = 0,
    ) -> None:
        avail_split = _split_amount(available_cash_cents, ACCOUNT_CASH_BUCKETS)
        locked_split = _split_amount(locked_cash_cents, ACCOUNT_CASH_BUCKETS)
        await conn.executemany(
            """
            INSERT INTO account_cash_buckets (
                account_id, bucket_id, available_cash_cents, locked_cash_cents
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (account_id, bucket_id) DO NOTHING
            """,
            [
                (account_id, b, avail_split[b], locked_split[b])
                for b in range(ACCOUNT_CASH_BUCKETS)
            ],
        )

    async def authenticate_account(self, payload: dict[str, Any]) -> dict[str, Any]:
        username = _require_non_empty_str(payload.get("username"), "username")
        password = _require_non_empty_str(payload.get("password"), "password")

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT account_id, username, status, password_hash, password_salt, password_iterations, is_admin
                    FROM accounts
                    WHERE username = $1
                    """,
                    username,
                )
                if not row:
                    raise SubmissionError("invalid username or password")
                if str(row["status"]) != "ACTIVE":
                    raise SubmissionError("account is not active")
                if not row["password_hash"] or not row["password_salt"] or not row["password_iterations"]:
                    raise SubmissionError("account has no password configured")
                if not verify_password(
                    password,
                    str(row["password_hash"]),
                    str(row["password_salt"]),
                    int(row["password_iterations"]),
                ):
                    raise SubmissionError("invalid username or password")

                await conn.execute(
                    """
                    DELETE FROM account_auth_sessions
                    WHERE account_id = $1
                      AND (expires_at <= now() OR revoked_at IS NOT NULL)
                    """,
                    row["account_id"],
                )
                self.session_cache.forget(str(row["account_id"]))

                account_session_token = generate_session_token()
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.account_session_ttl_seconds)
                await conn.execute(
                    """
                    INSERT INTO account_auth_sessions (
                        session_id,
                        account_id,
                        token_hash,
                        expires_at
                    )
                    VALUES ($1, $2, $3, $4)
                    """,
                    str(uuid.uuid4()),
                    row["account_id"],
                    hash_session_token(account_session_token),
                    expires_at,
                )
        except asyncpg.PostgresError as exc:
            raise SubmissionError(f"database error: {exc}") from exc

        return {
            "account_id": str(row["account_id"]),
            "username": str(row["username"]),
            "account_session_token": account_session_token,
            "expires_at": expires_at.isoformat(),
            "is_admin": bool(row["is_admin"]),
        }

    async def _require_account_session(
        self,
        conn: asyncpg.Connection,
        account_id: str,
        account_session_token: str,
    ) -> None:
        token_hash = hash_session_token(account_session_token)
        if self.session_cache.is_valid(account_id, token_hash):
            return
        session_row = await conn.fetchrow(
            """
            UPDATE account_auth_sessions
            SET last_used_at = now()
            WHERE account_id = $1
              AND token_hash = $2
              AND revoked_at IS NULL
              AND expires_at > now()
            RETURNING session_id
            """,
            account_id,
            token_hash,
        )
        if not session_row:
            raise SubmissionError("account authentication failed")
        self.session_cache.remember(account_id, token_hash)

    async def _require_admin_account_session(
        self,
        conn: asyncpg.Connection,
        account_id: str,
        account_session_token: str,
    ) -> dict[str, Any]:
        await self._require_account_session(conn, account_id, account_session_token)
        account = await conn.fetchrow(
            """
            SELECT account_id, username, is_admin, status
            FROM accounts
            WHERE account_id = $1
            """,
            account_id,
        )
        if not account or str(account["status"]) != "ACTIVE":
            raise SubmissionError("account not found or inactive")
        if not bool(account["is_admin"]):
            raise SubmissionError("admin privileges required")
        return {
            "account_id": str(account["account_id"]),
            "username": str(account["username"]),
            "is_admin": True,
        }

    async def submit_order(self, req: SubmitOrderRequest) -> dict[str, Any]:
        with Timer(METRICS, "listener_submit_latency_ms"):
            try:
                result = await self._with_occ_retry(lambda: self._submit_order_once(req))
            except SubmissionError:
                METRICS.incr("listener_submit_errors_total")
                raise
            METRICS.incr("listener_submit_total")
            if result.get("idempotent"):
                METRICS.incr("listener_submit_idempotent_total")
            return result

    async def cancel_order(self, req: CancelOrderRequest) -> dict[str, Any]:
        with Timer(METRICS, "listener_cancel_latency_ms"):
            try:
                result = await self._with_occ_retry(lambda: self._cancel_order_once(req))
            except SubmissionError:
                METRICS.incr("listener_cancel_errors_total")
                raise
            METRICS.incr("listener_cancel_total")
            return result

    async def get_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        order_id = _require_uuid(payload.get("order_id"), "order_id")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT request_id, global_seq, account_id, market_id, side, qty,
                       remaining_qty, price_cents, time_in_force, status,
                       reject_reason, created_at, updated_at
                FROM orders
                WHERE request_id = $1
                """,
                order_id,
            )
            if not row:
                raise SubmissionError("order not found")

            cancel_row = await conn.fetchrow(
                """
                SELECT cancel_id, reason, created_at
                FROM order_cancels
                WHERE order_id = $1
                ORDER BY cancel_seq DESC
                LIMIT 1
                """,
                order_id,
            )
            result = {
                "request_id": str(row["request_id"]),
                "global_seq": int(row["global_seq"]),
                "account_id": str(row["account_id"]),
                "market_id": str(row["market_id"]),
                "side": str(row["side"]),
                "qty": int(row["qty"]),
                "remaining_qty": int(row["remaining_qty"]),
                "price_cents": int(row["price_cents"]),
                "time_in_force": str(row["time_in_force"]),
                "status": str(row["status"]),
                "reject_reason": row["reject_reason"],
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }
            if cancel_row:
                result["cancel"] = {
                    "cancel_id": str(cancel_row["cancel_id"]),
                    "reason": cancel_row["reason"],
                    "created_at": cancel_row["created_at"].isoformat(),
                }
            return result

    async def list_orders(self, payload: dict[str, Any]) -> dict[str, Any]:
        limit = _safe_limit(payload.get("limit"), 100)
        status_filter = payload.get("statuses")

        clauses = []
        args: list[Any] = []
        idx = 1

        if payload.get("account_id") is not None:
            clauses.append(f"account_id = ${idx}")
            args.append(_require_uuid(payload.get("account_id"), "account_id"))
            idx += 1
        if payload.get("market_id") is not None:
            clauses.append(f"market_id = ${idx}")
            args.append(_require_uuid(payload.get("market_id"), "market_id"))
            idx += 1
        if status_filter is not None:
            if not isinstance(status_filter, Sequence) or isinstance(status_filter, (str, bytes)):
                raise SubmissionError("statuses must be an array of strings")
            statuses = [str(s).upper() for s in status_filter]
            if statuses:
                placeholders = ",".join(f"${idx + i}" for i in range(len(statuses)))
                clauses.append(f"status IN ({placeholders})")
                args.extend(statuses)
                idx += len(statuses)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        args.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT request_id, account_id, market_id, side, qty, remaining_qty,
                       price_cents, status, global_seq, created_at, updated_at
                FROM orders
                {where_sql}
                ORDER BY global_seq DESC
                LIMIT ${idx}
                """,
                *args,
            )
        return {
            "orders": [
                {
                    "request_id": str(r["request_id"]),
                    "account_id": str(r["account_id"]),
                    "market_id": str(r["market_id"]),
                    "side": str(r["side"]),
                    "qty": int(r["qty"]),
                    "remaining_qty": int(r["remaining_qty"]),
                    "price_cents": int(r["price_cents"]),
                    "status": str(r["status"]),
                    "global_seq": int(r["global_seq"]),
                    "created_at": r["created_at"].isoformat(),
                    "updated_at": r["updated_at"].isoformat(),
                }
                for r in rows
            ]
        }

    async def list_open_orders(self, payload: dict[str, Any]) -> dict[str, Any]:
        account_id = _require_uuid(payload.get("account_id"), "account_id")
        market_id = payload.get("market_id")
        limit = _safe_limit(payload.get("limit"), 100)

        args: list[Any] = [account_id]
        clause = ""
        if market_id is not None:
            clause = "AND market_id = $2"
            args.append(_require_uuid(market_id, "market_id"))
            limit_idx = 3
        else:
            limit_idx = 2
        args.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT request_id, market_id, side, qty, remaining_qty, price_cents, status, global_seq
                FROM orders
                WHERE account_id = $1
                  {clause}
                  AND status IN ('ACCEPTED','OPEN','PARTIALLY_FILLED')
                  AND remaining_qty > 0
                ORDER BY global_seq DESC
                LIMIT ${limit_idx}
                """,
                *args,
            )
        return {
            "orders": [
                {
                    "request_id": str(r["request_id"]),
                    "market_id": str(r["market_id"]),
                    "side": str(r["side"]),
                    "qty": int(r["qty"]),
                    "remaining_qty": int(r["remaining_qty"]),
                    "price_cents": int(r["price_cents"]),
                    "status": str(r["status"]),
                    "global_seq": int(r["global_seq"]),
                }
                for r in rows
            ]
        }

    async def cancel_all_orders(self, payload: dict[str, Any]) -> dict[str, Any]:
        account_id = _require_uuid(payload.get("account_id"), "account_id")
        account_session_token = _require_non_empty_str(payload.get("account_session_token"), "account_session_token")
        market_id = payload.get("market_id")
        if market_id is not None:
            market_id = _require_uuid(market_id, "market_id")
        reason = payload.get("reason")
        if reason is not None:
            reason = str(reason)

        async def _op():
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    args: list[Any] = [account_id]
                    clause = ""
                    if market_id is not None:
                        clause = "AND market_id = $2"
                        args.append(market_id)
                    rows = await conn.fetch(
                        f"""
                        SELECT request_id
                        FROM orders
                        WHERE account_id = $1
                          {clause}
                          AND status IN ('ACCEPTED','OPEN','PARTIALLY_FILLED')
                          AND remaining_qty > 0
                        ORDER BY global_seq
                        """,
                        *args,
                    )
                    cancelled = []
                    for row in rows:
                        result = await self._cancel_order_tx(
                            conn,
                            cancel_id=str(uuid.uuid4()),
                            order_id=str(row["request_id"]),
                            account_id=account_id,
                            account_session_token=account_session_token,
                            reason=reason or "cancel_all_orders",
                        )
                        cancelled.append(result)
                    return {
                        "cancelled_count": len(cancelled),
                        "orders": cancelled,
                    }

        return await self._with_occ_retry(_op)

    async def replace_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        cancel_id = _require_uuid(payload.get("cancel_id"), "cancel_id")
        old_order_id = _require_uuid(payload.get("old_order_id"), "old_order_id")
        account_id = _require_uuid(payload.get("account_id"), "account_id")

        request_payload = payload.get("new_order")
        if not isinstance(request_payload, dict):
            raise SubmissionError("new_order payload must be an object")
        req = SubmitOrderRequest.from_dict(request_payload)

        if req.account_id != account_id:
            raise SubmissionError("new_order account_id must match account_id")

        async def _op():
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    old_order = await conn.fetchrow(
                        """
                        SELECT market_id
                        FROM orders
                        WHERE request_id = $1
                        """,
                        old_order_id,
                    )
                    if not old_order:
                        raise SubmissionError("old_order_id not found")
                    if str(old_order["market_id"]) != req.market_id:
                        raise SubmissionError("new_order market_id must match old order market")

                    cancel_result = await self._cancel_order_tx(
                        conn,
                        cancel_id=cancel_id,
                        order_id=old_order_id,
                        account_id=account_id,
                        account_session_token=req.account_session_token,
                        reason="replace_order",
                    )
                    submit_result = await self._submit_order_tx(conn, req)
                    return {
                        "cancel": cancel_result,
                        "new_order": submit_result,
                    }

        return await self._with_occ_retry(_op)

    async def get_trades(self, payload: dict[str, Any]) -> dict[str, Any]:
        limit = _safe_limit(payload.get("limit"), 100)
        market_id = payload.get("market_id")
        account_id = payload.get("account_id")
        order_id = payload.get("order_id")

        clauses = []
        args: list[Any] = []
        idx = 1

        if market_id is not None:
            clauses.append(f"t.market_id = ${idx}")
            args.append(_require_uuid(market_id, "market_id"))
            idx += 1
        if order_id is not None:
            oid = _require_uuid(order_id, "order_id")
            clauses.append(f"(t.resting_order_id = ${idx} OR t.aggressing_order_id = ${idx})")
            args.append(oid)
            idx += 1
        if account_id is not None:
            clauses.append(
                f"EXISTS (SELECT 1 FROM trade_parties tp WHERE tp.trade_id = t.trade_id AND tp.account_id = ${idx})"
            )
            args.append(_require_uuid(account_id, "account_id"))
            idx += 1

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        args.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT t.trade_id, t.market_id, t.resting_order_id, t.aggressing_order_id,
                       t.qty, t.yes_price_cents, t.match_seq, t.created_at
                FROM trades t
                {where_sql}
                ORDER BY t.match_seq DESC
                LIMIT ${idx}
                """,
                *args,
            )

        return {
            "trades": [
                {
                    "trade_id": str(r["trade_id"]),
                    "market_id": str(r["market_id"]),
                    "resting_order_id": str(r["resting_order_id"]),
                    "aggressing_order_id": str(r["aggressing_order_id"]),
                    "qty": int(r["qty"]),
                    "yes_price_cents": int(r["yes_price_cents"]),
                    "match_seq": int(r["match_seq"]),
                    "created_at": r["created_at"].isoformat(),
                }
                for r in rows
            ]
        }

    async def get_positions(self, payload: dict[str, Any]) -> dict[str, Any]:
        account_id = _require_uuid(payload.get("account_id"), "account_id")
        market_id = payload.get("market_id")

        args: list[Any] = [account_id]
        clause = ""
        if market_id is not None:
            clause = "AND market_id = $2"
            args.append(_require_uuid(market_id, "market_id"))

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT market_id, yes_shares, no_shares, locked_yes_shares, locked_no_shares, updated_at
                FROM positions
                WHERE account_id = $1
                  {clause}
                ORDER BY market_id
                """,
                *args,
            )

        return {
            "positions": [
                {
                    "market_id": str(r["market_id"]),
                    "yes_shares": int(r["yes_shares"]),
                    "no_shares": int(r["no_shares"]),
                    "locked_yes_shares": int(r["locked_yes_shares"]),
                    "locked_no_shares": int(r["locked_no_shares"]),
                    "updated_at": r["updated_at"].isoformat(),
                }
                for r in rows
            ]
        }

    async def get_account_balances(self, payload: dict[str, Any]) -> dict[str, Any]:
        account_id = _require_uuid(payload.get("account_id"), "account_id")
        async with self.pool.acquire() as conn:
            account = await conn.fetchrow(
                """
                SELECT account_id, username, is_admin, status, updated_at
                FROM accounts
                WHERE account_id = $1
                """,
                account_id,
            )
            if not account:
                raise SubmissionError("account not found")

            bucket_totals = await conn.fetchrow(
                """
                SELECT
                    COALESCE(SUM(available_cash_cents), 0) AS available,
                    COALESCE(SUM(locked_cash_cents),    0) AS locked
                FROM account_cash_buckets
                WHERE account_id = $1
                """,
                account_id,
            )

            dep_sum = await conn.fetchval(
                """
                SELECT COALESCE(SUM(amount_cents), 0)
                FROM cash_transactions
                WHERE account_id = $1 AND type = 'DEPOSIT' AND status = 'COMPLETED'
                """,
                account_id,
            )
            wd_sum = await conn.fetchval(
                """
                SELECT COALESCE(SUM(amount_cents), 0)
                FROM cash_transactions
                WHERE account_id = $1 AND type = 'WITHDRAWAL' AND status = 'COMPLETED'
                """,
                account_id,
            )

        return {
            "account_id": str(account["account_id"]),
            "username": str(account["username"]),
            "is_admin": bool(account["is_admin"]),
            "status": str(account["status"]),
            "available_cash_cents": int(bucket_totals["available"]),
            "locked_cash_cents": int(bucket_totals["locked"]),
            "total_deposits_cents": int(dep_sum),
            "total_withdrawals_cents": int(wd_sum),
            "updated_at": account["updated_at"].isoformat(),
        }

    async def get_wallet_history(self, payload: dict[str, Any]) -> dict[str, Any]:
        account_id = _require_uuid(payload.get("account_id"), "account_id")
        limit = _safe_limit(payload.get("limit"), 500, max_limit=5000)

        async with self.pool.acquire() as conn:
            account = await conn.fetchrow(
                """
                SELECT account_id, username
                FROM accounts
                WHERE account_id = $1
                """,
                account_id,
            )
            if not account:
                raise SubmissionError("account not found")

            bucket_totals = await conn.fetchrow(
                """
                SELECT
                    COALESCE(SUM(available_cash_cents), 0) AS available,
                    COALESCE(SUM(locked_cash_cents),    0) AS locked
                FROM account_cash_buckets
                WHERE account_id = $1
                """,
                account_id,
            )

            rows = await conn.fetch(
                """
                SELECT entry_id, market_id, order_id, trade_id, cash_txn_id,
                       cash_delta_cents, locked_cash_delta_cents,
                       yes_share_delta, no_share_delta,
                       locked_yes_delta, locked_no_delta,
                       reason, notes, created_at
                FROM ledger_entries
                WHERE account_id = $1
                ORDER BY created_at DESC, entry_id DESC
                LIMIT $2
                """,
                account_id,
                limit,
            )

        current_total_cash = int(bucket_totals["available"]) + int(bucket_totals["locked"])
        baseline_cash = current_total_cash - sum(_wallet_cash_effect(row) for row in rows)
        running_cash = baseline_cash
        points: list[dict[str, Any]] = []

        for row in reversed(rows):
            running_cash += _wallet_cash_effect(row)
            points.append(
                {
                    "entry_id": str(row["entry_id"]),
                    "created_at": row["created_at"].isoformat(),
                    "reason": str(row["reason"]),
                    "notes": row["notes"],
                    "market_id": None if row["market_id"] is None else str(row["market_id"]),
                    "order_id": None if row["order_id"] is None else str(row["order_id"]),
                    "trade_id": None if row["trade_id"] is None else str(row["trade_id"]),
                    "cash_txn_id": None if row["cash_txn_id"] is None else str(row["cash_txn_id"]),
                    "cash_effect_cents": _wallet_cash_effect(row),
                    "cash_delta_cents": int(row["cash_delta_cents"]),
                    "locked_cash_delta_cents": int(row["locked_cash_delta_cents"]),
                    "yes_share_delta": int(row["yes_share_delta"]),
                    "no_share_delta": int(row["no_share_delta"]),
                    "locked_yes_delta": int(row["locked_yes_delta"]),
                    "locked_no_delta": int(row["locked_no_delta"]),
                    "total_cash_cents": running_cash,
                }
            )

        return {
            "account_id": str(account["account_id"]),
            "username": str(account["username"]),
            "current_total_cash_cents": current_total_cash,
            "current_locked_cash_cents": int(bucket_totals["locked"]),
            "baseline_total_cash_cents": baseline_cash,
            "points": points,
        }

    async def deposit_cash(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._with_occ_retry(lambda: self._cash_tx(payload, tx_type="DEPOSIT"))

    async def withdraw_cash(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._with_occ_retry(lambda: self._cash_tx(payload, tx_type="WITHDRAWAL"))

    async def _cash_tx(self, payload: dict[str, Any], tx_type: str) -> dict[str, Any]:
        cash_txn_id = _require_uuid(payload.get("cash_txn_id"), "cash_txn_id")
        account_id = _require_uuid(payload.get("account_id"), "account_id")
        account_session_token = _require_non_empty_str(payload.get("account_session_token"), "account_session_token")
        try:
            amount = int(payload.get("amount_cents"))
        except Exception as exc:
            raise SubmissionError("amount_cents must be an integer") from exc
        if amount <= 0:
            raise SubmissionError("amount_cents must be > 0")
        external_ref = payload.get("external_ref")
        notes = payload.get("notes")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await self._require_account_session(conn, account_id, account_session_token)
                existing = await conn.fetchrow(
                    """
                    SELECT cash_txn_id, type, amount_cents, status
                    FROM cash_transactions
                    WHERE cash_txn_id = $1
                    """,
                    cash_txn_id,
                )
                if existing:
                    if str(existing["type"]) != tx_type or int(existing["amount_cents"]) != amount:
                        raise SubmissionError("cash_txn_id already used with different payload")
                    return {
                        "idempotent": True,
                        "cash_txn_id": str(existing["cash_txn_id"]),
                        "type": str(existing["type"]),
                        "amount_cents": int(existing["amount_cents"]),
                        "status": str(existing["status"]),
                    }

                account = await conn.fetchrow(
                    "SELECT status FROM accounts WHERE account_id = $1",
                    account_id,
                )
                if not account or str(account["status"]) != "ACTIVE":
                    raise SubmissionError("account update failed (inactive or insufficient funds)")

                await self._seed_account_cash_buckets(conn, account_id, 0, 0)

                if tx_type == "DEPOSIT":
                    deposit_split = _split_amount(amount, ACCOUNT_CASH_BUCKETS)
                    await conn.executemany(
                        """
                        UPDATE account_cash_buckets
                        SET available_cash_cents = available_cash_cents + $3,
                            updated_at = now()
                        WHERE account_id = $1 AND bucket_id = $2
                        """,
                        [(account_id, b, deposit_split[b]) for b in range(ACCOUNT_CASH_BUCKETS)],
                    )
                    cash_delta = amount
                else:
                    bucket_balances = await conn.fetch(
                        """
                        SELECT bucket_id, available_cash_cents
                        FROM account_cash_buckets
                        WHERE account_id = $1
                        ORDER BY available_cash_cents DESC, bucket_id ASC
                        """,
                        account_id,
                    )
                    total_avail = sum(int(r["available_cash_cents"]) for r in bucket_balances)
                    if total_avail < amount:
                        raise SubmissionError("account update failed (inactive or insufficient funds)")
                    remaining = amount
                    for r in bucket_balances:
                        if remaining <= 0:
                            break
                        avail = int(r["available_cash_cents"])
                        take = avail if avail < remaining else remaining
                        if take <= 0:
                            continue
                        result = await conn.execute(
                            """
                            UPDATE account_cash_buckets
                            SET available_cash_cents = available_cash_cents - $3,
                                updated_at = now()
                            WHERE account_id = $1
                              AND bucket_id  = $2
                              AND available_cash_cents >= $3
                            """,
                            account_id,
                            int(r["bucket_id"]),
                            take,
                        )
                        if result != "UPDATE 1":
                            raise RetryableOCCError("bucket changed concurrently during withdraw")
                        remaining -= take
                    if remaining > 0:
                        raise RetryableOCCError("withdraw could not be satisfied across buckets")
                    cash_delta = -amount

                await conn.execute(
                    """
                    INSERT INTO cash_transactions (
                        cash_txn_id, account_id, type, amount_cents, status,
                        external_ref, notes, completed_at
                    )
                    VALUES ($1, $2, $3, $4, 'COMPLETED', $5, $6, now())
                    """,
                    cash_txn_id,
                    account_id,
                    tx_type,
                    amount,
                    external_ref,
                    notes,
                )

                await conn.execute(
                    """
                    INSERT INTO ledger_entries (
                        entry_id, account_id, cash_txn_id,
                        cash_delta_cents, locked_cash_delta_cents,
                        yes_share_delta, no_share_delta, locked_yes_delta, locked_no_delta,
                        reason, notes
                    )
                    VALUES ($1, $2, $3, $4, 0, 0, 0, 0, 0, $5, $6)
                    """,
                    str(uuid.uuid4()),
                    account_id,
                    cash_txn_id,
                    cash_delta,
                    tx_type,
                    notes or f"{tx_type} via rpc",
                )

                return {
                    "idempotent": False,
                    "cash_txn_id": cash_txn_id,
                    "type": tx_type,
                    "amount_cents": amount,
                    "status": "COMPLETED",
                }

    async def list_markets(self, payload: dict[str, Any]) -> dict[str, Any]:
        limit = _safe_limit(payload.get("limit"), 100)
        search = str(payload.get("search", "")).strip()
        statuses_payload = payload.get("statuses")

        clauses = []
        args: list[Any] = []
        idx = 1
        if search:
            clauses.append(f"(m.slug ILIKE ${idx} OR m.title ILIKE ${idx})")
            args.append(f"%{search}%")
            idx += 1
        if statuses_payload is not None:
            if not isinstance(statuses_payload, Sequence) or isinstance(statuses_payload, (str, bytes)):
                raise SubmissionError("statuses must be an array of strings")
            statuses = [str(s).upper() for s in statuses_payload]
            if statuses:
                placeholders = ",".join(f"${idx + i}" for i in range(len(statuses)))
                clauses.append(f"m.status IN ({placeholders})")
                args.extend(statuses)
                idx += len(statuses)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        args.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    m.market_id,
                    m.slug,
                    m.title,
                    m.description,
                    m.status,
                    m.tick_size_cents,
                    m.min_price_cents,
                    m.max_price_cents,
                    m.close_time,
                    m.resolve_time,
                    m.created_by,
                    m.created_at,
                    (
                        SELECT price_cents
                        FROM orders o
                        WHERE o.market_id = m.market_id
                          AND o.side = 'YES'
                          AND o.status IN ('ACCEPTED','OPEN','PARTIALLY_FILLED')
                          AND o.remaining_qty > 0
                        ORDER BY o.price_cents DESC, o.global_seq ASC
                        LIMIT 1
                    ) AS best_yes_bid,
                    (
                        SELECT price_cents
                        FROM orders o
                        WHERE o.market_id = m.market_id
                          AND o.side = 'NO'
                          AND o.status IN ('ACCEPTED','OPEN','PARTIALLY_FILLED')
                          AND o.remaining_qty > 0
                        ORDER BY o.price_cents DESC, o.global_seq ASC
                        LIMIT 1
                    ) AS best_no_bid,
                    (
                        SELECT yes_price_cents
                        FROM trades t
                        WHERE t.market_id = m.market_id
                        ORDER BY t.match_seq DESC
                        LIMIT 1
                    ) AS last_trade_yes_price,
                    (
                        SELECT created_at
                        FROM trades t
                        WHERE t.market_id = m.market_id
                        ORDER BY t.match_seq DESC
                        LIMIT 1
                    ) AS last_trade_at,
                    (
                        SELECT COALESCE(SUM(o.remaining_qty), 0)
                        FROM orders o
                        WHERE o.market_id = m.market_id
                          AND o.status IN ('ACCEPTED','OPEN','PARTIALLY_FILLED')
                          AND o.remaining_qty > 0
                    ) AS open_interest_qty
                FROM markets m
                {where_sql}
                ORDER BY
                    CASE m.status
                        WHEN 'ACTIVE' THEN 0
                        WHEN 'HALTED' THEN 1
                        WHEN 'CLOSED' THEN 2
                        WHEN 'DRAFT' THEN 3
                        ELSE 4
                    END,
                    COALESCE(m.resolve_time, m.close_time, m.created_at) DESC
                LIMIT ${idx}
                """,
                *args,
            )

        markets = []
        for row in rows:
            best_yes_bid = None if row["best_yes_bid"] is None else int(row["best_yes_bid"])
            best_no_bid = None if row["best_no_bid"] is None else int(row["best_no_bid"])
            last_trade_yes_price = None if row["last_trade_yes_price"] is None else int(row["last_trade_yes_price"])
            implied_yes_ask = None if best_no_bid is None else 100 - best_no_bid
            if last_trade_yes_price is not None:
                live_yes_price = last_trade_yes_price
            elif best_yes_bid is not None and implied_yes_ask is not None:
                live_yes_price = int(round((best_yes_bid + implied_yes_ask) / 2))
            else:
                live_yes_price = best_yes_bid if best_yes_bid is not None else implied_yes_ask
            markets.append(
                {
                    "market_id": str(row["market_id"]),
                    "slug": str(row["slug"]),
                    "title": str(row["title"]),
                    "description": row["description"],
                    "status": str(row["status"]),
                    "tick_size_cents": int(row["tick_size_cents"]),
                    "min_price_cents": int(row["min_price_cents"]),
                    "max_price_cents": int(row["max_price_cents"]),
                    "close_time": None if row["close_time"] is None else row["close_time"].isoformat(),
                    "resolve_time": None if row["resolve_time"] is None else row["resolve_time"].isoformat(),
                    "created_by": None if row["created_by"] is None else str(row["created_by"]),
                    "created_at": row["created_at"].isoformat(),
                    "best_yes_bid": best_yes_bid,
                    "best_no_bid": best_no_bid,
                    "implied_yes_ask": implied_yes_ask,
                    "last_trade_yes_price": last_trade_yes_price,
                    "last_trade_no_price": None if last_trade_yes_price is None else 100 - last_trade_yes_price,
                    "last_trade_at": None if row["last_trade_at"] is None else row["last_trade_at"].isoformat(),
                    "open_interest_qty": int(row["open_interest_qty"]),
                    "live_yes_price": live_yes_price,
                    "live_no_price": None if live_yes_price is None else 100 - live_yes_price,
                }
            )
        return {"markets": markets}

    async def get_market_history(self, payload: dict[str, Any]) -> dict[str, Any]:
        market_id = _require_uuid(payload.get("market_id"), "market_id")
        limit = _safe_limit(payload.get("limit"), 250, max_limit=2000)

        async with self.pool.acquire() as conn:
            market = await conn.fetchrow(
                """
                SELECT market_id, slug, title, status
                FROM markets
                WHERE market_id = $1
                """,
                market_id,
            )
            if not market:
                raise SubmissionError("market not found")

            rows = await conn.fetch(
                """
                SELECT match_seq, yes_price_cents, qty, created_at
                FROM trades
                WHERE market_id = $1
                ORDER BY match_seq DESC
                LIMIT $2
                """,
                market_id,
                limit,
            )

        points = [
            {
                "match_seq": int(r["match_seq"]),
                "yes_price_cents": int(r["yes_price_cents"]),
                "no_price_cents": 100 - int(r["yes_price_cents"]),
                "qty": int(r["qty"]),
                "created_at": r["created_at"].isoformat(),
            }
            for r in reversed(rows)
        ]
        return {
            "market_id": str(market["market_id"]),
            "slug": str(market["slug"]),
            "title": str(market["title"]),
            "status": str(market["status"]),
            "points": points,
        }

    async def create_market(self, payload: dict[str, Any]) -> dict[str, Any]:
        market_id = _require_uuid(payload.get("market_id"), "market_id")
        slug = str(payload.get("slug", "")).strip()
        title = str(payload.get("title", "")).strip()
        if not slug:
            raise SubmissionError("slug is required")
        if not title:
            raise SubmissionError("title is required")

        status = str(payload.get("status", "DRAFT")).upper()
        if status not in {"DRAFT", "ACTIVE", "HALTED", "CLOSED", "RESOLVED", "CANCELLED"}:
            raise SubmissionError("invalid market status")

        tick_size = int(payload.get("tick_size_cents", 1))
        min_price = int(payload.get("min_price_cents", 1))
        max_price = int(payload.get("max_price_cents", 99))
        account_id = _require_uuid(payload.get("account_id"), "account_id")
        account_session_token = _require_non_empty_str(payload.get("account_session_token"), "account_session_token")

        if tick_size <= 0:
            raise SubmissionError("tick_size_cents must be > 0")
        if min_price < 0 or max_price > 100 or min_price > max_price:
            raise SubmissionError("invalid price bounds")

        description = payload.get("description")
        close_time = payload.get("close_time")
        resolve_time = payload.get("resolve_time")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                admin_account = await self._require_admin_account_session(conn, account_id, account_session_token)
                created_by = admin_account["account_id"]
                existing = await conn.fetchrow(
                    "SELECT market_id, slug, status FROM markets WHERE market_id = $1",
                    market_id,
                )
                if existing:
                    return {
                        "idempotent": True,
                        "market_id": str(existing["market_id"]),
                        "slug": str(existing["slug"]),
                        "status": str(existing["status"]),
                    }

                row = await conn.fetchrow(
                    """
                    INSERT INTO markets (
                        market_id, slug, title, description, status,
                        tick_size_cents, min_price_cents, max_price_cents,
                        close_time, resolve_time, created_by
                    )
                    VALUES (
                        $1, $2, $3, $4, $5,
                        $6, $7, $8,
                        $9::timestamptz, $10::timestamptz, $11
                    )
                    RETURNING market_id, slug, status
                    """,
                    market_id,
                    slug,
                    title,
                    description,
                    status,
                    tick_size,
                    min_price,
                    max_price,
                    close_time,
                    resolve_time,
                    created_by,
                )
                self.market_cache.invalidate(market_id)
                return {
                    "idempotent": False,
                    "market_id": str(row["market_id"]),
                    "slug": str(row["slug"]),
                    "status": str(row["status"]),
                }

    async def update_market_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        market_id = _require_uuid(payload.get("market_id"), "market_id")
        new_status = str(payload.get("new_status", "")).upper()
        if new_status not in {"DRAFT", "ACTIVE", "HALTED", "CLOSED", "RESOLVED", "CANCELLED"}:
            raise SubmissionError("invalid new_status")
        account_id = _require_uuid(payload.get("account_id"), "account_id")
        account_session_token = _require_non_empty_str(payload.get("account_session_token"), "account_session_token")

        allowed = {
            "DRAFT": {"ACTIVE", "CANCELLED"},
            "ACTIVE": {"HALTED", "CLOSED", "RESOLVED", "CANCELLED"},
            "HALTED": {"ACTIVE", "CLOSED", "CANCELLED"},
            "CLOSED": {"RESOLVED", "CANCELLED"},
            "RESOLVED": set(),
            "CANCELLED": set(),
        }

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await self._require_admin_account_session(conn, account_id, account_session_token)
                market = await conn.fetchrow(
                    "SELECT market_id, status FROM markets WHERE market_id = $1",
                    market_id,
                )
                if not market:
                    raise SubmissionError("market not found")
                current = str(market["status"])

                if current == new_status:
                    return {
                        "idempotent": True,
                        "market_id": market_id,
                        "old_status": current,
                        "new_status": new_status,
                    }

                if new_status not in allowed.get(current, set()):
                    raise SubmissionError(f"invalid transition: {current} -> {new_status}")

                updated = await conn.execute(
                    "UPDATE markets SET status = $2 WHERE market_id = $1",
                    market_id,
                    new_status,
                )
                if updated != "UPDATE 1":
                    raise RetryableOCCError("market changed concurrently")

                self.market_cache.invalidate(market_id)
                return {
                    "idempotent": False,
                    "market_id": market_id,
                    "old_status": current,
                    "new_status": new_status,
                }

    async def resolve_market(self, payload: dict[str, Any]) -> dict[str, Any]:
        resolution_id = _require_uuid(payload.get("resolution_id"), "resolution_id")
        market_id = _require_uuid(payload.get("market_id"), "market_id")
        outcome = str(payload.get("outcome", "")).upper()
        if outcome not in {"YES", "NO", "CANCELLED"}:
            raise SubmissionError("outcome must be YES, NO, or CANCELLED")
        account_id = _require_uuid(payload.get("account_id"), "account_id")
        account_session_token = _require_non_empty_str(payload.get("account_session_token"), "account_session_token")
        notes = payload.get("notes")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                admin_account = await self._require_admin_account_session(conn, account_id, account_session_token)
                resolved_by = admin_account["account_id"]
                market = await conn.fetchrow(
                    "SELECT market_id, status FROM markets WHERE market_id = $1",
                    market_id,
                )
                if not market:
                    raise SubmissionError("market not found")

                existing = await conn.fetchrow(
                    "SELECT resolution_id, outcome FROM market_resolutions WHERE market_id = $1",
                    market_id,
                )
                if existing:
                    if str(existing["outcome"]) != outcome:
                        raise SubmissionError("market already resolved with a different outcome")
                    return {
                        "idempotent": True,
                        "resolution_id": str(existing["resolution_id"]),
                        "market_id": market_id,
                        "outcome": str(existing["outcome"]),
                    }

                await conn.execute(
                    """
                    INSERT INTO market_resolutions (resolution_id, market_id, outcome, resolved_by, notes)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    resolution_id,
                    market_id,
                    outcome,
                    resolved_by,
                    notes,
                )
                market_status = "CANCELLED" if outcome == "CANCELLED" else "RESOLVED"
                await conn.execute(
                    "UPDATE markets SET status = $2, resolve_time = COALESCE(resolve_time, now()) WHERE market_id = $1",
                    market_id,
                    market_status,
                )
                self.market_cache.invalidate(market_id)
                return {
                    "idempotent": False,
                    "resolution_id": resolution_id,
                    "market_id": market_id,
                    "outcome": outcome,
                    "market_status": market_status,
                }

    async def get_order_book(self, payload: dict[str, Any]) -> dict[str, Any]:
        market_id = _require_uuid(payload.get("market_id"), "market_id")
        depth = _safe_limit(payload.get("depth"), 10, max_limit=100)

        async with self.pool.acquire() as conn:
            yes_rows = await conn.fetch(
                """
                SELECT price_cents, SUM(remaining_qty) AS qty
                FROM orders
                WHERE market_id = $1
                  AND side = 'YES'
                  AND status IN ('ACCEPTED','OPEN','PARTIALLY_FILLED')
                  AND remaining_qty > 0
                GROUP BY price_cents
                ORDER BY price_cents DESC
                LIMIT $2
                """,
                market_id,
                depth,
            )
            no_rows = await conn.fetch(
                """
                SELECT price_cents, SUM(remaining_qty) AS qty
                FROM orders
                WHERE market_id = $1
                  AND side = 'NO'
                  AND status IN ('ACCEPTED','OPEN','PARTIALLY_FILLED')
                  AND remaining_qty > 0
                GROUP BY price_cents
                ORDER BY price_cents DESC
                LIMIT $2
                """,
                market_id,
                depth,
            )

        yes_levels = [{"price_cents": int(r["price_cents"]), "qty": int(r["qty"])} for r in yes_rows]
        no_levels = [{"price_cents": int(r["price_cents"]), "qty": int(r["qty"])} for r in no_rows]

        return {
            "market_id": market_id,
            "yes": yes_levels,
            "no": no_levels,
            "best_yes_bid": yes_levels[0]["price_cents"] if yes_levels else None,
            "best_no_bid": no_levels[0]["price_cents"] if no_levels else None,
        }

    async def _submit_order_once(self, req: SubmitOrderRequest) -> dict[str, Any]:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                return await self._submit_order_tx(conn, req)

    async def _submit_order_tx(self, conn: asyncpg.Connection, req: SubmitOrderRequest) -> dict[str, Any]:
        needed_cash = req.qty * req.price_cents
        await self.market_cache.validate(req.market_id, req.price_cents)

        token_hash = hash_session_token(req.account_session_token)
        session_pre_validated = self.session_cache.is_valid(req.account_id, token_hash)

        entry_id = str(uuid.uuid4())
        notes = f"TCP order lock for {req.request_id}"
        bucket_id = _bucket_for_uuid(req.request_id)

        try:
            if session_pre_validated:
                row = await self._submit_order_cte(
                    conn,
                    req=req,
                    needed_cash=needed_cash,
                    bucket_id=bucket_id,
                    entry_id=entry_id,
                    notes=notes,
                    token_hash=None,
                )
            else:
                row = await self._submit_order_cte(
                    conn,
                    req=req,
                    needed_cash=needed_cash,
                    bucket_id=bucket_id,
                    entry_id=entry_id,
                    notes=notes,
                    token_hash=token_hash,
                )
        except asyncpg.PostgresError as exc:
            if _is_occ_error(exc):
                raise RetryableOCCError from exc
            raise SubmissionError(f"database error: {exc}") from exc

        if row is None:
            raise SubmissionError("submit_order returned no result row")

        idempotent = bool(row["idempotent"])
        session_ok = bool(row["session_ok"])
        funds_ok = bool(row["funds_ok"])

        if not idempotent and not session_ok:
            raise SubmissionError("account authentication failed")

        if not idempotent and not funds_ok:
            # Picked bucket may be empty even though sum-of-buckets covers it.
            row = await self._submit_order_cte_fallback(
                conn,
                req=req,
                needed_cash=needed_cash,
                entry_id=entry_id,
                notes=notes,
            )
            if row is None or not bool(row["funds_ok"]):
                raise SubmissionError("insufficient funds or inactive account")
            idempotent = False

        if not session_pre_validated and session_ok:
            self.session_cache.remember(req.account_id, token_hash)

        return {
            "idempotent": idempotent,
            "request_id": str(row["request_id"]),
            "global_seq": int(row["global_seq"]),
            "status": str(row["status"]),
            "remaining_qty": int(row["remaining_qty"]),
        }

    async def _submit_order_cte(
        self,
        conn: asyncpg.Connection,
        *,
        req: SubmitOrderRequest,
        needed_cash: int,
        bucket_id: int,
        entry_id: str,
        notes: str,
        token_hash: str | None,
    ) -> Any:
        if token_hash is None:
            sql = """
                WITH
                  existing AS (
                    SELECT request_id, global_seq, status, remaining_qty
                    FROM orders
                    WHERE request_id = $1
                  ),
                  acct_bkt AS (
                    UPDATE account_cash_buckets
                    SET available_cash_cents = available_cash_cents - $9,
                        locked_cash_cents    = locked_cash_cents    + $9,
                        updated_at = now()
                    WHERE account_id = $2
                      AND bucket_id  = $12
                      AND available_cash_cents >= $9
                      AND NOT EXISTS (SELECT 1 FROM existing)
                    RETURNING bucket_id
                  ),
                  ord AS (
                    INSERT INTO orders (
                        request_id, account_id, market_id, side, order_type,
                        time_in_force, qty, remaining_qty, price_cents,
                        ingress_ts_ns, status, lock_bucket_id
                    )
                    SELECT $1, $2, $3, $4, 'LIMIT', $5, $6, $6, $7, $8, 'ACCEPTED', bucket_id
                    FROM acct_bkt
                    RETURNING request_id, global_seq, status, remaining_qty
                  ),
                  led AS (
                    INSERT INTO ledger_entries (
                        entry_id, account_id, market_id, order_id,
                        cash_delta_cents, locked_cash_delta_cents,
                        yes_share_delta, no_share_delta, locked_yes_delta, locked_no_delta,
                        reason, notes
                    )
                    SELECT $10, $2, $3, $1, 0, $9, 0, 0, 0, 0, 'ORDER_LOCK', $11
                    FROM ord
                    RETURNING entry_id
                  ),
                  q AS (
                    INSERT INTO match_work_queue (market_id, queued_at)
                    SELECT $3, now()
                    FROM ord
                    ON CONFLICT (market_id) DO NOTHING
                    RETURNING market_id
                  )
                SELECT
                  EXISTS(SELECT 1 FROM existing) AS idempotent,
                  TRUE                           AS session_ok,
                  EXISTS(SELECT 1 FROM acct_bkt) AS funds_ok,
                  EXISTS(SELECT 1 FROM led)      AS ledger_inserted,
                  COALESCE((SELECT request_id::text FROM existing),
                           (SELECT request_id::text FROM ord))    AS request_id,
                  COALESCE((SELECT global_seq      FROM existing),
                           (SELECT global_seq      FROM ord))     AS global_seq,
                  COALESCE((SELECT status          FROM existing),
                           (SELECT status          FROM ord))     AS status,
                  COALESCE((SELECT remaining_qty   FROM existing),
                           (SELECT remaining_qty   FROM ord))     AS remaining_qty
            """
            return await conn.fetchrow(
                sql,
                req.request_id,
                req.account_id,
                req.market_id,
                req.side,
                req.time_in_force,
                req.qty,
                req.price_cents,
                req.ingress_ts_ns,
                needed_cash,
                entry_id,
                notes,
                bucket_id,
            )

        sql = """
            WITH
              existing AS (
                SELECT request_id, global_seq, status, remaining_qty
                FROM orders
                WHERE request_id = $1
              ),
              session_check AS (
                UPDATE account_auth_sessions
                SET last_used_at = now()
                WHERE account_id = $2
                  AND token_hash = $13
                  AND revoked_at IS NULL
                  AND expires_at > now()
                RETURNING session_id
              ),
              acct_bkt AS (
                UPDATE account_cash_buckets
                SET available_cash_cents = available_cash_cents - $9,
                    locked_cash_cents    = locked_cash_cents    + $9,
                    updated_at = now()
                WHERE account_id = $2
                  AND bucket_id  = $12
                  AND available_cash_cents >= $9
                  AND NOT EXISTS (SELECT 1 FROM existing)
                  AND EXISTS (SELECT 1 FROM session_check)
                RETURNING bucket_id
              ),
              ord AS (
                INSERT INTO orders (
                    request_id, account_id, market_id, side, order_type,
                    time_in_force, qty, remaining_qty, price_cents,
                    ingress_ts_ns, status, lock_bucket_id
                )
                SELECT $1, $2, $3, $4, 'LIMIT', $5, $6, $6, $7, $8, 'ACCEPTED', bucket_id
                FROM acct_bkt
                RETURNING request_id, global_seq, status, remaining_qty
              ),
              led AS (
                INSERT INTO ledger_entries (
                    entry_id, account_id, market_id, order_id,
                    cash_delta_cents, locked_cash_delta_cents,
                    yes_share_delta, no_share_delta, locked_yes_delta, locked_no_delta,
                    reason, notes
                )
                SELECT $10, $2, $3, $1, 0, $9, 0, 0, 0, 0, 'ORDER_LOCK', $11
                FROM ord
                RETURNING entry_id
              ),
              q AS (
                INSERT INTO match_work_queue (market_id, queued_at)
                SELECT $3, now()
                FROM ord
                ON CONFLICT (market_id) DO NOTHING
                RETURNING market_id
              )
            SELECT
              EXISTS(SELECT 1 FROM existing)      AS idempotent,
              EXISTS(SELECT 1 FROM session_check) AS session_ok,
              EXISTS(SELECT 1 FROM acct_bkt)      AS funds_ok,
              EXISTS(SELECT 1 FROM led)           AS ledger_inserted,
              COALESCE((SELECT request_id::text FROM existing),
                       (SELECT request_id::text FROM ord))    AS request_id,
              COALESCE((SELECT global_seq      FROM existing),
                       (SELECT global_seq      FROM ord))     AS global_seq,
              COALESCE((SELECT status          FROM existing),
                       (SELECT status          FROM ord))     AS status,
              COALESCE((SELECT remaining_qty   FROM existing),
                       (SELECT remaining_qty   FROM ord))     AS remaining_qty
        """
        return await conn.fetchrow(
            sql,
            req.request_id,
            req.account_id,
            req.market_id,
            req.side,
            req.time_in_force,
            req.qty,
            req.price_cents,
            req.ingress_ts_ns,
            needed_cash,
            entry_id,
            notes,
            bucket_id,
            token_hash,
        )

    async def _submit_order_cte_fallback(
        self,
        conn: asyncpg.Connection,
        *,
        req: SubmitOrderRequest,
        needed_cash: int,
        entry_id: str,
        notes: str,
    ) -> Any:
        sql = """
            WITH
              existing AS (
                SELECT request_id, global_seq, status, remaining_qty
                FROM orders
                WHERE request_id = $1
              ),
              chosen AS (
                SELECT bucket_id
                FROM account_cash_buckets
                WHERE account_id = $2
                  AND available_cash_cents >= $9
                ORDER BY available_cash_cents DESC, bucket_id ASC
                LIMIT 1
              ),
              acct_bkt AS (
                UPDATE account_cash_buckets
                SET available_cash_cents = available_cash_cents - $9,
                    locked_cash_cents    = locked_cash_cents    + $9,
                    updated_at = now()
                WHERE account_id = $2
                  AND bucket_id  = (SELECT bucket_id FROM chosen)
                  AND available_cash_cents >= $9
                  AND NOT EXISTS (SELECT 1 FROM existing)
                RETURNING bucket_id
              ),
              ord AS (
                INSERT INTO orders (
                    request_id, account_id, market_id, side, order_type,
                    time_in_force, qty, remaining_qty, price_cents,
                    ingress_ts_ns, status, lock_bucket_id
                )
                SELECT $1, $2, $3, $4, 'LIMIT', $5, $6, $6, $7, $8, 'ACCEPTED', bucket_id
                FROM acct_bkt
                RETURNING request_id, global_seq, status, remaining_qty
              ),
              led AS (
                INSERT INTO ledger_entries (
                    entry_id, account_id, market_id, order_id,
                    cash_delta_cents, locked_cash_delta_cents,
                    yes_share_delta, no_share_delta, locked_yes_delta, locked_no_delta,
                    reason, notes
                )
                SELECT $10, $2, $3, $1, 0, $9, 0, 0, 0, 0, 'ORDER_LOCK', $11
                FROM ord
                RETURNING entry_id
              ),
              q AS (
                INSERT INTO match_work_queue (market_id, queued_at)
                SELECT $3, now()
                FROM ord
                ON CONFLICT (market_id) DO NOTHING
                RETURNING market_id
              )
            SELECT
              EXISTS(SELECT 1 FROM existing) AS idempotent,
              TRUE                           AS session_ok,
              EXISTS(SELECT 1 FROM acct_bkt) AS funds_ok,
              EXISTS(SELECT 1 FROM led)      AS ledger_inserted,
              COALESCE((SELECT request_id::text FROM existing),
                       (SELECT request_id::text FROM ord))    AS request_id,
              COALESCE((SELECT global_seq      FROM existing),
                       (SELECT global_seq      FROM ord))     AS global_seq,
              COALESCE((SELECT status          FROM existing),
                       (SELECT status          FROM ord))     AS status,
              COALESCE((SELECT remaining_qty   FROM existing),
                       (SELECT remaining_qty   FROM ord))     AS remaining_qty
        """
        return await conn.fetchrow(
            sql,
            req.request_id,
            req.account_id,
            req.market_id,
            req.side,
            req.time_in_force,
            req.qty,
            req.price_cents,
            req.ingress_ts_ns,
            needed_cash,
            entry_id,
            notes,
        )

    async def _cancel_order_once(self, req: CancelOrderRequest) -> dict[str, Any]:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                return await self._cancel_order_tx(
                    conn,
                    cancel_id=req.cancel_id,
                    order_id=req.order_id,
                    account_id=req.account_id,
                    account_session_token=req.account_session_token,
                    reason=req.reason,
                )

    async def _cancel_order_tx(
        self,
        conn: asyncpg.Connection,
        *,
        cancel_id: str,
        order_id: str,
        account_id: str,
        account_session_token: str,
        reason: str | None,
    ) -> dict[str, Any]:
        try:
            existing_cancel = await conn.fetchrow(
                """
                SELECT cancel_id, order_id
                FROM order_cancels
                WHERE cancel_id = $1
                """,
                cancel_id,
            )
            if existing_cancel:
                if str(existing_cancel["order_id"]) != order_id:
                    raise SubmissionError("cancel_id already used for a different order")
                existing_order = await conn.fetchrow(
                    """
                    SELECT request_id, status, remaining_qty
                    FROM orders
                    WHERE request_id = $1
                    """,
                    existing_cancel["order_id"],
                )
                if not existing_order:
                    raise SubmissionError("idempotent cancel references missing order")
                return {
                    "idempotent": True,
                    "cancel_id": str(existing_cancel["cancel_id"]),
                    "order_id": str(existing_order["request_id"]),
                    "status": str(existing_order["status"]),
                    "remaining_qty": int(existing_order["remaining_qty"]),
                    "unlocked_cash_cents": 0,
                }

            await self._require_account_session(conn, account_id, account_session_token)

            order_row = await conn.fetchrow(
                """
                SELECT request_id, account_id, market_id, price_cents, remaining_qty, status, lock_bucket_id
                FROM orders
                WHERE request_id = $1
                """,
                order_id,
            )
            if not order_row:
                raise SubmissionError("order not found")
            if str(order_row["account_id"]) != account_id:
                raise SubmissionError("order does not belong to account")
            if order_row["status"] not in OPEN_ORDER_STATUSES:
                raise SubmissionError(f"order status is {order_row['status']}, not cancellable")
            if int(order_row["remaining_qty"]) <= 0:
                raise SubmissionError("order has no remaining quantity to cancel")

            unlock_cash = int(order_row["remaining_qty"]) * int(order_row["price_cents"])
            # Legacy orders may have a NULL lock_bucket_id.
            lock_bucket_id = order_row["lock_bucket_id"]
            if lock_bucket_id is None:
                lock_bucket_id = _bucket_for_uuid(order_id)
            else:
                lock_bucket_id = int(lock_bucket_id)

            order_update = await conn.execute(
                """
                UPDATE orders
                SET
                    remaining_qty = 0,
                    status = 'CANCELLED',
                    updated_at = now()
                WHERE request_id = $1
                  AND account_id = $2
                  AND status IN ('ACCEPTED','OPEN','PARTIALLY_FILLED')
                  AND remaining_qty > 0
                """,
                order_id,
                account_id,
            )
            if order_update != "UPDATE 1":
                raise RetryableOCCError("order changed concurrently during cancel")

            account_update = await conn.execute(
                """
                UPDATE account_cash_buckets
                SET
                    available_cash_cents = available_cash_cents + $3,
                    locked_cash_cents    = locked_cash_cents    - $3,
                    updated_at = now()
                WHERE account_id = $1
                  AND bucket_id  = $2
                  AND locked_cash_cents >= $3
                """,
                account_id,
                lock_bucket_id,
                unlock_cash,
            )
            if account_update != "UPDATE 1":
                raise RetryableOCCError("cash bucket changed concurrently during cancel")

            await conn.execute(
                """
                INSERT INTO order_cancels (
                    cancel_id, order_id, account_id, reason
                )
                VALUES ($1, $2, $3, $4)
                """,
                cancel_id,
                order_id,
                account_id,
                reason,
            )

            await conn.execute(
                """
                INSERT INTO ledger_entries (
                    entry_id, account_id, market_id, order_id,
                    cash_delta_cents, locked_cash_delta_cents,
                    yes_share_delta, no_share_delta, locked_yes_delta, locked_no_delta,
                    reason, notes
                )
                VALUES ($1, $2, $3, $4, 0, $5, 0, 0, 0, 0, 'ORDER_UNLOCK', $6)
                """,
                str(uuid.uuid4()),
                account_id,
                str(order_row["market_id"]),
                order_id,
                -unlock_cash,
                f"Cancel order unlock for {order_id}",
            )

            return {
                "idempotent": False,
                "cancel_id": cancel_id,
                "order_id": order_id,
                "status": "CANCELLED",
                "remaining_qty": 0,
                "unlocked_cash_cents": unlock_cash,
            }
        except asyncpg.PostgresError as exc:
            if _is_occ_error(exc):
                raise RetryableOCCError from exc
            raise SubmissionError(f"database error: {exc}") from exc


async def create_pool(dsn: str, min_size: int, max_size: int) -> asyncpg.Pool:
    async def _dsql_safe_reset(_conn: asyncpg.Connection) -> None:
        return None

    return await asyncpg.create_pool(
        dsn=dsn,
        min_size=min_size,
        max_size=max_size,
        reset=_dsql_safe_reset,
    )
