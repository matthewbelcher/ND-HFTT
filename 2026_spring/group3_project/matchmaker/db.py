import asyncio
import random
import uuid
from datetime import datetime, timezone

import asyncpg

from metrics import METRICS
from models import Order


# Must agree with client/client-listener/db.py ACCOUNT_CASH_BUCKETS.
ACCOUNT_CASH_BUCKETS = 16


def _bucket_for_uuid(value: str) -> int:
    try:
        return int(uuid.UUID(value).int) % ACCOUNT_CASH_BUCKETS
    except Exception:
        return abs(hash(value)) % ACCOUNT_CASH_BUCKETS


def _market_shard(market_id: str, total_instances: int) -> int:
    if total_instances <= 1:
        return 0
    try:
        return int(uuid.UUID(market_id).int) % total_instances
    except Exception:
        return abs(hash(market_id)) % total_instances


class MatchError(Exception):
    """Raised when a match cannot be executed against the database.

    Surfaced by :meth:`MatchRepository.execute_match` when optimistic
    concurrency retries are exhausted or a non-retryable database error
    occurs. Callers should treat this as a transient failure for the
    specific market and continue scanning other candidate markets.

    The optional ``market_id`` attribute records which market the failed
    match belonged to, which is the most useful diagnostic context for a
    matcher that services many markets concurrently.
    """

    def __init__(self, message: str, *, market_id: str | None = None) -> None:
        super().__init__(message)
        self.market_id = market_id

    def __str__(self) -> str:
        base = super().__str__()
        if self.market_id is not None:
            return f"{base} (market_id={self.market_id})"
        return base


class RetryableOCCError(Exception):
    pass


def _is_occ_error(exc: Exception) -> bool:
    return isinstance(exc, asyncpg.PostgresError) and exc.sqlstate in {"40001", "40P01"}


async def create_pool(dsn: str, min_size: int, max_size: int) -> asyncpg.Pool:
    async def _dsql_safe_reset(_conn: asyncpg.Connection) -> None:
        # Aurora DSQL does not support pg_advisory_unlock_all().
        return None

    return await asyncpg.create_pool(
        dsn=dsn,
        min_size=min_size,
        max_size=max_size,
        reset=_dsql_safe_reset,
    )


class MatchRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def ensure_schema(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS matchmaker_market_leases (
                    market_id UUID PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    lease_expires_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX ASYNC IF NOT EXISTS matchmaker_market_leases_exp_idx
                ON matchmaker_market_leases(lease_expires_at)
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS match_work_queue (
                    market_id UUID PRIMARY KEY,
                    queued_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX ASYNC IF NOT EXISTS match_work_queue_queued_idx
                ON match_work_queue(queued_at)
                """
            )

    async def claim_pending_markets(
        self,
        limit: int,
        *,
        instance_index: int = 0,
        total_instances: int = 1,
    ) -> list[str]:
        if total_instances <= 0:
            total_instances = 1
        if limit <= 0:
            return []
        async with self.pool.acquire() as conn:
            # Over-fetch so the shard filter still has enough rows to work with.
            over_fetch = max(limit * max(1, total_instances), limit)
            rows = await conn.fetch(
                """
                SELECT market_id
                FROM match_work_queue
                ORDER BY queued_at
                LIMIT $1
                """,
                over_fetch,
            )
            mine: list[str] = []
            for r in rows:
                mid = str(r["market_id"])
                if total_instances == 1 or _market_shard(mid, total_instances) == instance_index:
                    mine.append(mid)
                    if len(mine) >= limit:
                        break
            if not mine:
                METRICS.set_gauge("matchmaker_queue_depth_sample", float(len(rows)))
                return []
            # IN-list with positional placeholders avoids asyncpg's array
            # type introspection, which DSQL rejects (it tries SET jit=off).
            placeholders = ",".join(f"${i + 1}::uuid" for i in range(len(mine)))
            deleted = await conn.fetch(
                f"""
                DELETE FROM match_work_queue
                WHERE market_id IN ({placeholders})
                RETURNING market_id, queued_at
                """,
                *mine,
            )
            METRICS.incr("matchmaker_claimed_markets_total", float(len(deleted)))
            METRICS.set_gauge("matchmaker_queue_depth_sample", float(len(rows)))
            now = datetime.now(timezone.utc)
            max_lag_ms = 0.0
            for r in deleted:
                qd = r["queued_at"]
                lag_ms = (now - qd).total_seconds() * 1000.0 if qd else 0.0
                if lag_ms > max_lag_ms:
                    max_lag_ms = lag_ms
            METRICS.set_gauge("matchmaker_queue_max_lag_ms", max_lag_ms)
            return [str(r["market_id"]) for r in deleted]

    async def requeue_market(self, market_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO match_work_queue (market_id, queued_at)
                VALUES ($1, now())
                ON CONFLICT (market_id) DO NOTHING
                """,
                market_id,
            )

    async def requeue_unmatched_markets(
        self,
        limit: int,
        *,
        instance_index: int = 0,
        total_instances: int = 1,
    ) -> int:
        # Safety net for queue entries lost to a matcher crash.
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT o.market_id
                FROM orders o
                JOIN markets m ON m.market_id = o.market_id
                LEFT JOIN match_work_queue q ON q.market_id = o.market_id
                WHERE o.status IN ('ACCEPTED', 'OPEN', 'PARTIALLY_FILLED')
                  AND o.remaining_qty > 0
                  AND m.status = 'ACTIVE'
                  AND (m.close_time IS NULL OR m.close_time > now())
                  AND q.market_id IS NULL
                LIMIT $1
                """,
                limit * max(1, total_instances),
            )
            inserted = 0
            for r in rows:
                mid = str(r["market_id"])
                if total_instances > 1 and _market_shard(mid, total_instances) != instance_index:
                    continue
                result = await conn.execute(
                    """
                    INSERT INTO match_work_queue (market_id, queued_at)
                    VALUES ($1, now())
                    ON CONFLICT (market_id) DO NOTHING
                    """,
                    mid,
                )
                if result.startswith("INSERT 0 1"):
                    inserted += 1
            METRICS.incr("matchmaker_requeued_markets_total", float(inserted))
            return inserted

    async def try_acquire_market_lease(
        self,
        market_id: str,
        owner_id: str,
        lease_seconds: int,
    ) -> bool:
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO matchmaker_market_leases (
                            market_id,
                            owner_id,
                            lease_expires_at,
                            updated_at
                        )
                        VALUES ($1, $2, now() + ($3::text || ' seconds')::interval, now())
                        ON CONFLICT (market_id)
                        DO UPDATE
                        SET owner_id = EXCLUDED.owner_id,
                            lease_expires_at = EXCLUDED.lease_expires_at,
                            updated_at = now()
                        WHERE matchmaker_market_leases.lease_expires_at < now()
                           OR matchmaker_market_leases.owner_id = EXCLUDED.owner_id
                        RETURNING owner_id
                        """,
                        market_id,
                        owner_id,
                        f"{lease_seconds} seconds",
                    )
                    return bool(row and row["owner_id"] == owner_id)
            except asyncpg.PostgresError as exc:
                if not _is_occ_error(exc):
                    raise
                if attempt == max_attempts:
                    return False
                delay = (0.02 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.02)
                await asyncio.sleep(delay)
        return False

    async def load_market_orders(self, market_id: str, limit: int) -> list[Order]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT request_id, global_seq, account_id, market_id, side,
                       remaining_qty, price_cents, status
                FROM orders
                WHERE market_id = $1
                  AND status IN ('ACCEPTED', 'OPEN', 'PARTIALLY_FILLED')
                  AND remaining_qty > 0
                ORDER BY side, price_cents DESC, global_seq
                LIMIT $2
                """,
                market_id,
                limit,
            )
        return [
            Order(
                request_id=str(r["request_id"]),
                global_seq=r["global_seq"],
                account_id=str(r["account_id"]),
                market_id=str(r["market_id"]),
                side=str(r["side"]),
                remaining_qty=r["remaining_qty"],
                price_cents=r["price_cents"],
                status=str(r["status"]),
            )
            for r in rows
        ]

    async def execute_match(
        self,
        market_id: str,
        yes_order_id: str,
        no_order_id: str,
        qty: int,
    ) -> bool:
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._execute_match_once(market_id, yes_order_id, no_order_id, qty)
            except RetryableOCCError:
                if attempt == max_attempts:
                    raise MatchError(
                        "database conflict after retries", market_id=market_id
                    )
                delay = (0.02 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.02)
                await asyncio.sleep(delay)

    async def _execute_match_once(
        self,
        market_id: str,
        yes_order_id: str,
        no_order_id: str,
        qty: int,
    ) -> bool:
        if qty <= 0:
            return False

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    market = await conn.fetchrow(
                        """
                        SELECT market_id, status, close_time
                        FROM markets
                        WHERE market_id = $1
                        """,
                        market_id,
                    )
                    if not market:
                        return False
                    if market["status"] != "ACTIVE":
                        return False
                    if market["close_time"] is not None and market["close_time"] <= datetime.now(timezone.utc):
                        return False

                    yes_row = await conn.fetchrow(
                        """
                        SELECT request_id, account_id, market_id, side, remaining_qty,
                               price_cents, status, global_seq, lock_bucket_id
                        FROM orders
                        WHERE request_id = $1
                        """,
                        yes_order_id,
                    )
                    no_row = await conn.fetchrow(
                        """
                        SELECT request_id, account_id, market_id, side, remaining_qty,
                               price_cents, status, global_seq, lock_bucket_id
                        FROM orders
                        WHERE request_id = $1
                        """,
                        no_order_id,
                    )

                    if not yes_row or not no_row:
                        return False
                    if str(yes_row["market_id"]) != market_id or str(no_row["market_id"]) != market_id:
                        return False
                    if yes_row["side"] != "YES" or no_row["side"] != "NO":
                        return False
                    if yes_row["status"] not in {"ACCEPTED", "OPEN", "PARTIALLY_FILLED"}:
                        return False
                    if no_row["status"] not in {"ACCEPTED", "OPEN", "PARTIALLY_FILLED"}:
                        return False
                    if yes_row["remaining_qty"] < qty or no_row["remaining_qty"] < qty:
                        return False
                    if yes_row["price_cents"] + no_row["price_cents"] < 100:
                        return False
                    if yes_row["account_id"] == no_row["account_id"]:
                        # Prevent self-trading and avoid duplicate trade_parties keys.
                        return False

                    yes_fill_cost = qty * yes_row["price_cents"]
                    no_fill_cost = qty * no_row["price_cents"]
                    yes_bucket = (
                        int(yes_row["lock_bucket_id"])
                        if yes_row["lock_bucket_id"] is not None
                        else _bucket_for_uuid(yes_order_id)
                    )
                    no_bucket = (
                        int(no_row["lock_bucket_id"])
                        if no_row["lock_bucket_id"] is not None
                        else _bucket_for_uuid(no_order_id)
                    )

                    update_row = await conn.fetchrow(
                        """
                        WITH
                          yes_bkt AS (
                            UPDATE account_cash_buckets
                            SET locked_cash_cents = locked_cash_cents - $2,
                                updated_at = now()
                            WHERE account_id = $1 AND bucket_id = $3
                              AND locked_cash_cents >= $2
                            RETURNING bucket_id
                          ),
                          no_bkt AS (
                            UPDATE account_cash_buckets
                            SET locked_cash_cents = locked_cash_cents - $5,
                                updated_at = now()
                            WHERE account_id = $4 AND bucket_id = $6
                              AND locked_cash_cents >= $5
                            RETURNING bucket_id
                          ),
                          yes_ord AS (
                            UPDATE orders
                            SET remaining_qty = remaining_qty - $8,
                                status = CASE
                                    WHEN remaining_qty - $8 = 0 THEN 'FILLED'
                                    ELSE 'PARTIALLY_FILLED'
                                END,
                                updated_at = now()
                            WHERE request_id = $7
                              AND status IN ('ACCEPTED', 'OPEN', 'PARTIALLY_FILLED')
                              AND remaining_qty >= $8
                            RETURNING request_id
                          ),
                          no_ord AS (
                            UPDATE orders
                            SET remaining_qty = remaining_qty - $8,
                                status = CASE
                                    WHEN remaining_qty - $8 = 0 THEN 'FILLED'
                                    ELSE 'PARTIALLY_FILLED'
                                END,
                                updated_at = now()
                            WHERE request_id = $9
                              AND status IN ('ACCEPTED', 'OPEN', 'PARTIALLY_FILLED')
                              AND remaining_qty >= $8
                            RETURNING request_id
                          )
                        SELECT
                          EXISTS(SELECT 1 FROM yes_bkt) AS yes_bkt_ok,
                          EXISTS(SELECT 1 FROM no_bkt)  AS no_bkt_ok,
                          EXISTS(SELECT 1 FROM yes_ord) AS yes_ord_ok,
                          EXISTS(SELECT 1 FROM no_ord)  AS no_ord_ok
                        """,
                        yes_row["account_id"], yes_fill_cost, yes_bucket,
                        no_row["account_id"], no_fill_cost, no_bucket,
                        yes_order_id, qty, no_order_id,
                    )
                    if update_row is None:
                        raise RetryableOCCError("match update returned no row")
                    if not bool(update_row["yes_bkt_ok"]) or not bool(update_row["no_bkt_ok"]):
                        return False
                    if not bool(update_row["yes_ord_ok"]) or not bool(update_row["no_ord_ok"]):
                        raise RetryableOCCError("order changed concurrently")

                    trade_id = str(uuid.uuid4())
                    resting_order_id = yes_order_id
                    aggressing_order_id = no_order_id
                    if yes_row["global_seq"] > no_row["global_seq"]:
                        resting_order_id = no_order_id
                        aggressing_order_id = yes_order_id

                    yes_entry_id = str(uuid.uuid4())
                    no_entry_id = str(uuid.uuid4())
                    yes_notes = f"Match execution for YES order {yes_order_id}"
                    no_notes = f"Match execution for NO order {no_order_id}"

                    await conn.execute(
                        """
                        WITH
                          trade_ins AS (
                            INSERT INTO trades (
                                trade_id, market_id, resting_order_id, aggressing_order_id,
                                qty, yes_price_cents
                            )
                            VALUES ($1, $2, $3, $4, $5, $6)
                            RETURNING trade_id
                          ),
                          yes_pos AS (
                            INSERT INTO positions (account_id, market_id, yes_shares, no_shares, updated_at)
                            VALUES ($7, $2, $5, 0, now())
                            ON CONFLICT (account_id, market_id) DO UPDATE
                            SET yes_shares = positions.yes_shares + EXCLUDED.yes_shares,
                                updated_at = now()
                            RETURNING account_id
                          ),
                          no_pos AS (
                            INSERT INTO positions (account_id, market_id, yes_shares, no_shares, updated_at)
                            VALUES ($8, $2, 0, $5, now())
                            ON CONFLICT (account_id, market_id) DO UPDATE
                            SET no_shares = positions.no_shares + EXCLUDED.no_shares,
                                updated_at = now()
                            RETURNING account_id
                          ),
                          parties AS (
                            INSERT INTO trade_parties (
                                trade_id, account_id, role, side_acquired, qty, cash_delta_cents
                            )
                            VALUES
                              ($1, $7, 'BUYER', 'YES', $5, $9),
                              ($1, $8, 'BUYER', 'NO',  $5, $10)
                            RETURNING trade_id
                          )
                        INSERT INTO ledger_entries (
                            entry_id, account_id, market_id, order_id, trade_id,
                            cash_delta_cents, locked_cash_delta_cents,
                            yes_share_delta, no_share_delta,
                            locked_yes_delta, locked_no_delta,
                            reason, notes
                        )
                        VALUES
                          ($11, $7, $2, $12, $1, 0, $9, $5, 0, 0, 0, 'TRADE_EXECUTION', $13),
                          ($14, $8, $2, $15, $1, 0, $10, 0, $5, 0, 0, 'TRADE_EXECUTION', $16)
                        """,
                        trade_id, market_id, resting_order_id, aggressing_order_id,
                        qty, yes_row["price_cents"],
                        yes_row["account_id"], no_row["account_id"],
                        -yes_fill_cost, -no_fill_cost,
                        yes_entry_id, yes_order_id, yes_notes,
                        no_entry_id, no_order_id, no_notes,
                    )

                    return True
        except asyncpg.PostgresError as exc:
            if _is_occ_error(exc):
                raise RetryableOCCError from exc
            raise MatchError(f"database error: {exc}", market_id=market_id) from exc
